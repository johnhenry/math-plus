/**
 * @johnhenry/math-plus-tensor-cpu — the CPU reference implementation of the
 * @johnhenry/tensor-backend `Backend` contract (issue #144; decided by
 * RFC 0001 §12 Q3: math-plus owns the CPU reference).
 *
 *   import { createCpuBackend } from "@johnhenry/math-plus-tensor-cpu";
 *   const cpu = createCpuBackend();
 *   const x = await cpu.fromHost(host("f32", [2, 3], [1, 2, 3, 4, 5, 6]));
 *   const y = cpu.scope(() => cpu.softmax(cpu.scale(x, 2), -1));
 *   const out = await cpu.read(y);
 *
 * Every computation runs on tensor-core's flat kernels
 * (`@johnhenry/math-plus-tensor-core/kernels`) or on tensor-core's `Tensor`
 * (`matmul`), so GEMM, softmax, LayerNorm, RoPE, attention, GELU/erf and the
 * reductions have one implementation in math-plus. This file only does
 * shapes, dtypes, broadcasting bookkeeping and tensor lifetime.
 *
 * - Storage: float tensors are Float32Array (f16/bf16 host data is widened
 *   on `fromHost`, per the contract's widening rule); i32 → Int32Array;
 *   bool → Uint8Array (0/1). `supports("f16" | "bf16")` is false and
 *   `cast` to them throws.
 * - Contiguous row-major storage; `reshape` and same-dtype `cast` share the
 *   buffer (tensors are immutable, so sharing is safe).
 * - Eager and synchronous; `fromHost` copies at call time and `read`
 *   resolves immediately (both return Promises, per the contract).
 * - Reductions, softmax, LayerNorm, attention and matmul accumulate in f64.
 *
 * Drop-in compatible with laya-js's `@johnhenry/backend-cpu@0.2` (same
 * `createCpuBackend`, `CpuTensor`, `CpuBackend`, dtype rules), which is to
 * become a re-export of this package.
 */
import type { Backend, DType, HostTensor, NumericsOp, Shape, Tensor } from "@johnhenry/tensor-backend";
import { sizeOf, toF32 } from "@johnhenry/tensor-backend";
import { broadcastShapes, Tensor as MpTensor } from "@johnhenry/math-plus-tensor-core";
import * as K from "@johnhenry/math-plus-tensor-core/kernels";

/** The canonical scalar erf / erfc / exact GELU the backend uses (from @johnhenry/math-plus-special via tensor-core). */
export { erf, erfc, geluErf as geluScalar } from "@johnhenry/math-plus-tensor-core";

type CpuDType = "f32" | "i32" | "bool";
type Data = Float32Array | Int32Array | Uint8Array;

export class CpuTensor implements Tensor {
  readonly shape: readonly number[];
  readonly dtype: DType;
  #data: Data | null;
  constructor(shape: readonly number[], dtype: CpuDType, data: Data) {
    this.shape = shape;
    this.dtype = dtype;
    this.#data = data;
  }
  /** Backing storage (row-major, possibly shared with other tensors: never mutate). Throws after dispose. */
  get data(): Data {
    const d = this.#data;
    if (d === null) throw new Error("tensor-cpu: tensor used after dispose");
    return d;
  }
  get disposed(): boolean {
    return this.#data === null;
  }
  /** @internal */
  _release(): void {
    this.#data = null;
  }
}

/** The CPU backend: the contract with every optional op implemented natively (except `compile`). */
export type CpuBackend = Backend<CpuTensor> &
  Required<Pick<Backend<CpuTensor>, NumericsOp | "geglu" | "meanPool" | "flush" | "destroy">>;

/** Creates an independent CPU backend (no shared global state). */
export function createCpuBackend(): CpuBackend {
  return new CpuBackendImpl();
}

// ---------------------------------------------------------------- helpers
function normAxis(axis: number, rank: number): number {
  const a = axis < 0 ? axis + rank : axis;
  if (!Number.isInteger(a) || a < 0 || a >= rank) throw new RangeError(`tensor-cpu: axis ${axis} out of range for rank ${rank}`);
  return a;
}

function stridesOf(shape: readonly number[]): number[] {
  const s = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i]!;
  }
  return s;
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function alloc(dtype: CpuDType, n: number): Data {
  return dtype === "f32" ? new Float32Array(n) : dtype === "i32" ? new Int32Array(n) : new Uint8Array(n);
}

/** `[outer, dim, inner]` around a (normalized) axis. */
function around(shape: readonly number[], a: number): [number, number, number] {
  let outer = 1, inner = 1;
  for (let i = 0; i < a; i++) outer *= shape[i]!;
  for (let i = a + 1; i < shape.length; i++) inner *= shape[i]!;
  return [outer, shape[a]!, inner];
}

function reducedShape(shape: readonly number[], a: number, keepDims: boolean): number[] {
  return keepDims ? shape.map((d, i) => (i === a ? 1 : d)) : shape.filter((_, i) => i !== a);
}

/**
 * Walks `outShape` row by row (all axes but the last) for every operand
 * broadcast to it, calling `row(offsets, lastStrides, outOffset, n)`.
 */
function broadcastRows(
  outShape: readonly number[],
  shapes: readonly (readonly number[])[],
  row: (offs: number[], steps: number[], oo: number, n: number) => void,
): void {
  const rank = outShape.length;
  const n = rank ? outShape[rank - 1]! : 1;
  const strides = shapes.map((s) => K.alignedStrides(s, outShape));
  const rows = strides.map((st) => K.rowOffsets(outShape, st));
  const steps = strides.map((st) => (rank ? st[rank - 1]! : 0));
  const offs = new Array<number>(shapes.length);
  const count = rows[0]!.length;
  for (let r = 0; r < count; r++) {
    for (let k = 0; k < rows.length; k++) offs[k] = rows[k]![r]!;
    row(offs, steps, r * n, n);
  }
}

const ZERO = new Float32Array(1);

// ---------------------------------------------------------------- backend
class CpuBackendImpl implements CpuBackend {
  readonly name = "cpu";
  readonly #scopes: Set<CpuTensor>[] = [];

  supports(dtype: DType): boolean {
    return dtype === "f32" || dtype === "i32" || dtype === "bool";
  }

  #make(shape: readonly number[], dtype: CpuDType, data: Data): CpuTensor {
    const t = new CpuTensor(Object.freeze([...shape]), dtype, data);
    const top = this.#scopes[this.#scopes.length - 1];
    if (top) top.add(t);
    return t;
  }

  // ---- transfer / lifetime
  async fromHost(t: HostTensor): Promise<CpuTensor> {
    return this.#upload(t);
  }

  #upload(t: HostTensor): CpuTensor {
    const n = sizeOf(t.shape);
    if (t.data.length !== n) throw new RangeError(`tensor-cpu: fromHost ${t.data.length} values for shape [${t.shape}]`);
    switch (t.dtype) {
      case "f32":
        return this.#make(t.shape, "f32", Float32Array.from(t.data as Float32Array));
      case "f16":
      case "bf16":
        // Widened to f32 storage (the contract's documented widening rule); toF32 allocates.
        return this.#make(t.shape, "f32", toF32(t));
      case "i32":
        return this.#make(t.shape, "i32", Int32Array.from(t.data as Int32Array));
      case "bool": {
        const out = new Uint8Array(n);
        K.compareStrided(K.CMP_NE, t.data as Uint8Array, 0, 1, ZERO, 0, 0, out, 0, n);
        return this.#make(t.shape, "bool", out);
      }
      default:
        throw new TypeError(`tensor-cpu: fromHost: unknown dtype ${String((t as HostTensor).dtype)}`);
    }
  }

  async read(t: CpuTensor): Promise<HostTensor> {
    return { dtype: t.dtype, shape: [...t.shape], data: t.data.slice() };
  }

  dispose(t: CpuTensor): void {
    t._release();
  }

  scope<R>(fn: () => R): R {
    const set = new Set<CpuTensor>();
    this.#scopes.push(set);
    let result: R;
    try {
      result = fn();
    } catch (e) {
      this.#scopes.pop();
      for (const t of set) t._release();
      throw e;
    }
    this.#scopes.pop();
    const keep = new Set<CpuTensor>();
    const visit = (v: unknown) => {
      if (v instanceof CpuTensor) keep.add(v);
    };
    if (result instanceof CpuTensor) keep.add(result);
    else if (Array.isArray(result)) result.forEach(visit);
    else if (result && typeof result === "object") Object.values(result as object).forEach(visit);
    const parent = this.#scopes[this.#scopes.length - 1];
    for (const t of set) {
      if (keep.has(t)) parent?.add(t);
      else t._release();
    }
    return result;
  }

  flush(): void {}

  destroy(): void {
    for (const s of this.#scopes) for (const t of s) t._release();
    this.#scopes.length = 0;
  }

  // ---- shape
  reshape(x: CpuTensor, shape: Shape): CpuTensor {
    const n = sizeOf(x.shape);
    const out = [...shape];
    const neg = out.indexOf(-1);
    if (neg >= 0) {
      let known = 1;
      out.forEach((d, i) => {
        if (i !== neg) known *= d;
      });
      out[neg] = n / known;
    }
    if (sizeOf(out) !== n || out.some((d) => !Number.isInteger(d) || d < 0)) {
      throw new RangeError(`tensor-cpu: reshape [${x.shape}] -> [${shape}]`);
    }
    return this.#make(out, x.dtype as CpuDType, x.data);
  }

  transpose(x: CpuTensor, perm: readonly number[]): CpuTensor {
    const rank = x.shape.length;
    if (perm.length !== rank) throw new RangeError(`tensor-cpu: transpose perm [${perm}] for rank ${rank}`);
    const p = perm.map((a) => normAxis(a, rank));
    if (new Set(p).size !== rank) throw new RangeError(`tensor-cpu: transpose perm [${perm}] is not a permutation`);
    const inStr = stridesOf(x.shape);
    const outShape = p.map((a) => x.shape[a]!);
    const out = alloc(x.dtype as CpuDType, x.data.length);
    if (out.length) K.stridedCopy(x.data, 0, outShape, p.map((a) => inStr[a]!), out);
    return this.#make(outShape, x.dtype as CpuDType, out);
  }

  slice(x: CpuTensor, begin: readonly number[], end: readonly number[]): CpuTensor {
    const rank = x.shape.length;
    const inStr = stridesOf(x.shape);
    const outShape: number[] = [];
    let base = 0;
    for (let i = 0; i < rank; i++) {
      const d = x.shape[i]!;
      let lo = i < begin.length ? begin[i]! : 0;
      let hi = i < end.length ? end[i]! : d;
      if (lo < 0) lo += d;
      if (hi < 0) hi += d;
      lo = Math.min(Math.max(lo, 0), d);
      hi = Math.min(Math.max(hi, lo), d);
      base += lo * inStr[i]!;
      outShape.push(hi - lo);
    }
    const out = alloc(x.dtype as CpuDType, sizeOf(outShape));
    if (out.length) K.stridedCopy(x.data, base, outShape, inStr, out);
    return this.#make(outShape, x.dtype as CpuDType, out);
  }

  split(x: CpuTensor, parts: number, axis: number): CpuTensor[] {
    const a = normAxis(axis, x.shape.length);
    const d = x.shape[a]!;
    if (!(parts > 0) || d % parts) throw new RangeError(`tensor-cpu: cannot split ${d} into ${parts}`);
    const step = d / parts;
    const res: CpuTensor[] = [];
    for (let p = 0; p < parts; p++) {
      const begin = x.shape.map((_, i) => (i === a ? p * step : 0));
      const end = x.shape.map((s, i) => (i === a ? (p + 1) * step : s));
      res.push(this.slice(x, begin, end));
    }
    return res;
  }

  concat(xs: readonly CpuTensor[], axis: number): CpuTensor {
    if (xs.length === 0) throw new RangeError("tensor-cpu: concat of nothing");
    const first = xs[0]!;
    const rank = first.shape.length;
    const a = normAxis(axis, rank);
    // Mixed inputs promote like the elementwise ops: any f32 → f32, else i32 if any i32.
    let dtype = first.dtype as CpuDType;
    for (const x of xs) {
      if (x.shape.length !== rank) throw new RangeError("tensor-cpu: concat rank mismatch");
      for (let i = 0; i < rank; i++) if (i !== a && x.shape[i] !== first.shape[i]) throw new RangeError("tensor-cpu: concat shape mismatch");
      if (x.dtype === "f32") dtype = "f32";
      else if (x.dtype === "i32" && dtype === "bool") dtype = "i32";
    }
    const outShape = [...first.shape];
    outShape[a] = xs.reduce((s, x) => s + x.shape[a]!, 0);
    const [outer, , inner] = around(outShape, a);
    const out = alloc(dtype, sizeOf(outShape));
    const rowLen = outShape[a]! * inner;
    let colOff = 0;
    for (const x of xs) {
      const chunk = x.shape[a]! * inner;
      const src = x.data;
      for (let o = 0; o < outer; o++) out.set(src.subarray(o * chunk, (o + 1) * chunk), o * rowLen + colOff);
      colOff += chunk;
    }
    return this.#make(outShape, dtype, out);
  }

  cast(x: CpuTensor, dtype: DType): CpuTensor {
    if (dtype === "f16" || dtype === "bf16") throw new TypeError(`tensor-cpu: ${dtype} is not supported (f32 reference backend)`);
    if (dtype !== "f32" && dtype !== "i32" && dtype !== "bool") throw new TypeError(`tensor-cpu: unknown dtype ${String(dtype)}`);
    if (dtype === x.dtype) return this.#make(x.shape, dtype, x.data);
    const src = x.data;
    const out = alloc(dtype, src.length);
    // Typed-array stores truncate toward zero for i32 (NaN → 0), exactly Math.trunc + ToInt32.
    if (dtype === "bool") K.compareStrided(K.CMP_NE, src, 0, 1, ZERO, 0, 0, out as Uint8Array, 0, src.length);
    else out.set(src);
    return this.#make(x.shape, dtype, out);
  }

  // ---- elementwise
  /** Result dtype of an arithmetic op: f32 if either side is (or for div), else i32. */
  #arith(a: CpuTensor, b: CpuTensor, op: number): CpuDType {
    return a.dtype === "f32" || b.dtype === "f32" || op === K.OP_DIV || op === K.OP_POW ? "f32" : "i32";
  }

  #binary(a: CpuTensor, b: CpuTensor, op: number): CpuTensor {
    const dtype = this.#arith(a, b, op);
    const A = a.data, B = b.data;
    if (sameShape(a.shape, b.shape)) {
      const out = alloc(dtype, A.length);
      if (op <= K.OP_DIV) K.binaryFlat(op, A, 0, B, 0, out, 0, A.length);
      else K.binaryStrided(op, A, 0, 1, B, 0, 1, out, 0, A.length);
      return this.#make(a.shape, dtype, out);
    }
    const outShape = broadcastShapes(a.shape, b.shape);
    const out = alloc(dtype, sizeOf(outShape));
    if (out.length === 0) return this.#make(outShape, dtype, out);
    if (op <= K.OP_DIV && B.length === 1 && sameShape(a.shape, outShape)) {
      K.binaryScalarRight(op, A, 0, B[0]!, out, 0, out.length);
      return this.#make(outShape, dtype, out);
    }
    broadcastRows(outShape, [a.shape, b.shape], (o, s, oo, n) => K.binaryStrided(op, A, o[0]!, s[0]!, B, o[1]!, s[1]!, out, oo, n));
    return this.#make(outShape, dtype, out);
  }

  add(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#binary(a, b, K.OP_ADD); }
  sub(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#binary(a, b, K.OP_SUB); }
  mul(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#binary(a, b, K.OP_MUL); }
  div(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#binary(a, b, K.OP_DIV); }
  maximum(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#binary(a, b, K.OP_MAXIMUM); }
  pow(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#binary(a, b, K.OP_POW); }

  where(cond: CpuTensor, a: CpuTensor, b: CpuTensor): CpuTensor {
    const outShape = broadcastShapes(broadcastShapes(cond.shape, a.shape), b.shape);
    const dtype: CpuDType = a.dtype === "f32" || b.dtype === "f32" ? "f32" : a.dtype === "i32" || b.dtype === "i32" ? "i32" : "bool";
    const out = alloc(dtype, sizeOf(outShape));
    if (out.length === 0) return this.#make(outShape, dtype, out);
    const C = cond.data, A = a.data, B = b.data;
    broadcastRows(outShape, [cond.shape, a.shape, b.shape], (o, s, oo, n) =>
      K.whereStrided(C, o[0]!, s[0]!, A, o[1]!, s[1]!, B, o[2]!, s[2]!, out, oo, n),
    );
    return this.#make(outShape, dtype, out);
  }

  scale(x: CpuTensor, s: number): CpuTensor {
    const src = x.data, out = new Float32Array(src.length);
    K.binaryScalarRight(K.OP_MUL, src, 0, s, out, 0, src.length);
    return this.#make(x.shape, "f32", out);
  }

  /** Float-valued unary op (integer/bool input computes in f32), evaluated in f64 and rounded once. */
  #unaryF(x: CpuTensor, op: number): CpuTensor {
    const src = x.data, out = new Float32Array(src.length);
    K.unaryFlat(op, src, 0, out, 0, src.length);
    return this.#make(x.shape, "f32", out);
  }
  /** Keeps i32 (two's-complement wrap, like MLX); f32 stays f32; bool → i32. */
  #unaryKeep(x: CpuTensor, op: number): CpuTensor {
    if (x.dtype === "f32") return this.#unaryF(x, op);
    const src = x.data, out = new Int32Array(src.length);
    K.unaryFlat(op, src, 0, out, 0, src.length);
    return this.#make(x.shape, "i32", out);
  }

  exp(x: CpuTensor): CpuTensor { return this.#unaryF(x, K.UN_EXP); }
  log(x: CpuTensor): CpuTensor { return this.#unaryF(x, K.UN_LOG); }
  relu(x: CpuTensor): CpuTensor { return this.#unaryF(x, K.UN_RELU); }
  gelu(x: CpuTensor): CpuTensor { return this.#unaryF(x, K.UN_GELU); }
  sqrt(x: CpuTensor): CpuTensor { return this.#unaryF(x, K.UN_SQRT); }
  rsqrt(x: CpuTensor): CpuTensor { return this.#unaryF(x, K.UN_RSQRT); }
  tanh(x: CpuTensor): CpuTensor { return this.#unaryF(x, K.UN_TANH); }
  sigmoid(x: CpuTensor): CpuTensor { return this.#unaryF(x, K.UN_SIGMOID); }
  erf(x: CpuTensor): CpuTensor { return this.#unaryF(x, K.UN_ERF); }
  neg(x: CpuTensor): CpuTensor { return this.#unaryKeep(x, K.UN_NEG); }
  abs(x: CpuTensor): CpuTensor { return this.#unaryKeep(x, K.UN_ABS); }

  // ---- comparisons / logical (general numerics)
  #compare(a: CpuTensor, b: CpuTensor, op: number): CpuTensor {
    const outShape = broadcastShapes(a.shape, b.shape);
    const out = new Uint8Array(sizeOf(outShape));
    if (out.length === 0) return this.#make(outShape, "bool", out);
    const A = a.data, B = b.data;
    if (sameShape(a.shape, b.shape)) K.compareStrided(op, A, 0, 1, B, 0, 1, out, 0, out.length);
    else broadcastRows(outShape, [a.shape, b.shape], (o, s, oo, n) => K.compareStrided(op, A, o[0]!, s[0]!, B, o[1]!, s[1]!, out, oo, n));
    return this.#make(outShape, "bool", out);
  }
  equal(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#compare(a, b, K.CMP_EQ); }
  notEqual(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#compare(a, b, K.CMP_NE); }
  less(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#compare(a, b, K.CMP_LT); }
  lessEqual(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#compare(a, b, K.CMP_LTE); }
  greater(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#compare(a, b, K.CMP_GT); }
  greaterEqual(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#compare(a, b, K.CMP_GTE); }
  logicalAnd(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#compare(a, b, K.CMP_AND); }
  logicalOr(a: CpuTensor, b: CpuTensor): CpuTensor { return this.#compare(a, b, K.CMP_OR); }
  logicalNot(x: CpuTensor): CpuTensor {
    const src = x.data, out = new Uint8Array(src.length);
    K.compareStrided(K.CMP_EQ, src, 0, 1, ZERO, 0, 0, out, 0, src.length);
    return this.#make(x.shape, "bool", out);
  }

  // ---- reductions
  #axisReduce(x: CpuTensor, axis: number, keepDims: boolean, dtype: CpuDType, what: string): { out: Data; shape: number[]; outer: number; dim: number; inner: number } {
    const a = normAxis(axis, x.shape.length);
    const [outer, dim, inner] = around(x.shape, a);
    if (dim === 0 && what !== "sum" && what !== "mean") throw new RangeError(`tensor-cpu: ${what} over an empty axis`);
    return { out: alloc(dtype, outer * inner), shape: reducedShape(x.shape, a, keepDims), outer, dim, inner };
  }

  sum(x: CpuTensor, axis: number, keepDims = false): CpuTensor {
    const dtype: CpuDType = x.dtype === "f32" ? "f32" : "i32";
    const r = this.#axisReduce(x, axis, keepDims, dtype, "sum");
    K.sumAxis(x.data, 0, r.outer, r.dim, r.inner, r.out, false);
    return this.#make(r.shape, dtype, r.out);
  }
  mean(x: CpuTensor, axis: number, keepDims = false): CpuTensor {
    const r = this.#axisReduce(x, axis, keepDims, "f32", "mean");
    K.sumAxis(x.data, 0, r.outer, r.dim, r.inner, r.out, true);
    return this.#make(r.shape, "f32", r.out);
  }
  max(x: CpuTensor, axis: number, keepDims = false): CpuTensor {
    const r = this.#axisReduce(x, axis, keepDims, x.dtype as CpuDType, "max");
    K.extremumAxis(x.data, 0, r.outer, r.dim, r.inner, r.out, true);
    return this.#make(r.shape, x.dtype as CpuDType, r.out);
  }
  min(x: CpuTensor, axis: number, keepDims = false): CpuTensor {
    const r = this.#axisReduce(x, axis, keepDims, x.dtype as CpuDType, "min");
    K.extremumAxis(x.data, 0, r.outer, r.dim, r.inner, r.out, false);
    return this.#make(r.shape, x.dtype as CpuDType, r.out);
  }
  argmax(x: CpuTensor, axis: number, keepDims = false): CpuTensor {
    const r = this.#axisReduce(x, axis, keepDims, "i32", "argmax");
    K.argExtremumAxis(x.data, 0, r.outer, r.dim, r.inner, r.out, true);
    return this.#make(r.shape, "i32", r.out);
  }
  argmin(x: CpuTensor, axis: number, keepDims = false): CpuTensor {
    const r = this.#axisReduce(x, axis, keepDims, "i32", "argmin");
    K.argExtremumAxis(x.data, 0, r.outer, r.dim, r.inner, r.out, false);
    return this.#make(r.shape, "i32", r.out);
  }

  cumsum(x: CpuTensor, axis: number): CpuTensor {
    const [outer, dim, inner] = around(x.shape, normAxis(axis, x.shape.length));
    const dtype: CpuDType = x.dtype === "f32" ? "f32" : "i32";
    const out = alloc(dtype, x.data.length);
    K.cumsumAxis(x.data, 0, outer, dim, inner, out);
    return this.#make(x.shape, dtype, out);
  }

  softmax(x: CpuTensor, axis: number): CpuTensor {
    const [outer, dim, inner] = around(x.shape, normAxis(axis, x.shape.length));
    const src = x.data instanceof Float32Array ? x.data : Float32Array.from(x.data);
    const out = new Float32Array(src.length);
    if (dim > 0) K.softmaxAxis(src, 0, outer, dim, inner, out, true);
    return this.#make(x.shape, "f32", out);
  }

  sort(x: CpuTensor, axis: number): CpuTensor {
    const [outer, dim, inner] = around(x.shape, normAxis(axis, x.shape.length));
    const out = alloc(x.dtype as CpuDType, x.data.length);
    K.sortAxis(x.data, 0, outer, dim, inner, out, alloc(x.dtype as CpuDType, dim));
    return this.#make(x.shape, x.dtype as CpuDType, out);
  }

  // ---- linear algebra & NN
  /** Float32 view of a tensor's values (integer/bool inputs convert). */
  #f32(x: CpuTensor): Float32Array {
    const d = x.data;
    return d instanceof Float32Array ? d : Float32Array.from(d);
  }

  matmul(a: CpuTensor, b: CpuTensor): CpuTensor {
    if (a.shape.length < 2 || b.shape.length < 2) throw new RangeError("tensor-cpu: matmul needs rank >= 2");
    // tensor-core's batched, broadcasting matmul (packed-f64 blocked GEMM); wraps without copying.
    const ta = MpTensor.fromTypedArray(this.#f32(a), a.shape, { dtype: "f32" });
    const tb = MpTensor.fromTypedArray(this.#f32(b), b.shape, { dtype: "f32" });
    const r = ta.matmul(tb);
    return this.#make(r.shape, "f32", r.data as Float32Array);
  }

  linear(x: CpuTensor, w: CpuTensor, b?: CpuTensor | null): CpuTensor {
    if (w.shape.length !== 2) throw new RangeError("tensor-cpu: linear weight must be [out, in]");
    const [nOut, nIn] = w.shape as [number, number];
    if (x.shape.length < 1 || x.shape[x.shape.length - 1] !== nIn) throw new RangeError(`tensor-cpu: linear [${x.shape}] with weight [${w.shape}]`);
    if (b && sizeOf(b.shape) !== nOut) throw new RangeError(`tensor-cpu: linear bias [${b.shape}] for ${nOut} outputs`);
    const X = x.data;
    const rows = nIn === 0 ? sizeOf(x.shape.slice(0, -1)) : X.length / nIn;
    const out = new Float32Array(rows * nOut);
    K.linearNT(X, 0, rows, nIn, w.data, nOut, b ? b.data : null, out);
    return this.#make([...x.shape.slice(0, -1), nOut], "f32", out);
  }

  layerNorm(x: CpuTensor, weight: CpuTensor | null, bias: CpuTensor | null, eps: number): CpuTensor {
    const D = x.shape[x.shape.length - 1]!;
    const X = x.data;
    const out = new Float32Array(X.length);
    if (D > 0) K.layerNormRows(X, 0, X.length / D, D, weight ? weight.data : null, bias ? bias.data : null, eps, out);
    return this.#make(x.shape, "f32", out);
  }

  embedding(table: CpuTensor, ids: CpuTensor): CpuTensor {
    if (table.shape.length !== 2) throw new RangeError("tensor-cpu: embedding table must be [V, D]");
    const [V, D] = table.shape as [number, number];
    const I = ids.data;
    const out = new Float32Array(I.length * D);
    K.takeRows(this.#f32(table), D, V, I, out);
    return this.#make([...ids.shape, D], "f32", out);
  }

  gatherRows(x: CpuTensor, idx: CpuTensor): CpuTensor {
    const [B, L, D] = x.shape as [number, number, number];
    const [B2, M] = idx.shape as [number, number];
    if (x.shape.length !== 3 || idx.shape.length !== 2 || B2 !== B) throw new RangeError(`tensor-cpu: gatherRows [${x.shape}] idx [${idx.shape}]`);
    const I = idx.data;
    const flat = new Float64Array(B * M);
    for (let b = 0; b < B; b++) {
      for (let m = 0; m < M; m++) {
        const raw = I[b * M + m]!;
        const r = raw < 0 ? raw + L : raw;
        if (!(r >= 0 && r < L)) throw new RangeError(`tensor-cpu: gatherRows index ${raw} out of range for length ${L}`);
        flat[b * M + m] = b * L + r;
      }
    }
    const out = new Float32Array(B * M * D);
    K.takeRows(this.#f32(x), D, B * L, flat, out);
    return this.#make([B, M, D], "f32", out);
  }

  rope(x: CpuTensor, base: number): CpuTensor {
    if (x.shape.length !== 4) throw new RangeError(`tensor-cpu: rope expects [B, H, L, Dh], got [${x.shape}]`);
    const [B, H, L, Dh] = x.shape as [number, number, number, number];
    if (Dh % 2) throw new RangeError(`tensor-cpu: rope needs an even head dim, got ${Dh}`);
    const out = new Float32Array(x.data.length);
    K.ropeHalf(x.data, 0, B * H, L, Dh, base, out);
    return this.#make(x.shape, "f32", out);
  }

  sdpa(q: CpuTensor, k: CpuTensor, v: CpuTensor, mask: CpuTensor | null, scale: number): CpuTensor {
    if (q.shape.length !== 4 || k.shape.length !== 4 || v.shape.length !== 4) throw new RangeError("tensor-cpu: sdpa expects rank-4 q/k/v");
    const [B, H, Lq, Dh] = q.shape as [number, number, number, number];
    const Hk = k.shape[1]!, Lk = k.shape[2]!, Dv = v.shape[3]!;
    if (k.shape[0] !== B || v.shape[0] !== B || v.shape[1] !== Hk || v.shape[2] !== Lk || k.shape[3] !== Dh || H % Hk) {
      throw new RangeError(`tensor-cpu: sdpa q [${q.shape}] k [${k.shape}] v [${v.shape}]`);
    }
    let m: K.AttentionMask | null = null;
    if (mask) {
      const ms = [...mask.shape];
      if (ms.length > 4) throw new RangeError(`tensor-cpu: sdpa mask [${mask.shape}] has rank > 4`);
      while (ms.length < 4) ms.unshift(1);
      const target = [B, H, Lq, Lk];
      if (broadcastShapes(ms, target).some((d, i) => d !== target[i])) throw new RangeError(`tensor-cpu: sdpa mask [${mask.shape}] does not broadcast to [${target}]`);
      m = { data: mask.data, strides: K.alignedStrides(ms, target), boolean: mask.dtype === "bool" };
    }
    const out = new Float32Array(B * H * Lq * Dv);
    K.attention(q.data, k.data, v.data, B, H, Hk, Lq, Lk, Dh, Dv, scale, m, out);
    return this.#make([B, H, Lq, Dv], "f32", out);
  }

  geglu(x: CpuTensor): CpuTensor {
    const D2 = x.shape[x.shape.length - 1]!;
    if (D2 % 2) throw new RangeError(`tensor-cpu: geglu needs an even last axis, got ${D2}`);
    const D = D2 / 2;
    const rows = D2 === 0 ? 0 : x.data.length / D2;
    const out = new Float32Array(rows * D);
    K.gegluRows(x.data, 0, rows, D, out);
    return this.#make([...x.shape.slice(0, -1), D], "f32", out);
  }

  meanPool(x: CpuTensor, mask: CpuTensor): CpuTensor {
    const [B, L, D] = x.shape as [number, number, number];
    if (x.shape.length !== 3 || mask.shape.length !== 2 || mask.shape[0] !== B || mask.shape[1] !== L) {
      throw new RangeError(`tensor-cpu: meanPool x [${x.shape}] mask [${mask.shape}]`);
    }
    const out = new Float32Array(B * D);
    K.maskedMeanPool(x.data, B, L, D, mask.data, out);
    return this.#make([B, D], "f32", out);
  }
}
