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
import { broadcastShapes, Tensor as MpTensor, type AnyTypedArray } from "@johnhenry/math-plus-tensor-core";
import * as K from "@johnhenry/math-plus-tensor-core/kernels";
import { ArrayDevice, type DeviceArray } from "./device-array.ts";

/** The canonical scalar erf / erfc / exact GELU the backend uses (from @johnhenry/math-plus-special via tensor-core). */
export { erf, erfc, geluErf as geluScalar } from "@johnhenry/math-plus-tensor-core";
/** The tensor-core `Tensor` <-> `HostTensor` bridge every math-plus device package (tensor-mlx, tensor-webgpu) uses; see host.ts. */
export { DEVICE_DTYPES, hostFromTensor, isDeviceDType, tensorFromHost, type DeviceDType } from "./host.ts";
/** The chainable device-array API every math-plus device facade (CPU, MLX, WebGPU) shares; see device-array.ts. */
export { ArrayDevice, DeviceArray, type ArrayDeviceNames, type DeviceArrayClass, type HandleOf } from "./device-array.ts";

/**
 * f16/bf16 stay excluded by design (this is an f32-based reference backend;
 * see the file docstring). Every other tensor-core dtype is supported as of
 * 2026-09-25 -- u64/i64 storage is BigUint64Array/BigInt64Array (bigint
 * elements), which the flat numeric kernels below (`K.*`) cannot operate on
 * directly, so bigint dtypes are routed through tensor-core's own `Tensor`
 * arithmetic instead (`#bigintUnary`/`#bigintBinary`), the same delegation
 * pattern `matmul` already uses for f32 GEMM.
 */
type CpuDType = "f32" | "i32" | "bool" | "u8" | "i8" | "u16" | "i16" | "u32" | "u64" | "i64" | "f64";
type Data = Float32Array | Int32Array | Uint8Array | Int8Array | Int16Array | Uint16Array | Uint32Array | Float64Array | BigUint64Array | BigInt64Array;

function isBigIntCpuDType(dtype: CpuDType): dtype is "u64" | "i64" {
  return dtype === "u64" || dtype === "i64";
}

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
  switch (dtype) {
    case "f32":
      return new Float32Array(n);
    case "i32":
      return new Int32Array(n);
    case "bool":
    case "u8":
      return new Uint8Array(n);
    case "i8":
      return new Int8Array(n);
    case "u16":
      return new Uint16Array(n);
    case "i16":
      return new Int16Array(n);
    case "u32":
      return new Uint32Array(n);
    case "f64":
      return new Float64Array(n);
    case "u64":
      return new BigUint64Array(n);
    case "i64":
      return new BigInt64Array(n);
  }
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

/** K.OP_* constant -> tensor-core Tensor method name, for the bigint/new-dtype delegation path in #binary. */
function opName(op: number): "add" | "sub" | "mul" | "div" {
  if (op === K.OP_ADD) return "add";
  if (op === K.OP_SUB) return "sub";
  if (op === K.OP_MUL) return "mul";
  if (op === K.OP_DIV) return "div";
  throw new TypeError(`tensor-cpu: internal: opName called for op ${op} (maximum/pow are intercepted before reaching this)`);
}

/** Never throws (unlike opName) -- for error-message labels only, called before maximum/pow are intercepted. */
function opLabel(op: number): string {
  if (op === K.OP_MAXIMUM) return "maximum";
  if (op === K.OP_POW) return "pow";
  try {
    return opName(op);
  } catch {
    return `op${op}`;
  }
}

// ---------------------------------------------------------------- backend
class CpuBackendImpl implements CpuBackend {
  readonly name = "cpu";
  readonly #scopes: Set<CpuTensor>[] = [];

  supports(dtype: DType): boolean {
    return dtype !== "f16" && dtype !== "bf16";
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
      case "u8":
        return this.#make(t.shape, "u8", Uint8Array.from(t.data as Uint8Array));
      case "i8":
        return this.#make(t.shape, "i8", Int8Array.from(t.data as Int8Array));
      case "u16":
        return this.#make(t.shape, "u16", Uint16Array.from(t.data as Uint16Array));
      case "i16":
        return this.#make(t.shape, "i16", Int16Array.from(t.data as Int16Array));
      case "u32":
        return this.#make(t.shape, "u32", Uint32Array.from(t.data as Uint32Array));
      case "f64":
        return this.#make(t.shape, "f64", Float64Array.from(t.data as Float64Array));
      case "u64":
        return this.#make(t.shape, "u64", BigUint64Array.from(t.data as BigUint64Array));
      case "i64":
        return this.#make(t.shape, "i64", BigInt64Array.from(t.data as BigInt64Array));
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
    // stridedCopy is a pure index-based element copy (no value arithmetic --
    // see its source), so it's safe for bigint dtypes despite K.NumArray's
    // (overly conservative) type not including them.
    if (out.length) K.stridedCopy(x.data as K.NumArray, 0, outShape, p.map((a) => inStr[a]!), out as K.NumArray);
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
    if (out.length) K.stridedCopy(x.data as K.NumArray, base, outShape, inStr, out as K.NumArray);
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
    // Mixed inputs promote like the elementwise ops, but only among the
    // original 3 dtypes (unchanged behavior): any f32 → f32, else i32 if any
    // i32. Any new (2026-09-25) dtype requires an exact match across every
    // input -- no implicit promotion rule was designed for 13-way mixing.
    let dtype = first.dtype as CpuDType;
    for (const x of xs) {
      if (x.shape.length !== rank) throw new RangeError("tensor-cpu: concat rank mismatch");
      for (let i = 0; i < rank; i++) if (i !== a && x.shape[i] !== first.shape[i]) throw new RangeError("tensor-cpu: concat shape mismatch");
      const legacy = (d: CpuDType) => d === "f32" || d === "i32" || d === "bool";
      if (legacy(x.dtype as CpuDType) && legacy(dtype)) {
        if (x.dtype === "f32") dtype = "f32";
        else if (x.dtype === "i32" && dtype === "bool") dtype = "i32";
      } else if (x.dtype !== dtype) {
        throw new TypeError(`tensor-cpu: concat requires matching dtypes for ${dtype}/${x.dtype} operands (no implicit dtype promotion) -- cast explicitly first`);
      }
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
      // Pure element copy between same-dtype arrays (enforced above for the
      // new dtypes) -- safe for bigint despite TypedArray.set()'s ArrayLike<number> typing.
      for (let o = 0; o < outer; o++) (out as Exclude<Data, BigUint64Array | BigInt64Array>).set(src.subarray(o * chunk, (o + 1) * chunk) as ArrayLike<number>, o * rowLen + colOff);
      colOff += chunk;
    }
    return this.#make(outShape, dtype, out);
  }

  cast(x: CpuTensor, dtype: DType): CpuTensor {
    if (dtype === "f16" || dtype === "bf16") throw new TypeError(`tensor-cpu: ${dtype} is not supported (f32 reference backend)`);
    if (dtype === x.dtype) return this.#make(x.shape, dtype as CpuDType, x.data);
    // bool/i32/f32 keep the original fast path (no tensor-core round-trip,
    // same behavior as before 2026-09-25). Any pair touching a new dtype
    // (including bool/i32/f32 -> a new one) delegates to tensor-core's own
    // Tensor.cast, which already handles every dtype pair correctly
    // (including bigint <-> number conversions) -- reusing the reference
    // implementation instead of re-deriving truncation/rounding rules here.
    if ((dtype === "bool" || dtype === "i32" || dtype === "f32") && !isBigIntCpuDType(x.dtype as CpuDType)) {
      const src = x.data as K.NumArray;
      const out = alloc(dtype as CpuDType, src.length);
      // Typed-array stores truncate toward zero for i32 (NaN → 0), exactly Math.trunc + ToInt32.
      if (dtype === "bool") K.compareStrided(K.CMP_NE, src, 0, 1, ZERO, 0, 0, out as Uint8Array, 0, src.length);
      else (out as Float32Array | Int32Array).set(src as ArrayLike<number>);
      return this.#make(x.shape, dtype as CpuDType, out);
    }
    const t = MpTensor.fromTypedArray(x.data as AnyTypedArray, x.shape, { dtype: x.dtype }).cast(dtype);
    return this.#make(x.shape, dtype as CpuDType, t.data as Data);
  }

  // ---- elementwise
  /** Result dtype of an arithmetic op: f32 if either side is (or for div), else i32. */
  #arith(a: CpuTensor, b: CpuTensor, op: number): CpuDType {
    return a.dtype === "f32" || b.dtype === "f32" || op === K.OP_DIV || op === K.OP_POW ? "f32" : "i32";
  }

  /** Wraps a and b in tensor-core Tensors (no copy) for delegation. Requires matching dtypes -- no implicit promotion for the new (2026-09-25) dtypes, mirroring RFC 0001 §7.2's "no implicit dtype promotion" rule. */
  #wrap(x: CpuTensor): MpTensor {
    return MpTensor.fromTypedArray(x.data as AnyTypedArray, x.shape, { dtype: x.dtype });
  }
  #requireSameDtype(a: CpuTensor, b: CpuTensor, what: string): void {
    if (a.dtype !== b.dtype) throw new TypeError(`tensor-cpu: ${what} requires matching dtypes for ${a.dtype}/${b.dtype} operands (no implicit dtype promotion) -- cast explicitly first`);
  }

  #binary(a: CpuTensor, b: CpuTensor, op: number): CpuTensor {
    // New (2026-09-25) integer/float dtypes and bigint dtypes bypass the
    // f32-or-i32 promotion rule below (that rule predates them and is kept
    // unchanged for bool/i32/f32 for backward compatibility) -- delegate to
    // tensor-core's own arithmetic, which already implements every dtype
    // pair correctly, including bigint (K's flat kernels below cannot: JS
    // throws mixing bigint with plain-number arithmetic).
    if (a.dtype !== "f32" && a.dtype !== "i32" && a.dtype !== "bool") {
      this.#requireSameDtype(a, b, opLabel(op));
      if (op === K.OP_MAXIMUM) return this.#bigintOrWideMaximum(a, b);
      if (op === K.OP_POW) throw new TypeError(`tensor-cpu: pow is not supported for dtype ${a.dtype} (no tensor-tensor pow in the host reference; only a float dtype)`);
      const r = (this.#wrap(a) as any)[opName(op)](this.#wrap(b));
      return this.#make(r.shape, r.dtype as CpuDType, r.data as Data);
    }
    if (b.dtype !== "f32" && b.dtype !== "i32" && b.dtype !== "bool") {
      this.#requireSameDtype(a, b, opLabel(op));
      return this.#binary(b, a, op); // symmetric ops only reach here (add/mul/maximum); div/sub/pow already required a.dtype match above
    }
    const dtype = this.#arith(a, b, op);
    // Both operands are provably non-bigint here (the two branches above
    // already delegated and returned for any bigint dtype).
    const A = a.data as K.NumArray, B = b.data as K.NumArray;
    if (sameShape(a.shape, b.shape)) {
      const out = alloc(dtype, A.length) as K.NumArray;
      if (op <= K.OP_DIV) K.binaryFlat(op, A, 0, B, 0, out, 0, A.length);
      else K.binaryStrided(op, A, 0, 1, B, 0, 1, out, 0, A.length);
      return this.#make(a.shape, dtype, out);
    }
    const outShape = broadcastShapes(a.shape, b.shape);
    const out = alloc(dtype, sizeOf(outShape)) as K.NumArray;
    if (out.length === 0) return this.#make(outShape, dtype, out);
    if (op <= K.OP_DIV && B.length === 1 && sameShape(a.shape, outShape)) {
      K.binaryScalarRight(op, A, 0, B[0]!, out, 0, out.length);
      return this.#make(outShape, dtype, out);
    }
    broadcastRows(outShape, [a.shape, b.shape], (o, s, oo, n) => K.binaryStrided(op, A, o[0]!, s[0]!, B, o[1]!, s[1]!, out, oo, n));
    return this.#make(outShape, dtype, out);
  }

  /** maximum has no tensor-core tensor-tensor equivalent; broadcasts and compares directly (bigint `>` works natively, no arithmetic mixing). */
  #bigintOrWideMaximum(a: CpuTensor, b: CpuTensor): CpuTensor {
    const outShape = broadcastShapes(a.shape, b.shape);
    const dtype = a.dtype as CpuDType;
    const out = alloc(dtype, sizeOf(outShape)) as Data;
    const A = a.data, B = b.data;
    const strideA = K.alignedStrides(a.shape, outShape), strideB = K.alignedStrides(b.shape, outShape);
    const idx = new Array<number>(outShape.length).fill(0);
    for (let i = 0; i < out.length; i++) {
      let oa = 0, ob = 0;
      for (let d = 0; d < outShape.length; d++) {
        oa += idx[d]! * strideA[d]!;
        ob += idx[d]! * strideB[d]!;
      }
      const av = A[oa]!, bv = B[ob]!;
      (out as any)[i] = av > bv ? av : bv;
      for (let d = outShape.length - 1; d >= 0; d--) {
        if (++idx[d]! < outShape[d]!) break;
        idx[d] = 0;
      }
    }
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
    if (isBigIntCpuDType(a.dtype as CpuDType) || isBigIntCpuDType(b.dtype as CpuDType)) {
      this.#requireSameDtype(a, b, "where");
      const dtype = a.dtype as CpuDType;
      const out = alloc(dtype, sizeOf(outShape)) as Data;
      const C = cond.data, A = a.data, B = b.data;
      const strideC = K.alignedStrides(cond.shape, outShape), strideA = K.alignedStrides(a.shape, outShape), strideB = K.alignedStrides(b.shape, outShape);
      const idx = new Array<number>(outShape.length).fill(0);
      for (let i = 0; i < out.length; i++) {
        let oc = 0, oa = 0, ob = 0;
        for (let d = 0; d < outShape.length; d++) {
          oc += idx[d]! * strideC[d]!;
          oa += idx[d]! * strideA[d]!;
          ob += idx[d]! * strideB[d]!;
        }
        (out as any)[i] = C[oc] ? A[oa] : B[ob];
        for (let d = outShape.length - 1; d >= 0; d--) {
          if (++idx[d]! < outShape[d]!) break;
          idx[d] = 0;
        }
      }
      return this.#make(outShape, dtype, out);
    }
    const dtype: CpuDType =
      a.dtype === "f32" || b.dtype === "f32" ? "f32" : a.dtype === "i32" || b.dtype === "i32" ? "i32" : a.dtype === "bool" && b.dtype === "bool" ? "bool" : (a.dtype as CpuDType);
    const out = alloc(dtype, sizeOf(outShape)) as K.NumArray;
    if (out.length === 0) return this.#make(outShape, dtype, out);
    const C = cond.data as K.NumArray, A = a.data as K.NumArray, B = b.data as K.NumArray;
    broadcastRows(outShape, [cond.shape, a.shape, b.shape], (o, s, oo, n) =>
      K.whereStrided(C, o[0]!, s[0]!, A, o[1]!, s[1]!, B, o[2]!, s[2]!, out, oo, n),
    );
    return this.#make(outShape, dtype, out);
  }

  scale(x: CpuTensor, s: number): CpuTensor {
    if (isBigIntCpuDType(x.dtype as CpuDType)) throw new TypeError(`tensor-cpu: scale is not supported for dtype ${x.dtype} (fractional scalar multiply has no bigint meaning) -- cast to a float dtype first`);
    // f64 keeps f64 precision (not downcast to f32 -- same class of bug
    // #unaryF/mean had; this one cascades further, since compose.ts's neg
    // uses scale(x,-1) for float dtypes).
    const src = x.data as K.NumArray;
    const out = x.dtype === "f64" ? new Float64Array(src.length) : new Float32Array(src.length);
    K.binaryScalarRight(K.OP_MUL, src, 0, s, out, 0, src.length);
    return this.#make(x.shape, x.dtype === "f64" ? "f64" : "f32", out);
  }

  /** Float-valued unary op (integer/bool input computes in f32), evaluated in f64 and rounded once. */
  /** f64 input keeps f64 precision throughout (not downcast to f32 -- this bug would otherwise fail exactly the tight-tolerance f64 conformance cases meant to catch it); every other dtype computes in f32 as before. */
  #unaryF(x: CpuTensor, op: number): CpuTensor {
    if (x.dtype === "f64") {
      const src = x.data as Float64Array, out = new Float64Array(src.length);
      K.unaryFlat(op, src, 0, out, 0, src.length);
      return this.#make(x.shape, "f64", out);
    }
    const src = this.#numArray(x);
    const out = new Float32Array(src.length);
    K.unaryFlat(op, src, 0, out, 0, src.length);
    return this.#make(x.shape, "f32", out);
  }
  /** Keeps the input's own dtype (two's-complement wrap for integers, like MLX); f32/f64 stay float; bool → i32. */
  #unaryKeep(x: CpuTensor, op: number): CpuTensor {
    if (x.dtype === "f32" || x.dtype === "f64") return this.#unaryFKeep(x, op);
    if (isBigIntCpuDType(x.dtype as CpuDType)) {
      const method = op === K.UN_NEG ? "neg" : "abs";
      const r = (this.#wrap(x) as any)[method]();
      return this.#make(x.shape, x.dtype as CpuDType, r.data as Data);
    }
    if (x.dtype === "bool") {
      const src = x.data as K.NumArray, out = new Int32Array(src.length);
      K.unaryFlat(op, src, 0, out, 0, src.length);
      return this.#make(x.shape, "i32", out);
    }
    // i32 and the new narrower integer dtypes (u8/i8/u16/i16/u32) all wrap
    // within their own width, matching cumsum's documented wrap rule above.
    const src = x.data as K.NumArray, out = alloc(x.dtype as CpuDType, src.length) as K.NumArray;
    K.unaryFlat(op, src, 0, out, 0, src.length);
    return this.#make(x.shape, x.dtype as CpuDType, out);
  }
  #unaryFKeep(x: CpuTensor, op: number): CpuTensor {
    const src = x.data as K.NumArray, out = x.dtype === "f64" ? new Float64Array(src.length) : new Float32Array(src.length);
    K.unaryFlat(op, src, 0, out, 0, src.length);
    return this.#make(x.shape, x.dtype as CpuDType, out);
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
    if (isBigIntCpuDType(a.dtype as CpuDType) || isBigIntCpuDType(b.dtype as CpuDType)) {
      this.#requireSameDtype(a, b, "comparison");
      const method = { [K.CMP_EQ]: "eq", [K.CMP_LT]: "lt", [K.CMP_LTE]: "lte", [K.CMP_GT]: "gt", [K.CMP_GTE]: "gte" }[op];
      if (!method) {
        // CMP_NE: no direct tensor-core method -- synthesize by inverting eq's bool output.
        const eq = (this.#wrap(a) as any).eq(this.#wrap(b));
        const out = (eq.data as Uint8Array).map((v: number) => (v ? 0 : 1));
        return this.#make(eq.shape, "bool", out);
      }
      const r = (this.#wrap(a) as any)[method](this.#wrap(b));
      return this.#make(r.shape, "bool", r.data as Uint8Array);
    }
    const out = new Uint8Array(sizeOf(outShape));
    if (out.length === 0) return this.#make(outShape, "bool", out);
    const A = a.data as K.NumArray, B = b.data as K.NumArray;
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
    const src = this.#numArray(x);
    const out = new Uint8Array(src.length);
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

  /** Delegates an axis reduction that must preserve dtype exactly (bigint dtypes only) to tensor-core, then restores keepDims via reshape. */
  #bigintReduce(x: CpuTensor, axis: number, keepDims: boolean, method: "sum" | "max" | "min" | "argmax" | "argmin" | "cumsum"): CpuTensor {
    const a = normAxis(axis, x.shape.length);
    const r = (this.#wrap(x) as any)[method](method === "cumsum" ? a : a);
    const outDtype = method === "argmax" || method === "argmin" ? "i32" : (x.dtype as CpuDType);
    let t = this.#make(r.shape as readonly number[], outDtype, r.data as Data);
    if (method !== "cumsum" && keepDims) t = this.reshape(t, reducedShape(x.shape, a, true));
    return t;
  }

  sum(x: CpuTensor, axis: number, keepDims = false): CpuTensor {
    if (isBigIntCpuDType(x.dtype as CpuDType)) return this.#bigintReduce(x, axis, keepDims, "sum");
    const dtype: CpuDType = x.dtype === "f32" ? "f32" : x.dtype === "bool" || x.dtype === "i32" ? "i32" : (x.dtype as CpuDType);
    const r = this.#axisReduce(x, axis, keepDims, dtype, "sum");
    K.sumAxis(x.data as K.NumArray, 0, r.outer, r.dim, r.inner, r.out as K.NumArray, false);
    return this.#make(r.shape, dtype, r.out);
  }
  mean(x: CpuTensor, axis: number, keepDims = false): CpuTensor {
    // f64 keeps f64 precision (not downcast to f32 -- same class of bug
    // #unaryF had); bigint converts through Number() first, like #f32/#numArray.
    const outDtype: CpuDType = x.dtype === "f64" ? "f64" : "f32";
    const src = isBigIntCpuDType(x.dtype as CpuDType) ? this.#make(x.shape, "f32", Float32Array.from(x.data as BigInt64Array | BigUint64Array, (v) => Number(v))) : x;
    const r = this.#axisReduce(src, axis, keepDims, outDtype, "mean");
    K.sumAxis(src.data as K.NumArray, 0, r.outer, r.dim, r.inner, r.out as K.NumArray, true);
    return this.#make(r.shape, outDtype, r.out);
  }
  max(x: CpuTensor, axis: number, keepDims = false): CpuTensor {
    if (isBigIntCpuDType(x.dtype as CpuDType)) return this.#bigintReduce(x, axis, keepDims, "max");
    const r = this.#axisReduce(x, axis, keepDims, x.dtype as CpuDType, "max");
    K.extremumAxis(x.data as K.NumArray, 0, r.outer, r.dim, r.inner, r.out as K.NumArray, true);
    return this.#make(r.shape, x.dtype as CpuDType, r.out);
  }
  min(x: CpuTensor, axis: number, keepDims = false): CpuTensor {
    if (isBigIntCpuDType(x.dtype as CpuDType)) return this.#bigintReduce(x, axis, keepDims, "min");
    const r = this.#axisReduce(x, axis, keepDims, x.dtype as CpuDType, "min");
    K.extremumAxis(x.data as K.NumArray, 0, r.outer, r.dim, r.inner, r.out as K.NumArray, false);
    return this.#make(r.shape, x.dtype as CpuDType, r.out);
  }
  argmax(x: CpuTensor, axis: number, keepDims = false): CpuTensor {
    if (isBigIntCpuDType(x.dtype as CpuDType)) return this.#bigintReduce(x, axis, keepDims, "argmax");
    const r = this.#axisReduce(x, axis, keepDims, "i32", "argmax");
    K.argExtremumAxis(x.data as K.NumArray, 0, r.outer, r.dim, r.inner, r.out as K.NumArray, true);
    return this.#make(r.shape, "i32", r.out);
  }
  argmin(x: CpuTensor, axis: number, keepDims = false): CpuTensor {
    if (isBigIntCpuDType(x.dtype as CpuDType)) return this.#bigintReduce(x, axis, keepDims, "argmin");
    const r = this.#axisReduce(x, axis, keepDims, "i32", "argmin");
    K.argExtremumAxis(x.data as K.NumArray, 0, r.outer, r.dim, r.inner, r.out as K.NumArray, false);
    return this.#make(r.shape, "i32", r.out);
  }

  cumsum(x: CpuTensor, axis: number): CpuTensor {
    if (isBigIntCpuDType(x.dtype as CpuDType)) return this.#bigintReduce(x, axis, false, "cumsum");
    const [outer, dim, inner] = around(x.shape, normAxis(axis, x.shape.length));
    // i32/bool -> i32 (unchanged); every other dtype (f32/f64 and the new
    // narrower integer dtypes) keeps its own dtype and wraps within its own
    // width on overflow -- confirmed against real MLX (u8 cumsum wraps at
    // 256, it does not auto-promote); see the contract's cumsum doc comment.
    const dtype: CpuDType = x.dtype === "bool" || x.dtype === "i32" ? "i32" : (x.dtype as CpuDType);
    const out = alloc(dtype, x.data.length);
    K.cumsumAxis(x.data as K.NumArray, 0, outer, dim, inner, out as K.NumArray);
    return this.#make(x.shape, dtype, out);
  }

  softmax(x: CpuTensor, axis: number): CpuTensor {
    const [outer, dim, inner] = around(x.shape, normAxis(axis, x.shape.length));
    const src = x.data instanceof Float32Array ? x.data : this.#f32(x);
    const out = new Float32Array(src.length);
    if (dim > 0) K.softmaxAxis(src, 0, outer, dim, inner, out, true);
    return this.#make(x.shape, "f32", out);
  }

  sort(x: CpuTensor, axis: number): CpuTensor {
    if (isBigIntCpuDType(x.dtype as CpuDType)) {
      const a = normAxis(axis, x.shape.length);
      const r = (this.#wrap(x) as any).sort(a);
      return this.#make(r.shape as readonly number[], x.dtype as CpuDType, r.data as Data);
    }
    const [outer, dim, inner] = around(x.shape, normAxis(axis, x.shape.length));
    const out = alloc(x.dtype as CpuDType, x.data.length);
    K.sortAxis(x.data as K.NumArray, 0, outer, dim, inner, out as K.NumArray, alloc(x.dtype as CpuDType, dim) as K.NumArray);
    return this.#make(x.shape, x.dtype as CpuDType, out);
  }

  // ---- linear algebra & NN
  /** Float32 view of a tensor's values (integer/bool/bigint inputs convert). */
  #f32(x: CpuTensor): Float32Array {
    const d = x.data;
    if (d instanceof Float32Array) return d;
    if (isBigIntCpuDType(x.dtype as CpuDType)) return Float32Array.from(d as BigInt64Array | BigUint64Array, (v) => Number(v));
    return Float32Array.from(d as ArrayLike<number>);
  }
  /**
   * `x.data` widened to K's `NumArray` (every dtype the flat kernels accept
   * natively -- f32/f64/i32/u32/i16/u16/i8/u8, i.e. everything except
   * bigint). bigint dtypes convert through `Number()` (lossy, but these NN
   * ops already compute in f32 for any non-float input, per the contract's
   * "accumulates in f32 regardless of storage dtype" doc comments).
   */
  #numArray(x: CpuTensor): K.NumArray {
    return isBigIntCpuDType(x.dtype as CpuDType) ? Float32Array.from(x.data as BigInt64Array | BigUint64Array, (v) => Number(v)) : (x.data as K.NumArray);
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
    const X = this.#numArray(x);
    const rows = nIn === 0 ? sizeOf(x.shape.slice(0, -1)) : X.length / nIn;
    const out = new Float32Array(rows * nOut);
    K.linearNT(X, 0, rows, nIn, this.#numArray(w), nOut, b ? this.#numArray(b) : null, out);
    return this.#make([...x.shape.slice(0, -1), nOut], "f32", out);
  }

  layerNorm(x: CpuTensor, weight: CpuTensor | null, bias: CpuTensor | null, eps: number): CpuTensor {
    const D = x.shape[x.shape.length - 1]!;
    const X = this.#numArray(x);
    const out = new Float32Array(X.length);
    if (D > 0) K.layerNormRows(X, 0, X.length / D, D, weight ? this.#numArray(weight) : null, bias ? this.#numArray(bias) : null, eps, out);
    return this.#make(x.shape, "f32", out);
  }

  embedding(table: CpuTensor, ids: CpuTensor): CpuTensor {
    if (table.shape.length !== 2) throw new RangeError("tensor-cpu: embedding table must be [V, D]");
    const [V, D] = table.shape as [number, number];
    const I = ids.data as Int32Array; // ids are always i32 per the contract ("ids i32 [...]")
    const out = new Float32Array(I.length * D);
    K.takeRows(this.#f32(table), D, V, I, out);
    return this.#make([...ids.shape, D], "f32", out);
  }

  gatherRows(x: CpuTensor, idx: CpuTensor): CpuTensor {
    const [B, L, D] = x.shape as [number, number, number];
    const [B2, M] = idx.shape as [number, number];
    if (x.shape.length !== 3 || idx.shape.length !== 2 || B2 !== B) throw new RangeError(`tensor-cpu: gatherRows [${x.shape}] idx [${idx.shape}]`);
    const I = idx.data as Int32Array; // idx is always i32 per the contract ("idx i32 [B, M]")
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
    K.ropeHalf(this.#numArray(x), 0, B * H, L, Dh, base, out);
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
      m = { data: this.#numArray(mask), strides: K.alignedStrides(ms, target), boolean: mask.dtype === "bool" };
    }
    const out = new Float32Array(B * H * Lq * Dv);
    K.attention(this.#numArray(q), this.#numArray(k), this.#numArray(v), B, H, Hk, Lq, Lk, Dh, Dv, scale, m, out);
    return this.#make([B, H, Lq, Dv], "f32", out);
  }

  geglu(x: CpuTensor): CpuTensor {
    const D2 = x.shape[x.shape.length - 1]!;
    if (D2 % 2) throw new RangeError(`tensor-cpu: geglu needs an even last axis, got ${D2}`);
    const D = D2 / 2;
    const rows = D2 === 0 ? 0 : x.data.length / D2;
    const out = new Float32Array(rows * D);
    K.gegluRows(this.#numArray(x), 0, rows, D, out);
    return this.#make([...x.shape.slice(0, -1), D], "f32", out);
  }

  meanPool(x: CpuTensor, mask: CpuTensor): CpuTensor {
    const [B, L, D] = x.shape as [number, number, number];
    if (x.shape.length !== 3 || mask.shape.length !== 2 || mask.shape[0] !== B || mask.shape[1] !== L) {
      throw new RangeError(`tensor-cpu: meanPool x [${x.shape}] mask [${mask.shape}]`);
    }
    const out = new Float32Array(B * D);
    K.maskedMeanPool(this.#numArray(x), B, L, D, this.#numArray(mask), out);
    return this.#make([B, D], "f32", out);
  }
}

// ---------------------------------------------------------------- device facade
/**
 * The CPU device: the chainable {@link DeviceArray} API (shared with
 * tensor-mlx's `MlxDevice` and tensor-webgpu's `WebGpuDevice`) over a
 * {@link CpuBackend}. Eager, f32/i32/bool: f16/bf16 uploads and casts are
 * refused (the backend would widen them to f32 silently), so cast to f32
 * explicitly first.
 */
export class CpuDevice extends ArrayDevice<CpuBackend> {
  declare readonly name: "cpu";

  /** @internal Use `createCpuDevice()`. */
  constructor(backend: CpuBackend) {
    super(backend, { label: "tensor-cpu", device: "CpuDevice", array: "a CpuArray" });
  }
}

/** An array of a {@link CpuDevice}. */
export type CpuArray = DeviceArray<CpuDevice>;

/**
 * Creates a CPU device (its own backend; no shared global state):
 *
 *     const cpu = createCpuDevice();
 *     const x = await cpu.fromTensor(Tensor.from([1, 2, 3]));
 *     const t = await x.mul(2).softmax().toTensor();
 */
export function createCpuDevice(): CpuDevice {
  return new CpuDevice(createCpuBackend());
}
