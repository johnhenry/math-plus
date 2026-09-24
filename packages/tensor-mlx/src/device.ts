/**
 * `MlxDevice` / `MlxArray` — a math-plus-flavoured array API over the
 * `@johnhenry/tensor-backend` contract, implemented by `@johnhenry/backend-mlx`
 * (Apple's mlx-c over FFI: koffi on Node, bun:ffi on Bun). This package binds
 * no native symbols itself (canonical-implementation rule): every op below is
 * one or a few `Backend` calls.
 *
 * Rules carried over from docs/PLAN.md and RFC 0001:
 * - No global default device: you create one with `createMlxDevice()` and
 *   every array remembers the device that owns it.
 * - No implicit transfers: data enters only through `fromTensor`/`fromHost`
 *   and leaves only through `toTensor`/`toHost`, and both directions are
 *   async (RFC 0001 §12 Q2), so a transfer is never mistaken for a cheap
 *   synchronous op. Passing a tensor-core `Tensor` (or an array from another
 *   device) to an op throws. Number operands never upload: the constant is
 *   derived on the device from the array itself (see `_constLike`).
 * - No implicit dtype promotion: binary ops need matching dtypes, number
 *   operands take the array's dtype, and float-only ops refuse integer
 *   input. `cast()` is the only way across dtypes (MLX itself would promote).
 * - Lazy execution: ops append nodes to MLX's graph and return at once; shape
 *   and dtype errors still throw at the call site. Work runs at `eval()`,
 *   `toTensor()`/`toHost()`, or when MLX needs a value.
 * - The "general numerics" ops (`sqrt`, comparisons, `argmax`, `cumsum`, …)
 *   are called through tensor-backend's compose helpers, which use
 *   backend-mlx's native mlx-c kernels (every one is native there).
 */
import type { Tensor } from "@johnhenry/math-plus-tensor-core";
import { createMlxBackend, mlxPlatformSupported, resolveLib, type MlxBackend, type MlxTensor } from "@johnhenry/backend-mlx";
import * as ops from "@johnhenry/tensor-backend";
import type { HostTensor, Shape } from "@johnhenry/tensor-backend";
import { hostFromTensor, isDeviceDType, tensorFromHost, type DeviceDType } from "./convert.ts";

export interface MlxDeviceOptions {
  /** "gpu" (Metal, the default) or "cpu" (MLX's CPU backend). */
  device?: "gpu" | "cpu";
  /** Explicit path to libmlxc.dylib; otherwise @johnhenry/backend-mlx's resolution order applies. */
  libPath?: string;
  /** FinalizationRegistry safety net for leaked handles (default true). `dispose`/`scope` stay the real mechanism. */
  finalizers?: boolean;
}

/**
 * Why an MLX device cannot be created in this process, or `null` when it
 * can (darwin/arm64 and a libmlxc.dylib was found). Tests use this to skip,
 * never fail, where MLX is unavailable.
 */
export function mlxUnavailableReason(): string | null {
  if (!mlxPlatformSupported()) return "MLX needs macOS on Apple Silicon (darwin/arm64)";
  try {
    resolveLib(); // same resolution createMlxDevice uses, incl. a $LAYA_MLXC_PATH that points nowhere
  } catch (e) {
    return (e as Error).message.split("\n")[0]!;
  }
  return null;
}

/** Creates an MLX device. Throws off darwin/arm64 or when libmlxc cannot be found. */
export function createMlxDevice(opts: MlxDeviceOptions = {}): MlxDevice {
  return new MlxDevice(createMlxBackend(opts));
}

const FLOAT: ReadonlySet<DeviceDType> = new Set(["f32", "f16", "bf16"]);

export class MlxDevice {
  readonly name = "mlx" as const;
  /** The underlying `@johnhenry/tensor-backend` `Backend` — for running the conformance suite or backend-generic code. */
  readonly backend: MlxBackend;
  readonly #scopes: Set<MlxArray>[] = [];

  /** @internal Use `createMlxDevice()`. */
  constructor(backend: MlxBackend) {
    this.backend = backend;
  }

  /** "gpu" or "cpu". */
  get kind(): "gpu" | "cpu" {
    return this.backend.device;
  }

  /** Which libmlxc loaded, through which FFI (node/bun), and its mlx-c ABI. */
  get info(): MlxBackend["info"] {
    return this.backend.info;
  }

  // ---- transfers (the only way data crosses the boundary) -------------------

  /**
   * Explicit, async upload of a tensor-core `Tensor`: exactly one copy, from
   * the tensor's own storage into MLX unified memory. The tensor must be
   * C-contiguous (call `.contiguous()` first) and have a device dtype
   * (f32/f16/bf16/i32/bool — cast f64/i64/... explicitly first); those
   * checks throw synchronously, before any Promise. Do not mutate the
   * tensor until the Promise settles. Upload several with `Promise.all`.
   */
  fromTensor(t: Tensor): Promise<MlxArray> {
    return this.fromHost(hostFromTensor(t));
  }

  /** Explicit, async upload of a tensor-backend `HostTensor` (one copy). */
  async fromHost(h: HostTensor): Promise<MlxArray> {
    return this._wrap(await this.backend.fromHost(h));
  }

  // ---- graph control -------------------------------------------------------

  /** Evaluates the given arrays' pending graphs now (`mlx_eval`); with no arguments, waits for the device stream. */
  eval(...arrays: MlxArray[]): void {
    this.backend.flush(...arrays.map((a) => this._own(a, "eval")));
  }

  /**
   * Runs `fn`; every array created inside and not returned (directly, or one
   * level deep in a returned array/object) is disposed afterwards — also
   * when `fn` throws.
   */
  scope<R>(fn: () => R): R {
    const created = new Set<MlxArray>();
    this.#scopes.push(created);
    let result!: R;
    try {
      this.backend.scope(() => {
        result = fn();
        return handlesOf(result);
      });
    } catch (e) {
      this.#scopes.pop();
      for (const a of created) a._markDisposed();
      throw e;
    }
    this.#scopes.pop();
    const kept = new Set(arraysOf(result));
    for (const a of created) if (!kept.has(a)) a._markDisposed();
    const parent = this.#scopes[this.#scopes.length - 1];
    if (parent) for (const a of kept) if (created.has(a)) parent.add(a);
    return result;
  }

  /** `where(cond, a, b)`: `cond` is bool; `a` and `b` share a dtype; all three broadcast. */
  where(cond: MlxArray, a: MlxArray, b: MlxArray): MlxArray {
    if (cond.dtype !== "bool") throw new TypeError(`tensor-mlx where: cond must be bool, got ${cond.dtype}`);
    sameDtype("where", a, b);
    return this._wrap(this.backend.where(this._own(cond, "where"), this._own(a, "where"), this._own(b, "where")));
  }

  /** Live (not yet disposed) MLX handles owned by this device. */
  liveArrays(): number {
    return this.backend.liveTensors();
  }

  /** MLX allocator statistics, in bytes. */
  memory(): { active: number; peak: number } {
    return this.backend.memory();
  }

  /** Releases the device's stream. Arrays must not be used afterwards. */
  destroy(): void {
    this.backend.destroy();
  }

  // ---- internals shared with MlxArray ----------------------------------------

  /** @internal */
  _wrap(h: MlxTensor): MlxArray {
    const a = new MlxArray(this, h);
    this.#scopes[this.#scopes.length - 1]?.add(a);
    return a;
  }

  /** @internal Validates that `x` is a live array of THIS device and returns its handle. */
  _own(x: unknown, op: string): MlxTensor {
    if (!(x instanceof MlxArray)) {
      const what = x && typeof x === "object" && "strides" in x ? "a tensor-core Tensor" : typeof x;
      throw new TypeError(
        `tensor-mlx ${op}: expected an MlxArray, got ${what} — upload explicitly with device.fromTensor() (no implicit transfers)`,
      );
    }
    if (x.device !== this) throw new Error(`tensor-mlx ${op}: array belongs to a different MlxDevice (no implicit transfers)`);
    return x._handle(op);
  }

  /**
   * @internal The constant `v` in `dtype`, derived on the device from `seed`
   * (0-d, or seed-shaped when `seed` is empty) without any upload — uploads
   * are async, ops are not. Rounds like the old host upload did: floats round
   * `v` straight to `dtype`; i32 truncates toward zero and must fit (exact
   * for the whole i32 range, built from two 16-bit halves); bool is `v != 0`.
   * Call inside a `backend.scope` (the intermediates are left to it).
   */
  _constLike(seed: MlxTensor, v: number, dtype: DeviceDType): MlxTensor {
    const b = this.backend;
    let n = 1;
    for (const d of seed.shape) n *= d;
    const s = n > 0 ? b.reshape(b.slice(b.reshape(seed, [n]), [0], [1]), []) : seed;
    if (dtype === "bool") return v !== 0 && !Number.isNaN(v) ? ops.onesLike(b, s, "bool") : ops.zerosLike(b, s, "bool");
    if (dtype !== "i32") return b.scale(ops.onesLike(b, s, dtype), v); // scale's scalar is `dtype`: one rounding
    const t = Math.trunc(v);
    if (!(t >= -(2 ** 31) && t < 2 ** 31)) throw new RangeError(`tensor-mlx: ${v} does not fit in i32 (no implicit narrowing)`);
    const one = ops.onesLike(b, s, "f32");
    const i32 = (k: number) => b.cast(b.scale(one, k), "i32"); // exact: |k| <= 2^16
    const hi = Math.trunc(t / 65536);
    const lo = i32(t - hi * 65536);
    return hi === 0 ? lo : b.add(b.mul(i32(hi), i32(65536)), lo);
  }
}

type Operand = MlxArray | number;
type Axis = number | undefined;

export class MlxArray {
  readonly device: MlxDevice;
  #h: MlxTensor | null;

  /** @internal Arrays come from `MlxDevice.fromTensor`/`fromHost` or from ops. */
  constructor(device: MlxDevice, h: MlxTensor) {
    this.device = device;
    this.#h = h;
  }

  get shape(): Shape {
    return this._handle("shape").shape;
  }
  get dtype(): DeviceDType {
    return this._handle("dtype").dtype;
  }
  get ndim(): number {
    return this.shape.length;
  }
  get size(): number {
    let n = 1;
    for (const d of this.shape) n *= d;
    return n;
  }
  get disposed(): boolean {
    return this.#h === null;
  }

  // ---- transfers & lifetime ------------------------------------------------

  /** Explicit download: evaluates, then copies into a new tensor-core `Tensor` (one copy, async). */
  async toTensor(): Promise<Tensor> {
    return tensorFromHost(await this.toHost());
  }

  /** Explicit download as a tensor-backend `HostTensor` (f16 as `Float16Array`, bf16 as raw bits). */
  toHost(): Promise<HostTensor> {
    return this.device.backend.read(this._handle("toHost"));
  }

  /** Forces this array's pending graph to execute now. Returns `this`. */
  eval(): this {
    this.device.eval(this);
    return this;
  }

  /** Frees the MLX handle now. Idempotent. A pending graph that uses this array keeps its data alive. */
  dispose(): void {
    if (this.#h) this.device.backend.dispose(this.#h);
    this.#h = null;
  }

  // ---- elementwise arithmetic (NumPy broadcasting, matching dtypes) ---------

  add(other: Operand): MlxArray {
    return this.#binary("add", other, "numeric", (b, x, y) => b.add(x, y));
  }
  sub(other: Operand): MlxArray {
    return this.#binary("sub", other, "numeric", (b, x, y) => b.sub(x, y));
  }
  mul(other: Operand): MlxArray {
    return this.#binary("mul", other, "numeric", (b, x, y) => b.mul(x, y));
  }
  div(other: Operand): MlxArray {
    return this.#binary("div", other, "numeric", (b, x, y) => b.div(x, y));
  }
  maximum(other: Operand): MlxArray {
    return this.#binary("maximum", other, "numeric", (b, x, y) => b.maximum(x, y));
  }
  /** Composed as `-maximum(-a, -b)` (the contract has no elementwise `minimum`; `neg` is native). */
  minimum(other: Operand): MlxArray {
    return this.#binary("minimum", other, "numeric", (b, x, y) => ops.neg(b, b.maximum(ops.neg(b, x), ops.neg(b, y))));
  }
  /** `this ** other`, float dtypes only (MLX/C `pow`: a negative base is fine for integral exponents). */
  pow(other: Operand): MlxArray {
    return this.#binary("pow", other, "float", (b, x, y) => ops.pow(b, x, y));
  }

  // ---- unary math --------------------------------------------------------------

  /** −x; f32/f16/bf16/i32. */
  neg(): MlxArray {
    return this.#unary("neg", "numeric", ops.neg);
  }
  /** |x|; f32/f16/bf16/i32. */
  abs(): MlxArray {
    return this.#unary("abs", "numeric", ops.abs);
  }
  exp(): MlxArray {
    return this.#unary("exp", "float", (b, x) => b.exp(x));
  }
  log(): MlxArray {
    return this.#unary("log", "float", (b, x) => b.log(x));
  }
  sqrt(): MlxArray {
    return this.#unary("sqrt", "float", ops.sqrt);
  }
  /** 1 / √x. */
  rsqrt(): MlxArray {
    return this.#unary("rsqrt", "float", ops.rsqrt);
  }
  tanh(): MlxArray {
    return this.#unary("tanh", "float", ops.tanh);
  }
  /** 1 / (1 + e⁻ˣ). */
  sigmoid(): MlxArray {
    return this.#unary("sigmoid", "float", ops.sigmoid);
  }
  /** The error function (MLX's `erf`, accurate to f32). */
  erf(): MlxArray {
    return this.#unary("erf", "float", ops.erf);
  }
  relu(): MlxArray {
    return this.#unary("relu", "numeric", (b, x) => b.relu(x));
  }
  /** Exact erf GELU, 0.5·x·(1 + erf(x/√2)) — not tensor-core's tanh approximation (see README). */
  gelu(): MlxArray {
    return this.#unary("gelu", "float", (b, x) => b.gelu(x));
  }

  // ---- comparisons & logic (bool results) --------------------------------------

  /** `this == other` → bool. Any dtype; operands must match. */
  equal(other: Operand): MlxArray {
    return this.#binary("equal", other, "any", (b, x, y) => ops.equal(b, x, y));
  }
  notEqual(other: Operand): MlxArray {
    return this.#binary("notEqual", other, "any", (b, x, y) => ops.notEqual(b, x, y));
  }
  less(other: Operand): MlxArray {
    return this.#binary("less", other, "any", (b, x, y) => ops.less(b, x, y));
  }
  lessEqual(other: Operand): MlxArray {
    return this.#binary("lessEqual", other, "any", (b, x, y) => ops.lessEqual(b, x, y));
  }
  greater(other: Operand): MlxArray {
    return this.#binary("greater", other, "any", (b, x, y) => ops.greater(b, x, y));
  }
  greaterEqual(other: Operand): MlxArray {
    return this.#binary("greaterEqual", other, "any", (b, x, y) => ops.greaterEqual(b, x, y));
  }
  /** Elementwise AND of two bool arrays (cast("bool") first: no implicit truthiness). */
  logicalAnd(other: MlxArray): MlxArray {
    return this.#binary("logicalAnd", other, "bool", (b, x, y) => ops.logicalAnd(b, x, y));
  }
  logicalOr(other: MlxArray): MlxArray {
    return this.#binary("logicalOr", other, "bool", (b, x, y) => ops.logicalOr(b, x, y));
  }
  logicalNot(): MlxArray {
    return this.#unary("logicalNot", "bool", ops.logicalNot);
  }

  // ---- reductions & scans --------------------------------------------------------

  /** Sum over `axis`, or over every element when `axis` is omitted. */
  sum(axis?: number, opts: { keepDims?: boolean } = {}): MlxArray {
    return this.#reduce("sum", "numeric", axis, opts, (b, x, a, k) => b.sum(x, a, k));
  }
  /** Mean over `axis` (or all elements); float dtypes, result in the same dtype (MLX accumulates in f32). */
  mean(axis?: number, opts: { keepDims?: boolean } = {}): MlxArray {
    return this.#reduce("mean", "float", axis, opts, ops.mean);
  }
  max(axis?: number, opts: { keepDims?: boolean } = {}): MlxArray {
    return this.#reduce("max", "numeric", axis, opts, (b, x, a, k) => b.max(x, a, k));
  }
  min(axis?: number, opts: { keepDims?: boolean } = {}): MlxArray {
    return this.#reduce("min", "numeric", axis, opts, ops.min);
  }
  /** Index (i32) of the first maximum along `axis`, or into the flattened array when `axis` is omitted. */
  argmax(axis?: number, opts: { keepDims?: boolean } = {}): MlxArray {
    return this.#reduce("argmax", "numeric", axis, opts, ops.argmax);
  }
  /** Index (i32) of the first minimum along `axis`, or into the flattened array when `axis` is omitted. */
  argmin(axis?: number, opts: { keepDims?: boolean } = {}): MlxArray {
    return this.#reduce("argmin", "numeric", axis, opts, ops.argmin);
  }
  /**
   * Inclusive prefix sum along `axis`; with no axis, over the flattened
   * array (1-D result, like `numpy.cumsum`). Keeps f32/f16/bf16/i32.
   */
  cumsum(axis?: number): MlxArray {
    this.#check("cumsum", "numeric");
    const d = this.device;
    const b = d.backend;
    const x = this._handle("cumsum");
    if (axis !== undefined) return d._wrap(ops.cumsum(b, x, normAxis(axis, this.ndim, "cumsum")));
    return d._wrap(b.scope(() => ops.cumsum(b, b.reshape(x, [this.size]), 0)));
  }
  /** Numerically stable softmax along `axis` (f32 accumulation for f16/bf16). */
  softmax(axis = -1): MlxArray {
    return this.#unary("softmax", "float", (b, x) => b.softmax(x, axis));
  }

  // ---- linear algebra & NN ------------------------------------------------

  /** Batched matmul with broadcasting of leading dims: [..., m, k] @ [..., k, n]. */
  matmul(other: MlxArray): MlxArray {
    sameDtype("matmul", this, other);
    const d = this.device;
    return d._wrap(d.backend.matmul(this._handle("matmul"), d._own(other, "matmul")));
  }
  /** LayerNorm over the last axis (`mlx.fast.layer_norm`); weight/bias optional. */
  layerNorm(weight: MlxArray | null = null, bias: MlxArray | null = null, eps = 1e-5): MlxArray {
    this.#check("layerNorm", "float");
    const d = this.device;
    for (const p of [weight, bias]) if (p) sameDtype("layerNorm", this, p);
    return d._wrap(
      d.backend.layerNorm(this._handle("layerNorm"), weight ? d._own(weight, "layerNorm") : null, bias ? d._own(bias, "layerNorm") : null, eps),
    );
  }

  // ---- shape & dtype -----------------------------------------------------------

  /** Explicit dtype conversion (f32/f16/bf16/i32/bool) — the only way across dtypes. */
  cast(dtype: DeviceDType): MlxArray {
    if (!isDeviceDType(dtype)) throw new TypeError(`tensor-mlx cast: unsupported dtype ${dtype as string}`);
    return this.device._wrap(this.device.backend.cast(this._handle("cast"), dtype));
  }
  reshape(shape: Shape): MlxArray {
    return this.device._wrap(this.device.backend.reshape(this._handle("reshape"), shape));
  }
  /** Permute axes (default: reverse them), like `numpy.transpose`. */
  transpose(axes?: readonly number[]): MlxArray {
    const perm = axes ?? Array.from({ length: this.ndim }, (_, i) => this.ndim - 1 - i);
    return this.device._wrap(this.device.backend.transpose(this._handle("transpose"), perm));
  }

  // ---- internals ----------------------------------------------------------------

  /** @internal */
  _handle(op: string): MlxTensor {
    if (!this.#h) throw new Error(`tensor-mlx ${op}: array used after dispose`);
    return this.#h;
  }

  /** @internal Called when an enclosing scope freed the handle. */
  _markDisposed(): void {
    this.#h = null;
  }

  /** Validates this array's dtype for `op` (see the dtype rules in the README). */
  #check(op: string, kind: Kind): DeviceDType {
    const dt = this._handle(op).dtype;
    if (kind === "numeric" && dt === "bool") throw new TypeError(`tensor-mlx ${op}: not defined for bool; cast() first`);
    if (kind === "float" && !FLOAT.has(dt)) throw new TypeError(`tensor-mlx ${op}: needs a float dtype, got ${dt}; cast("f32") first (no implicit promotion)`);
    if (kind === "bool" && dt !== "bool") throw new TypeError(`tensor-mlx ${op}: needs bool operands, got ${dt}; cast("bool") first (no implicit truthiness)`);
    return dt;
  }

  #unary(op: string, kind: Kind, f: (b: MlxBackend, x: MlxTensor) => MlxTensor): MlxArray {
    this.#check(op, kind);
    const b = this.device.backend;
    const x = this._handle(op);
    return this.device._wrap(b.scope(() => f(b, x)));
  }

  /** A binary op; a number operand becomes an on-device constant of this array's dtype. */
  #binary(op: string, other: Operand, kind: Kind, f: (b: MlxBackend, x: MlxTensor, y: MlxTensor) => MlxTensor): MlxArray {
    const dtype = this.#check(op, kind);
    const d = this.device;
    const b = d.backend;
    const x = this._handle(op);
    if (typeof other === "number") {
      if (kind === "bool") throw new TypeError(`tensor-mlx ${op}: needs an MlxArray operand`);
      return d._wrap(b.scope(() => f(b, x, d._constLike(x, other, dtype))));
    }
    const y = d._own(other, op);
    sameDtype(op, this, other);
    return d._wrap(b.scope(() => f(b, x, y)));
  }

  /** A reduction over one axis, or over every element (via a flat reshape) when `axis` is undefined. */
  #reduce(
    op: string,
    kind: Kind,
    axis: Axis,
    opts: { keepDims?: boolean },
    f: (b: MlxBackend, x: MlxTensor, axis: number, keepDims: boolean) => MlxTensor,
  ): MlxArray {
    this.#check(op, kind);
    const d = this.device;
    const b = d.backend;
    const x = this._handle(op);
    const keepDims = opts.keepDims ?? false;
    if (axis !== undefined) return d._wrap(b.scope(() => f(b, x, normAxis(axis, this.ndim, op), keepDims)));
    const ndim = this.ndim;
    return d._wrap(
      b.scope(() => {
        const r = f(b, b.reshape(x, [this.size]), 0, false);
        return keepDims ? b.reshape(r, new Array<number>(ndim).fill(1)) : r;
      }),
    );
  }
}

/** What an op accepts: any dtype, numeric (not bool), float only, or bool only. */
type Kind = "any" | "numeric" | "float" | "bool";

function sameDtype(op: string, a: MlxArray, b: MlxArray): void {
  if (!(b instanceof MlxArray)) return; // _own reports the better error
  if (a.dtype !== b.dtype) {
    throw new TypeError(`tensor-mlx ${op}: dtype mismatch ${a.dtype} vs ${b.dtype} (no implicit promotion; cast() first)`);
  }
}

function normAxis(axis: number, ndim: number, op: string): number {
  const a = axis < 0 ? axis + ndim : axis;
  if (!Number.isInteger(a) || a < 0 || a >= ndim) throw new RangeError(`tensor-mlx ${op}: axis ${axis} out of range for ndim ${ndim}`);
  return a;
}

function arraysOf(v: unknown): MlxArray[] {
  if (v instanceof MlxArray) return [v];
  if (Array.isArray(v)) return v.filter((x): x is MlxArray => x instanceof MlxArray);
  if (v && typeof v === "object") return Object.values(v).filter((x): x is MlxArray => x instanceof MlxArray);
  return [];
}

/** The handles of the arrays a scope returns, in a shape tensor-backend's `scope` recognises. */
function handlesOf(v: unknown): MlxTensor[] {
  return arraysOf(v).map((a) => a._handle("scope"));
}
