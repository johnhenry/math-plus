/**
 * `ArrayDevice` / `DeviceArray` — the one chainable device-array API of
 * math-plus, over ANY `@johnhenry/tensor-backend` `Backend`. MLX
 * (`@johnhenry/math-plus-tensor-mlx`'s `MlxDevice`/`MlxArray`), WebGPU
 * (`@johnhenry/math-plus-tensor-webgpu`'s `WebGpuDevice`) and the CPU
 * reference (`createCpuDevice()` here) are thin subclasses of these two
 * classes (AGENTS.md's canonical-implementation rule: the wrapper exists
 * once). It lives in this package next to the tensor-core <-> `HostTensor`
 * conversion (host.ts) because both device packages already depend on it,
 * and it needs nothing else: every op is one or a few `Backend` calls, the
 * optional ones through tensor-backend's compose helpers (native kernel when
 * the backend has one, the default composition otherwise).
 *
 * Rules (docs/PLAN.md, RFC 0001 §12):
 * - No global default device: you create one and every array remembers the
 *   device that owns it.
 * - No implicit transfers: data enters only through `fromTensor`/`fromHost`
 *   and leaves only through `toTensor`/`toHost`, and both directions are
 *   async (§12 Q2). Validation errors (non-contiguous tensor, dtype the
 *   device cannot hold) throw synchronously, before any Promise. Passing a
 *   tensor-core `Tensor` (or an array from another device) to an op throws.
 *   Number operands never upload: the constant is derived on the device
 *   from the array itself (`_constLike`).
 * - No implicit dtype promotion: binary ops need matching dtypes, number
 *   operands take the array's dtype, float-only ops refuse integer input,
 *   and `cast()` is the only way across dtypes. A dtype the backend does not
 *   `supports()` is refused at upload and at `cast`, never widened silently.
 * - Lazy inside, eager-observable (§12 Q1): a backend may queue work (MLX
 *   graphs, batched WebGPU passes); shape and dtype errors still throw at the
 *   call site, and values appear at `eval()`/`toTensor()`/`toHost()`.
 *
 * Not included (scope boundary): indexing/slicing, concat/split, sort,
 * the fused transformer ops (`linear`, `rope`, `sdpa`, …) and quantized
 * weights. Use `device.backend` with `array.handle` for those, and
 * `device.wrap(handle)` to bring a result back into the chainable API.
 */
import type { Tensor } from "@johnhenry/math-plus-tensor-core";
import * as ops from "@johnhenry/tensor-backend";
import type { Backend, DType, HostTensor, Shape, Tensor as Handle } from "@johnhenry/tensor-backend";
import { hostFromTensor, isDeviceDType, tensorFromHost } from "./host.ts";

/** The backend tensor handle type of a `Backend`. */
export type HandleOf<B> = B extends Backend<infer T> ? T : never;

/** How a device names itself in error messages. */
export interface ArrayDeviceNames {
  /** Error-message prefix, e.g. `"tensor-mlx"`. */
  label: string;
  /** The device class, e.g. `"MlxDevice"` ("array belongs to a different MlxDevice"). */
  device: string;
  /** The array type with its article, e.g. `"an MlxArray"` ("expected an MlxArray"). */
  array: string;
}

const FLOAT: ReadonlySet<DType> = new Set(["f32", "f16", "bf16"]);

/** A `DeviceArray` subclass a facade creates its arrays as (e.g. tensor-mlx's `MlxArray`). */
export type DeviceArrayClass = new (device: any, h: any) => DeviceArray<any>;

/**
 * A device: a `Backend` plus the bookkeeping the chainable arrays need
 * (scopes, ownership checks, on-device constants). Subclass it for a device
 * facade; pass `arrayClass` when the facade has its own array subclass.
 */
export class ArrayDevice<B extends Backend<any> = Backend> {
  /** The underlying `@johnhenry/tensor-backend` `Backend`, for backend-generic code and the ops the arrays do not wrap. */
  readonly backend: B;
  /** The backend's name ("cpu", "mlx", "webgpu"). */
  readonly name: string;
  /** @internal */
  readonly _names: ArrayDeviceNames;
  readonly #scopes: Set<DeviceArray<any>>[] = [];
  readonly #Array: DeviceArrayClass;

  constructor(backend: B, names: ArrayDeviceNames, arrayClass: DeviceArrayClass = DeviceArray) {
    this.backend = backend;
    this.name = backend.name;
    this._names = names;
    this.#Array = arrayClass;
  }

  /** Whether the device stores and computes `dtype`. Uploads and casts to any other dtype throw. */
  supports(dtype: DType): boolean {
    return this.backend.supports(dtype);
  }

  // ---- transfers (the only way data crosses the boundary) -------------------

  /**
   * Explicit, async upload of a tensor-core `Tensor`: one copy, from the
   * tensor's own storage into the device. The tensor must be C-contiguous
   * (call `.contiguous()` first) and have a dtype this device `supports()`
   * (cast f64/i64/… explicitly first); those checks throw synchronously,
   * before any Promise. Do not mutate the tensor until the Promise settles.
   * Upload several with `Promise.all`.
   */
  fromTensor(t: Tensor): Promise<DeviceArray<this>> {
    return this.fromHost(hostFromTensor(t, this._names.label));
  }

  /** Explicit, async upload of a tensor-backend `HostTensor` (one copy). Throws synchronously for a dtype the device does not support. */
  fromHost(h: HostTensor): Promise<DeviceArray<this>> {
    this.#checkDtype(h.dtype, "fromHost");
    return this.backend.fromHost(h).then((x: HandleOf<B>) => this.wrap(x));
  }

  /**
   * Adopts a backend handle (from `device.backend.*`) as an array of this
   * device. The array is tracked by the enclosing `scope` like any op result,
   * so wrap handles created in the current scope (or outside every scope).
   */
  wrap(h: HandleOf<B>): DeviceArray<this> {
    const a = new this.#Array(this, h) as DeviceArray<this>;
    this.#scopes[this.#scopes.length - 1]?.add(a);
    return a;
  }

  // ---- graph control & lifetime ------------------------------------------------

  /** Runs the given arrays' pending work now (`backend.flush`); with no arguments, flushes everything queued. */
  eval(...arrays: DeviceArray<this>[]): void {
    const hs = arrays.map((a) => this._own(a, "eval"));
    this.backend.flush?.(...hs);
  }

  /**
   * Runs `fn`; every array created inside and not returned (directly, or one
   * level deep in a returned array/object) is disposed afterwards — also
   * when `fn` throws. Backend handles returned the same way are kept too.
   */
  scope<R>(fn: () => R): R {
    const created = new Set<DeviceArray<any>>();
    this.#scopes.push(created);
    let result!: R;
    try {
      this.backend.scope(() => {
        result = fn();
        return keptOf(result);
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
  where(cond: DeviceArray<this>, a: DeviceArray<this>, b: DeviceArray<this>): DeviceArray<this> {
    const c = this._own(cond, "where");
    if (c.dtype !== "bool") throw new TypeError(`${this._names.label} where: cond must be bool, got ${c.dtype}`);
    const x = this._own(a, "where");
    const y = this._own(b, "where");
    sameDtype(this._names.label, "where", x, y);
    return this.wrap(this.backend.where(c, x, y));
  }

  /** Releases the backend's resources. Arrays must not be used afterwards. */
  destroy(): void {
    this.backend.destroy?.();
  }

  // ---- internals shared with DeviceArray ----------------------------------------

  /** @internal Validates that `x` is a live array of THIS device and returns its handle. */
  _own(x: unknown, op: string): HandleOf<B> {
    const { label, device, array } = this._names;
    if (!(x instanceof DeviceArray)) {
      const what = x && typeof x === "object" && "strides" in x ? "a tensor-core Tensor" : typeof x;
      throw new TypeError(`${label} ${op}: expected ${array}, got ${what} — upload explicitly with device.fromTensor() (no implicit transfers)`);
    }
    if (x.device !== this) {
      const other = (x.device as ArrayDevice<any>)._names.device;
      throw new Error(`${label} ${op}: array belongs to a different ${other === device ? device : `device (${other})`} (no implicit transfers)`);
    }
    return x._handle(op) as HandleOf<B>;
  }

  /** @internal Throws unless the device holds `dtype` (no silent widening). */
  _checkDtype(dtype: DType, op: string): void {
    this.#checkDtype(dtype, op);
  }

  #checkDtype(dtype: DType, op: string): void {
    const label = this._names.label;
    if (!isDeviceDType(dtype)) throw new TypeError(`${label} ${op}: unsupported dtype ${dtype as string}`);
    if (!this.backend.supports(dtype)) {
      throw new TypeError(`${label} ${op}: this ${this.name} device does not support ${dtype}; cast to a supported dtype (e.g. "f32") explicitly first`);
    }
  }

  /**
   * @internal The constant `v` in `dtype`, derived on the device from `seed`
   * (0-d, or seed-shaped when `seed` is empty) without any upload — uploads
   * are async, ops are not. Floats round `v` straight to `dtype` (`scale`'s
   * scalar is applied in the tensor's dtype); i32 truncates toward zero and
   * must fit (exact over the whole i32 range, built from two 16-bit halves);
   * bool is `v != 0`. Call inside a `backend.scope` (the intermediates are
   * left to it).
   */
  _constLike(seed: HandleOf<B>, v: number, dtype: DType): HandleOf<B> {
    const b = this.backend as Backend<Handle>;
    let n = 1;
    for (const d of seed.shape) n *= d;
    const s = n > 0 ? b.reshape(b.slice(b.reshape(seed, [n]), [0], [1]), []) : seed;
    if (dtype === "bool") return (v !== 0 && !Number.isNaN(v) ? ops.onesLike(b, s, "bool") : ops.zerosLike(b, s, "bool")) as HandleOf<B>;
    if (dtype !== "i32") return b.scale(ops.onesLike(b, s, dtype), v) as HandleOf<B>;
    const t = Math.trunc(v);
    if (!(t >= -(2 ** 31) && t < 2 ** 31)) throw new RangeError(`${this._names.label}: ${v} does not fit in i32 (no implicit narrowing)`);
    const one = ops.onesLike(b, s, "f32");
    const i32 = (k: number) => b.cast(b.scale(one, k), "i32"); // exact: |k| <= 2^16
    const hi = Math.trunc(t / 65536);
    const lo = i32(t - hi * 65536);
    return (hi === 0 ? lo : b.add(b.mul(i32(hi), i32(65536)), lo)) as HandleOf<B>;
  }
}

type Operand<D extends ArrayDevice<any>> = DeviceArray<D> | number;
type AnyBackend = Backend<Handle>;

/**
 * A chainable array on an {@link ArrayDevice}. Arrays come from
 * `device.fromTensor`/`fromHost`/`wrap` or from ops; every op returns a new
 * array of the same device.
 */
export class DeviceArray<D extends ArrayDevice<any> = ArrayDevice> {
  readonly device: D;
  #h: Handle | null;

  /** @internal Arrays come from `device.fromTensor`/`fromHost`/`wrap` or from ops. */
  constructor(device: D, h: HandleOf<D["backend"]>) {
    this.device = device;
    this.#h = h;
  }

  get shape(): Shape {
    return this._handle("shape").shape;
  }
  get dtype(): DType {
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
  /** The backend handle, for `device.backend.*` ops this API does not wrap. Throws after dispose. */
  get handle(): HandleOf<D["backend"]> {
    return this._handle("handle") as HandleOf<D["backend"]>;
  }

  // ---- transfers & lifetime ------------------------------------------------

  /** Explicit download: runs pending work, then copies into a new tensor-core `Tensor` (one copy, async). */
  async toTensor(): Promise<Tensor> {
    return tensorFromHost(await this.toHost());
  }

  /** Explicit download as a tensor-backend `HostTensor` (f16 as `Float16Array`, bf16 as raw bits). */
  toHost(): Promise<HostTensor> {
    return this.#b.read(this._handle("toHost"));
  }

  /** Runs this array's pending work now. Returns `this`. */
  eval(): this {
    this.device.eval(this);
    return this;
  }

  /** Frees the handle now. Idempotent. Pending work that uses this array keeps its data alive. */
  dispose(): void {
    if (this.#h) this.#b.dispose(this.#h);
    this.#h = null;
  }

  // ---- elementwise arithmetic (NumPy broadcasting, matching dtypes) ---------

  add(other: Operand<D>): DeviceArray<D> {
    return this.#binary("add", other, "numeric", (b, x, y) => b.add(x, y));
  }
  sub(other: Operand<D>): DeviceArray<D> {
    return this.#binary("sub", other, "numeric", (b, x, y) => b.sub(x, y));
  }
  mul(other: Operand<D>): DeviceArray<D> {
    return this.#binary("mul", other, "numeric", (b, x, y) => b.mul(x, y));
  }
  div(other: Operand<D>): DeviceArray<D> {
    return this.#binary("div", other, "numeric", (b, x, y) => b.div(x, y));
  }
  maximum(other: Operand<D>): DeviceArray<D> {
    return this.#binary("maximum", other, "numeric", (b, x, y) => b.maximum(x, y));
  }
  /** Composed as `-maximum(-a, -b)` (the contract has no elementwise `minimum`). */
  minimum(other: Operand<D>): DeviceArray<D> {
    return this.#binary("minimum", other, "numeric", (b, x, y) => ops.neg(b, b.maximum(ops.neg(b, x), ops.neg(b, y))));
  }
  /** `this ** other`, float dtypes only (C `pow`: a negative base is fine for integral exponents). */
  pow(other: Operand<D>): DeviceArray<D> {
    return this.#binary("pow", other, "float", (b, x, y) => ops.pow(b, x, y));
  }

  // ---- unary math --------------------------------------------------------------

  /** −x; f32/f16/bf16/i32. */
  neg(): DeviceArray<D> {
    return this.#unary("neg", "numeric", ops.neg);
  }
  /** |x|; f32/f16/bf16/i32. */
  abs(): DeviceArray<D> {
    return this.#unary("abs", "numeric", ops.abs);
  }
  exp(): DeviceArray<D> {
    return this.#unary("exp", "float", (b, x) => b.exp(x));
  }
  log(): DeviceArray<D> {
    return this.#unary("log", "float", (b, x) => b.log(x));
  }
  sqrt(): DeviceArray<D> {
    return this.#unary("sqrt", "float", ops.sqrt);
  }
  /** 1 / √x. */
  rsqrt(): DeviceArray<D> {
    return this.#unary("rsqrt", "float", ops.rsqrt);
  }
  tanh(): DeviceArray<D> {
    return this.#unary("tanh", "float", ops.tanh);
  }
  /** 1 / (1 + e⁻ˣ). */
  sigmoid(): DeviceArray<D> {
    return this.#unary("sigmoid", "float", ops.sigmoid);
  }
  /** The error function (accurate to f32). */
  erf(): DeviceArray<D> {
    return this.#unary("erf", "float", ops.erf);
  }
  relu(): DeviceArray<D> {
    return this.#unary("relu", "numeric", (b, x) => b.relu(x));
  }
  /** Exact erf GELU, 0.5·x·(1 + erf(x/√2)) — not tensor-core's tanh approximation. */
  gelu(): DeviceArray<D> {
    return this.#unary("gelu", "float", (b, x) => b.gelu(x));
  }

  // ---- comparisons & logic (bool results) --------------------------------------

  /** `this == other` → bool. Any dtype; operands must match. */
  equal(other: Operand<D>): DeviceArray<D> {
    return this.#binary("equal", other, "any", (b, x, y) => ops.equal(b, x, y));
  }
  notEqual(other: Operand<D>): DeviceArray<D> {
    return this.#binary("notEqual", other, "any", (b, x, y) => ops.notEqual(b, x, y));
  }
  less(other: Operand<D>): DeviceArray<D> {
    return this.#binary("less", other, "any", (b, x, y) => ops.less(b, x, y));
  }
  lessEqual(other: Operand<D>): DeviceArray<D> {
    return this.#binary("lessEqual", other, "any", (b, x, y) => ops.lessEqual(b, x, y));
  }
  greater(other: Operand<D>): DeviceArray<D> {
    return this.#binary("greater", other, "any", (b, x, y) => ops.greater(b, x, y));
  }
  greaterEqual(other: Operand<D>): DeviceArray<D> {
    return this.#binary("greaterEqual", other, "any", (b, x, y) => ops.greaterEqual(b, x, y));
  }
  /** Elementwise AND of two bool arrays (cast("bool") first: no implicit truthiness). */
  logicalAnd(other: DeviceArray<D>): DeviceArray<D> {
    return this.#binary("logicalAnd", other, "bool", (b, x, y) => ops.logicalAnd(b, x, y));
  }
  logicalOr(other: DeviceArray<D>): DeviceArray<D> {
    return this.#binary("logicalOr", other, "bool", (b, x, y) => ops.logicalOr(b, x, y));
  }
  logicalNot(): DeviceArray<D> {
    return this.#unary("logicalNot", "bool", ops.logicalNot);
  }

  // ---- reductions & scans --------------------------------------------------------

  /** Sum over `axis`, or over every element when `axis` is omitted. */
  sum(axis?: number, opts: { keepDims?: boolean } = {}): DeviceArray<D> {
    return this.#reduce("sum", "numeric", axis, opts, (b, x, a, k) => b.sum(x, a, k));
  }
  /** Mean over `axis` (or all elements); float dtypes, result in the same dtype (f32 accumulation). */
  mean(axis?: number, opts: { keepDims?: boolean } = {}): DeviceArray<D> {
    return this.#reduce("mean", "float", axis, opts, ops.mean);
  }
  max(axis?: number, opts: { keepDims?: boolean } = {}): DeviceArray<D> {
    return this.#reduce("max", "numeric", axis, opts, (b, x, a, k) => b.max(x, a, k));
  }
  min(axis?: number, opts: { keepDims?: boolean } = {}): DeviceArray<D> {
    return this.#reduce("min", "numeric", axis, opts, ops.min);
  }
  /** Index (i32) of the first maximum along `axis`, or into the flattened array when `axis` is omitted. */
  argmax(axis?: number, opts: { keepDims?: boolean } = {}): DeviceArray<D> {
    return this.#reduce("argmax", "numeric", axis, opts, ops.argmax);
  }
  /** Index (i32) of the first minimum along `axis`, or into the flattened array when `axis` is omitted. */
  argmin(axis?: number, opts: { keepDims?: boolean } = {}): DeviceArray<D> {
    return this.#reduce("argmin", "numeric", axis, opts, ops.argmin);
  }
  /**
   * Inclusive prefix sum along `axis`; with no axis, over the flattened
   * array (1-D result, like `numpy.cumsum`). Keeps f32/f16/bf16/i32.
   */
  cumsum(axis?: number): DeviceArray<D> {
    this.#check("cumsum", "numeric");
    const b = this.#b;
    const x = this._handle("cumsum");
    if (axis !== undefined) return this.#wrap(ops.cumsum(b, x, normAxis(this.#label, axis, this.ndim, "cumsum")));
    const n = this.size;
    return this.#wrap(b.scope(() => ops.cumsum(b, b.reshape(x, [n]), 0)));
  }
  /** Numerically stable softmax along `axis` (f32 accumulation for f16/bf16). */
  softmax(axis = -1): DeviceArray<D> {
    return this.#unary("softmax", "float", (b, x) => b.softmax(x, axis));
  }

  // ---- linear algebra & NN ------------------------------------------------

  /** Batched matmul with broadcasting of leading dims: [..., m, k] @ [..., k, n]. */
  matmul(other: DeviceArray<D>): DeviceArray<D> {
    const x = this._handle("matmul");
    const y = this.device._own(other, "matmul");
    sameDtype(this.#label, "matmul", x, y);
    return this.#wrap(this.#b.matmul(x, y));
  }
  /** LayerNorm over the last axis; weight/bias optional (statistics in f32). */
  layerNorm(weight: DeviceArray<D> | null = null, bias: DeviceArray<D> | null = null, eps = 1e-5): DeviceArray<D> {
    this.#check("layerNorm", "float");
    const x = this._handle("layerNorm");
    const w = weight ? this.device._own(weight, "layerNorm") : null;
    const bi = bias ? this.device._own(bias, "layerNorm") : null;
    for (const p of [w, bi]) if (p) sameDtype(this.#label, "layerNorm", x, p);
    return this.#wrap(this.#b.layerNorm(x, w, bi, eps));
  }

  // ---- shape & dtype -----------------------------------------------------------

  /** Explicit dtype conversion (f32/f16/bf16/i32/bool, where the device supports it) — the only way across dtypes. */
  cast(dtype: DType): DeviceArray<D> {
    this.device._checkDtype(dtype, "cast");
    return this.#wrap(this.#b.cast(this._handle("cast"), dtype));
  }
  reshape(shape: Shape): DeviceArray<D> {
    return this.#wrap(this.#b.reshape(this._handle("reshape"), shape));
  }
  /** Permute axes (default: reverse them), like `numpy.transpose`. */
  transpose(axes?: readonly number[]): DeviceArray<D> {
    const perm = axes ?? Array.from({ length: this.ndim }, (_, i) => this.ndim - 1 - i);
    return this.#wrap(this.#b.transpose(this._handle("transpose"), perm));
  }

  // ---- internals ----------------------------------------------------------------

  /** @internal */
  _handle(op: string): Handle {
    if (!this.#h) throw new Error(`${this.#label} ${op}: array used after dispose`);
    return this.#h;
  }

  /** @internal Called when an enclosing scope freed the handle. */
  _markDisposed(): void {
    this.#h = null;
  }

  get #b(): AnyBackend {
    return this.device.backend as AnyBackend;
  }
  get #label(): string {
    return this.device._names.label;
  }
  #wrap(h: Handle): DeviceArray<D> {
    return this.device.wrap(h) as DeviceArray<D>;
  }

  /** Validates this array's dtype for `op`. */
  #check(op: string, kind: Kind): DType {
    const dt = this._handle(op).dtype;
    const label = this.#label;
    if (kind === "numeric" && dt === "bool") throw new TypeError(`${label} ${op}: not defined for bool; cast() first`);
    if (kind === "float" && !FLOAT.has(dt)) throw new TypeError(`${label} ${op}: needs a float dtype, got ${dt}; cast("f32") first (no implicit promotion)`);
    if (kind === "bool" && dt !== "bool") throw new TypeError(`${label} ${op}: needs bool operands, got ${dt}; cast("bool") first (no implicit truthiness)`);
    return dt;
  }

  #unary(op: string, kind: Kind, f: (b: AnyBackend, x: Handle) => Handle): DeviceArray<D> {
    this.#check(op, kind);
    const b = this.#b;
    const x = this._handle(op);
    return this.#wrap(b.scope(() => f(b, x)));
  }

  /** A binary op; a number operand becomes an on-device constant of this array's dtype. */
  #binary(op: string, other: Operand<D>, kind: Kind, f: (b: AnyBackend, x: Handle, y: Handle) => Handle): DeviceArray<D> {
    const dtype = this.#check(op, kind);
    const d = this.device;
    const b = this.#b;
    const x = this._handle(op);
    if (typeof other === "number") {
      if (kind === "bool") throw new TypeError(`${this.#label} ${op}: needs ${d._names.array} operand`);
      return this.#wrap(b.scope(() => f(b, x, d._constLike(x, other, dtype))));
    }
    const y = d._own(other, op);
    sameDtype(this.#label, op, x, y);
    return this.#wrap(b.scope(() => f(b, x, y)));
  }

  /** A reduction over one axis, or over every element (via a flat reshape) when `axis` is undefined. */
  #reduce(
    op: string,
    kind: Kind,
    axis: number | undefined,
    opts: { keepDims?: boolean },
    f: (b: AnyBackend, x: Handle, axis: number, keepDims: boolean) => Handle,
  ): DeviceArray<D> {
    this.#check(op, kind);
    const b = this.#b;
    const x = this._handle(op);
    const keepDims = opts.keepDims ?? false;
    const ndim = this.ndim;
    if (axis !== undefined) return this.#wrap(b.scope(() => f(b, x, normAxis(this.#label, axis, ndim, op), keepDims)));
    const n = this.size;
    return this.#wrap(
      b.scope(() => {
        const r = f(b, b.reshape(x, [n]), 0, false);
        return keepDims ? b.reshape(r, new Array<number>(ndim).fill(1)) : r;
      }),
    );
  }
}

/** What an op accepts: any dtype, numeric (not bool), float only, or bool only. */
type Kind = "any" | "numeric" | "float" | "bool";

function sameDtype(label: string, op: string, a: Handle, b: Handle): void {
  if (a.dtype !== b.dtype) throw new TypeError(`${label} ${op}: dtype mismatch ${a.dtype} vs ${b.dtype} (no implicit promotion; cast() first)`);
}

function normAxis(label: string, axis: number, ndim: number, op: string): number {
  const a = axis < 0 ? axis + ndim : axis;
  if (!Number.isInteger(a) || a < 0 || a >= ndim) throw new RangeError(`${label} ${op}: axis ${axis} out of range for ndim ${ndim}`);
  return a;
}

function arraysOf(v: unknown): DeviceArray<any>[] {
  if (v instanceof DeviceArray) return [v];
  if (Array.isArray(v)) return v.filter((x): x is DeviceArray<any> => x instanceof DeviceArray);
  if (v && typeof v === "object") return Object.values(v).filter((x): x is DeviceArray<any> => x instanceof DeviceArray);
  return [];
}

/**
 * What the backend's `scope` must keep, as a flat list it recognises: the
 * handles of returned arrays, plus any backend handles returned alongside
 * them (one level deep), which the backend recognises itself.
 */
function keptOf(v: unknown): unknown[] {
  const handle = (x: unknown) => (x instanceof DeviceArray ? (x.disposed ? null : x._handle("scope")) : x);
  if (v instanceof DeviceArray) return [handle(v)];
  if (Array.isArray(v)) return v.map(handle);
  // `v` itself may be a raw backend handle; otherwise its values are what is kept.
  if (v && typeof v === "object") return [v, ...Object.values(v).map(handle)];
  return [];
}
