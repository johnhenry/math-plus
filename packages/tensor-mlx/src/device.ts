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
 *   and leaves only through `toTensor`/`toHost`. Passing a tensor-core
 *   `Tensor` (or an array from another device) to an op throws.
 * - No implicit dtype promotion: binary ops need matching dtypes, number
 *   operands take the array's dtype, and float-only ops refuse integer
 *   input. `cast()` is the only way across dtypes (MLX itself would promote).
 * - Lazy execution: ops append nodes to MLX's graph and return at once; shape
 *   and dtype errors still throw at the call site. Work runs at `eval()`,
 *   `toTensor()`/`toHost()`, or when MLX needs a value.
 */
import type { Tensor } from "@johnhenry/math-plus-tensor-core";
import { createMlxBackend, mlxPlatformSupported, resolveLib, type MlxBackend, type MlxTensor } from "@johnhenry/backend-mlx";
import { host, type HostTensor, type Shape } from "@johnhenry/tensor-backend";
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
   * Explicit upload of a tensor-core `Tensor`: exactly one copy, from the
   * tensor's own storage into MLX unified memory. The tensor must be
   * C-contiguous (call `.contiguous()` first) and have a device dtype
   * (f32/f16/bf16/i32/bool — cast f64/i64/... explicitly first).
   */
  fromTensor(t: Tensor): MlxArray {
    return this.fromHost(hostFromTensor(t));
  }

  /** Explicit upload of a tensor-backend `HostTensor` (one copy). */
  fromHost(h: HostTensor): MlxArray {
    return this._wrap(this.backend.fromHost(h));
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

  /** @internal A 0-d array of `dtype` holding `v`; caller disposes it. */
  _scalar(v: number, dtype: DeviceDType): MlxTensor {
    return this.backend.fromHost(host(dtype, [], [v]));
  }
}

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

  // ---- elementwise (NumPy broadcasting, matching dtypes) -------------------

  add(other: MlxArray | number): MlxArray {
    return this.#binary("add", other);
  }
  sub(other: MlxArray | number): MlxArray {
    return this.#binary("sub", other);
  }
  mul(other: MlxArray | number): MlxArray {
    return this.#binary("mul", other);
  }
  div(other: MlxArray | number): MlxArray {
    return this.#binary("div", other);
  }
  maximum(other: MlxArray | number): MlxArray {
    return this.#binary("maximum", other);
  }
  /** Composed as `-maximum(-a, -b)` (the contract has no `minimum`). */
  minimum(other: MlxArray | number): MlxArray {
    const d = this.device;
    const dtype = this.#numericDtype("minimum");
    if (typeof other !== "number") {
      d._own(other, "minimum");
      sameDtype("minimum", this, other);
    }
    return d._wrap(
      d.backend.scope(() => {
        const nb = typeof other === "number" ? d._scalar(-other, dtype) : other.#negH();
        return this.#negOf(d.backend.maximum(this.#negH(), nb));
      }),
    );
  }
  neg(): MlxArray {
    return this.device._wrap(this.#negH());
  }
  exp(): MlxArray {
    return this.#unary("exp");
  }
  log(): MlxArray {
    return this.#unary("log");
  }
  relu(): MlxArray {
    this.#numericDtype("relu");
    return this.device._wrap(this.device.backend.relu(this._handle("relu")));
  }
  /** Exact erf GELU, 0.5·x·(1 + erf(x/√2)) — not tensor-core's tanh approximation (see README). */
  gelu(): MlxArray {
    return this.#unary("gelu");
  }

  // ---- reductions ------------------------------------------------------------

  /** Sum over `axis`, or over every element when `axis` is omitted. */
  sum(axis?: number, opts: { keepDims?: boolean } = {}): MlxArray {
    this.#numericDtype("sum");
    return this.#reduce("sum", axis, opts.keepDims ?? false);
  }
  /** `sum / n` in the array's (float) dtype. */
  mean(axis?: number, opts: { keepDims?: boolean } = {}): MlxArray {
    this.#floatDtype("mean");
    const n = axis === undefined ? this.size : this.shape[normAxis(axis, this.ndim, "mean")]!;
    const d = this.device;
    return d._wrap(
      d.backend.scope(() => {
        const s = this.#reduceH("sum", axis, opts.keepDims ?? false);
        const inv = d._scalar(1 / n, this.dtype);
        return d.backend.mul(s, inv);
      }),
    );
  }
  max(axis?: number, opts: { keepDims?: boolean } = {}): MlxArray {
    this.#numericDtype("max");
    return this.#reduce("max", axis, opts.keepDims ?? false);
  }
  /** Composed as `-max(-x)`. */
  min(axis?: number, opts: { keepDims?: boolean } = {}): MlxArray {
    this.#numericDtype("min");
    const d = this.device;
    return d._wrap(
      d.backend.scope(() => {
        const neg = new MlxArray(d, this.#negH());
        return this.#negOf(neg.#reduceH("max", axis, opts.keepDims ?? false));
      }),
    );
  }
  /** Numerically stable softmax along `axis` (f32 accumulation for f16/bf16). */
  softmax(axis = -1): MlxArray {
    this.#floatDtype("softmax");
    return this.device._wrap(this.device.backend.softmax(this._handle("softmax"), axis));
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
    this.#floatDtype("layerNorm");
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

  #numericDtype(op: string): DeviceDType {
    const dt = this.dtype;
    if (dt === "bool") throw new TypeError(`tensor-mlx ${op}: not defined for bool; cast() first`);
    return dt;
  }

  #floatDtype(op: string): DeviceDType {
    const dt = this.dtype;
    if (!FLOAT.has(dt)) throw new TypeError(`tensor-mlx ${op}: needs a float dtype, got ${dt}; cast("f32") first (no implicit promotion)`);
    return dt;
  }

  #unary(op: "exp" | "log" | "gelu"): MlxArray {
    this.#floatDtype(op);
    return this.device._wrap(this.device.backend[op](this._handle(op)));
  }

  #binary(op: "add" | "sub" | "mul" | "div" | "maximum", other: MlxArray | number): MlxArray {
    const d = this.device;
    const b = d.backend;
    const x = this._handle(op);
    if (typeof other === "number") {
      const s = d._scalar(other, this.#numericDtype(op));
      try {
        return d._wrap(b[op](x, s));
      } finally {
        b.dispose(s);
      }
    }
    const y = d._own(other, op);
    sameDtype(op, this, other);
    return d._wrap(b[op](x, y));
  }

  /** Raw handle of -this (caller owns it). */
  #negH(): MlxTensor {
    return this.#negOf(this._handle("neg"), this.#numericDtype("neg"));
  }

  #negOf(h: MlxTensor, dtype: DeviceDType = h.dtype): MlxTensor {
    const d = this.device;
    const m1 = d._scalar(-1, dtype);
    try {
      return d.backend.mul(h, m1);
    } finally {
      d.backend.dispose(m1);
    }
  }

  #reduce(op: "sum" | "max", axis: number | undefined, keepDims: boolean): MlxArray {
    const d = this.device;
    return d._wrap(d.backend.scope(() => this.#reduceH(op, axis, keepDims)));
  }

  /** Reduction over one axis, or all axes when `axis` is undefined (via a flat reshape). */
  #reduceH(op: "sum" | "max", axis: number | undefined, keepDims: boolean): MlxTensor {
    const b = this.device.backend;
    const x = this._handle(op);
    if (axis !== undefined) return b[op](x, normAxis(axis, this.ndim, op), keepDims);
    const r = b[op](b.reshape(x, [this.size]), 0, false);
    return keepDims ? b.reshape(r, new Array<number>(this.ndim).fill(1)) : r;
  }
}

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
