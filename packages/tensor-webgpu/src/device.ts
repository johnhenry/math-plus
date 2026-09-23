/**
 * WebGPU capability detection + the GPU-resident tensor type (issue #12, v1
 * scope items 4 & 5: "`await x.to('webgpu')` stays explicit and async").
 *
 * Design decision (documented per the issue's "you decide the exact shape"):
 * this package does NOT monkey-patch `@johnhenry/math-plus-tensor-core`'s `Tensor` class
 * with a `.to()` method. `Tensor` has no device-transfer hook today, and
 * adding one from a downstream package would mean either (a) mutating
 * `Tensor.prototype` from outside its own module — fragile, and invisible to
 * anyone reading tensor-core in isolation — or (b) tensor-core growing a
 * dependency on this package to define it as a real instance method, which
 * inverts the intended dependency direction (tensor-webgpu depends on
 * tensor-core, never the reverse, matching every other adapter/accelerator
 * package in this repo). Instead, the explicit-and-async transfer the issue
 * asks for is a free function: `toWebGPU(tensor)` returns a `Promise<GPUTensor>`,
 * mirroring `@johnhenry/math-plus-frame-arrow`'s `Series.toTensor()` / `Frame.toTensor()`
 * pattern of "device/format transfer is always an awaited call, never a
 * property access" — same spirit as `x.to("webgpu")`, different spelling.
 * `GPUTensor.toTensor()` is the inverse (GPU -> CPU), completing the pair.
 *
 * Dtype scope: f32, plus f16 *storage* (added with the tiled GEMM). f16 data
 * crosses the host boundary as raw IEEE-754 binary16 bits in a `Uint16Array`
 * — exactly how `@johnhenry/math-plus-tensor-core` stores `"f16"` tensors on
 * main — so this package never needs its own f32<->f16 codec (a second copy
 * of one would violate AGENTS.md's canonical-implementation rule). f64 has no
 * WebGPU representation; integer dtypes aren't part of the GEMM/attention/
 * fusion surface. Only GEMM computes on f16 today (see gemm.ts); attention
 * and elementwise fusion remain f32-only.
 */
import { Tensor, type Shape } from "@johnhenry/math-plus-tensor-core";
import { paddedByteLength, readBackBytes, writePadded } from "./gpu-runtime.ts";
import { registerGemmAdapter, SUBGROUP_MATRIX_FEATURE, type GemmCapabilities } from "./gemm-caps.ts";

/** Storage dtypes a {@link GPUTensor} can hold. */
export type GPUDType = "f32" | "f16";

export interface WebGPUCapability {
  available: boolean;
  /** Present only when `available` is true. */
  adapter?: GPUAdapter;
  /** Present only when `available` is true. */
  device?: GPUDevice;
  /** Present only when `available` is true: what the GEMM kernels can use on `device` (f16 storage, subgroup matrices). */
  gemm?: GemmCapabilities;
  /** Present only when `available` is false — always a human-readable explanation, never silently empty. */
  reason?: string;
}

export interface DetectWebGPUOptions {
  /**
   * The `GPU` entry point to use instead of `navigator.gpu` — e.g. Dawn's in
   * Node/Bun via `requestDawnGPU()` from this package's `./dawn` subpath.
   * Defaults to `navigator.gpu`.
   */
  gpu?: GPU;
  /** Default `"high-performance"`. */
  powerPreference?: GPUPowerPreference;
  /** Request `shader-f16` when the adapter offers it (default true) — required for f16 GEMM. */
  f16?: boolean;
  /**
   * Request Dawn's experimental `chromium-experimental-subgroup-matrix` when
   * the adapter offers it (default true). Only adapters with an f32 8x8x8
   * config and a fixed subgroup size of 32 (Apple GPUs via Metal) actually
   * enable the subgroup-matrix GEMM kernel; everything else uses the
   * portable tiled kernel. The feature is only exposed at all by Dawn
   * instances created with `allow_unsafe_apis` (Node: `requestDawnGPU({
   * unsafe: true })`; Chrome: `--enable-unsafe-webgpu`).
   */
  subgroupMatrix?: boolean;
}

/** Adapter limits worth raising from their spec defaults when the hardware allows (large GEMM operands need big storage bindings). */
const RAISED_LIMITS = ["maxStorageBufferBindingSize", "maxBufferSize"] as const;

/**
 * Feature-detect WebGPU and, if present, actually request an adapter +
 * device (not just check `"gpu" in navigator`) — a browser can expose
 * `navigator.gpu` while `requestAdapter()` still resolves `null` (no
 * compatible adapter, software or hardware), which is exactly the failure
 * mode this function's `reason` string distinguishes from "API not present
 * at all" so callers/tests can tell the two apart.
 *
 * Also requests `shader-f16` (and, where offered, subgroup matrices) and
 * records the result for the GEMM kernel selector (`gemm` in the result).
 * A device you create yourself works too — call `registerGemmAdapter(device,
 * adapter)` to let GEMM use those features on it.
 */
export async function detectWebGPU(options: DetectWebGPUOptions = {}): Promise<WebGPUCapability> {
  const nav = (globalThis as { navigator?: { gpu?: GPU } }).navigator;
  const gpu = options.gpu ?? nav?.gpu;
  if (!gpu) {
    return {
      available: false,
      reason:
        "navigator.gpu is not present — this requires a WebGPU-capable browser, Deno, " +
        "or (Node/Bun) Dawn via this package's ./dawn subpath passed as { gpu } (see README.md \"Node and Bun\")",
    };
  }
  let adapter: GPUAdapter | null;
  try {
    adapter = await gpu.requestAdapter({ powerPreference: options.powerPreference ?? "high-performance" });
  } catch (err) {
    return { available: false, reason: `requestAdapter() threw: ${String(err)}` };
  }
  if (!adapter) {
    return { available: false, reason: "requestAdapter() resolved null (no compatible GPUAdapter)" };
  }
  const requiredFeatures: GPUFeatureName[] = [];
  if ((options.f16 ?? true) && adapter.features.has("shader-f16")) requiredFeatures.push("shader-f16");
  if ((options.subgroupMatrix ?? true) && adapter.features.has(SUBGROUP_MATRIX_FEATURE)) {
    requiredFeatures.push(SUBGROUP_MATRIX_FEATURE as GPUFeatureName);
  }
  const requiredLimits: Record<string, number> = {};
  for (const name of RAISED_LIMITS) {
    const v = (adapter.limits as unknown as Record<string, number>)[name];
    if (typeof v === "number") requiredLimits[name] = v;
  }
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  const gemm = registerGemmAdapter(device, adapter);
  return { available: true, adapter, device, gemm };
}

function shapeSize(shape: Shape): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * A tensor whose data (f32, or f16 bits) lives in a `GPUBuffer` (STORAGE |
 * COPY_SRC | COPY_DST usage) rather than a JS `TypedArray`. Created via
 * {@link toWebGPU}; `.free()` releases the underlying `GPUBuffer` — WebGPU
 * buffers are NOT garbage collected on a predictable schedule, so (like
 * `@johnhenry/math-plus-tensor-wasm`'s `WasmTensor`) this is manual memory
 * management, not GC'd JS storage.
 */
export class GPUTensor {
  readonly device: GPUDevice;
  readonly buffer: GPUBuffer;
  readonly shape: Shape;
  readonly dtype: GPUDType;
  #freed = false;

  private constructor(device: GPUDevice, buffer: GPUBuffer, shape: Shape, dtype: GPUDType) {
    this.device = device;
    this.buffer = buffer;
    this.shape = Object.freeze([...shape]);
    this.dtype = dtype;
  }

  static fromFloat32Array(device: GPUDevice, data: Float32Array, shape: Shape): GPUTensor {
    if (data.length !== shapeSize(shape)) {
      throw new RangeError(
        `GPUTensor.fromFloat32Array: shape [${shape}] (${shapeSize(shape)} elements) does not match data length ${data.length}`,
      );
    }
    return GPUTensor.#upload(device, data, shape, "f32");
  }

  /**
   * f16 counterpart of {@link fromFloat32Array}: `bits` are raw IEEE-754
   * binary16 values (tensor-core's `"f16"` storage; a `Float16Array`'s
   * underlying bytes are the same thing — pass `new Uint16Array(f16.buffer,
   * f16.byteOffset, f16.length)`). Storing f16 needs no device feature;
   * *computing* on it (GEMM) needs `shader-f16`.
   */
  static fromFloat16Bits(device: GPUDevice, bits: Uint16Array, shape: Shape): GPUTensor {
    if (bits.length !== shapeSize(shape)) {
      throw new RangeError(
        `GPUTensor.fromFloat16Bits: shape [${shape}] (${shapeSize(shape)} elements) does not match data length ${bits.length}`,
      );
    }
    return GPUTensor.#upload(device, bits, shape, "f16");
  }

  static #upload(device: GPUDevice, data: Float32Array | Uint16Array, shape: Shape, dtype: GPUDType): GPUTensor {
    const buffer = device.createBuffer({
      size: paddedByteLength(data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    writePadded(device, buffer, data);
    return new GPUTensor(device, buffer, shape, dtype);
  }

  /**
   * Wrap an ALREADY-POPULATED GPU buffer as a `GPUTensor` with no host
   * round-trip (issue #100) — for op implementations (attention.ts, gemm.ts,
   * elementwise.ts) that compute directly into a buffer they allocated (e.g.
   * a compute shader's output) and want the result to stay GPU-resident for
   * chaining into further dispatches, rather than reading it back to a
   * `Float32Array` just to re-upload it via {@link fromFloat32Array}. `buffer`
   * must already be sized for `shape` (`shapeSize(shape) * 4` bytes for f32,
   * `* 2` rounded up to a multiple of 4 for f16) and
   * usable both as a dispatch output and, if the caller ever calls
   * {@link toTensor}/{@link toFloat32Array} on the result or reuses it as an
   * upload target, as a copy source/destination too — i.e. it should carry at
   * least `STORAGE`, and typically `COPY_SRC`/`COPY_DST` as well, matching
   * what {@link fromFloat32Array} itself allocates (`gpu-runtime.ts`'s
   * `allocateGPUResidentBuffer` returns exactly that combination).
   */
  static fromBuffer(device: GPUDevice, buffer: GPUBuffer, shape: Shape, dtype: GPUDType = "f32"): GPUTensor {
    return new GPUTensor(device, buffer, shape, dtype);
  }

  async #readBytes(): Promise<ArrayBuffer> {
    if (this.#freed) throw new Error("GPUTensor: use after free()");
    return readBackBytes(this.device, this.buffer, this.buffer.size);
  }

  /**
   * Read the buffer back into a plain `Float32Array` (host copy — for
   * `.toTensor()` or inspection/testing). For an f16 tensor this decodes via
   * the platform's `Float16Array` (Chrome >= 135, Node >= 24, Deno, Bun) and
   * throws where that's missing rather than shipping a second f16 codec —
   * use {@link toUint16Array} for the raw bits.
   */
  async toFloat32Array(): Promise<Float32Array> {
    if (this.dtype === "f16") {
      const F16 = (globalThis as { Float16Array?: new (b: ArrayBuffer, o: number, n: number) => ArrayLike<number> }).Float16Array;
      if (!F16) {
        throw new TypeError("GPUTensor.toFloat32Array: decoding f16 needs a runtime with Float16Array; use toUint16Array() for the raw bits");
      }
      return Float32Array.from(new F16(await this.#readBytes(), 0, shapeSize(this.shape)));
    }
    return new Float32Array(await this.#readBytes(), 0, shapeSize(this.shape));
  }

  /** f16 only: read the raw binary16 bits back (tensor-core's f16 representation). */
  async toUint16Array(): Promise<Uint16Array> {
    if (this.dtype !== "f16") throw new TypeError(`GPUTensor.toUint16Array: tensor is ${this.dtype}, not f16`);
    return new Uint16Array(await this.#readBytes(), 0, shapeSize(this.shape));
  }

  /**
   * GPU -> CPU: the inverse of {@link toWebGPU}. Always a copy (never aliases
   * the `GPUBuffer`), and always explicit/async — no implicit CPU<->GPU
   * copying (this repo's non-goal 5), same as `toWebGPU` itself. An f16
   * `GPUTensor` comes back as an `"f16"` tensor-core `Tensor` (Uint16Array
   * bits), not widened to f32.
   */
  async toTensor(): Promise<Tensor> {
    if (this.dtype === "f16") {
      return Tensor.fromTypedArray(await this.toUint16Array(), this.shape, { dtype: "f16" });
    }
    return Tensor.fromTypedArray(await this.toFloat32Array(), this.shape, { dtype: "f32" });
  }

  free(): void {
    if (this.#freed) return;
    this.buffer.destroy();
    this.#freed = true;
  }
}

/**
 * CPU -> GPU: the explicit, awaited device transfer the issue calls for
 * (v1 non-goal 5 — no implicit copying). Requires an `"f32"` or `"f16"`
 * tensor and a contiguous one (call `.contiguous()` first on a
 * view/transposed tensor — matches `@johnhenry/math-plus-tensor-wasm`'s
 * `WasmTensor.fromArray` contract).
 */
export async function toWebGPU(tensor: Tensor, device: GPUDevice): Promise<GPUTensor> {
  if (tensor.dtype !== "f32" && tensor.dtype !== "f16") {
    throw new TypeError(`toWebGPU: supports f32 and f16 only, got ${tensor.dtype} (cast() first)`);
  }
  if (!tensor.isContiguous) {
    throw new TypeError("toWebGPU: tensor must be contiguous (call .contiguous() first)");
  }
  const whole = tensor.offset === 0 && tensor.data.length === tensor.size;
  if (tensor.dtype === "f16") {
    const bits = tensor.data as Uint16Array;
    return GPUTensor.fromFloat16Bits(device, whole ? bits : bits.subarray(tensor.offset, tensor.offset + tensor.size), tensor.shape);
  }
  const data = tensor.data as Float32Array;
  return GPUTensor.fromFloat32Array(device, whole ? data : data.subarray(tensor.offset, tensor.offset + tensor.size), tensor.shape);
}
