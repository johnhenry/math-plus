/**
 * WebGPU capability detection + the legacy GPU-resident tensor type (issue
 * #12, v1 scope items 4 & 5: "`await x.to('webgpu')` stays explicit and
 * async").
 *
 * DEPRECATED surface (issue #146): `GPUTensor` and `toWebGPU` keep working
 * through a deprecation window, now as a thin layer over
 * `@johnhenry/backend-webgpu` (a `GPUTensor` is a `WebGpuTensor` of the
 * device's backend plus the legacy methods). New code should use
 * `createWebGpuDevice()` (facade.ts): `await gpu.fromTensor(t)` /
 * `await gpu.toTensor(x)` and the backend's ops. `detectWebGPU` stays
 * supported — it is how you get a device with subgroup matrices detected
 * when you want to own the device yourself.
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
import type { WebGpuTensor } from "@johnhenry/backend-webgpu";
import { backendFor, readRaw, uploadSync, wrapBuffer } from "./bridge.ts";
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
  /**
   * Request `timestamp-query` when the adapter offers it (default false),
   * which the GPU profiler (`startProfiling`/`stopProfiling`) needs.
   */
  timestampQuery?: boolean;
}

/**
 * Adapter limits raised from their spec defaults when the hardware allows —
 * the same list `@johnhenry/backend-webgpu` raises on devices it creates
 * (large GEMM operands need big storage bindings; the head-dim-64 fused
 * attention kernel needs ~20 KiB of workgroup memory; wide fused
 * expressions need more than 8 storage buffers).
 */
const RAISED_LIMITS = [
  "maxStorageBufferBindingSize",
  "maxBufferSize",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupsPerDimension",
  "maxStorageBuffersPerShaderStage",
] as const;

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
  if (options.timestampQuery && adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
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
 * A tensor whose data (f32, or f16 bits) lives on the GPU — since issue
 * #146 a `WebGpuTensor` of `@johnhenry/backend-webgpu` (see {@link handle}),
 * held in that runtime's buffer pool. `.free()` returns the buffer to the
 * pool (or, for {@link fromBuffer}, destroys the caller's buffer) — manual
 * memory management, like `@johnhenry/math-plus-tensor-wasm`'s `WasmTensor`.
 *
 * @deprecated Use `createWebGpuDevice()`: `await gpu.fromTensor(t)` returns
 * a backend `WebGpuTensor` directly, ops are `gpu.backend.*`, and
 * `await gpu.toTensor(x)` reads it back. `GPUTensor` stays through the
 * deprecation window announced in the 0.2.0 changelog.
 */
export class GPUTensor {
  readonly device: GPUDevice;
  readonly shape: Shape;
  readonly dtype: GPUDType;
  /**
   * The backend-webgpu tensor behind this `GPUTensor` — pass it to
   * `backendFor(device)`'s / `createWebGpuDevice({ device }).backend`'s ops
   * to migrate incrementally. Owned by this `GPUTensor`: don't dispose it.
   */
  readonly handle: WebGpuTensor;
  readonly #external: boolean;
  #freed = false;

  private constructor(device: GPUDevice, handle: WebGpuTensor, dtype: GPUDType, external: boolean) {
    this.device = device;
    this.handle = handle;
    this.shape = Object.freeze([...handle.shape]);
    this.dtype = dtype;
    this.#external = external;
  }

  /** The `GPUBuffer` holding the data (starting at element 0 of this tensor; pooled buffers may be larger than the data). */
  get buffer(): GPUBuffer {
    return this.handle.storage.buffer;
  }

  static fromFloat32Array(device: GPUDevice, data: Float32Array, shape: Shape): GPUTensor {
    if (data.length !== shapeSize(shape)) {
      throw new RangeError(
        `GPUTensor.fromFloat32Array: shape [${shape}] (${shapeSize(shape)} elements) does not match data length ${data.length}`,
      );
    }
    return new GPUTensor(device, uploadSync(backendFor(device), data, shape, "f32"), "f32", false);
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
    return new GPUTensor(device, uploadSync(backendFor(device), bits, shape, "f16"), "f16", false);
  }

  /**
   * Wrap an ALREADY-POPULATED `GPUBuffer` you own as a `GPUTensor`, with no
   * host round-trip (issue #100). It must hold the data from byte 0
   * (`shapeSize(shape) * 4` bytes for f32, `* 2` rounded up to a multiple of
   * 4 for f16) and carry at least `STORAGE | COPY_SRC`. `.free()` destroys it.
   * f16 needs a device with `shader-f16` (throws otherwise: backend-webgpu
   * stores f16 as f32 there).
   */
  static fromBuffer(device: GPUDevice, buffer: GPUBuffer, shape: Shape, dtype: GPUDType = "f32"): GPUTensor {
    return new GPUTensor(device, wrapBuffer(backendFor(device), buffer, shape, dtype), dtype, true);
  }

  /** @internal Wrap a backend-owned result (ownership moves to the `GPUTensor`). */
  static _fromHandle(device: GPUDevice, handle: WebGpuTensor): GPUTensor {
    if (handle.dtype !== "f32" && handle.dtype !== "f16") throw new TypeError(`GPUTensor: unsupported dtype ${handle.dtype}`);
    return new GPUTensor(device, handle, handle.dtype, false);
  }

  /** @internal The live backend tensor, for the deprecated op shims. */
  _live(op: string): WebGpuTensor {
    if (this.#freed) throw new Error(`${op}: GPUTensor used after free()`);
    return this.handle;
  }

  async #readBytes(): Promise<ArrayBuffer> {
    return readRaw(backendFor(this.device), this._live("GPUTensor"));
  }

  /**
   * Read the data back into a plain `Float32Array` (host copy). For an f16
   * tensor this decodes via the platform's `Float16Array` (Chrome >= 135,
   * Node >= 24, Deno, Bun) and throws where that's missing rather than
   * shipping a second f16 codec — use {@link toUint16Array} for the raw bits.
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
   * GPU -> CPU: the inverse of {@link toWebGPU}. Always a copy, and always
   * explicit/async (non-goal 5). An f16 `GPUTensor` comes back as an
   * `"f16"` tensor-core `Tensor` (Uint16Array bits), not widened to f32.
   */
  async toTensor(): Promise<Tensor> {
    if (this.dtype === "f16") {
      return Tensor.fromTypedArray(await this.toUint16Array(), this.shape, { dtype: "f16" });
    }
    return Tensor.fromTypedArray(await this.toFloat32Array(), this.shape, { dtype: "f32" });
  }

  /** Release the GPU memory. Idempotent. Pending (already encoded) work that reads it is submitted first. */
  free(): void {
    if (this.#freed) return;
    this.#freed = true;
    const b = backendFor(this.device);
    // A wrapped buffer (`fromBuffer`) is never pooled by the backend; this API destroys it.
    b.dispose(this.handle);
    if (this.#external) {
      b.flush();
      this.handle.storage.buffer.destroy();
    }
  }
}

/**
 * CPU -> GPU: the explicit, awaited device transfer (v1 non-goal 5 — no
 * implicit copying). Requires an `"f32"` or `"f16"` tensor and a contiguous
 * one (call `.contiguous()` first on a view/transposed tensor).
 *
 * @deprecated Use `createWebGpuDevice()` and `await gpu.fromTensor(tensor)`
 * (every device dtype, not just f32/f16).
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
