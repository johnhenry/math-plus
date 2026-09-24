/**
 * WebGPU capability detection (issue #12). `detectWebGPU` requests an
 * adapter and a device with the features and limits this package's
 * kernels can use, and records the adapter for the GEMM kernel selector
 * (gemm-caps.ts). It is how you get a device with subgroup matrices
 * detected when you want to own the device yourself; pass it on with
 * `createWebGpuDevice({ device })`.
 *
 * The `GPUTensor` type and `toWebGPU` transfer that used to live here were
 * removed in 0.3.0: `createWebGpuDevice()`'s `await gpu.fromTensor(t)` /
 * `await gpu.toTensor(x)` are the explicit, async transfers now.
 */
import { registerGemmAdapter, SUBGROUP_MATRIX_FEATURE, type GemmCapabilities } from "./gemm-caps.ts";

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
   * Node/Bun via `getGpu({ unsafe: true })` from `@johnhenry/backend-webgpu`.
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
   * instances created with `allow_unsafe_apis` (Node: backend-webgpu's
   * `getGpu({ unsafe: true })`; Chrome: `--enable-unsafe-webgpu`).
   */
  subgroupMatrix?: boolean;
  /**
   * Request `timestamp-query` when the adapter offers it (default false),
   * which the runtime's GPU profiler (`gpu.backend.rt.startProfiling()`)
   * needs.
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
        "or (Node/Bun) Dawn via @johnhenry/backend-webgpu's getGpu() passed as { gpu } (createWebGpuDevice() finds Dawn itself)",
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

