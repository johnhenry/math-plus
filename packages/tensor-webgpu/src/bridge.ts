/**
 * The one place this package touches `@johnhenry/backend-webgpu`'s runtime
 * directly (issue #146, RFC 0001 §12 Q6 path (a)): backend-webgpu is the
 * single WebGPU runtime (buffer pool, pipeline and bind-group caches,
 * uniform arena, dispatch batching, readback sleep), and every GPU op this
 * package runs — GEMM, attention, IR fusion — goes through it.
 *
 * One `WebGpuBackend` per `GPUDevice`. That is a correctness rule, not an
 * optimisation: a backend batches dispatches into a compute pass it submits
 * later, so two runtimes on one device could reorder each other's work.
 * `createWebGpuDevice()` registers the backend it creates, and the
 * deprecated `GPUDevice`-taking functions (`runGemm`, `runAttention`, …)
 * look it up here (or create one for a device from `detectWebGPU()`).
 *
 * Since backend-webgpu 0.3.1 this uses its documented hooks:
 * `createWebGpuBackend({ device, adapter })`, `empty`, `wrapBuffer`,
 * `elementwise` (the fusion, elementwise.ts), the exported `Runtime`'s
 * `write` / `readBytes`, and the `sleepThresholdMs` option. One exception
 * remains: {@link backendFor} must be synchronous (the deprecated
 * `GPUTensor` constructors are), so for a device without a backend it calls
 * the `WebGpuBackend` constructor, which 0.3.1 types as public but does not
 * document. It goes away with the deprecated surface.
 */
import { createWebGpuBackend, WebGpuBackend, type AdapterSummary, type CreateWebGpuBackendOptions, type WebGpuTensor } from "@johnhenry/backend-webgpu";
import type { DType, Shape } from "@johnhenry/tensor-backend";
import { gemmAdapter, gemmCapabilities } from "./gemm-caps.ts";

/**
 * Readback-sleep threshold for backends this package creates: 15 ms.
 * backend-webgpu sleeps before a readback (instead of letting Dawn
 * busy-poll `mapAsync`, ≈100% of a core under Bun) when the expected wait
 * exceeds `sleepThresholdMs`, 3 ms by default. At 3 ms this package's
 * typical 2–6 ms GEMM/attention readbacks slept and lost 15–60% latency, so
 * 0.2.0 turned sleeping off entirely; at 15 ms those readbacks keep polling
 * (measured in docs/spikes/webgpu-runtime.md) while long waits sleep again,
 * as they did before #146. Whether sleeping is on at all stays
 * backend-webgpu's default: on under Dawn, off for `navigator.gpu`.
 */
export const SLEEP_THRESHOLD_MS_DEFAULT = 15;

/** The adapter summary `createWebGpuBackend({ device, adapter })` computes, for {@link backendFor}'s synchronous construction. */
function adapterSummary(device: GPUDevice, adapter: GPUAdapter | undefined): AdapterSummary {
  const info = adapter?.info ?? (device as { adapterInfo?: GPUAdapterInfo }).adapterInfo;
  const nav = (globalThis as { navigator?: { gpu?: unknown } }).navigator;
  const limits: Record<string, number> = {};
  for (const k of ["maxStorageBufferBindingSize", "maxBufferSize", "maxComputeWorkgroupStorageSize"]) {
    const v = (device.limits as unknown as Record<string, number>)[k];
    if (typeof v === "number") limits[k] = v;
  }
  return {
    vendor: info?.vendor ?? "",
    architecture: info?.architecture ?? "",
    device: info?.device ?? "",
    description: info?.description ?? "",
    // backend-webgpu keys its readback-sleep default on this (off for navigator.gpu).
    source: nav?.gpu ? "navigator.gpu" : "webgpu (Dawn)",
    features: [...device.features].map(String).sort(),
    limits,
  };
}

const backends = new WeakMap<GPUDevice, WebGpuBackend>();

/**
 * The backend that owns `device`'s GPU work — the one registered by
 * `createWebGpuDevice()`, else one created now for a device you requested
 * yourself (e.g. `detectWebGPU().device`). Subgroup-matrix GEMM is enabled
 * when `detectWebGPU()`/`registerGemmAdapter()` saw the adapter (Dawn does
 * not expose the needed adapter info on the device). The backend never
 * destroys a device it did not create.
 */
export function backendFor(device: GPUDevice): WebGpuBackend {
  let b = backends.get(device);
  if (!b) {
    const caps = gemmCapabilities(device);
    b = new WebGpuBackend(device, adapterSummary(device, gemmAdapter(device)), {
      f16: caps.f16,
      ownsDevice: false,
      subgroupMatrix: caps.subgroupMatrix,
      sleepThresholdMs: SLEEP_THRESHOLD_MS_DEFAULT,
    });
    backends.set(device, b);
  }
  return b;
}

/**
 * A new, unregistered backend from backend-webgpu's `createWebGpuBackend`
 * (which requests a device unless `opts.device` is given), with this
 * package's readback-sleep threshold and, for a device `detectWebGPU()`
 * created, the adapter it came from (subgroup-matrix detection needs it).
 */
export function createBackend(opts: CreateWebGpuBackendOptions): Promise<WebGpuBackend> {
  const adapter = opts.adapter ?? (opts.device ? gemmAdapter(opts.device) : undefined);
  return createWebGpuBackend({ sleepThresholdMs: SLEEP_THRESHOLD_MS_DEFAULT, ...opts, adapter });
}

/** Make `b` the backend for its device. Throws if another backend already owns that device (see the module doc). */
export function registerBackend(b: WebGpuBackend): void {
  const prev = backends.get(b.device);
  if (prev && prev !== b) {
    throw new Error("tensor-webgpu: this GPUDevice already has a backend (one per device; use createWebGpuDevice({ device }) to share it)");
  }
  backends.set(b.device, b);
}

/** Forget `b` (after `destroy()`), so a later lookup for its device does not return a destroyed backend. */
export function unregisterBackend(b: WebGpuBackend): void {
  if (backends.get(b.device) === b) backends.delete(b.device);
}

/** Bytes per element of `dtype` as the deprecated `GPUTensor` API stores it (f16 is always 2-byte binary16 bits). */
export function bytesPer(dtype: DType): number {
  return dtype === "f16" ? 2 : 4;
}

function numel(shape: Shape): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * Synchronous upload into a pooled tensor (`backend.empty` + the runtime's
 * `write`; `fromHost` is async by contract). `data`'s bytes are stored
 * as-is (f16 as binary16 bits). Pending dispatches are submitted first when
 * there are any, because the pooled buffer may be one they still read (the
 * backend's own upload tracks that per buffer; this path is conservative).
 * Tracked by the enclosing backend `scope`, if any; otherwise the caller
 * disposes it.
 */
export function uploadSync(b: WebGpuBackend, data: Float32Array | Uint16Array, shape: Shape, dtype: "f32" | "f16"): WebGpuTensor {
  const t = b.empty(shape, dtype);
  if (data.byteLength) b.rt.write(t.storage.buffer, b.rt.hasPending, data);
  return t;
}

/**
 * View a caller-owned `GPUBuffer` holding `dtype` data from byte 0 as a
 * tensor, without copying (`backend.wrapBuffer`: `dispose` never pools or
 * destroys it). f16 needs a device with `shader-f16`: without it
 * backend-webgpu stores f16 as f32, which a buffer of binary16 bits is not.
 */
export function wrapBuffer(b: WebGpuBackend, buffer: GPUBuffer, shape: Shape, dtype: "f32" | "f16"): WebGpuTensor {
  if (dtype === "f16" && !b.hasF16) {
    throw new TypeError("GPUTensor.fromBuffer: f16 needs a device with shader-f16 (backend-webgpu stores f16 as f32 without it); upload with fromFloat16Bits instead");
  }
  return b.wrapBuffer(buffer, shape, dtype);
}

/** Read `t`'s raw bytes back (flushes pending work first). */
export async function readRaw(b: WebGpuBackend, t: WebGpuTensor): Promise<ArrayBuffer> {
  const n = numel(t.shape) * bytesPer(t.dtype);
  if (n === 0) return new ArrayBuffer(0);
  return b.rt.readBytes(t.storage.buffer, t.offset * bytesPer(t.dtype), n);
}

/** The backend registered for `device`, if any (unlike {@link backendFor}, never creates one). */
export function lookupBackend(device: GPUDevice): WebGpuBackend | undefined {
  return backends.get(device);
}
