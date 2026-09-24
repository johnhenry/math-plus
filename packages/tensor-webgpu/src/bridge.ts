/**
 * The one place this package creates `@johnhenry/backend-webgpu` backends
 * (issue #146, RFC 0001 §12 Q6 path (a)): backend-webgpu is the single
 * WebGPU runtime (buffer pool, pipeline and bind-group caches, uniform
 * arena, dispatch batching, readback sleep), and every GPU op this package
 * runs goes through it.
 *
 * One `WebGpuBackend` per `GPUDevice`. That is a correctness rule, not an
 * optimisation: a backend batches dispatches into a compute pass it submits
 * later, so two runtimes on one device could reorder each other's work.
 * `createWebGpuDevice()` registers the backend it creates here, and a later
 * `createWebGpuDevice({ device })` for the same device shares it.
 *
 * Only backend-webgpu's documented API is used: `createWebGpuBackend({
 * device, adapter })` and its `sleepThresholdMs` option. (The synchronous
 * `backendFor(device)` of the API removed in 0.3.0 was the one caller of
 * the undocumented `WebGpuBackend` constructor.)
 */
import { createWebGpuBackend, type CreateWebGpuBackendOptions, type WebGpuBackend } from "@johnhenry/backend-webgpu";
import { gemmAdapter } from "./gemm-caps.ts";

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

const backends = new WeakMap<GPUDevice, WebGpuBackend>();

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

/** The backend registered for `device`, if any (never creates one). */
export function lookupBackend(device: GPUDevice): WebGpuBackend | undefined {
  return backends.get(device);
}
