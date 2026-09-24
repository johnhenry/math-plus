/**
 * Deprecated per-`GPUDevice` runtime knobs, kept through the deprecation
 * window (issue #146). They act on the device's `@johnhenry/backend-webgpu`
 * runtime (bridge.ts `backendFor`); new code uses
 * `createWebGpuDevice().backend.rt` directly (`startProfiling()`,
 * `stopProfiling()`, `sleepWhileWaiting`, `stats`, `trim()`).
 */
import { backendFor } from "./bridge.ts";

/** GPU time spent in one kernel while profiling. `kernel` is backend-webgpu's pipeline key (e.g. `gemmsg:…`, `sdpafast:…`), not this package's old labels. */
export interface KernelTiming {
  kernel: string;
  /** Total GPU milliseconds across `count` dispatches. */
  ms: number;
  count: number;
}

/** @deprecated Use `backend.rt.sleepWhileWaiting` on `createWebGpuDevice().backend`. */
export interface GPURuntimeOptions {
  /** Sleep for most of the expected GPU time before a readback instead of letting Dawn busy-poll `mapAsync`. Default: off (see bridge.ts `SLEEP_WHILE_WAITING_DEFAULT`; before #146 it was on under Dawn for waits over 15 ms). */
  sleepWhileWaiting?: boolean;
}

/**
 * Set runtime options for `device`'s backend.
 *
 * @deprecated See the module doc. (`sleepThresholdMs` is gone: backend-webgpu
 * sleeps only for expected waits over 3 ms.)
 */
export function configureGPURuntime(device: GPUDevice, options: GPURuntimeOptions): void {
  if (options.sleepWhileWaiting !== undefined) backendFor(device).rt.sleepWhileWaiting = options.sleepWhileWaiting;
}

/**
 * Start timing every dispatch on `device` (needs the `timestamp-query`
 * feature: `detectWebGPU({ timestampQuery: true })`; throws otherwise).
 *
 * @deprecated See the module doc.
 */
export function startProfiling(device: GPUDevice): void {
  backendFor(device).rt.startProfiling();
}

/**
 * Stop profiling and return GPU time per kernel, slowest first (`[]` when not profiling).
 *
 * @deprecated See the module doc.
 */
export function stopProfiling(device: GPUDevice): Promise<KernelTiming[]> {
  return backendFor(device).rt.stopProfiling();
}
