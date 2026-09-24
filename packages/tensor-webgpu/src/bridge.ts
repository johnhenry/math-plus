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
 * backend-webgpu@0.3.0 has no documented hook for custom kernels or for
 * wrapping existing buffers, so this module uses members that 0.3.0 does
 * make public but does not document: `backend.rt.kernel/dispatch/acquire/
 * write/readBytes`, the `WebGpuTensor` constructor and the `WebGpuBackend`
 * constructor. The documented replacement (`WebGpuBackend.elementwise`,
 * `empty`, `wrapBuffer`, the `adapter` option and exported runtime types)
 * is proposed in johnhenry/laya-js (feat/runtime-hooks); once released,
 * only this file changes.
 */
import { WebGpuBackend, WebGpuTensor, type AdapterSummary } from "@johnhenry/backend-webgpu";
import type { DType, Shape } from "@johnhenry/tensor-backend";
import { gemmCapabilities } from "./gemm-caps.ts";

/** backend-webgpu's runtime (not exported by name from 0.3.0). */
export type Runtime = WebGpuBackend["rt"];
/** A kernel description the runtime compiles: bindings + uniform params + a WGSL body with entry point `main`. */
export type KernelSource = ReturnType<Parameters<Runtime["kernel"]>[0]>;
type Storage = WebGpuTensor["storage"];

const backends = new WeakMap<GPUDevice, WebGpuBackend>();

/**
 * Readback sleep default for backends this package creates: OFF. backend-
 * webgpu@0.3 sleeps (instead of letting Dawn busy-poll `mapAsync`) for any
 * expected wait over 3 ms, which measured +15–60% latency on this package's
 * 2–6 ms GEMM/attention readbacks (Apple M2, Dawn; the pre-#146 runtime only
 * slept above 15 ms for exactly this reason, docs/spikes/webgpu-runtime.md).
 * Opt in with `configureGPURuntime(device, { sleepWhileWaiting: true })` or
 * `createWebGpuDevice({ sleepWhileWaiting: true })` when CPU matters more
 * than latency (long waits under Bun). A threshold option is proposed
 * upstream (johnhenry/laya-js, feat/runtime-hooks).
 */
export const SLEEP_WHILE_WAITING_DEFAULT = false;

function adapterSummary(device: GPUDevice): AdapterSummary {
  const info = (device as { adapterInfo?: GPUAdapterInfo }).adapterInfo;
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
    // Informational (backend-webgpu keys its own sleep default on it; backendFor sets that flag explicitly).
    source: nav?.gpu ? "navigator.gpu" : "webgpu (Dawn)",
    features: [...device.features].map(String).sort(),
    limits,
  };
}

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
    b = new WebGpuBackend(device, adapterSummary(device), {
      f16: caps.f16,
      ownsDevice: false,
      subgroupMatrix: caps.subgroupMatrix,
      sleepWhileWaiting: SLEEP_WHILE_WAITING_DEFAULT,
    });
    backends.set(device, b);
  }
  return b;
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

/** Bytes per element of `dtype` as this package stores it (f16 is always 2-byte bits; backend-webgpu stores f16 natively when the device has shader-f16). */
export function bytesPer(dtype: DType): number {
  return dtype === "f16" ? 2 : 4;
}

function numel(shape: Shape): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * Synchronous upload into a pooled runtime buffer (the backend's own upload
 * path; `fromHost` is async only by contract). The result is NOT tracked by
 * any backend scope: the caller owns it (`backend.dispose`). `data`'s bytes
 * are stored as-is (f16 as binary16 bits).
 */
export function uploadSync(b: WebGpuBackend, data: Float32Array | Uint16Array, shape: Shape, dtype: "f32" | "f16"): WebGpuTensor {
  const { buffer, bytes, writeHazard } = b.rt.acquire(Math.max(4, data.byteLength));
  if (data.byteLength) b.rt.write(buffer, writeHazard, data);
  return new WebGpuTensor([...shape], dtype, { refs: 1, buffer, bytes } as Storage, 0);
}

/** An uninitialised, untracked output tensor from the runtime's pool (caller owns it). */
export function allocate(b: WebGpuBackend, shape: Shape, dtype: DType): WebGpuTensor {
  const { buffer, bytes } = b.rt.acquire(Math.max(4, numel(shape) * bytesPer(dtype)));
  return new WebGpuTensor([...shape], dtype, { refs: 1, buffer, bytes } as Storage, 0);
}

/**
 * View a caller-owned `GPUBuffer` as a tensor, without copying. Never pass
 * the result to `backend.dispose` (that would put the caller's buffer into
 * the runtime's pool); just drop it.
 */
export function wrapBuffer(buffer: GPUBuffer, shape: Shape, dtype: DType): WebGpuTensor {
  return new WebGpuTensor([...shape], dtype, { refs: 1, buffer, bytes: buffer.size } as Storage, 0);
}

/** Hand an untracked tensor to `b`'s scope machinery (as if an op had produced it): a tracked view replaces it. */
export function adopt(b: WebGpuBackend, t: WebGpuTensor): WebGpuTensor {
  const tracked = b.reshape(t, t.shape);
  b.dispose(t);
  return tracked;
}

/** Encode one dispatch of a custom kernel on `b`'s runtime (compiled once per `key`). */
export function dispatchCustom(
  b: WebGpuBackend,
  key: string,
  src: () => KernelSource,
  buffers: GPUBuffer[],
  params: Record<string, number | readonly number[]>,
  groups: readonly [number, number?, number?],
): void {
  b.rt.dispatch(b.rt.kernel(src, key), buffers, params, groups);
}

/** Workgroup grid for `n` workgroups within the 65535-per-dimension limit (x·y ≥ n; kernels index `wid.x + wid.y * nwg.x`). */
export function flatGrid(n: number): [number, number, number] {
  if (n <= 65535) return [Math.max(1, n), 1, 1];
  const y = Math.ceil(n / 65535);
  return [Math.ceil(n / y), y, 1];
}

/** Read `t`'s raw bytes back (flushes pending work first). */
export async function readRaw(b: WebGpuBackend, t: WebGpuTensor): Promise<ArrayBuffer> {
  const n = numel(t.shape) * bytesPer(t.dtype);
  if (n === 0) return new ArrayBuffer(0);
  return b.rt.readBytes(t.storage.buffer, t.offset * bytesPer(t.dtype), n);
}
