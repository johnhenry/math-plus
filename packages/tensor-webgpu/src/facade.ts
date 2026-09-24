/**
 * `createWebGpuDevice()` — math-plus's WebGPU device (issue #146, RFC 0001
 * §12 Q6 path (a)), the same shape as `@johnhenry/math-plus-tensor-mlx`'s
 * `createMlxDevice()`:
 *
 * - The runtime is `@johnhenry/backend-webgpu` (the one WebGPU
 *   implementation of the `@johnhenry/tensor-backend` contract); `backend`
 *   exposes it, and its ops (`matmul`, `linear`, `sdpa`, `softmax`,
 *   `layerNorm`, elementwise, reductions, the general-numerics section, …)
 *   are the device's ops.
 * - No global default device: you create one, and pass it around.
 * - Transfers are explicit and async in both directions (PLAN.md non-goal
 *   5; RFC §12 Q2): `await gpu.fromTensor(t)` uploads a tensor-core
 *   `Tensor`, `await gpu.toTensor(x)` downloads into a new one. Nothing is
 *   copied implicitly.
 * - What this package adds on top is the tensor-compile IR -> WGSL
 *   elementwise fusion (`fuse`, `compile`), running on the same runtime.
 */
import { isWebGpuAvailable, type AdapterSummary, type CreateWebGpuBackendOptions, type WebGpuBackend, type WebGpuTensor } from "@johnhenry/backend-webgpu";
import { Traced, type IRNode } from "@johnhenry/math-plus-tensor-compile";
import type { Tensor } from "@johnhenry/math-plus-tensor-core";
import { hostFromTensor, tensorFromHost } from "@johnhenry/math-plus-tensor-cpu";
import type { DType, HostTensor } from "@johnhenry/tensor-backend";
import { createBackend, registerBackend, unregisterBackend, lookupBackend } from "./bridge.ts";
import { encodeFused } from "./elementwise.ts";

export type { WebGpuBackend, WebGpuTensor };

export interface WebGpuDeviceOptions extends CreateWebGpuBackendOptions {
  /**
   * Use this `GPUDevice` (e.g. `detectWebGPU().device`) instead of
   * requesting one. The device's existing backend is shared if it has one
   * (one runtime per device). Subgroup-matrix GEMM is used when `adapter`
   * is given or `detectWebGPU()`/`registerGemmAdapter()` saw the device's
   * adapter. The device is never destroyed by `destroy()`. The other
   * options are ignored when a backend already exists for the device.
   */
  device?: GPUDevice;
}

/**
 * Creates a WebGPU device: requests an adapter and device (navigator.gpu in
 * browsers and Deno; Dawn in Node/Bun, created with `allow_unsafe_apis` so
 * subgroup matrices are available on Apple GPUs) unless `opts.device` is
 * given. Rejects when no adapter is available — check
 * {@link webGpuUnavailableReason} first to skip instead.
 */
export async function createWebGpuDevice(opts: WebGpuDeviceOptions = {}): Promise<WebGpuDevice> {
  const existing = opts.device && lookupBackend(opts.device);
  if (existing) return new WebGpuDevice(existing, false);
  const backend = await createBackend(opts);
  // A concurrent call for the same device may have registered one meanwhile: share it.
  const raced = opts.device && lookupBackend(opts.device);
  if (raced) {
    backend.destroy(); // releases only its (empty) pools: it does not own the device
    return new WebGpuDevice(raced, false);
  }
  registerBackend(backend);
  return new WebGpuDevice(backend, !opts.device);
}

/** Why a WebGPU device cannot be created in this runtime, or `null` when an adapter is available. For skipping tests and falling back. */
export async function webGpuUnavailableReason(): Promise<string | null> {
  try {
    return (await isWebGpuAvailable()) ? null : "no WebGPU adapter (navigator.gpu is absent or returned none, and Dawn via the `webgpu` package is unavailable or found no GPU)";
  } catch (e) {
    return `WebGPU probe failed: ${(e as Error).message}`;
  }
}

const LABEL = "tensor-webgpu";

export class WebGpuDevice {
  readonly name = "webgpu" as const;
  /** The `@johnhenry/tensor-backend` `Backend` (backend-webgpu): the device's ops, and the target of the conformance suite. */
  readonly backend: WebGpuBackend;
  readonly #owns: boolean;

  /** @internal Use `createWebGpuDevice()`. */
  constructor(backend: WebGpuBackend, owns: boolean) {
    this.backend = backend;
    this.#owns = owns;
  }

  /** The underlying `GPUDevice` (to share with other WebGPU code). */
  get device(): GPUDevice {
    return this.backend.device;
  }

  /** Adapter vendor/architecture, whether this is navigator.gpu or Dawn, features and limits. */
  get info(): AdapterSummary {
    return this.backend.adapterInfo;
  }

  /** Whether the device stores and computes `dtype` natively (f16 needs `shader-f16`; bf16 is stored as f32). */
  supports(dtype: DType): boolean {
    return this.backend.supports(dtype);
  }

  // ---- transfers (the only way data crosses the boundary) -------------------

  /**
   * Explicit upload of a tensor-core `Tensor` (one copy, from its own
   * storage into a GPU buffer). The tensor must be C-contiguous (call
   * `.contiguous()` first) and have a device dtype (f32/f16/bf16/i32/bool —
   * cast f64/i64/… explicitly first).
   */
  fromTensor(t: Tensor): Promise<WebGpuTensor> {
    let h: HostTensor;
    try {
      h = hostFromTensor(t, LABEL);
    } catch (e) {
      return Promise.reject(e);
    }
    return this.backend.fromHost(h);
  }

  /** Explicit upload of a tensor-backend `HostTensor` (one copy). */
  fromHost(h: HostTensor): Promise<WebGpuTensor> {
    return this.backend.fromHost(h);
  }

  /** Explicit download into a new tensor-core `Tensor` (submits pending work, one copy). */
  async toTensor(x: WebGpuTensor): Promise<Tensor> {
    return tensorFromHost(await this.backend.read(x));
  }

  /** Explicit download as a tensor-backend `HostTensor` (f16 as `Float16Array`, bf16 as raw bits). */
  toHost(x: WebGpuTensor): Promise<HostTensor> {
    return this.backend.read(x);
  }

  // ---- fusion (tensor-compile IR -> one WGSL dispatch) ------------------------

  /**
   * Evaluates a traced elementwise expression (an `IRNode`, or a `Traced`
   * built with `Traced.input(i)` / `compile`'s tracer) over `inputs` in ONE
   * dispatch: every op the expression chains is fused, with no intermediate
   * GPU buffer. Inputs are f32 and broadcast against each other like
   * NumPy's (and like tensor-compile's CPU `forward`): e.g. `[B, N]` with
   * `[N]` or `[B, 1]`. The result is a new f32 tensor of the broadcast
   * shape, tracked by the enclosing `scope`. Runs on backend-webgpu's
   * `elementwise` hook.
   */
  fuse(expr: IRNode | Traced, inputs: readonly WebGpuTensor[]): WebGpuTensor {
    const node = expr instanceof Traced ? expr.node : expr;
    if (!inputs.length) throw new RangeError("tensor-webgpu fuse: needs at least one input");
    return encodeFused(this.backend, node, inputs);
  }

  /**
   * Traces `fn` once (like tensor-compile's `compile`) and returns a
   * function that runs it fused on this device:
   *
   *     const f = gpu.compile(2, (x, y) => x.mul(y).add(1).gelu());
   *     const z = f(a, b); // one dispatch
   */
  compile(numInputs: number, fn: (...args: Traced[]) => Traced): (...inputs: WebGpuTensor[]) => WebGpuTensor {
    // The same trace tensor-compile's `compile` performs (its IR is private there).
    const node = fn(...Array.from({ length: numInputs }, (_, i) => Traced.input(i))).node;
    return (...inputs) => {
      if (inputs.length !== numInputs) throw new RangeError(`tensor-webgpu compile: expects ${numInputs} input(s), got ${inputs.length}`);
      return this.fuse(node, inputs);
    };
  }

  // ---- lifetime ----------------------------------------------------------------

  /** Runs `fn`; every tensor created inside and not returned (directly, or one level deep in an array/object) is disposed afterwards — also when `fn` throws. */
  scope<R>(fn: () => R): R {
    return this.backend.scope(fn);
  }

  /** Frees a tensor now (idempotent). */
  dispose(x: WebGpuTensor): void {
    this.backend.dispose(x);
  }

  /** Resolves when all submitted GPU work is done (benchmarking); throws the first uncaptured device error. */
  sync(): Promise<void> {
    return this.backend.sync();
  }

  /**
   * Releases the runtime's pooled buffers and pipelines, and the
   * `GPUDevice` if `createWebGpuDevice()` requested it. Tensors must not be
   * used afterwards. A device passed in with `{ device }` is left alive, and
   * its backend stays registered, so a later `createWebGpuDevice({ device })`
   * shares it.
   */
  destroy(): void {
    if (!this.#owns) {
      this.backend.rt.trim();
      return;
    }
    unregisterBackend(this.backend);
    this.backend.destroy();
  }
}
