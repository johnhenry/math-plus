/**
 * `createWebGpuDevice()` — math-plus's WebGPU device (issue #146, RFC 0001
 * §12 Q6 path (a)), the same shape as `@johnhenry/math-plus-tensor-mlx`'s
 * `createMlxDevice()`, because it is the same class: `WebGpuDevice` extends
 * `@johnhenry/math-plus-tensor-cpu`'s `ArrayDevice`, so its arrays are the
 * chainable `DeviceArray` every math-plus device shares (one implementation
 * for CPU, MLX and WebGPU — AGENTS.md's canonical-implementation rule).
 *
 * - The runtime is `@johnhenry/backend-webgpu` (the one WebGPU
 *   implementation of the `@johnhenry/tensor-backend` contract); `backend`
 *   exposes it for the ops the arrays do not wrap (`linear`, `sdpa`, `rope`,
 *   slicing, …) — use `array.handle` to pass an array in and
 *   `gpu.wrap(handle)` to bring a result back.
 * - No global default device: you create one, and pass it around.
 * - Transfers are explicit and async in both directions (PLAN.md non-goal
 *   5; RFC §12 Q2): `await gpu.fromTensor(t)` uploads a tensor-core
 *   `Tensor` as a `WebGpuArray`, `await x.toTensor()` (or
 *   `gpu.toTensor(x)`) downloads into a new one. Nothing is copied
 *   implicitly.
 * - What this package adds on top is the tensor-compile IR -> WGSL
 *   elementwise fusion (`fuse`, `compile`), running on the same runtime.
 */
import { isWebGpuAvailable, type AdapterSummary, type CreateWebGpuBackendOptions, type WebGpuBackend, type WebGpuTensor } from "@johnhenry/backend-webgpu";
import { Traced, type IRNode } from "@johnhenry/math-plus-tensor-compile";
import type { Tensor } from "@johnhenry/math-plus-tensor-core";
import { ArrayDevice, DeviceArray, tensorFromHost } from "@johnhenry/math-plus-tensor-cpu";
import type { HostTensor } from "@johnhenry/tensor-backend";
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

/** An array of a {@link WebGpuDevice}: the chainable `DeviceArray` shared by every math-plus device. */
export type WebGpuArray = DeviceArray<WebGpuDevice>;

/** What the transfer, `dispose` and fusion methods accept: an array of this device, or a raw backend tensor. */
export type WebGpuInput = WebGpuArray | WebGpuTensor;

/** A fused function from `compile`: arrays in, an array out; raw backend tensors in, a raw tensor out. */
export interface FusedFunction {
  (...inputs: WebGpuArray[]): WebGpuArray;
  (...inputs: WebGpuTensor[]): WebGpuTensor;
}

/**
 * The WebGPU device. Transfers (`fromTensor`/`fromHost` → `WebGpuArray`),
 * `scope`, `eval`, `where`, `wrap` and `supports` are the shared
 * `ArrayDevice` ones; this class adds the GPU specifics (`device`, `info`,
 * `sync`, `destroy`) and the IR -> WGSL fusion.
 */
export class WebGpuDevice extends ArrayDevice<WebGpuBackend> {
  declare readonly name: "webgpu";
  readonly #owns: boolean;

  /** @internal Use `createWebGpuDevice()`. */
  constructor(backend: WebGpuBackend, owns: boolean) {
    super(backend, { label: "tensor-webgpu", device: "WebGpuDevice", array: "a WebGpuArray" });
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

  // ---- downloads (arrays also have their own toTensor/toHost) ----------------

  /** Explicit download into a new tensor-core `Tensor` (submits pending work, one copy). Same as `x.toTensor()` for an array. */
  async toTensor(x: WebGpuInput): Promise<Tensor> {
    return tensorFromHost(await this.toHost(x));
  }

  /** Explicit download as a tensor-backend `HostTensor` (f16 as `Float16Array`, bf16 as raw bits). */
  toHost(x: WebGpuInput): Promise<HostTensor> {
    return this.backend.read(this.#raw(x, "toHost"));
  }

  // ---- fusion (tensor-compile IR -> one WGSL dispatch) ------------------------

  /**
   * Evaluates a traced elementwise expression (an `IRNode`, or a `Traced`
   * built with `Traced.input(i)` / `compile`'s tracer) over `inputs` in ONE
   * dispatch: every op the expression chains is fused, with no intermediate
   * GPU buffer. Inputs are f32 and broadcast against each other like
   * NumPy's (and like tensor-compile's CPU `forward`): e.g. `[B, N]` with
   * `[N]` or `[B, 1]`. The result is a new f32 array of the broadcast
   * shape (a raw backend tensor when the inputs are raw tensors), tracked by
   * the enclosing `scope`. Runs on backend-webgpu's `elementwise` hook.
   */
  fuse(expr: IRNode | Traced, inputs: readonly WebGpuArray[]): WebGpuArray;
  fuse(expr: IRNode | Traced, inputs: readonly WebGpuTensor[]): WebGpuTensor;
  /** Mixed inputs: the result is an array when the first input is one. */
  fuse(expr: IRNode | Traced, inputs: readonly WebGpuInput[]): WebGpuInput;
  fuse(expr: IRNode | Traced, inputs: readonly WebGpuInput[]): WebGpuInput {
    const node = expr instanceof Traced ? expr.node : expr;
    if (!inputs.length) throw new RangeError("tensor-webgpu fuse: needs at least one input");
    const arrays = inputs[0] instanceof DeviceArray;
    const raw = inputs.map((x) => this.#raw(x, "fuse"));
    const out = encodeFused(this.backend, node, raw);
    return arrays ? this.wrap(out) : out;
  }

  /**
   * Traces `fn` once (like tensor-compile's `compile`) and returns a
   * function that runs it fused on this device:
   *
   *     const f = gpu.compile(2, (x, y) => x.mul(y).add(1).gelu());
   *     const z = f(a, b); // one dispatch
   */
  compile(numInputs: number, fn: (...args: Traced[]) => Traced): FusedFunction {
    // The same trace tensor-compile's `compile` performs (its IR is private there).
    const node = fn(...Array.from({ length: numInputs }, (_, i) => Traced.input(i))).node;
    return ((...inputs: WebGpuInput[]) => {
      if (inputs.length !== numInputs) throw new RangeError(`tensor-webgpu compile: expects ${numInputs} input(s), got ${inputs.length}`);
      return this.fuse(node, inputs as WebGpuArray[]);
    }) as FusedFunction;
  }

  // ---- lifetime ----------------------------------------------------------------

  /** Frees an array or a raw backend tensor now (idempotent). Same as `x.dispose()` for an array. */
  dispose(x: WebGpuInput): void {
    if (x instanceof DeviceArray) x.dispose();
    else this.backend.dispose(x);
  }

  /** Resolves when all submitted GPU work is done (benchmarking); throws the first uncaptured device error. */
  sync(): Promise<void> {
    return this.backend.sync();
  }

  /**
   * Releases the runtime's pooled buffers and pipelines, and the
   * `GPUDevice` if `createWebGpuDevice()` requested it. Arrays must not be
   * used afterwards. A device passed in with `{ device }` is left alive, and
   * its backend stays registered, so a later `createWebGpuDevice({ device })`
   * shares it.
   */
  override destroy(): void {
    if (!this.#owns) {
      this.backend.rt.trim();
      return;
    }
    unregisterBackend(this.backend);
    this.backend.destroy();
  }

  /** The backend tensor of an array of this device, or a raw backend tensor as is. */
  #raw(x: WebGpuInput, op: string): WebGpuTensor {
    return x instanceof DeviceArray ? this._own(x, op) : x;
  }
}
