/**
 * @johnhenry/math-plus-tensor-webgpu — math-plus's WebGPU device.
 *
 * Since issue #146 (RFC 0001 §12 Q6, path (a)) this is a device facade over
 * `@johnhenry/backend-webgpu`, the single WebGPU runtime: GEMM, fused
 * attention, the op set and the runtime (buffer pool, pipeline and
 * bind-group caches, batching) live there. This package adds explicit
 * async transfers to and from tensor-core `Tensor`s (`createWebGpuDevice`,
 * whose arrays are the chainable `DeviceArray` shared with tensor-mlx and
 * tensor-cpu),
 * the tensor-compile IR -> WGSL elementwise fusion on that runtime, and the
 * measured WASM-vs-WebGPU GEMM threshold. Browsers via `navigator.gpu`;
 * Node/Bun via Dawn.
 *
 * The pre-#146 `GPUDevice` + `GPUTensor` API (`toWebGPU`, `runGemm*`,
 * `runAttention`, `runQKT`/`runSoftmax`/`runWeightedSum`,
 * `runElementwiseWGSL`, the profiling functions, `backendFor`, the `./dawn`
 * subpath) was removed in 0.3.0: see the README's "Removed in 0.3.0".
 */
export {
  createWebGpuDevice,
  webGpuUnavailableReason,
  WebGpuDevice,
  type WebGpuArray,
  type WebGpuInput,
  type FusedFunction,
  type WebGpuDeviceOptions,
  type WebGpuBackend,
  type WebGpuTensor,
} from "./facade.ts";
export { detectWebGPU, type WebGPUCapability, type DetectWebGPUOptions } from "./device.ts";
export { GEMM_ELEMENT_THRESHOLD, GEMM_WORK_THRESHOLD, chooseGemmBackend } from "./threshold.ts";
export { registerGemmAdapter, gemmCapabilities, subgroupMatrixUsable, type GemmCapabilities } from "./gemm-caps.ts";
export { compileIRToWGSL, compileIRToElementwise, compileIRToKernel, type ElementwiseExpr, type ElementwiseWGSL } from "./fusion-wgsl.ts";
