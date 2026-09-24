/**
 * @johnhenry/math-plus-tensor-webgpu — math-plus's WebGPU device.
 *
 * Since issue #146 (RFC 0001 §12 Q6, path (a)) this is a device facade over
 * `@johnhenry/backend-webgpu`, the single WebGPU runtime: GEMM, fused
 * attention, the op set and the runtime (buffer pool, pipeline and
 * bind-group caches, batching) live there. This package adds explicit
 * async transfers to and from tensor-core `Tensor`s (`createWebGpuDevice`),
 * the tensor-compile IR -> WGSL elementwise fusion on that runtime, and the
 * measured WASM-vs-WebGPU GEMM threshold. Browsers via `navigator.gpu`;
 * Node/Bun via Dawn.
 *
 * The pre-#146 `GPUDevice` + `GPUTensor` API (`toWebGPU`, `runGemm*`,
 * `runAttention`, `runQKT`/`runSoftmax`/`runWeightedSum`,
 * `runElementwiseWGSL`, profiling, the `./dawn` subpath) still works, on
 * the same runtime, and is deprecated: see the README's migration table.
 */
export {
  createWebGpuDevice,
  webGpuUnavailableReason,
  WebGpuDevice,
  type WebGpuDeviceOptions,
  type WebGpuBackend,
  type WebGpuTensor,
} from "./facade.ts";
export { backendFor } from "./bridge.ts";
export {
  detectWebGPU,
  toWebGPU,
  GPUTensor,
  type WebGPUCapability,
  type DetectWebGPUOptions,
  type GPUDType,
} from "./device.ts";
export { GEMM_ELEMENT_THRESHOLD, GEMM_WORK_THRESHOLD, chooseGemmBackend } from "./threshold.ts";
export {
  runGemm,
  runGemmWGSL,
  runGemmF16WGSL,
  gemmKernelApplicable,
  type GemmDType,
  type GemmKernel,
  type GemmOptions,
} from "./gemm.ts";
export { registerGemmAdapter, gemmCapabilities, subgroupMatrixUsable, type GemmCapabilities } from "./gemm-caps.ts";
export { runQKT, runSoftmax, runWeightedSum, runAttention, type AttentionOptions } from "./attention.ts";
export { compileIRToWGSL, compileIRToElementwise, compileIRToKernel, type ElementwiseExpr, type ElementwiseWGSL } from "./fusion-wgsl.ts";
export { runElementwiseWGSL } from "./elementwise.ts";
export { configureGPURuntime, startProfiling, stopProfiling, type GPURuntimeOptions, type KernelTiming } from "./profiling.ts";
