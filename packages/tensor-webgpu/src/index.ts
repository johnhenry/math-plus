/**
 * @johnhenry/math-plus-tensor-webgpu (issue #12) — WebGPU-accelerated GEMM and
 * attention-adjacent primitives for Math Plus tensors. Browsers via
 * `navigator.gpu`; Node/Bun via Dawn through the separate `./dawn` subpath
 * (optional `webgpu` peer dependency — see README.md "Node and Bun").
 *
 * See docs/spikes/webgpu-tiled-gemm.md (and the v1 baseline it supersedes,
 * docs/spikes/webgpu-baseline.md) for the measured GEMM WASM-vs-WebGPU
 * crossover this package's `chooseGemmBackend` is built on.
 */
export {
  detectWebGPU,
  toWebGPU,
  GPUTensor,
  type WebGPUCapability,
  type DetectWebGPUOptions,
  type GPUDType,
} from "./device.ts";
export { GEMM_ELEMENT_THRESHOLD, chooseGemmBackend } from "./threshold.ts";
export {
  runGemm,
  runGemmWGSL,
  runGemmF16WGSL,
  selectGemmKernel,
  gemmKernelApplicable,
  planGemm,
  GEMM_CONFIG,
  type GemmKernel,
  type GemmOptions,
  type GemmPlan,
} from "./gemm.ts";
export { registerGemmAdapter, gemmCapabilities, subgroupMatrixUsable, type GemmCapabilities } from "./gemm-caps.ts";
export {
  tiledGemmWGSL,
  skinnyGemmWGSL,
  subgroupMatrixGemmWGSL,
  type GemmDType,
  type TiledGemmConfig,
  type SkinnyGemmConfig,
  type SubgroupMatrixGemmConfig,
  type SubgroupMatrixSyntax,
} from "./gemm-kernels.ts";
export { runQKT, runSoftmax, runWeightedSum } from "./attention.ts";
export { compileIRToWGSL, type ElementwiseWGSL } from "./fusion-wgsl.ts";
export { runElementwiseWGSL } from "./elementwise.ts";
export {
  uploadStorageBuffer,
  writePadded,
  paddedByteLength,
  bindingOf,
  allocateOutputBuffer,
  allocateGPUResidentBuffer,
  acquireBuffer,
  releaseBuffer,
  destroyBufferPool,
  readBackFloat32,
  readBackBytes,
  dispatchCompute,
  getOrCreateComputePipeline,
  pipelineCacheSize,
  getKernel,
  getKernelChecked,
  parseWGSLBindings,
  dispatchKernel,
  writeBytes,
  configureGPURuntime,
  gpuRuntimeStats,
  startProfiling,
  stopProfiling,
  type BindingKind,
  type ComputeKernel,
  type DispatchOptions,
  type GPURuntimeOptions,
  type GPURuntimeStats,
  type KernelTiming,
  workgroupsFor,
  type SizedBuffer,
} from "./gpu-runtime.ts";
