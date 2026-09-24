---
"@johnhenry/math-plus-tensor-webgpu": minor
---

WebGPU convergence (#146, RFC 0001 §12 Q6 path (a)): `@johnhenry/backend-webgpu` is now the single WebGPU runtime, and this package is a device facade over it.

**New:**
- `createWebGpuDevice(opts?)` returns a `WebGpuDevice`, the same shape as tensor-mlx's `createMlxDevice`. There is no global default device, and transfers are explicit and async: `await gpu.fromTensor(t)` / `fromHost(h)`, `await gpu.toTensor(x)` / `toHost(x)`. `gpu.backend` holds the tensor-backend ops. There are also `scope`, `dispose`, `sync`, `destroy`, and `webGpuUnavailableReason()`.
- IR -> WGSL elementwise fusion now runs on backend-webgpu's runtime: `gpu.fuse(irNode | traced, tensors)`, `gpu.compile(n, fn)`, and `compileIRToKernel`.
- `backendFor(device)` enforces one backend per `GPUDevice`. `GPUTensor.handle` exposes the backend tensor, for step-by-step migration.
- The tensor-backend conformance suite runs through the facade.

**Deprecated** (kept working on backend-webgpu; JSDoc `@deprecated`; migration table in the README):
- `toWebGPU`, `GPUTensor` and `gpu.toTensor()`
- `runGemm`, `runGemmWGSL`, `runGemmF16WGSL`
- `runAttention`, `runQKT`, `runSoftmax`, `runWeightedSum`
- `runElementwiseWGSL`
- `startProfiling`, `stopProfiling`, `configureGPURuntime`
- the `./dawn` subpath (`requestDawnGPU` now delegates to backend-webgpu's `getGpu`)

`detectWebGPU`, `registerGemmAdapter`, `gemmCapabilities`, `chooseGemmBackend` and the threshold constants stay supported.

**Removal plan:** the deprecated surface will be removed in the first minor release after laya-js publishes the documented runtime hooks (johnhenry/laya-js#10: `WebGpuBackend.elementwise`, `empty`, `wrapBuffer`, the `adapter` option, a limit-aware `sdpa` and a readback-sleep threshold). It will be at least one minor release after this one.

**Removed** (the duplicated kernels and runtime, per the canonical-implementation rule):
- modules: `gemm-kernels.ts`, `attention-kernels.ts`, `gpu-runtime.ts`
- GEMM exports: `planGemm`, `selectGemmKernel`, `GEMM_CONFIG`, `tiledGemmWGSL`, `skinnyGemmWGSL`, `subgroupMatrixGemmWGSL`, `GemmPlan`, the `*GemmConfig` types
- attention exports: `planAttention`, `fastAttentionWGSL`, `genericAttentionWGSL`, `genericAttentionConfig`, `AttentionPlan`, `AttentionKernel`, `AttentionVariant`
- runtime exports: `uploadStorageBuffer`, `writePadded`, `paddedByteLength`, `bindingOf`, `allocateOutputBuffer`, `allocateGPUResidentBuffer`, `acquireBuffer`, `releaseBuffer`, `destroyBufferPool`, `readBackFloat32`, `readBackBytes`, `dispatchCompute`, `getOrCreateComputePipeline`, `pipelineCacheSize`, `getKernel`, `getKernelChecked`, `parseWGSLBindings`, `dispatchKernel`, `writeBytes`, `gpuRuntimeStats`, `workgroupsFor` and their types
- `scripts/measure-runtime.ts`

**Behaviour changes in the shims:**
- GEMM uses backend-webgpu's kernels. Forced `kernel` families map onto its tuning table. Skinny needs M ≤ 64. For A·B, B is transposed once to reach the subgroup-matrix `linear` path.
- `runAttention` ignores `skipMaskedTiles` and `kernel: "generic"`, and still returns 0 for fully masked rows. Devices with less than 32 KiB of workgroup memory use a composed matmul/softmax path.
- Readback sleep is off by default for backends created here, because backend-webgpu's 3 ms threshold measured 15–60% more latency on typical readbacks. `sleepThresholdMs` is gone.
- `webgpu` is no longer a peer or dev dependency: it comes with backend-webgpu.

**Performance** (Apple M2, Dawn, old and new interleaved in one process; median per call including readback):
- GEMM 1024³ resident: A·B 4.5 → 2.4 ms, A·Bᵀ 4.5 → 2.3 ms
- attention B16×L512×D64: 3.1 → 2.7 ms, with a ±64 window 2.9 → 2.1 ms

`chooseGemmBackend` was re-measured on the new path and is unchanged (docs/spikes/webgpu-tiled-gemm.md).
