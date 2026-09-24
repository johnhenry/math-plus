---
"@johnhenry/math-plus-tensor-webgpu": minor
---

Removes the API deprecated in 0.2.0, as its changeset promised for the first minor release after backend-webgpu's runtime hooks shipped (they did, in 0.2.1). `createWebGpuDevice()` is the package's API; the replacement for each removed name is in the README's "Removed in 0.3.0".

**Removed:**

- `toWebGPU`, `GPUTensor` (and with it `gpu.toTensor()` on a `GPUTensor`, `fromFloat32Array` / `fromFloat16Bits` / `fromBuffer`, `toFloat32Array` / `toUint16Array`, `free`, `handle`) and the `GPUDType` type. Use `await gpu.fromTensor(t)` / `fromHost(h)`, `await gpu.toTensor(x)` / `toHost(x)`, `gpu.dispose(x)` / `gpu.scope(fn)`, and `gpu.backend.wrapBuffer(...)`.
- `runGemm`, `runGemmWGSL`, `runGemmF16WGSL`, `gemmKernelApplicable`, and the `GemmOptions` / `GemmKernel` / `GemmDType` types. Use `gpu.backend.matmul(a, b)` / `gpu.backend.linear(x, w, bias?)`.
- `runAttention`, `runQKT`, `runSoftmax`, `runWeightedSum`, and `AttentionOptions`. Use `gpu.backend.sdpa(q, k, v, boolMask, scale)` on `[B, H, L, D]` tensors, or `matmul` / `softmax`. Fully masked query rows are undefined there; the shim returned 0.
- `runElementwiseWGSL`. Use `gpu.fuse(node, tensors)` / `gpu.compile(n, fn)`.
- `startProfiling`, `stopProfiling`, `configureGPURuntime`, and the `KernelTiming` / `GPURuntimeOptions` types. Use `gpu.backend.rt.startProfiling()` / `stopProfiling()` / `sleepWhileWaiting` / `sleepThresholdMs`.
- `backendFor(device)`. Use `(await createWebGpuDevice({ device })).backend`: one backend per device is still enforced.
- The `./dawn` subpath (`requestDawnGPU`, `DawnOptions`). `createWebGpuDevice()` finds Dawn itself; backend-webgpu's `getGpu({ unsafe })` returns the raw `GPU`.
- Internals only those used: `src/gemm.ts`, `src/attention.ts`, `src/profiling.ts`, `src/dawn.ts`, and in `src/bridge.ts` the synchronous `backendFor` with its call to backend-webgpu's undocumented `WebGpuBackend` constructor, `uploadSync`, `wrapBuffer`, `readRaw`. The package now uses only backend-webgpu's documented API.

**Kept:** `createWebGpuDevice` / `WebGpuDevice` (`fuse`, `compile`, transfers, `scope`, `destroy`), `webGpuUnavailableReason`, `detectWebGPU`, `registerGemmAdapter`, `gemmCapabilities`, `subgroupMatrixUsable`, `compileIRToWGSL` / `compileIRToElementwise` / `compileIRToKernel`, `chooseGemmBackend` and the threshold constants.

The GEMM (every kernel family × f32/f16 × both B layouts) and fused-attention NumPy oracles and the fusion cross-checks now run through the facade; tests that only exercised the removed shims were deleted. `scripts/measure-gemm-threshold.ts` measures the facade (`gpu.backend.matmul` / `linear`).
