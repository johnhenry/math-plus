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

**GEMM threshold re-measured (patch-level note):** `chooseGemmBackend` now picks WebGPU when `m·n >= 256²` and `m·n·k >= 2²⁴` (was `192²` and `2²²`): `GEMM_ELEMENT_THRESHOLD = 256 * 256`, `GEMM_WORK_THRESHOLD = 2 ** 24`. Measured end to end through the facade (`matmul` / `linear`) against tensor-wasm's SIMD GEMM on 43 shapes, with the thermal-aware method, in Dawn, headless Chrome and a real, visible Chromium without subgroup matrices (new page: `scripts/gemm-threshold-page/`). Both browsers lost at 192³ (0.88x / 0.89x), and the visible one also lost 192x256x192, 192x1024x192 and the 16-row Linear (0.95-0.96x), all of which the old rule sent to WebGPU. The new rule sends no measured shape to a slower WebGPU in any of the three. Tables in `docs/spikes/webgpu-tiled-gemm.md`.
