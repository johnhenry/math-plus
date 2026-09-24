# @johnhenry/math-plus-tensor-webgpu

## 0.3.0

### Minor Changes

- 857ee36: Removes the API deprecated in 0.2.0, as its changeset promised for the first minor release after backend-webgpu's runtime hooks shipped (they did, in 0.2.1). `createWebGpuDevice()` is the package's API; the replacement for each removed name is in the README's "Removed in 0.3.0".

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

## 0.2.1

### Patch Changes

- d122aef: Published declarations no longer import `./x.ts` (closes #157). tsc's `rewriteRelativeImportExtensions` rewrites `.ts` specifiers to `.js` in emitted JS but not in emitted `.d.ts`, so `dist/*.d.ts` referenced files that aren't in the package, and Deno's type check failed on them. Every package's build now runs `scripts/rewrite-dts-extensions.mjs` after `tsc`:

  - relative `.ts` / `.mts` / `.cts` specifiers in `dist/**/*.d.ts` become `.js` / `.mjs` / `.cjs` (as in laya-js),
  - each `dist/*.js` with a declaration file starts with `// @ts-self-types="./x.d.ts"` (after the `#!` line of a bin), so Deno finds the types when it loads `dist/` as plain files or from a URL instead of through `npm:`. Source maps are shifted by the inserted line.

  No runtime change. The manifest drift test checks every built `dist/` for both.

- d122aef: Moves to `@johnhenry/backend-webgpu` 0.3.1's documented runtime hooks (dependency `^0.3.1`).

  - **Fusion broadcasts.** `gpu.fuse` / `gpu.compile` run on the backend's `elementwise` hook, so inputs broadcast with NumPy's rules, as in tensor-compile's CPU `forward` (for example `[B, N]` with `[N]` or `[B, 1]`). Before, every input had to have the same shape. It is still one dispatch per expression, and inputs are still f32. New export: `compileIRToElementwise(node, n)`, the lowering as the hook's expression and helpers. `compileIRToKernel` stays but `fuse` no longer uses it.
  - **Attention is fused on every device.** backend-webgpu 0.3.1 fits `sdpa` to the device's workgroup-memory limit, so `runAttention` no longer composes attention from matmul and softmax on devices below 32 KiB, such as one with the 16 KiB WebGPU default.
  - **Readback sleep defaults to a 15 ms threshold.** It was off. Backends created here now follow backend-webgpu's default (sleep under Dawn, not for `navigator.gpu`) with `sleepThresholdMs: 15`. Readbacks of a few milliseconds keep polling at full speed, and waits over 15 ms use about 3× less CPU (measured in `docs/spikes/webgpu-runtime.md`). `configureGPURuntime` takes `sleepThresholdMs` again.
  - `createWebGpuDevice({ device, adapter })`: a device you pass in goes through `createWebGpuBackend`, so subgroup-matrix GEMM is detected from `adapter`, or from the adapter `detectWebGPU()` used.
  - `GPUTensor.fromBuffer` wraps the buffer with the backend's `wrapBuffer`, so it is never pooled. f16 now needs a device with `shader-f16`, because without it backend-webgpu stores f16 as f32.
  - `src/bridge.ts` no longer uses undocumented backend internals, except for one call: the synchronous `backendFor(device)` of the deprecated API still calls the `WebGpuBackend` constructor.

  This is a patch release because the deprecated 0.1 API is scheduled for removal in the first minor release after these hooks shipped (see 0.2.0). That removal is a separate change.

- Updated dependencies [d122aef]
  - @johnhenry/math-plus-tensor-compile@0.1.3
  - @johnhenry/math-plus-tensor-core@0.2.1
  - @johnhenry/math-plus-tensor-cpu@0.2.1

## 0.2.0

### Minor Changes

- 8ec9c91: WebGPU convergence (#146, RFC 0001 §12 Q6 path (a)): `@johnhenry/backend-webgpu` is now the single WebGPU runtime, and this package is a device facade over it.

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

### Patch Changes

- Updated dependencies [8ec9c91]
  - @johnhenry/math-plus-tensor-cpu@0.2.0

## 0.1.3

### Patch Changes

- Updated dependencies [f68063d]
  - @johnhenry/math-plus-tensor-core@0.2.0
  - @johnhenry/math-plus-tensor-compile@0.1.2

## 0.1.2

### Patch Changes

- c1e15b9: Re-measure the WebGPU-vs-WASM GEMM crossover against tensor-wasm's SIMD128 GEMM (#130) with the thermal-aware method (docs/BENCHMARKING.md). `chooseGemmBackend(m, n, k?)` now takes an optional `k` and picks WebGPU only when `m·n >= GEMM_ELEMENT_THRESHOLD` (now `192 * 192`, was `128 * 128`) **and**, when `k` is given, `m·n·k >= GEMM_WORK_THRESHOLD` (new export, `2 ** 22`). On an Apple M2 the square crossover moved to n = 160 (Dawn) / n = 192 (headless Chrome), and small-k or small-output products at those sizes still lose to WASM. Numbers: docs/spikes/webgpu-tiled-gemm.md.
- Updated dependencies [739e3be]
  - @johnhenry/math-plus-tensor-core@0.1.2

## 0.1.1

### Patch Changes

- 80123df: Republish with an npm provenance attestation. The previous versions were published from a local machine on 2026-09-23 without provenance (the CI `NPM_TOKEN` secret had expired). No code changes.
- Updated dependencies [80123df]
  - @johnhenry/math-plus-tensor-compile@0.1.1
  - @johnhenry/math-plus-tensor-core@0.1.1

## 0.1.0

### Minor Changes

- 866f3ef: **BREAKING (pre-1.0, so a minor bump): `gelu()` now defaults to EXACT erf-GELU, not the tanh approximation.** One canonical double-precision `erf`/`erfc` for the monorepo (#122).

  - **Behaviour change — read this if you call `gelu()`:** `Tensor.gelu()`, `Variable.gelu()`, `Traced.gelu()` and the WGSL `gelu` lowering now compute exact `x·Φ(x) = 0.5·x·(1 + erf(x/√2))` by default, matching PyTorch's `nn.GELU()` / `F.gelu(x)` and what BERT/ModernBERT were trained with. Previously they always used the tanh approximation. Outputs move by up to ~4.7e-4 (at |x| ≈ 2.7). To keep the old numbers, pass `{ approximate: "tanh" }` (same option name and values as `torch.nn.functional.gelu(approximate=...)`). `Variable.gelu()`'s backward differentiates whichever mode ran.
  - **@johnhenry/math-plus-tensor-core:** new `src/special.ts`, the single canonical implementation: `erf`, `erfc` (~1e-15 / ~3.5e-15 relative; Maclaurin series below |x| = 1, Laplace continued fraction above, with an fdlibm-style split `exp(-z²)`), `gelu(x, approximate)`, `geluErf`, `geluTanh`, `geluDerivative`, plus `ERF_F32_PARAMS` for f32 lowerings. New `Tensor.erf()` / `Tensor.erfc()` (op-table parity with the compiled IR's `erf`) and `Tensor.gelu({ approximate })`. Exact GELU is computed as `0.5·x·erfc(-x/√2)` with the exponent taken from `x` directly, so it keeps full relative accuracy down its left tail instead of cancelling to 0.
  - **@johnhenry/math-plus-tensor-compile:** the Abramowitz & Stegun 7.1.26 `erf` copy (~1.5e-7 absolute) is gone; `erf` and both GELU modes evaluate tensor-core's canonical functions, so compiled and eager results are bit-identical. IR op `"gelu"` now means exact GELU; new `UnaryOp` `"gelu_tanh"` for the tanh form (`Traced.gelu({ approximate: "tanh" })`). Code that switches exhaustively over `UnaryOp` must add a `"gelu_tanh"` case.
  - **@johnhenry/math-plus-tensor-webgpu:** WGSL `math_plus_erf` is now an f32 lowering of the canonical algorithm (loop counts from `ERF_F32_PARAMS`, measured ~1.4e-7 absolute on a real adapter) with `math_plus_erfc` / `math_plus_gelu` alongside; IR `"gelu"` lowers to exact GELU, `"gelu_tanh"` to the tanh form.

  Not changed: `@johnhenry/math-plus-frame-arrow`'s `fn.erf` keeps its local A&S 7.1.26 copy (frame-arrow has no static dependency on tensor-core by design), so it now differs from the tensor-compile path by up to ~1.5e-7 — documented in `eval-expr.ts`. `@johnhenry/math`'s `SpecialFunctions.erf` (separate repo) is unchanged.

- 94527ec: WebGPU runtime improvements ported from laya-js (issue #126); measurements are in `docs/spikes/webgpu-runtime.md`.

  - New `runAttention(device, q, k, v, { mask, scale, skipMaskedTiles, kernel })`: fused flash-style attention in one dispatch, with an optional f32 mask (nonzero = attend) that broadcasts against `(batch, seqQ, seqK)`. Key tiles that no query can see are skipped; a ±64 sliding window at B=16, L=512, D=64 runs about 1.5-2x faster on an Apple M2. Two kernels: `fast` for head dim 32/64 and `generic` for any head dim. `planAttention` and the WGSL generators are exported. `scale` defaults to `1/sqrt(dim)`. Query rows where every key is masked return 0.
  - Dispatches now go through `dispatchKernel`. Bind-group layouts are parsed from each kernel's WGSL, bind groups are cached, and uniforms are written to a per-device ring buffer and bound with dynamic offsets. The cache key includes the uniform size (regression-tested).
  - Readbacks no longer busy-poll under Dawn. When there is no `navigator.gpu` and the expected wait is over 15 ms, `readBackBytes` sleeps for most of the wait before polling. Configure this with `configureGPURuntime(device, { sleepWhileWaiting, sleepThresholdMs })`.
  - Every upload goes through the new `writeBytes`, which always passes `(arrayBuffer, byteOffset, byteLength)`. Bun's Dawn binding ignores a view's `byteOffset` if you pass the view itself.
  - New GPU timestamp profiler, `startProfiling` / `stopProfiling`. It needs `timestamp-query`, which the new `detectWebGPU({ timestampQuery: true })` option requests.
  - `detectWebGPU()` also requests the adapter's maximum `maxComputeWorkgroupStorageSize`.
  - New `gpuRuntimeStats`, `getKernel`, `getKernelChecked` and `parseWGSLBindings`. `getOrCreateComputePipeline` now returns pipelines with explicit layouts for kernels whose bindings it can parse.

- 57b4669: Fast WebGPU GEMM: tiled, small-M "skinny", and (experimental, Dawn/Chromium) subgroup-matrix kernels ported from laya-js, replacing the naive one-thread-per-output shader.

  - f32, and f16 storage with f32 accumulation on devices with `shader-f16` (f16 data crosses the host boundary as binary16 bits in a `Uint16Array`, tensor-core's f16 representation).
  - New `runGemm` keeps GEMM GPU-resident (`GPUTensor` in/out); `runGemmWGSL` keeps its signature and gains `{ transB, kernel }` options; new `runGemmF16WGSL`.
  - `GPUTensor` can hold f16 (`fromFloat16Bits`, `toUint16Array`; `toTensor` returns an `"f16"` tensor); `toWebGPU` accepts f16 tensors. Its `dtype` field widens from `"f32"` to `"f32" | "f16"`.
  - `detectWebGPU(options)` accepts `{ gpu, powerPreference, f16, subgroupMatrix }`, requests `shader-f16` / subgroup matrices when offered, and reports `gemm` capabilities.
  - Node/Bun support via Dawn: new `@johnhenry/math-plus-tensor-webgpu/dawn` subpath (`requestDawnGPU`), with `webgpu` as an optional peer dependency.
  - `GEMM_ELEMENT_THRESHOLD` changes from `Infinity` to `128 * 128`: re-measured on an Apple M2 (docs/spikes/webgpu-tiled-gemm.md), so `chooseGemmBackend` now routes large GEMMs to WebGPU. One machine's number — re-measure on your hardware.

### Patch Changes

- 5d7172b: Lower `engines.node` from `>=26.0.0` to `>=24.0.0`. Nothing in these packages needs Node 26: the full test suite passes on Node 24.9, and CI now tests Node 24. Every suite also runs under Bun 1.2.17 (`npm run test:bun`).

  `@johnhenry/math-plus-frame-parquet`: `scanParquet`/`scanParquetLazy` now accept an absolute file path without wildcards under Bun. Bun 1.2's `fs.promises.glob` returns no matches for such a path, so a pattern with no glob metacharacters is now resolved with `stat` on every runtime.

- 57b4669: Fused WGSL `tanh` and `gelu_tanh` clamp tanh's argument to ±15 (exactly 1.0 in f32 beyond that): Metal via Dawn computes tanh through `exp` and returned NaN once it overflowed. Found by running the #122 accuracy tests on the new in-process Dawn harness.
- Updated dependencies [866f3ef]
- Updated dependencies [5d7172b]
- Updated dependencies [648d5e0]
- Updated dependencies [6beb547]
- Updated dependencies [c00998a]
  - @johnhenry/math-plus-tensor-core@0.1.0
  - @johnhenry/math-plus-tensor-compile@0.1.0

## 0.0.5

### Patch Changes

- Updated dependencies [262a154]
  - @johnhenry/math-plus-tensor-core@0.2.0
  - @johnhenry/math-plus-tensor-compile@0.0.5

## 0.0.4

### Patch Changes

- @johnhenry/math-plus-tensor-compile@0.0.4

## 0.0.3

### Patch Changes

- @johnhenry/math-plus-tensor-compile@0.0.3

## 0.0.2

### Patch Changes

- Updated dependencies [aeeeb35]
  - @johnhenry/math-plus-tensor-core@0.1.0
  - @johnhenry/math-plus-tensor-compile@0.0.2
