# @johnhenry/math-plus-tensor-webgpu

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
