# @johnhenry/math-plus-tensor-core

## 0.1.1

### Patch Changes

- 80123df: Republish with an npm provenance attestation. The previous versions were published from a local machine on 2026-09-23 without provenance (the CI `NPM_TOKEN` secret had expired). No code changes.

## 0.1.0

### Minor Changes

- 866f3ef: **BREAKING (pre-1.0, so a minor bump): `gelu()` now defaults to EXACT erf-GELU, not the tanh approximation.** One canonical double-precision `erf`/`erfc` for the monorepo (#122).

  - **Behaviour change — read this if you call `gelu()`:** `Tensor.gelu()`, `Variable.gelu()`, `Traced.gelu()` and the WGSL `gelu` lowering now compute exact `x·Φ(x) = 0.5·x·(1 + erf(x/√2))` by default, matching PyTorch's `nn.GELU()` / `F.gelu(x)` and what BERT/ModernBERT were trained with. Previously they always used the tanh approximation. Outputs move by up to ~4.7e-4 (at |x| ≈ 2.7). To keep the old numbers, pass `{ approximate: "tanh" }` (same option name and values as `torch.nn.functional.gelu(approximate=...)`). `Variable.gelu()`'s backward differentiates whichever mode ran.
  - **@johnhenry/math-plus-tensor-core:** new `src/special.ts`, the single canonical implementation: `erf`, `erfc` (~1e-15 / ~3.5e-15 relative; Maclaurin series below |x| = 1, Laplace continued fraction above, with an fdlibm-style split `exp(-z²)`), `gelu(x, approximate)`, `geluErf`, `geluTanh`, `geluDerivative`, plus `ERF_F32_PARAMS` for f32 lowerings. New `Tensor.erf()` / `Tensor.erfc()` (op-table parity with the compiled IR's `erf`) and `Tensor.gelu({ approximate })`. Exact GELU is computed as `0.5·x·erfc(-x/√2)` with the exponent taken from `x` directly, so it keeps full relative accuracy down its left tail instead of cancelling to 0.
  - **@johnhenry/math-plus-tensor-compile:** the Abramowitz & Stegun 7.1.26 `erf` copy (~1.5e-7 absolute) is gone; `erf` and both GELU modes evaluate tensor-core's canonical functions, so compiled and eager results are bit-identical. IR op `"gelu"` now means exact GELU; new `UnaryOp` `"gelu_tanh"` for the tanh form (`Traced.gelu({ approximate: "tanh" })`). Code that switches exhaustively over `UnaryOp` must add a `"gelu_tanh"` case.
  - **@johnhenry/math-plus-tensor-webgpu:** WGSL `math_plus_erf` is now an f32 lowering of the canonical algorithm (loop counts from `ERF_F32_PARAMS`, measured ~1.4e-7 absolute on a real adapter) with `math_plus_erfc` / `math_plus_gelu` alongside; IR `"gelu"` lowers to exact GELU, `"gelu_tanh"` to the tanh form.

  Not changed: `@johnhenry/math-plus-frame-arrow`'s `fn.erf` keeps its local A&S 7.1.26 copy (frame-arrow has no static dependency on tensor-core by design), so it now differs from the tensor-compile path by up to ~1.5e-7 — documented in `eval-expr.ts`. `@johnhenry/math`'s `SpecialFunctions.erf` (separate repo) is unchanged.

- 648d5e0: Real f16/bf16 support. `cast()` to/from `f16`/`bf16` now converts values (round-to-nearest-even, bit-for-bit equal to NumPy's `astype(float16)`) instead of truncating to integers and reinterpreting raw bits — a correctness fix. `from`/`full`/`arange`/`random.*` encode and `at`/`item`/`toArray` decode half dtypes; `.npy` read/write supports `<f2`. New exports `encodeHalf`/`decodeHalf`/`isHalfDType`. Arithmetic/comparison/reduction/sort/matmul kernels now throw a clear `TypeError` on half dtypes (they previously computed on bit patterns); `cast("f32")` first.
- c00998a: Half-precision follow-ups (issue #128):

  - New `withCompute("f32" | "f64", inputs, fn)`, an explicit opt-in for arithmetic on f16/bf16. It casts the half inputs to the compute dtype and passes other inputs through unchanged. It runs `fn`, then casts compute-dtype results back to the half dtype, rounding once when the region exits. A single op is bit-identical to NumPy's float16 ufuncs. Mixed f16/bf16 inputs, or no half input, throw. Plain half arithmetic still throws, and its error message now mentions `withCompute`.
  - `.npy` bf16 follows the `ml_dtypes` convention. `toNpy()` writes descr `'<V2'`, byte-identical to `np.save` of an `ml_dtypes.bfloat16` array. `Tensor.fromNpy(bytes, { voidAs: "bf16" })` reads such files. Without the option, a 2-byte void descr throws instead of being guessed.

### Patch Changes

- 5d7172b: Lower `engines.node` from `>=26.0.0` to `>=24.0.0`. Nothing in these packages needs Node 26: the full test suite passes on Node 24.9, and CI now tests Node 24. Every suite also runs under Bun 1.2.17 (`npm run test:bun`).

  `@johnhenry/math-plus-frame-parquet`: `scanParquet`/`scanParquetLazy` now accept an absolute file path without wildcards under Bun. Bun 1.2's `fs.promises.glob` returns no matches for such a path, so a pattern with no glob metacharacters is now resolved with `stat` on every runtime.

- 6beb547: Contiguous fast paths, blocked GEMM, and fused softmax/variance (no API change; fixes #120). Contiguous inputs to elementwise ops (same-shape, scalar, trailing-block "bias" and per-row broadcasts), unary ops, `cast`, `contiguous`, comparisons, and `sum`/`mean`/`min`/`max` now run flat typed-array loops instead of the per-element offset generator; `matmul` runs a register-blocked GEMM over packed f64 panels (~5–7× faster at 256²/1024² on an Apple M2); `softmax` and `variance`/`std` are fused single kernels with no temporaries. All fast paths are bit-identical to the general strided path, which still handles views, broadcasts, and i64/u64. Also fixes `broadcastShapes` turning a zero-size dim into 1 when broadcast against a size-1 dim (`[0, 4]` with `[4]` now gives `[0, 4]`, matching NumPy).

## 0.2.0

### Minor Changes

- 262a154: Add `Tensor.prototype.unfold(windowShape, axes?)`: sliding-window ("patch") view via NumPy's `sliding_window_view` stride trick, never copies. Upstream for the generalized Wang tile laboratory's patch-census machinery (johnhenry/mallory#92). Fixes #84 (item 1 of 4).

## 0.1.0

### Minor Changes

- aeeeb35: Gap-analysis backlog (issues #64-#72): additive new API surface across six packages, all backward-compatible.

  - **@johnhenry/math-plus-tensor-core**: eager `Tensor` unary op-table parity with the compiled IR (`exp`/`pow`/`abs`/`neg`/full trig+hyperbolic families/`cbrt`/`log10`/`log2`/`expm1`/`log1p`/`floor`/`ceil`/`round`/`trunc`), plus structural ops `clip`/`prod`/`pad`/`split`/`repeat`/`flip`/`roll`/`nonzero`.
  - **@johnhenry/math-plus-tensor-wasm**: `subInto`/`divInto` WASM kernels, parity with the existing `addInto`/`mulInto`.
  - **@johnhenry/math-plus-adapter-math**: `det`/`inv` (derived from the existing `lu`/`solve`), and `eigGeneral` — eigenvalues of a general non-symmetric real matrix via Hessenberg reduction + shifted QR, returned as `ComplexNumber[]` to support genuine complex-conjugate pairs.
  - **@johnhenry/math-plus-fft**: `fft2`/`ifft2` and `fftshift`/`ifftshift`.
  - **@johnhenry/math-plus-signal**: `correlate`/`correlate1D` (convolution's cross-correlation dual), `freqz` (SOS filter frequency response), `welch` (power spectral density).
  - **@johnhenry/math-plus-tensor-autograd**: `nn.Sequential`, `nn.Dropout`, `nn.huberLoss`, `nn.binaryCrossEntropy`; `optim.Adam`, `optim.RMSprop`, `optim.StepLR` (a learning-rate scheduler — `optim.SGD`/`optim.AdamW`'s `lr` field is now mutable rather than `readonly` to support this, a backward-compatible widening).
