# @johnhenry/math-plus-tensor-autograd

## 0.1.1

### Patch Changes

- 80123df: Republish with an npm provenance attestation. The previous versions were published from a local machine on 2026-09-23 without provenance (the CI `NPM_TOKEN` secret had expired). No code changes.
- Updated dependencies [80123df]
  - @johnhenry/math-plus-telemetry@0.0.2
  - @johnhenry/math-plus-tensor-core@0.1.1

## 0.1.0

### Minor Changes

- 254c35f: Transformer readiness (#123). **Breaking:** `nn.Linear` now stores PyTorch's `[out, in]` weight layout, and every layer's parameters default to **f32** (were f64) with a new `dtype` option (`"f32" | "f64" | "f16" | "bf16"`; half dtypes are storage-only, upcast on the fly). Old checkpoints keep loading: `io.writeCheckpoint` now writes MPCK version 2, and `loadCheckpoint` tags version-1 files so `Module.loadStateDict` transposes Linear weights and casts to the parameters' dtype (or pass `{ legacyLinearLayout: true }`). `loadStateDict` gains PyTorch semantics: shape checks, dtype casting, atomic updates, and `{ strict: false }`. Migration notes in the package README.

  New: `Variable` view ops `reshape`/`permute`/`transpose(dim0, dim1)`/`slice`/`narrow`/`Variable.concat`, plus `exp`, `tanh`, `maskedFill`, `cast`, `div(number)` and batched `matmul`; layers `scaledDotProductAttention` (bool/float masks, `isCausal`), `MultiheadAttention` (PyTorch state-dict keys, optional `rotary`), `RotaryEmbedding` (split-half RoPE), `LayerNorm({ bias: false })`, `TransformerEncoderLayer` (`normFirst`, `bias`, relu/gelu), `GeGLU`/`geglu` (both, and `TransformerEncoderLayer`'s `"gelu"`, use exact erf-GELU via `Variable.gelu({ approximate: "none" })` from #122, like PyTorch); `Module.namedModules()`; and a `@johnhenry/math-plus-tensor-autograd/safetensors` subpath (optional peer `@johnhenry/math-plus-safetensors`) with `saveSafetensors`, `stateDictFromSafetensors`, `loadSafetensors`, `loadSafetensorsInto`. Every new op and layer is differential-tested against PyTorch, forward and backward.

- 866f3ef: **BREAKING (pre-1.0, so a minor bump): `gelu()` now defaults to EXACT erf-GELU, not the tanh approximation.** One canonical double-precision `erf`/`erfc` for the monorepo (#122).

  - **Behaviour change — read this if you call `gelu()`:** `Tensor.gelu()`, `Variable.gelu()`, `Traced.gelu()` and the WGSL `gelu` lowering now compute exact `x·Φ(x) = 0.5·x·(1 + erf(x/√2))` by default, matching PyTorch's `nn.GELU()` / `F.gelu(x)` and what BERT/ModernBERT were trained with. Previously they always used the tanh approximation. Outputs move by up to ~4.7e-4 (at |x| ≈ 2.7). To keep the old numbers, pass `{ approximate: "tanh" }` (same option name and values as `torch.nn.functional.gelu(approximate=...)`). `Variable.gelu()`'s backward differentiates whichever mode ran.
  - **@johnhenry/math-plus-tensor-core:** new `src/special.ts`, the single canonical implementation: `erf`, `erfc` (~1e-15 / ~3.5e-15 relative; Maclaurin series below |x| = 1, Laplace continued fraction above, with an fdlibm-style split `exp(-z²)`), `gelu(x, approximate)`, `geluErf`, `geluTanh`, `geluDerivative`, plus `ERF_F32_PARAMS` for f32 lowerings. New `Tensor.erf()` / `Tensor.erfc()` (op-table parity with the compiled IR's `erf`) and `Tensor.gelu({ approximate })`. Exact GELU is computed as `0.5·x·erfc(-x/√2)` with the exponent taken from `x` directly, so it keeps full relative accuracy down its left tail instead of cancelling to 0.
  - **@johnhenry/math-plus-tensor-compile:** the Abramowitz & Stegun 7.1.26 `erf` copy (~1.5e-7 absolute) is gone; `erf` and both GELU modes evaluate tensor-core's canonical functions, so compiled and eager results are bit-identical. IR op `"gelu"` now means exact GELU; new `UnaryOp` `"gelu_tanh"` for the tanh form (`Traced.gelu({ approximate: "tanh" })`). Code that switches exhaustively over `UnaryOp` must add a `"gelu_tanh"` case.
  - **@johnhenry/math-plus-tensor-webgpu:** WGSL `math_plus_erf` is now an f32 lowering of the canonical algorithm (loop counts from `ERF_F32_PARAMS`, measured ~1.4e-7 absolute on a real adapter) with `math_plus_erfc` / `math_plus_gelu` alongside; IR `"gelu"` lowers to exact GELU, `"gelu_tanh"` to the tanh form.

  Not changed: `@johnhenry/math-plus-frame-arrow`'s `fn.erf` keeps its local A&S 7.1.26 copy (frame-arrow has no static dependency on tensor-core by design), so it now differs from the tensor-compile path by up to ~1.5e-7 — documented in `eval-expr.ts`. `@johnhenry/math`'s `SpecialFunctions.erf` (separate repo) is unchanged.

### Patch Changes

- 5d7172b: Lower `engines.node` from `>=26.0.0` to `>=24.0.0`. Nothing in these packages needs Node 26: the full test suite passes on Node 24.9, and CI now tests Node 24. Every suite also runs under Bun 1.2.17 (`npm run test:bun`).

  `@johnhenry/math-plus-frame-parquet`: `scanParquet`/`scanParquetLazy` now accept an absolute file path without wildcards under Bun. Bun 1.2's `fs.promises.glob` returns no matches for such a path, so a pattern with no glob metacharacters is now resolved with `stat` on every runtime.

- e9b691d: Widen internal peer-dependency ranges to `^0.0.0 || ^0.1.0` so the 0.1.0 releases of tensor-core and safetensors stay in range (Changesets would otherwise force a major bump on every peer dependent).
- c00998a: `io.loadCheckpoint` now reads bf16 entries. `writeCheckpoint` stores them as `.npy` with the `ml_dtypes` `'<V2'` descr, now that tensor-core's `toNpy()` supports bf16. Without this change a bf16 checkpoint could be saved but not loaded.
- Updated dependencies [866f3ef]
- Updated dependencies [5d7172b]
- Updated dependencies [648d5e0]
- Updated dependencies [6beb547]
- Updated dependencies [c00998a]
  - @johnhenry/math-plus-tensor-core@0.1.0
  - @johnhenry/math-plus-telemetry@0.0.1

## 0.2.2

### Patch Changes

- Updated dependencies [262a154]
  - @johnhenry/math-plus-tensor-core@0.2.0

## 0.2.1

### Patch Changes

- 3f3824a: Fix `nn.binaryCrossEntropy` returning NaN once a classifier converges well (saturated logits, `|z| >~ 37`). Reformulated using the standard numerically-stable BCEWithLogits formula, `relu(z) - z*y + log(1+exp(-|z|))`, built from existing `relu`/`sigmoid`/`log` ops (no `exp`/`abs` ops needed — `log(1+exp(-|z|))` rewritten as `-log(sigmoid(|z|))`, and `|z|` as `relu(z) + relu(-z)`). Byte-equivalent to the prior formula in the non-saturated regime (verified to ~1e-15); now finite everywhere. Fixes #85.

## 0.2.0

### Minor Changes

- 21981eb: Fixes johnhenry/math-plus#89: `optim.SGD` gains an optional `momentum`/`nesterov` option, following PyTorch's own update convention (`buf = momentum*buf + grad`, Nesterov's lookahead `d_p = grad + momentum*buf` applied after the buffer update). Both default to off (`0`/`false`), so `new SGD(params, { lr })` is byte-identical to the pre-#89 plain-SGD update -- no existing caller's behavior changes. Constructing with `nesterov: true` and no (or zero) `momentum` throws a `RangeError`, since Nesterov's lookahead is meaningless without a momentum term to look ahead with.

## 0.1.0

### Minor Changes

- aeeeb35: Gap-analysis backlog (issues #64-#72): additive new API surface across six packages, all backward-compatible.

  - **@johnhenry/math-plus-tensor-core**: eager `Tensor` unary op-table parity with the compiled IR (`exp`/`pow`/`abs`/`neg`/full trig+hyperbolic families/`cbrt`/`log10`/`log2`/`expm1`/`log1p`/`floor`/`ceil`/`round`/`trunc`), plus structural ops `clip`/`prod`/`pad`/`split`/`repeat`/`flip`/`roll`/`nonzero`.
  - **@johnhenry/math-plus-tensor-wasm**: `subInto`/`divInto` WASM kernels, parity with the existing `addInto`/`mulInto`.
  - **@johnhenry/math-plus-adapter-math**: `det`/`inv` (derived from the existing `lu`/`solve`), and `eigGeneral` — eigenvalues of a general non-symmetric real matrix via Hessenberg reduction + shifted QR, returned as `ComplexNumber[]` to support genuine complex-conjugate pairs.
  - **@johnhenry/math-plus-fft**: `fft2`/`ifft2` and `fftshift`/`ifftshift`.
  - **@johnhenry/math-plus-signal**: `correlate`/`correlate1D` (convolution's cross-correlation dual), `freqz` (SOS filter frequency response), `welch` (power spectral density).
  - **@johnhenry/math-plus-tensor-autograd**: `nn.Sequential`, `nn.Dropout`, `nn.huberLoss`, `nn.binaryCrossEntropy`; `optim.Adam`, `optim.RMSprop`, `optim.StepLR` (a learning-rate scheduler — `optim.SGD`/`optim.AdamW`'s `lr` field is now mutable rather than `readonly` to support this, a backward-compatible widening).

### Patch Changes

- Updated dependencies [aeeeb35]
  - @johnhenry/math-plus-tensor-core@0.1.0
