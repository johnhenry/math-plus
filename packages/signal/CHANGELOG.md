# @johnhenry/math-plus-signal

## 0.0.4

### Patch Changes

- Updated dependencies [f68063d]
  - @johnhenry/math-plus-tensor-core@0.2.0
  - @johnhenry/math-plus-fft@0.0.3

## 0.0.3

### Patch Changes

- c1e15b9: `resamplePoly`: the identity path (`up === down` after GCD reduction) no longer returns a dtype-mislabeled tensor. An f32 (or f16/integer) input used to come back labeled `f64` over its original `Float32Array` storage; it now returns a genuine f64 copy, like every other `resamplePoly` output. Inputs are cast to f64 up front, which also decodes f16/bf16 values instead of reading their bit patterns (#113).
- Updated dependencies [739e3be]
  - @johnhenry/math-plus-tensor-core@0.1.2

## 0.0.2

### Patch Changes

- 80123df: Republish with an npm provenance attestation. The previous versions were published from a local machine on 2026-09-23 without provenance (the CI `NPM_TOKEN` secret had expired). No code changes.
- Updated dependencies [80123df]
  - @johnhenry/math-plus-fft@0.0.2
  - @johnhenry/math-plus-scalar-types@0.0.2
  - @johnhenry/math-plus-tensor-core@0.1.1

## 0.0.1

### Patch Changes

- 5d7172b: Lower `engines.node` from `>=26.0.0` to `>=24.0.0`. Nothing in these packages needs Node 26: the full test suite passes on Node 24.9, and CI now tests Node 24. Every suite also runs under Bun 1.2.17 (`npm run test:bun`).

  `@johnhenry/math-plus-frame-parquet`: `scanParquet`/`scanParquetLazy` now accept an absolute file path without wildcards under Bun. Bun 1.2's `fs.promises.glob` returns no matches for such a path, so a pattern with no glob metacharacters is now resolved with `stat` on every runtime.

- Updated dependencies [866f3ef]
- Updated dependencies [5d7172b]
- Updated dependencies [648d5e0]
- Updated dependencies [6beb547]
- Updated dependencies [c00998a]
  - @johnhenry/math-plus-tensor-core@0.1.0
  - @johnhenry/math-plus-scalar-types@0.0.1
  - @johnhenry/math-plus-fft@0.0.1

## 0.3.0

### Minor Changes

- 262a154: Add `correlate2D`: true 2-D cross-correlation via FFT (existing `correlate` is 1-D with row/column batching, not genuine 2-D). Same `correlate(a,b) === convolve(a, flip(b))` convention as `correlate1D`. Upstream for the generalized Wang tile laboratory's autocorrelation-surface analysis (johnhenry/mallory#92). Fixes #84 (item 3 of 4).

### Patch Changes

- Updated dependencies [262a154]
- Updated dependencies [262a154]
  - @johnhenry/math-plus-fft@0.2.0
  - @johnhenry/math-plus-tensor-core@0.2.0

## 0.2.0

### Minor Changes

- 59162f6: Fixes johnhenry/math-plus#90: `butter()` gains `"bandpass"`/`"bandstop"` support, alongside the existing `"lowpass"`/`"highpass"`. `wn` takes a `[low, high]` pair for the two new types (a `number` still works for lowpass/highpass, unchanged and non-breaking) -- expressed via function overloads so the compiler enforces the right shape per `btype` at the call site.

  Implements scipy's `lp2bp_zpk`/`lp2bs_zpk` analog frequency transforms, and replaces the old lowpass/highpass-only `zpk2sos` shortcut (which assumed every digital zero was real and identical, a shape bandpass/bandstop's zeros don't have) with a general real-coefficient pairing: complex-conjugate pairs and leftover real values are grouped independently on the zero side and the pole side, which is provably always the same group count on both sides for any real-coefficient system with equal zero/pole counts -- see the module's own doc comment for the parity argument. Verified byte-identical output to the pre-#90 specialized `zpk2sos` for lowpass/highpass's own shape, and differentially tested end-to-end (via `sosFilter` vs. scipy's `sosfilt`) against real `scipy.signal.butter` for bandpass and bandstop across multiple orders, including the order-dependent edge case where the prototype's single real pole (odd order only) becomes either a complex-conjugate pair or two real poles depending on bandwidth/center-frequency.

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
  - @johnhenry/math-plus-fft@0.1.0
