# @johnhenry/math-plus-signal

## 0.3.0

### Minor Changes

- f89f166: BREAKING: closes the remaining SciPy-parity gaps (SciPy 1.16), and SciPy wins where it differs from 0.2. New: `periodogram`, `csd`, `spectrogram` (modes `psd`, `complex`, `magnitude`, `angle`, `phase`) and `coherence`, with SciPy's signatures, defaults, N-D `axis`, complex input and NumPy broadcasting of `x` and `y` (`{ freqs, Pxx }`, `{ freqs, Pxy }`, `{ freqs, times, Sxx }`, `{ freqs, Cxy }`), also exported from `./web`; they share their segmenting, windowing and DFT with `welch` and `stft` (`welch` is now `csd(x, x).real`, as in SciPy). Every window and alias in SciPy's own table works in `getWindow` and every `window` option, including the new `["chebwin", at]`, `["dpss", NW]`, `lanczos` and `["kbd", beta]` (`"kaiser bessel derived"`, symmetric only), with `WINDOW_NAMES` exported; all checked at 1e-12 against `get_window`. Empty input now returns empty arrays like SciPy (or raises where SciPy does) from every spectral function, `getWindow(name, 0)` returns an empty window, and an `nperseg` longer than the signal now warns with SciPy's text through `onWarning` (default `console.warn`). Breaking: `stft`/`istft` `scaling` is only `"spectrum"` or `"psd"` (`"density"` and `"none"` now throw, as in SciPy); `stft`'s `nperseg` defaults to 256 even with an array `window`; `nperseg`/`noverlap`/`nfft`/`axis` are truncated like Python's `int()` and a negative `noverlap` is allowed; `average` is only validated with more than one segment; a bare number is not a `window` for the spectral functions; wrong window parameter counts throw `TypeError` (SciPy's) instead of `RangeError`; warning messages are SciPy's own text, unprefixed. See the README's "Migrating from 0.2" and "Not supported" sections. Follow-up to #178. Unblocks johnhenry/viaframe#551.

## 0.2.0

### Minor Changes

- 74b8842: BREAKING: `welch`, `stft` and `istft` now match SciPy by default: with no options they equal `scipy.signal.welch(x)`, `scipy.signal.stft(x)` and `scipy.signal.istft(Zxx)`, and every option takes SciPy's default and meaning. Changed defaults: `welch` `onesided` false -> true (`floor(nfft/2)+1` bins), `welch` `detrend` false -> `"constant"`, `welch`/`stft` `nfft` next power of two -> `nperseg`, `stft` `onesided` false -> true, `stft` `scaling` `"none"` -> `"spectrum"` (`"psd"` added, `"density"` kept as an alias, `"none"` kept as a non-SciPy raw FFT), `stft` `boundary` none -> `"zeros"` (`"even"`, `"odd"`, `"constant"`, `null` also accepted), `stft` `padded` false -> true, and an `nperseg` longer than the signal is now clamped instead of throwing. New: `welch` `average` (`"mean"`/`"median"`), `nfft` may be any integer `>= nperseg` (Bluestein on top of the radix-2 FFT, no power-of-two restriction). New return shapes: `stft` returns `{ freqs, times, Zxx }` with `Zxx` a `[freqs, times]` ComplexTensor (the frame-major tensor with `.freqs`/`.times`/`.frames` is gone, and the `Spectrogram` type is replaced by `StftResult`); `istft(Zxx, opts)` takes that `Zxx` and returns `{ times, x }`, supports SciPy's `nperseg`, `noverlap`, `nfft`, `window`, `fs`, `inputOnesided`, `boundary` and `scaling`, and round-trips `stft` defaults over the whole signal. `detrend: "linear"` on segments of length 1-2 now gives zeros, as SciPy does. See the README's "Matches SciPy" and migration notes. Unblocks johnhenry/viaframe#551.
- 44ba419: BREAKING: `welch` now returns `{ freqs, Pxx }` (was `{ frequencies, psd }`), mirroring SciPy's `(f, Pxx)` and `stft`'s `{ freqs, times, Zxx }`; no alias is kept. `istft` with `inputOnesided: false` now returns a complex `x` (a `ComplexTensor`, as SciPy does) instead of only its real part, so take `.real` where you relied on the old result. `getWindow` is computed with SciPy's own formulas and can differ from `hannWindow` in the last bit. The rest is additive and closes the SciPy-parity gaps the README used to list: N-D input (`welch`/`stft` take `axis`, `istft` takes `freqAxis` and `timeAxis`, with SciPy's output shapes); complex input to `welch` and `stft` (a `ComplexTensor`; forced two-sided in FFT order, `stft` warns as SciPy does); callable `detrend` (`detrend(segment) -> segment`, per segment); `(name, param)` window tuples (`["kaiser", beta]`, `["gaussian", std]`, `["general_gaussian", p, sig]`, `["tukey", alpha]`, `["exponential", center, tau]`, `["general_hamming", alpha]`, `["general_cosine", coefficients]`, `["taylor", nbar, sll, norm]`) and the names `bartlett`, `flattop`, `parzen`, `bohman`, `nuttall`, `blackmanharris`, `barthann`, `cosine`, `triang`, `tukey`, `exponential` and `taylor`, with `getWindow(window, n, fftbins = true)` matching `scipy.signal.get_window`; new `checkNOLA` and `checkCOLA` (`scipy.signal.check_NOLA` / `check_COLA`); and an `onWarning(message)` option on `welch`, `stft` and `istft` (default `console.warn`) used for `istft`'s failed-NOLA warning and `stft`'s complex-input warning. See the README's "Matches SciPy" section for what is still not supported. Follow-up to #178. Unblocks johnhenry/viaframe#551.
- 7e7dfb9: `welch` and `stft` take SciPy's options: `fs`, `onesided`, `detrend` (`"constant"`, `"linear"`), `scaling`, any `nperseg` with a zero-padding `nfft`, and `window` as `"hann"`, `"hamming"`, `"blackman"`, `"boxcar"`/`"rectangular"` or a `Float64Array`. `stft` also carries `freqs`, `times` (segment centres) and `frames`. New `getWindow` (periodic, as `scipy.signal.get_window`). Defaults keep the previous behaviour. Fixes #178.

## 0.1.0

### Minor Changes

- 1c9b6e5: Add `sosfiltfilt` (zero-phase forward-backward SOS filtering with scipy's odd padding and steady-state initial conditions) and a browser/Web Worker entry, `@johnhenry/math-plus-signal/web`, exporting `rfft`, `welch`, `stft`, `butter`, `sosfiltfilt`, `findPeaks` and `resamplePoly` (plus `Tensor`/`ComplexTensor`). The entry's import graph contains no Node built-ins or Node globals; a test enforces this (#173).

## 0.0.5

### Patch Changes

- d122aef: Published declarations no longer import `./x.ts` (closes #157). tsc's `rewriteRelativeImportExtensions` rewrites `.ts` specifiers to `.js` in emitted JS but not in emitted `.d.ts`, so `dist/*.d.ts` referenced files that aren't in the package, and Deno's type check failed on them. Every package's build now runs `scripts/rewrite-dts-extensions.mjs` after `tsc`:

  - relative `.ts` / `.mts` / `.cts` specifiers in `dist/**/*.d.ts` become `.js` / `.mjs` / `.cjs` (as in laya-js),
  - each `dist/*.js` with a declaration file starts with `// @ts-self-types="./x.d.ts"` (after the `#!` line of a bin), so Deno finds the types when it loads `dist/` as plain files or from a URL instead of through `npm:`. Source maps are shifted by the inserted line.

  No runtime change. The manifest drift test checks every built `dist/` for both.

- Updated dependencies [d122aef]
  - @johnhenry/math-plus-fft@0.0.4
  - @johnhenry/math-plus-scalar-types@0.0.3
  - @johnhenry/math-plus-tensor-core@0.2.1

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
