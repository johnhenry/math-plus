# @johnhenry/math-plus-signal

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fmath-plus-signal.svg)](https://www.npmjs.com/package/@johnhenry/math-plus-signal)
[![license](https://img.shields.io/npm/l/%40johnhenry%2Fmath-plus-signal.svg)](../../LICENSE)

A SciPy-equivalent signal-processing slice for the math-plus tensor family:
`convolve`/`correlate`, `stft`/`istft`/`welch`, `findPeaks`, Butterworth
filter design (`butter`) + `sosFilter`/`freqz`, and `resamplePoly`. Pure JS,
`Tensor` in / `Tensor` out, differential-tested against a real SciPy oracle.

## Install

```bash
npm install @johnhenry/math-plus-signal
```

## Quick start

```js
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { butter, sosFilter, convolve1D, findPeaks } from "@johnhenry/math-plus-signal";

// Convolution, NumPy modes
const a = new Float64Array([1, 2, 3, 4]);
const b = new Float64Array([1, 1, 1]);
convolve1D(a, b, "full");  // [1, 3, 6, 9, 7, 4]
convolve1D(a, b, "same");  // [3, 6, 9, 7]
convolve1D(a, b, "valid"); // [6, 9]

// Filter design + zero-state filtering (scipy argument order: sos first)
const sos = butter(4, 0.2, { btype: "lowpass" }); // wn normalized to Nyquist=1
const out = sosFilter(sos, Tensor.from(noisy, { dtype: "f64" }));

// Peak finding
const x = [0, 5, 0, 8, 0, 3, 0, 0, 0, 6, 0];
findPeaks(Tensor.from(x, { dtype: "f64" }), { distance: 4 });
// { indices: [3, 9], heights: [8, 6], prominences: [...] }
```

## API surface

| Group | Exports |
|---|---|
| Convolution | `convolve`, `convolve1D`, `correlate`, `correlate1D`, `correlate2D` (true 2-D, FFT-based, `"full"` only), `applyTimeDomainOp` |
| Spectral | `stft`, `istft`, `welch`, `getWindow`, `hannWindow`, `hammingWindow` |
| Filters | `butter` (lowpass/highpass/bandpass/bandstop, SOS output only), `sosFilter`, `sosfiltfilt` (zero-phase, scipy's odd padding + steady-state `zi`; 1-D, `padlen` option), `freqz` |
| Peaks | `findPeaks` (`height`/`distance`/`prominence` options, scipy semantics) |
| Resampling | `resamplePoly(signal, up, down)` — positional integer args, GCD-reduced |

## Browser / Web Worker entry

```js
import { Tensor, rfft, welch, stft, butter, sosfiltfilt, findPeaks, resamplePoly }
  from "@johnhenry/math-plus-signal/web";
```

`/web` re-exports exactly the spectral + filtering surface above (plus
`Tensor`/`ComplexTensor`, `sosFilter`) from one specifier. Nothing reachable
from it uses Node built-ins or Node globals; `test/web-entry.test.ts` checks
the built import graph and loads the entry with `process`/`Buffer`/`require`
removed. It ships unbundled ESM: use a bundler or an ESM CDN for a single file.

## Traps

- **Argument order:** `sosFilter(sos, signal)` — filter first, matching scipy.
- **`welch` / `stft` options** follow SciPy: `fs` (default 1), `window`
  (`"hann"` default, `"hamming"`, `"blackman"`, `"boxcar"`/`"rectangular"`, all
  periodic like `get_window`, or a `Float64Array` of length `nperseg`),
  `nperseg` (any length; default `min(256, length)`), `noverlap` (default half),
  `nfft` (power of two `>= nperseg`, zero-padding each segment; default the
  next power of two), `detrend` (`false` default, `"constant"`, `"linear"`),
  `onesided` (`false` default: all `nfft` bins) and `scaling` (`welch`:
  `"density"` default or `"spectrum"`; `stft`: `"none"` default = raw FFT,
  `"spectrum"` = `/Σw`, `"density"` = `/sqrt(fs·Σw²)`). The defaults are the
  pre-options behaviour, which differs from SciPy's in two places: `onesided`
  (SciPy: true for real input) and `detrend` (SciPy: `"constant"`).
- **`welch` one-sided** doubles the bins between DC and Nyquist; **`stft`
  one-sided does not** (SciPy doesn't). `welch` returns `{ frequencies, psd }`
  with frequencies in `fs` units; two-sided order is `fftfreq`'s (Nyquist is
  negative for even `nfft`).
- **`stft`** returns the `[numFrames, bins]` ComplexTensor, which also carries
  `freqs`, `times` (segment centres, `(start + nperseg/2) / fs`) and `frames`
  (the tensor itself): `const { freqs, times, frames } = stft(x, opts)`. No
  boundary extension or padding (SciPy `boundary=None, padded=False`): samples
  past the last full frame are lost. Default `nperseg` is `min(256, length)`;
  default window periodic Hann with 50% overlap (COLA). `istft` inverts only
  a two-sided, un-padded (`nfft === nperseg`), un-detrended, `scaling: "none"`
  spectrogram, and is least accurate in the first/last half-window.
- **`butter`'s `wn` is normalized to Nyquist = 1**, strictly `0 < wn < 1`.
  SOS section coefficients are deliberately **not** byte-comparable to
  scipy's (scipy's own pole/zero-to-section grouping isn't fixed either);
  end-to-end filter behavior matches to 1e-6.
- **`resamplePoly` is not bit-compatible with scipy** — Hamming-windowed sinc
  (not Kaiser β=5.0), direct upsample-then-convolve rather than a true
  polyphase decomposition. A performance gap, not a correctness one.
- **Everything emits `f64`** regardless of input dtype — pass
  `{ dtype: "f64" }` to `Tensor.from` when building inputs.
- `findPeaks` never reports endpoints as peaks; filter order is
  `height` → `prominence` → `distance` (distance ties keep the taller,
  earliest-discarded).

## Tests

`npm test`. SciPy differential tests skip (not fail) unless a Python with
SciPy is found (`MATH_PLUS_SCIPY_ORACLE_PYTHON` / `MATH_PLUS_ORACLE_PYTHON`).
`findPeaks`' distance filter carries a regression perf test from issue #101
(was O(n²): 4594 ms at 40,000 candidate peaks; now near-linear).

## Provenance

Built across issues #44 (core slice), #70 (`correlate`/`freqz`/`welch`),
#84 (`correlate2D`), and #90 (bandpass/bandstop). Part of the
[math-plus](https://github.com/johnhenry/math-plus) monorepo; family docs at
<https://opensource.johnhenry.me/math/>.
