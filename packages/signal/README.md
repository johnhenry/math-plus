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

## Matches SciPy

`welch(x)`, `stft(x)` and `istft(Zxx)` with no options give the same results as
`scipy.signal.welch(x)`, `scipy.signal.stft(x)` and `scipy.signal.istft(Zxx)`
(checked against SciPy by differential tests, to ~1e-9 relative to the
spectrum's peak, over many lengths, including non-power-of-two and shorter
than 256), and every option takes SciPy's default and meaning:

| | `welch` | `stft` |
|---|---|---|
| `fs` / `window` / `nperseg` / `noverlap` / `nfft` | 1 / `"hann"` (periodic) / `min(256, n)` / `nperseg >> 1` / `nperseg` | same |
| `detrend` | `"constant"` | `false` |
| `onesided` | `true` (`floor(nfft/2)+1` bins) | `true` |
| `scaling` | `"density"` (or `"spectrum"`) | `"spectrum"` (`"psd"`; `"density"` is an alias; `"none"` is a non-SciPy raw FFT) |
| `average` | `"mean"` (or `"median"`) | n/a |
| `boundary` / `padded` | n/a | `"zeros"` (`"even"`, `"odd"`, `"constant"`, `null`) / `true` |

`nfft` may be any integer `>= nperseg`: powers of two use the radix-2 FFT, every
other length uses Bluestein's chirp-z algorithm on top of it (exact, a few ulps
of rounding). Like SciPy, an `nperseg` longer than the signal is clamped to the
signal's length.

Return shapes mirror SciPy's tuples as plain objects:

```js
const { frequencies, psd } = welch(x);       // psd: Tensor [floor(nfft/2)+1]
const { freqs, times, Zxx } = stft(x);       // Zxx: ComplexTensor [freqs.length, times.length]
const { times: t, x: y } = istft(Zxx);       // x: Tensor, round-trips stft(x) (whole signal, edges too)
```

Not implemented (1-D real `Tensor` input only): `axis`/`time_axis`/`freq_axis`
(and so N-D input), complex input, callable `detrend`, `(name, param)` window
tuples, SciPy's NOLA warning, and the complex output of `istft` with
`inputOnesided: false` (the real part is returned).

### Migrating from 0.1 (breaking)

The old defaults were not SciPy's. Pass the old value to keep the old result:

| Option | Before | Now (SciPy) | To keep the old behaviour |
|---|---|---|---|
| `welch` `onesided` | `false` (`nfft` bins) | `true` | `onesided: false` |
| `welch` `detrend` | `false` | `"constant"` | `detrend: false` |
| `welch`/`stft` `nfft` | next power of two `>= nperseg` | `nperseg` | `nfft: 2 ** Math.ceil(Math.log2(nperseg))` |
| `welch`/`stft` `nperseg` > signal length | `RangeError` | clamped to the length | n/a |
| `stft` `onesided` | `false` | `true` | `onesided: false` |
| `stft` `scaling` | `"none"` (raw FFT) | `"spectrum"` (`/Σw`) | `scaling: "none"` |
| `stft` `boundary` / `padded` | none / `false` | `"zeros"` / `true` | `boundary: null, padded: false` |
| `stft` return | ComplexTensor `[frames, bins]` with `.freqs`, `.times`, `.frames` | `{ freqs, times, Zxx }`, `Zxx` is `[bins, frames]` | index as `Zxx.real.at(bin, frame)` (frame-major access becomes bin-major) |
| `istft` | `istft(spec, opts)` -> `Tensor`; two-sided, `nfft === nperseg`, `scaling: "none"` only | `istft(Zxx, opts)` -> `{ times, x }`; SciPy's full option set | `istft(Zxx, { inputOnesided: false, scaling: "none", boundary: false }).x` |

`welch` still returns `{ frequencies, psd }`. `detrend: "linear"` on a
length-1 or length-2 segment now gives zeros (a line fits exactly), as SciPy does.

## Traps

- **Argument order:** `sosFilter(sos, signal)` — filter first, matching scipy.
- **`welch` / `stft` / `istft` match SciPy** (see "Matches SciPy" below). Same
  option names, camelCased (`return_onesided` -> `onesided`, `input_onesided`
  -> `inputOnesided`); window is `"hann"`, `"hamming"`, `"blackman"`,
  `"boxcar"`/`"rectangular"` (periodic, like `get_window`) or a `Float64Array`.
- **`welch` one-sided** doubles every bin but DC (and Nyquist for even `nfft`);
  **`stft` one-sided does not** (SciPy doesn't either). Two-sided order is
  `fftfreq`'s (Nyquist is negative for even `nfft`).
- **`istft` infers an even `nperseg`** (`2 * (bins - 1)`), exactly as SciPy
  does: for an odd segment length pass `nperseg` explicitly (and `inputOnesided:
  false` / `nfft` if you changed those in `stft`).
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
