# @johnhenry/math-plus-signal

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
| Spectral | `stft`, `istft`, `welch`, `hannWindow`, `hammingWindow` |
| Filters | `butter` (lowpass/highpass/bandpass/bandstop, SOS output only), `sosFilter`, `freqz` |
| Peaks | `findPeaks` (`height`/`distance`/`prominence` options, scipy semantics) |
| Resampling | `resamplePoly(signal, up, down)` — positional integer args, GCD-reduced |

## Traps

- **Argument order:** `sosFilter(sos, signal)` — filter first, matching scipy.
- **`stft`'s `nperseg` must be a power of two**, and the default is
  `min(256, signal.length)` — a 100-sample signal's *default* throws. Default
  window is periodic Hann with 50% overlap (COLA); `stft` returns the **full**
  `nperseg` bins per frame, not `nperseg/2+1`. No boundary padding: samples
  past the last full frame are lost, and `istft` reconstruction is least
  accurate in the first/last half-window.
- **`welch` returns the TWO-sided PSD** (`return_onesided=False`), `fs`
  fixed at 1.0; frequency labels follow `fftfreq` wraparound (Nyquist bin is
  `-0.5`).
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
