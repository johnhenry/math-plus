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
| Spectral | `stft`, `istft`, `welch`, `getWindow`, `checkNOLA`, `checkCOLA`, `hannWindow`, `hammingWindow` |
| Filters | `butter` (lowpass/highpass/bandpass/bandstop, SOS output only), `sosFilter`, `sosfiltfilt` (zero-phase, scipy's odd padding + steady-state `zi`; 1-D, `padlen` option), `freqz` |
| Peaks | `findPeaks` (`height`/`distance`/`prominence` options, scipy semantics) |
| Resampling | `resamplePoly(signal, up, down)` — positional integer args, GCD-reduced |

## Browser / Web Worker entry

```js
import { Tensor, rfft, welch, stft, istft, getWindow, checkNOLA, butter, sosfiltfilt, findPeaks, resamplePoly }
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
| `axis` | `-1` | `-1` |

`nfft` may be any integer `>= nperseg`: powers of two use the radix-2 FFT, every
other length uses Bluestein's chirp-z algorithm on top of it (exact, a few ulps
of rounding). Like SciPy, an `nperseg` longer than the signal is clamped to the
signal's length.

Return shapes mirror SciPy's tuples as plain objects:

```js
const { freqs, Pxx } = welch(x);             // Pxx: Tensor [floor(nfft/2)+1]
const { freqs, times, Zxx } = stft(x);       // Zxx: ComplexTensor [freqs.length, times.length]
const { times: t, x: y } = istft(Zxx);       // y: Tensor, round-trips stft(x) (whole signal, edges too)
```

### N-D input and axes

`welch` and `stft` take `axis` (default `-1`); `istft` takes `freqAxis` (default
`-2`) and `timeAxis` (default `-1`). Axes may be negative. Shapes follow SciPy
exactly:

| | input | output |
|---|---|---|
| `welch(x, { axis })` | `[a, n, b]`, `axis: 1` | `Pxx`: `[a, freqs, b]` (the frequency axis replaces `axis`) |
| `stft(x, { axis })` | `[a, n, b]`, `axis: 1` | `Zxx`: `[a, freqs, b, times]` (frequency axis replaces `axis`, time axis last) |
| `istft(Zxx, { freqAxis, timeAxis })` | `[a, freqs, b, times]`, `freqAxis: 1`, `timeAxis: 3` | `x`: `[a, b, samples]` (the frequency axis is removed, the signal takes the time axis's place among the rest) |

### Complex input

`welch` and `stft` accept a `ComplexTensor` as well as a `Tensor`. As in
SciPy, `onesided` is then forced to `false` (default `true` is overridden): the
output is two-sided in FFT order (`freqs` as `fftfreq`, non-negative first),
`nfft` bins long. SciPy 1.16's `stft` warns when it does this and its `welch`
does not; so `stft` calls `onWarning` / `console.warn` ("input data is complex,
switching to onesided=false") unless you passed `onesided: false`, and `welch`
is silent. `istft` with `inputOnesided: false` returns a **`ComplexTensor`**
`x` (as SciPy returns a complex array there, even when the imaginary part is
~0); one-sided input returns a real `Tensor`. Round trip for a complex signal:
`istft(stft(z, { onesided: false }).Zxx, { inputOnesided: false }).x`.

### Windows

`window` is a name, a `(name, param)` tuple, or a `Float64Array`; everything
but the array is periodic (`fftbins=True`), as in `scipy.signal.get_window`.
`getWindow(window, n, fftbins = true)` has `get_window`'s contract and also
accepts a bare number (a Kaiser `beta`). Supported, and checked against
`get_window` at 1e-12 for lengths 1 to 255 (periodic and symmetric):

| Form | Windows |
|---|---|
| name | `"hann"`, `"hamming"`, `"blackman"`, `"boxcar"` (`"rectangular"`), `"bartlett"`, `"flattop"`, `"parzen"`, `"bohman"`, `"nuttall"`, `"blackmanharris"`, `"barthann"`, `"cosine"`, `"triang"`, and, with SciPy's defaults, `"tukey"` (alpha 0.5), `"exponential"` (centred, tau 1), `"taylor"` (nbar 4, sll 30, norm) |
| tuple | `["kaiser", beta]`, `["gaussian", std]`, `["general_gaussian", p, sig]`, `["general_hamming", alpha]`, `["general_cosine", coefficients]`, `["tukey", alpha]`, `["exponential", center, tau]`, `["taylor", nbar, sll, norm]` |

Tuple parameters are positional, exactly as in SciPy: the first parameter of
`"exponential"` is the **centre**, so a centred window with a decay time is
`["exponential", null, tau]` (`["exponential", 3]` would put the centre at
sample 3, legal only for a periodic window). A bare `"kaiser"`, `"gaussian"`
etc. (parameters needed) and a wrong number of parameters throw a `RangeError`
where SciPy raises a `ValueError`/`TypeError`.

### Callable `detrend`

`detrend` may be a function as well as `false`, `"constant"` or `"linear"`:
`detrend(segment) -> segment`, called for each segment on its own with a 1-D
`Tensor` of `nperseg` samples (a 1-D `ComplexTensor` for complex input) and
returning the same length (a `Tensor`/`ComplexTensor`, or a `Float64Array` for
real input). SciPy passes its function every segment (and every other row) in
one array; a function that works along the segment axis independently, as
`scipy.signal.detrend` does, gives the same result, and the tests compare
against SciPy with such functions.

### NOLA and COLA

`checkNOLA(window, nperseg, noverlap, tol = 1e-10)` and
`checkCOLA(window, nperseg, noverlap, tol = 1e-10)` are `scipy.signal.check_NOLA`
and `check_COLA` (camelCased like every other name here); `window` is anything
`getWindow` takes. `istft` checks NOLA as SciPy's does: when some output
sample's window power is `<= 1e-10` (after the boundary is removed) it reports
`"istft: NOLA condition failed, STFT may not be invertible."` (plus
`" Possibly due to missing boundary"` when `boundary: false`) by calling
**`options.onWarning(message)` if you pass one, else `console.warn(message)`**,
and goes on, dividing only where the power is non-zero. SciPy's `stft` does not
check NOLA, so neither does this one.

### Warnings

`welch`, `stft` and `istft` take `onWarning?: (message: string) => void`
(SciPy's `warnings.warn`). Without it the message goes to `console.warn`; pass
`() => {}` to silence. The warnings are: `stft` given complex input with
`onesided` left on, and `istft`'s failed NOLA check.

### Not supported (and why)

- An empty signal throws a `RangeError` (SciPy returns empty arrays).
- SciPy's warning for an `nperseg` longer than the signal is not emitted: the
  value is clamped, silently (it would fire on every short input).
- Windows beyond the list above: `chebwin`, `dpss`, `lanczos`,
  `kaiser_bessel_derived`, and SciPy's short aliases (`"hanning"`, `"blk"`,
  `"gauss"`, ...). Use the canonical names.
- `istft`'s `times` for N-D input has one entry per output sample. SciPy 1.16
  builds it from `x.shape[0]`, which is the wrong length for N-D input unless
  the time axis comes first; matching that would return a `times` that does
  not line up with `x`.
- `stft` / `istft` `scaling: "none"` is not SciPy: it is the raw windowed DFT,
  kept from 0.1.
- The other spectral functions in SciPy (`periodogram`, `csd`, `spectrogram`,
  `coherence`, `ShortTimeFFT`) do not exist in this package. `freqz` returns
  `{ frequencies, response }` (SciPy's `(w, h)`): it is filter analysis, not a
  spectral estimate, and was left alone.
- Real input to `welch` / `stft` is transformed with the complex DFT (the
  imaginary part zero): correct, but about twice the work of SciPy's `rfft`.

### Migrating from 0.1 (breaking)

The old defaults were not SciPy's. Pass the old value to keep the old result:

| Option | Before | Now (SciPy) | To keep the old behaviour |
|---|---|---|---|
| `welch` return | `{ frequencies, psd }` | `{ freqs, Pxx }` | rename the fields |
| `welch` `onesided` | `false` (`nfft` bins) | `true` | `onesided: false` |
| `welch` `detrend` | `false` | `"constant"` | `detrend: false` |
| `welch`/`stft` `nfft` | next power of two `>= nperseg` | `nperseg` | `nfft: 2 ** Math.ceil(Math.log2(nperseg))` |
| `welch`/`stft` `nperseg` > signal length | `RangeError` | clamped to the length | n/a |
| `stft` `onesided` | `false` | `true` | `onesided: false` |
| `stft` `scaling` | `"none"` (raw FFT) | `"spectrum"` (`/Σw`) | `scaling: "none"` |
| `stft` `boundary` / `padded` | none / `false` | `"zeros"` / `true` | `boundary: null, padded: false` |
| `stft` return | ComplexTensor `[frames, bins]` with `.freqs`, `.times`, `.frames` | `{ freqs, times, Zxx }`, `Zxx` is `[bins, frames]` | index as `Zxx.real.at(bin, frame)` (frame-major access becomes bin-major) |
| `istft` | `istft(spec, opts)` -> `Tensor`; two-sided, `nfft === nperseg`, `scaling: "none"` only | `istft(Zxx, opts)` -> `{ times, x }`; SciPy's full option set | `istft(Zxx, { inputOnesided: false, scaling: "none", boundary: false }).x` (now a `ComplexTensor`: take `.real`) |
| `istft` `inputOnesided: false` | real part only | `x` is a `ComplexTensor`, as in SciPy | `.real` |
| `welch`/`stft` input | 1-D real `Tensor` | any shape (`axis`), real or complex | n/a |

`detrend: "linear"` on a length-1 or length-2 segment now gives zeros (a line
fits exactly), as SciPy does. `getWindow("hann", n)` and friends are computed
with SciPy's own formulas now and can differ from `hannWindow(n)` in the last
bit.

## Traps

- **Argument order:** `sosFilter(sos, signal)` — filter first, matching scipy.
- **`welch` / `stft` / `istft` match SciPy** (see "Matches SciPy" above). Same
  option names, camelCased (`return_onesided` -> `onesided`, `input_onesided`
  -> `inputOnesided`, `freq_axis` -> `freqAxis`, `check_NOLA` -> `checkNOLA`);
  window is a name, a `(name, param)` tuple (periodic, like `get_window`) or a
  `Float64Array`.
- **`welch` one-sided** doubles every bin but DC (and Nyquist for even `nfft`);
  **`stft` one-sided does not** (SciPy doesn't either). Two-sided order is
  `fftfreq`'s (Nyquist is negative for even `nfft`).
- **`istft` infers an even `nperseg`** (`2 * (bins - 1)`), exactly as SciPy
  does: for an odd segment length pass `nperseg` explicitly (and `inputOnesided:
  false` / `nfft` if you changed those in `stft`). `istft(..., { inputOnesided: false })`
  returns a `ComplexTensor`; take `.real` for the signal of a real input.
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
