# @johnhenry/math-plus-signal

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fmath-plus-signal.svg)](https://www.npmjs.com/package/@johnhenry/math-plus-signal)
[![license](https://img.shields.io/npm/l/%40johnhenry%2Fmath-plus-signal.svg)](../../LICENSE)

A SciPy-equivalent signal-processing slice for the math-plus tensor family:
`convolve`/`correlate`, `stft`/`istft`/`welch`/`periodogram`/`csd`/`spectrogram`/`coherence`, `findPeaks`, Butterworth
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
| Spectral | `welch`, `periodogram`, `csd`, `coherence`, `spectrogram`, `stft`, `istft`, `getWindow` (+ `WINDOW_NAMES`), `checkNOLA`, `checkCOLA`, `hannWindow`, `hammingWindow` |
| Filters | `butter` (lowpass/highpass/bandpass/bandstop, SOS output only), `sosFilter`, `sosfiltfilt` (zero-phase, scipy's odd padding + steady-state `zi`; 1-D, `padlen` option), `freqz` |
| Peaks | `findPeaks` (`height`/`distance`/`prominence` options, scipy semantics) |
| Resampling | `resamplePoly(signal, up, down)` — positional integer args, GCD-reduced |

## Browser / Web Worker entry

```js
import { Tensor, rfft, welch, periodogram, csd, coherence, spectrogram, stft, istft, getWindow, checkNOLA, butter, sosfiltfilt, findPeaks, resamplePoly }
  from "@johnhenry/math-plus-signal/web";
```

`/web` re-exports exactly the spectral + filtering surface above (plus
`Tensor`/`ComplexTensor`, `sosFilter`) from one specifier. Nothing reachable
from it uses Node built-ins or Node globals; `test/web-entry.test.ts` checks
the built import graph and loads the entry with `process`/`Buffer`/`require`
removed. It ships unbundled ESM: use a bundler or an ESM CDN for a single file.

## Matches SciPy

`welch(x)`, `periodogram(x)`, `csd(x, y)`, `coherence(x, y)`, `spectrogram(x)`,
`stft(x)` and `istft(Zxx)` with no options give the same results as the SciPy
functions of the same name (SciPy 1.16; checked against SciPy by differential
tests, to ~1e-9 relative to the spectrum's peak, over many lengths, including
non-power-of-two and shorter than 256), and every option takes SciPy's default
and meaning:

| | `welch` / `csd` / `coherence` | `periodogram` | `spectrogram` | `stft` |
|---|---|---|---|---|
| `fs` | 1 | 1 | 1 | 1 |
| `window` | `"hann"` (periodic) | `"boxcar"` | `["tukey", 0.25]` | `"hann"` |
| `nperseg` | `min(256, n)` | the signal length (one segment) | `min(256, n)` | 256 (always, even with an array `window`; clamped to `n`) |
| `noverlap` | `nperseg >> 1` | 0 | `nperseg >> 3` | `nperseg >> 1` |
| `nfft` | `nperseg` | the signal length | `nperseg` | `nperseg` |
| `detrend` | `"constant"` | `"constant"` | `"constant"` | `false` |
| `onesided` | `true` (`floor(nfft/2)+1` bins) | `true` | `true` | `true` |
| `scaling` | `"density"` (or `"spectrum"`) | `"density"` (or `"spectrum"`) | `"density"` (or `"spectrum"`) | `"spectrum"` (or `"psd"`) |
| `average` | `"mean"` (or `"median"`; `coherence` has none) | n/a | n/a | n/a |
| `boundary` / `padded` | n/a | n/a | n/a | `"zeros"` (`"even"`, `"odd"`, `"constant"`, `null`) / `true` |
| `mode` | n/a | n/a | `"psd"` (or `"complex"`, `"magnitude"`, `"angle"`, `"phase"`) | n/a |
| `axis` | `-1` | `-1` | `-1` | `-1` |

`nfft` may be any integer `>= nperseg`: powers of two use the radix-2 FFT, every
other length uses Bluestein's chirp-z algorithm on top of it (exact, a few ulps
of rounding). Like SciPy, `nperseg`, `noverlap`, `nfft` and `axis` are truncated
to integers (Python's `int()`), `noverlap` may be negative (gaps between
segments), `average` is only checked when there is more than one segment, and an
`nperseg` longer than the signal is clamped to the signal's length **with
SciPy's warning** (see [Warnings](#warnings)).

Return shapes mirror SciPy's tuples as plain objects:

```js
const { freqs, Pxx } = welch(x);                // Pxx: Tensor [floor(nfft/2)+1]
const { freqs, Pxx } = periodogram(x);          // one segment, boxcar window
const { freqs, Pxy } = csd(x, y);               // Pxy: ComplexTensor (conj(X) * Y), real Tensor if y is x
const { freqs, Cxy } = coherence(x, y);         // Cxy: Tensor in [0, 1]
const { freqs, times, Sxx } = spectrogram(x);   // Sxx: Tensor [freqs.length, times.length] (ComplexTensor for mode "complex")
const { freqs, times, Zxx } = stft(x);          // Zxx: ComplexTensor [freqs.length, times.length]
const { times: t, x: y } = istft(Zxx);          // y: Tensor, round-trips stft(x) (whole signal, edges too)
```

`welch` is `csd(x, x).real`, as in SciPy, and the two (with `periodogram` and
`coherence`) share one routine; `spectrogram` and `stft` share the segmenting,
windowing and DFT with it (`src/spectral.ts`), so they cannot drift apart.

### `csd`, `coherence` and shapes of `x` and `y`

`csd(x, y, options)` is SciPy's: `Pxy` is the mean over segments of
`conj(X) * Y`. Each of `x` and `y` is real or complex, of any dimension, with
the transform `axis` (one `axis` for both, negative counting from each one's
end). The rules are SciPy's:

- The other dimensions of `x` and `y` must **broadcast** (NumPy rules):
  `[3, n]` with `[n]` or `[1, n]` gives `[3, freqs]`; `[2, 3, n]` with `[3, n]`
  gives `[2, 3, freqs]`; `[3, n]` with `[2, n]` throws.
- If the lengths along `axis` differ, the shorter is zero-padded to the longer.
  (SciPy builds the padding from the *other* array's shape, so then the other
  dimensions must be equal, not just broadcastable: `[3, 40]` with `[30]`
  throws, as in SciPy.)
- The two sets of segment spectra are then multiplied with NumPy broadcasting
  *with their frequency and segment axes in place*, exactly as SciPy does. With
  equal ranks, or a negative `axis`, this is what you expect. With different
  ranks and a positive `axis` (`x` of `[n, 3]` with a 1-D `y`, `axis: 0`)
  SciPy's own result is an error, or, if the frequency count happens to equal
  the other dimension, a broadcast along the wrong axis; this package returns
  the same thing, since the comparison is the spec.
- A real `x` with a complex `y` and the default `onesided: true` throws
  (SciPy: "Complex-valued `x` not allowed"); a complex `x` is fine (two-sided,
  silently). Pass `onesided: false` for a complex `y`.
- `Pxy` is a `ComplexTensor`, except that `csd(x, x)` with the very same
  object returns the real auto-spectrum (SciPy tests `y is x`; so do we, with
  `===`). Two equal arrays that are not the same object give a complex result.
- `coherence(x, y)` is `|Pxy|² / Pxx / Pyy` from `welch(x)`, `welch(y)` and
  `csd(x, y)` with the same options (so an `nperseg` that is too long warns
  three times, as in SciPy), broadcast like any NumPy division. It has no
  `scaling`, `onesided` or `average` option, as in SciPy. Different lengths give
  spectra of different sizes, and SciPy's broadcasting error.

### `spectrogram` modes

`mode: "psd"` (default) is the power per segment, one-sided bins doubled as in
`welch`; `"complex"` is `stft`'s values (no boundary, no padding; a
`ComplexTensor`); `"magnitude"` is `|stft|`; `"angle"` is the phase in radians
(`atan2`); `"phase"` is that angle unwrapped (`np.unwrap`) **along the
frequency axis**, which is what SciPy's `spectrogram` does (not along time). The
phase of a bin that is exactly real and negative (DC; Nyquist of an even `nfft`)
is `pi` here; SciPy's is `+pi` or `-pi` depending on the sign of a zero in its
FFT, which differs even between its own one- and two-sided paths, so this is the
one place that can differ from SciPy by exactly `2*pi`.

### N-D input and axes

`welch`, `periodogram`, `csd`, `coherence`, `spectrogram` and `stft` take `axis`
(default `-1`); `istft` takes `freqAxis` (default
`-2`) and `timeAxis` (default `-1`). Axes may be negative. Shapes follow SciPy
exactly:

| | input | output |
|---|---|---|
| `welch(x, { axis })` (also `periodogram`, `csd`, `coherence`) | `[a, n, b]`, `axis: 1` | `Pxx`: `[a, freqs, b]` (the frequency axis replaces `axis`) |
| `spectrogram(x, { axis })` | `[a, n, b]`, `axis: 1` | `Sxx`: `[a, freqs, b, times]`, like `stft` |
| `stft(x, { axis })` | `[a, n, b]`, `axis: 1` | `Zxx`: `[a, freqs, b, times]` (frequency axis replaces `axis`, time axis last) |
| `istft(Zxx, { freqAxis, timeAxis })` | `[a, freqs, b, times]`, `freqAxis: 1`, `timeAxis: 3` | `x`: `[a, b, samples]` (the frequency axis is removed, the signal takes the time axis's place among the rest) |

### Complex input

Every spectral function accepts a `ComplexTensor` as well as a `Tensor`. As in
SciPy, `onesided` is then forced to `false` (default `true` is overridden): the
output is two-sided in FFT order (`freqs` as `fftfreq`, non-negative first),
`nfft` bins long. SciPy 1.16's `stft` and `spectrogram` warn when they do this
and its `welch`, `csd`, `periodogram` and `coherence` do not; so `stft` and
`spectrogram` call `onWarning` / `console.warn` ("Input data is complex,
switching to return_onesided=False") unless you passed `onesided: false`, and
the others are silent. `istft` with `inputOnesided: false` returns a **`ComplexTensor`**
`x` (as SciPy returns a complex array there, even when the imaginary part is
~0); one-sided input returns a real `Tensor`. Round trip for a complex signal:
`istft(stft(z, { onesided: false }).Zxx, { inputOnesided: false }).x`.

### Windows

`window` is a name, a `(name, param)` tuple, or a `Float64Array`; everything
but the array is periodic (`fftbins=True`), as in `scipy.signal.get_window`.
`getWindow(window, n, fftbins = true, { onWarning })` has `get_window`'s
contract and also accepts a bare number (a Kaiser `beta`, as `get_window` does;
the spectral functions do not take one, as in SciPy) and, as an extension, an
array of length `n` (returned as is). Every window and every alias in SciPy
1.16's own table is supported, and checked against `get_window` at 1e-12 for
lengths 0 to 255, periodic and symmetric, every alias with every parameter
set. `WINDOW_NAMES` is that table, `[names, needsParameters]` per row, and a
test compares it with the one in the installed SciPy.

| Window | Names (all accepted bare or as the first element of a tuple, except where parameters are needed) | Parameters |
|---|---|---|
| Hann | `"hann"`, `"han"` | |
| Hamming | `"hamming"`, `"hamm"`, `"ham"` | |
| Blackman | `"blackman"`, `"black"`, `"blk"` | |
| Blackman-Harris | `"blackmanharris"`, `"blackharr"`, `"bkh"` | |
| Nuttall | `"nuttall"`, `"nutl"`, `"nut"` | |
| Flat top | `"flattop"`, `"flat"`, `"flt"` | |
| Bartlett | `"bartlett"`, `"bart"`, `"brt"` | |
| Bartlett-Hann | `"barthann"`, `"brthan"`, `"bth"` | |
| Bohman | `"bohman"`, `"bman"`, `"bmn"` | |
| Parzen | `"parzen"`, `"parz"`, `"par"` | |
| Triangular | `"triang"`, `"triangle"`, `"tri"` | |
| Boxcar | `"boxcar"`, `"box"`, `"ones"`, `"rect"`, `"rectangular"` | |
| Cosine | `"cosine"`, `"halfcosine"` | |
| Lanczos | `"lanczos"`, `"sinc"` | |
| Exponential | `"exponential"`, `"poisson"` | optional `["exponential", center = null, tau = 1]` |
| Tukey | `"tukey"`, `"tuk"` | optional `["tukey", alpha = 0.5]` |
| Taylor | `"taylor"`, `"taylorwin"` | optional `["taylor", nbar = 4, sll = 30, norm = true]` |
| Kaiser | `"kaiser"`, `"ksr"` | `["kaiser", beta]` |
| Gaussian | `"gaussian"`, `"gauss"`, `"gss"` | `["gaussian", std]` |
| General Gaussian | `"general_gaussian"`, `"general gaussian"`, `"general_gauss"`, `"general gauss"`, `"ggs"` | `["general_gaussian", p, sig]` |
| General Hamming | `"general_hamming"`, `"general hamming"` | `["general_hamming", alpha]` |
| General cosine | `"general_cosine"`, `"general cosine"` | `["general_cosine", [a0, a1, ...]]` |
| Dolph-Chebyshev | `"chebwin"`, `"cheb"` | `["chebwin", at]`; warns under 45 dB, like SciPy |
| DPSS (Slepian) | `"dpss"` | `["dpss", NW]`: the first taper (`Kmax=None`, `norm="approximate"`) |
| Kaiser-Bessel derived | `"kaiser bessel derived"`, `"kbd"` | `["kbd", beta]`: symmetric only (`fftbins: false`) and an even length, as SciPy enforces |

The names are exactly SciPy 1.16's. In particular `"hanning"` is **not** one
(SciPy raises "Unknown window type"), nor is `"kaiser_bessel_derived"` with
underscores (its name is `"kaiser bessel derived"`), so both throw here.

Tuple parameters are positional, exactly as in SciPy: the first parameter of
`"exponential"` is the **centre**, so a centred window with a decay time is
`["exponential", null, tau]` (`["exponential", 3]` would put the centre at
sample 3, legal only for a periodic window). `["dpss", NW]` is all
`get_window` offers (one window); a `Kmax` is a `TypeError` there and here.
Errors follow SciPy's classes: an unknown name, a bare name that needs
parameters (`"kaiser"`, `"cheb"`, `"dpss"`, ...), a bad length or an invalid
value (`dpss` with `NW >= n / 2`, `kbd` with an odd length) throw a `RangeError`
(SciPy's `ValueError`); the wrong number or type of parameters throws a
`TypeError` (SciPy's `TypeError`). Length 0 gives an empty window.

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

Every spectral function (and `getWindow`) takes `onWarning?: (message: string) =>
void` (SciPy's `warnings.warn`), with SciPy's own message text. Without it the
message goes to `console.warn`; pass `() => {}` to silence. The warnings are:

- an `nperseg` longer than the signal (`welch`, `csd`, `coherence`, `spectrogram`,
  `stft`), which is then clamped to the signal's length. `welch`/`csd` say
  `nperseg=300 is greater than signal length max(len(x), len(y)) = 20, using
  nperseg = 20`; `spectrogram`/`stft` say `nperseg = 300 is greater than input
  length  = 20, using nperseg = 20` (two spaces, as SciPy has them);
- complex input to `stft` or `spectrogram` with `onesided` left on;
- `chebwin` with an attenuation under 45 dB;
- `istft`'s failed NOLA check.

### Empty input

An empty input does what SciPy does, in every spectral function. `welch`,
`periodogram`, `csd` and `coherence` return empty arrays shaped like the input
(for `csd` with different shapes, the broadcast outer dimensions with the
shorter length on `axis`); `stft` and `spectrogram` do too, and neither
looks at the window or segment options first except where SciPy does
(`spectrogram` sets up the window and `nperseg` before it notices, so it can
warn or raise first; `stft` checks only `scaling` and `boundary`). `freqs` and
`times` are `[]` (SciPy's are empty arrays of the input's shape, which a
`number[]` cannot carry). `Zxx` and a `mode: "complex"` `Sxx` are empty
`ComplexTensor`s (SciPy returns real empty arrays; the type here stays true to
the function's usual one). SciPy raises in a few places and so does this:
`csd` with outer dimensions that do not broadcast, `spectrogram` `mode: "phase"`
on a 1-D empty input (`np.unwrap` on an axis it does not have), `stft` with a bad
`scaling`/`boundary`. `istft` of an empty spectrogram returns an empty `x`.



### Not supported (and why)

- `ShortTimeFFT`, `lombscargle`, `czt`/`zoom_fft` and SciPy's other spectral
  tools are not provided; `freqz` returns `{ frequencies, response }` (SciPy's
  `(w, h)`): it is filter analysis, not a spectral estimate, and was left alone.
- `istft`'s `times` for N-D input has one entry per output sample. SciPy 1.16
  builds it from `x.shape[0]`, which is the wrong length for N-D input unless
  the time axis comes first; matching that would return a `times` that does
  not line up with `x`. (For a 2-D `Zxx` the two agree.)
- `fs <= 0` (or not finite) is rejected with a `RangeError` everywhere. SciPy
  does the same in `welch`/`csd`/`periodogram`/`coherence` (`ValueError`), but in
  `stft`, `istft` and `spectrogram` it raises `ZeroDivisionError` for 0 and
  returns negative or NaN output for a negative `fs`.
- Errors are `RangeError` for SciPy's `ValueError` and `IndexError` (an `axis`
  out of range, a 0-d input) and `TypeError` where SciPy raises one for the
  wrong number or type of window parameters; there is no `IndexError`. A window
  that SciPy would build from a NumPy scalar or a list is not accepted: use a
  name, a tuple or a `Float64Array`.
- `callable detrend` is called once per segment on a 1-D `Tensor`; SciPy
  hands it all segments at once (see "Callable `detrend`" above).
- `getWindow` of a window array returns it (checking its length); SciPy's
  `get_window` rejects arrays. The spectral functions take arrays in SciPy too.
- Real input to the spectral functions is transformed with the complex DFT (the
  imaginary part zero): correct, but about twice the work of SciPy's `rfft`.
- `resamplePoly` is not SciPy's filter (see Traps), and `butter` is lowpass,
  highpass, bandpass and bandstop with SOS output only.

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

### Migrating from 0.2 (breaking)

0.2 matched SciPy for `welch`, `stft` and `istft` and left gaps; closing them
changes a few results and errors:

| Before | Now (SciPy) | To keep the old behaviour |
|---|---|---|
| empty input threw `RangeError` | returns empty arrays like SciPy (`freqs: []`) | check the length first |
| `nperseg` > length clamped silently | clamped **with SciPy's warning** (`onWarning` / `console.warn`) | pass `onWarning: () => {}` |
| `stft` / `istft` `scaling: "density"` and `"none"` | `RangeError`: SciPy's take only `"spectrum"` and `"psd"` | `"psd"` for `"density"`; scale by `Σw` for `"none"` |
| `stft` with an array `window` and no `nperseg` took `nperseg = window.length` | `nperseg` defaults to 256, as in SciPy's `stft`: an array of another length needs its `nperseg` | pass `nperseg: window.length` |
| a non-integer `nperseg` / `noverlap` / `nfft` / `axis`, or a negative `noverlap`, was a `RangeError` | truncated to integers like Python's `int()`; a negative `noverlap` leaves gaps | pass integers |
| `average` was always validated | only when there is more than one segment | n/a |
| `window` of a bare number accepted by `welch` / `stft` | `TypeError` (`RangeError` for `stft`/`spectrogram`): SciPy rejects it; `getWindow(4.0, n)` still works | `["kaiser", beta]` |
| `getWindow(name, 0)` threw | returns an empty window | n/a |
| a wrong window parameter count was a `RangeError` | `TypeError`, as SciPy | catch both |
| warning texts began `stft: ` / `istft: ` | SciPy's own text, unprefixed | n/a |

## Traps

- **Argument order:** `sosFilter(sos, signal)` — filter first, matching scipy.
- **`welch` / `periodogram` / `csd` / `coherence` / `spectrogram` / `stft` /
  `istft` match SciPy** (see "Matches SciPy" above). Same option names,
  camelCased (`return_onesided` -> `onesided`, `input_onesided` ->
  `inputOnesided`, `freq_axis` -> `freqAxis`, `check_NOLA` -> `checkNOLA`);
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
This package follows **SciPy 1.16**. The tests run against any SciPy, and a
few expectations that depend on what 1.17 changed in `get_window` (it refuses a
length of 0 and numeric strings, also accepts `"kaiser_bessel_derived"` and
`"<name>_periodic"` / `"<name>_symmetric"`, and raises `ValueError` where 1.16
raises `TypeError`) are compared with an oracle older than 1.17 and checked by
hand against a newer one.
`findPeaks`' distance filter carries a regression perf test from issue #101
(was O(n²): 4594 ms at 40,000 candidate peaks; now near-linear).

## Provenance

Built across issues #44 (core slice), #70 (`correlate`/`freqz`/`welch`),
#84 (`correlate2D`), and #90 (bandpass/bandstop). Part of the
[math-plus](https://github.com/johnhenry/math-plus) monorepo; family docs at
<https://opensource.johnhenry.me/math/>.
