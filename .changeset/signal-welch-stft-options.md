---
"@johnhenry/math-plus-signal": minor
---

`welch` and `stft` take SciPy's options: `fs`, `onesided`, `detrend` (`"constant"`, `"linear"`), `scaling`, any `nperseg` with a zero-padding `nfft`, and `window` as `"hann"`, `"hamming"`, `"blackman"`, `"boxcar"`/`"rectangular"` or a `Float64Array`. `stft` also carries `freqs`, `times` (segment centres) and `frames`. New `getWindow` (periodic, as `scipy.signal.get_window`). Defaults keep the previous behaviour. Fixes #178.
