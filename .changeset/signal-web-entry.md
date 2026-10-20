---
"@johnhenry/math-plus-signal": minor
---

Add `sosfiltfilt` (zero-phase forward-backward SOS filtering with scipy's odd padding and steady-state initial conditions) and a browser/Web Worker entry, `@johnhenry/math-plus-signal/web`, exporting `rfft`, `welch`, `stft`, `butter`, `sosfiltfilt`, `findPeaks` and `resamplePoly` (plus `Tensor`/`ComplexTensor`). The entry's import graph contains no Node built-ins or Node globals; a test enforces this (#173).
