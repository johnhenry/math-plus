# @johnhenry/math-plus-fft

`ComplexTensor` plus `fft`/`ifft`/`rfft`/`irfft`/`fft2`/`fftn` for the
math-plus tensor family. Reference-speed pure JS — no WASM kernel in v1, same
"reference now, native later" framing as the rest of the family.

## Install

```bash
npm install @johnhenry/math-plus-fft
```

## Quick start

```js
import { ComplexNumber } from "@johnhenry/math-plus-scalar-types";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { ComplexTensor, fft, ifft, rfft, irfft } from "@johnhenry/math-plus-fft";

// Complex in, complex out
const input = ComplexTensor.fromComplexArray(
  [1, 2, 3, 4, 5, 6, 7, 8].map((r) => new ComplexNumber(r, 0)),
);
const spectrum = fft(input);
const back = ifft(spectrum); // round-trips to the input (ifft divides by N)

// Real convenience path
const real = Tensor.from([1, -2.5, 3, 0, 4.25, -1, 2, 7], { dtype: "f64" });
const rspec = rfft(real);    // NOTE: full N-point spectrum, not N/2+1
const rback = irfft(rspec);  // Tensor again
```

## API surface

| Export | What it is |
|---|---|
| `ComplexTensor` | Split-storage complex tensor (two `Tensor`s, `real` + `imag`). Statics: `fromParts`, `fromReal`, `fromComplexArray`, `zeros`. Boxed `ComplexNumber` at the edges (`at`/`item`/`toComplexArray`), flat typed arrays in the kernels. |
| `fft` / `ifft` | 1-D radix-2 Cooley-Tukey. `fft` unnormalized; `ifft` divides by N (NumPy convention). |
| `fftPadded` | Forward FFT with zero-padding to the next power of two — output has the padded length. |
| `fft2` / `ifft2` | 2-D separable FFT (rank-2 input only). |
| `fftn` / `ifftn` | n-D FFT over `axes` (defaults to all, ascending; negative axes OK). |
| `fftshift` / `ifftshift` | Move the zero-frequency bin to/from the center — NumPy-exact `roll(±floor(n/2))`. |
| `rfft` / `irfft` | Real→complex and back. `irfft` = `ifft(...).real`. |

## Traps

- **Power-of-two lengths only.** `fft`/`ifft` throw `RangeError` on anything
  else — use `fftPadded` for arbitrary 1-D lengths. `fft2`/`fftn` inherit the
  rule per transformed axis and there is **no** 2-D/n-D padded escape hatch.
- **`rfft` is not NumPy's `rfft`.** It returns the full N-point
  Hermitian-symmetric spectrum (`rfft(x).size === x.size`), not `N/2+1` bins.
  A documented v1 simplification.
- **Output dtype is always `f64`** regardless of input dtype.
- `irfft` trusts Hermitian symmetry and silently discards the imaginary part.
- `fftn` on a 2-D input matches `fft2` only to floating-point tolerance — the
  two use different axis-processing orders, so the last ULP can differ.
- `ComplexTensor.fromReal(t)` does **not** copy: `ct.real` is the same
  `Tensor` object you passed in.

## Tests

`npm test` — includes differential tests against `@johnhenry/math`'s scalar
`FFT` oracle, and NumPy oracle tests that skip (not fail) unless a Python
with NumPy is found (`MATH_PLUS_ORACLE_PYTHON`).

## Provenance

Built for issues #40 (1-D suite), #69 (`fft2`/`fftshift`), and #84 (`fftn`,
upstream of the Wang-tile diffraction-spectrum work). Part of the
[math-plus](https://github.com/johnhenry/math-plus) monorepo; family docs at
<https://opensource.johnhenry.me/math/>.
