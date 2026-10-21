/**
 * Exact DFT of any length on top of @johnhenry/math-plus-fft's radix-2 `fft`.
 *
 * Power-of-two lengths go straight to `fft`/`ifft`. Every other length uses
 * Bluestein's chirp-z algorithm: with `w_k = exp(-iπk²/n)`,
 * `X_k = w_k · Σ_j (x_j w_j) · conj(w_{k-j})`, a convolution evaluated with
 * power-of-two FFTs of length `m >= 2n - 1`. It is an exact identity (not an
 * approximation); the only error is floating-point rounding, a few ulps of
 * the spectrum's largest magnitude. The chirp angle uses `k² mod 2n` so it
 * stays accurate for long transforms.
 *
 * Pure typed-array code plus the FFT package: browser-safe (`./web`).
 */
import { ComplexTensor, fft, ifft } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";

export interface DftPlan {
  readonly n: number;
  /** Forward (`inverse=false`, unnormalised) or inverse (`inverse=true`, scaled by `1/n`) DFT. Returns new arrays; the inputs are not modified. */
  transform(re: Float64Array, im: Float64Array, inverse: boolean): { re: Float64Array; im: Float64Array };
}

const isPowerOfTwo = (n: number) => n > 0 && (n & (n - 1)) === 0;

function pack(re: Float64Array, im: Float64Array): ComplexTensor {
  const n = re.length;
  return ComplexTensor.fromParts(
    Tensor.fromTypedArray(Float64Array.from(re), [n], { dtype: "f64" }),
    Tensor.fromTypedArray(Float64Array.from(im), [n], { dtype: "f64" }),
  );
}

const unpack = (z: ComplexTensor) => ({
  re: Float64Array.from(z.real.contiguous().data as Float64Array),
  im: Float64Array.from(z.imag.contiguous().data as Float64Array),
});

/** Build a reusable plan for length-`n` transforms (the chirp and its FFT are computed once). */
export function planDft(n: number): DftPlan {
  if (!Number.isInteger(n) || n < 1) throw new RangeError(`dft: length must be a positive integer, got ${n}`);
  if (isPowerOfTwo(n)) {
    return {
      n,
      transform(re, im, inverse) {
        const z = pack(re, im);
        return unpack(inverse ? ifft(z) : fft(z));
      },
    };
  }

  let m = 1;
  while (m < 2 * n - 1) m *= 2;
  // chirp[k] = exp(-iπk²/n)
  const wRe = new Float64Array(n);
  const wIm = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const angle = (Math.PI * ((k * k) % (2 * n))) / n;
    wRe[k] = Math.cos(angle);
    wIm[k] = -Math.sin(angle);
  }
  // b = conj(chirp), wrapped symmetrically into length m, then transformed once
  const bRe = new Float64Array(m);
  const bIm = new Float64Array(m);
  for (let k = 0; k < n; k++) {
    bRe[k] = wRe[k] as number;
    bIm[k] = -(wIm[k] as number);
    if (k > 0) {
      bRe[m - k] = wRe[k] as number;
      bIm[m - k] = -(wIm[k] as number);
    }
  }
  const B = unpack(fft(pack(bRe, bIm)));

  function forward(re: Float64Array, im: Float64Array): { re: Float64Array; im: Float64Array } {
    const aRe = new Float64Array(m);
    const aIm = new Float64Array(m);
    for (let j = 0; j < n; j++) {
      const xr = re[j] as number;
      const xi = im[j] as number;
      aRe[j] = xr * (wRe[j] as number) - xi * (wIm[j] as number);
      aIm[j] = xr * (wIm[j] as number) + xi * (wRe[j] as number);
    }
    const A = unpack(fft(pack(aRe, aIm)));
    for (let k = 0; k < m; k++) {
      const ar = A.re[k] as number;
      const ai = A.im[k] as number;
      const br = B.re[k] as number;
      const bi = B.im[k] as number;
      A.re[k] = ar * br - ai * bi;
      A.im[k] = ar * bi + ai * br;
    }
    const c = unpack(ifft(pack(A.re, A.im)));
    const outRe = new Float64Array(n);
    const outIm = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      const cr = c.re[k] as number;
      const ci = c.im[k] as number;
      outRe[k] = cr * (wRe[k] as number) - ci * (wIm[k] as number);
      outIm[k] = cr * (wIm[k] as number) + ci * (wRe[k] as number);
    }
    return { re: outRe, im: outIm };
  }

  return {
    n,
    transform(re, im, inverse) {
      if (!inverse) return forward(re, im);
      // ifft(X) = conj(fft(conj(X))) / n
      const negIm = new Float64Array(n);
      for (let i = 0; i < n; i++) negIm[i] = -(im[i] as number);
      const y = forward(re, negIm);
      for (let i = 0; i < n; i++) {
        y.re[i] = (y.re[i] as number) / n;
        y.im[i] = -(y.im[i] as number) / n;
      }
      return y;
    },
  };
}

/** One-shot DFT of any length (see {@link planDft} to reuse the setup). */
export function dft(re: Float64Array, im: Float64Array, inverse: boolean): { re: Float64Array; im: Float64Array } {
  if (re.length !== im.length) throw new RangeError(`dft: real (${re.length}) and imaginary (${im.length}) lengths differ`);
  if (re.length === 0) throw new RangeError("dft: input must not be empty");
  return planDft(re.length).transform(re, im, inverse);
}
