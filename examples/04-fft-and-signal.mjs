// fft + signal: spectra, filter design, and peak finding — the SciPy-slice
// of the family. Pure JS; everything emits f64.
//
// Run: npm install && npm run build && node examples/04-fft-and-signal.mjs
import assert from "node:assert/strict";
import { ComplexTensor, fft, ifft, rfft, irfft } from "@johnhenry/math-plus-fft";
import { ComplexNumber } from "@johnhenry/math-plus-scalar-types";
import { butter, convolve1D, findPeaks, sosFilter } from "@johnhenry/math-plus-signal";
import { Tensor } from "@johnhenry/math-plus-tensor-core";

// fft/ifft — power-of-two lengths only (use fftPadded otherwise);
// ifft divides by N, fft doesn't (NumPy convention).
const input = ComplexTensor.fromComplexArray(
  [1, 2, 3, 4, 5, 6, 7, 8].map((r) => new ComplexNumber(r, 0)),
);
const roundTrip = ifft(fft(input)).toComplexArray();
console.log(roundTrip[2].re.toFixed(6)); // 3.000000

// rfft: real in, FULL N-point spectrum out (not NumPy's N/2+1 — v1 scope)
const real = Tensor.from([1, -2.5, 3, 0, 4.25, -1, 2, 7], { dtype: "f64" });
const spectrum = rfft(real);
assert.equal(spectrum.size, real.size);
console.log(irfft(spectrum).toArray().map((v) => Math.round(v * 1e6) / 1e6)); // the input back

// Convolution, NumPy modes
const a = new Float64Array([1, 2, 3, 4]);
const b = new Float64Array([1, 1, 1]);
console.log([...convolve1D(a, b, "full")]); // [1, 3, 6, 9, 7, 4]
console.log([...convolve1D(a, b, "same")]); // [3, 6, 9, 7]

// Butterworth lowpass + zero-state filtering. wn is normalized to
// Nyquist = 1; argument order is scipy's: sosFilter(sos, signal).
const N = 256;
const lo = Array.from({ length: N }, (_, i) => Math.sin((2 * Math.PI * 4 * i) / N));
const hi = Array.from({ length: N }, (_, i) => Math.sin((2 * Math.PI * 100 * i) / N));
const noisy = lo.map((v, i) => v + hi[i]);
const sos = butter(4, 0.2, { btype: "lowpass" });
const filtered = sosFilter(sos, Tensor.from(noisy, { dtype: "f64" })).toArray();
const rms = (xs) => Math.sqrt(xs.reduce((s, v) => s + v * v, 0) / xs.length);
console.log("rms noisy:", rms(noisy).toFixed(3), "-> filtered:", rms(filtered).toFixed(3));

// Peak finding, scipy semantics (endpoints are never peaks)
const sig = [0, 5, 0, 8, 0, 3, 0, 0, 0, 6, 0];
console.log(findPeaks(Tensor.from(sig, { dtype: "f64" }), { distance: 4 }).indices); // [3, 9]
