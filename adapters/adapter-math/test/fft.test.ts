import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { FFT } from "@johnhenry/math";
import { convolve, fft, fftPadded, ifft, realSignal } from "../src/index.ts";

function closeArrays(a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-9): void {
  assert.equal(a.length, b.length, `length mismatch: ${a.length} vs ${b.length}`);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs((a[i] as number) - (b[i] as number)) < tol, `index ${i}: ${a[i]} vs ${b[i]}`);
  }
}

test("fft: matches @johnhenry/math's own FFT.fft directly (bridge doesn't introduce its own numerical bug)", () => {
  const real = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const { real: outReal, imag: outImag } = fft(real);

  const reference = FFT.fft([...real]);
  closeArrays(
    outReal,
    reference.map((z) => z.re),
  );
  closeArrays(
    outImag,
    reference.map((z) => z.im),
  );
});

test("fft -> ifft round-trips a real signal (within float tolerance)", () => {
  const real = new Float64Array([1, -2.5, 3, 0, 4.25, -1, 2, 7]);
  const { real: fReal, imag: fImag } = fft(real);
  const { real: back } = ifft(fReal, fImag);
  closeArrays(back, real, 1e-9);
});

test("fft -> ifft round-trips a complex signal", () => {
  const real = new Float64Array([1, 2, 3, 4]);
  const imag = new Float64Array([0.5, -1, 2, 0]);
  const { real: fReal, imag: fImag } = fft(real, imag);
  const back = ifft(fReal, fImag);
  closeArrays(back.real, real, 1e-9);
  closeArrays(back.imag, imag, 1e-9);
});

test("fft: cross-checked against @johnhenry/math's own O(n^2) dft reference for a small input", () => {
  const real = new Float64Array([0, 1, 2, 3]);
  const { real: fastReal, imag: fastImag } = fft(real);
  const reference = FFT.dft([...real]);
  closeArrays(
    fastReal,
    reference.map((z) => z.re),
    1e-9,
  );
  closeArrays(
    fastImag,
    reference.map((z) => z.im),
    1e-9,
  );
});

test("fft: throws a clear error for a non-power-of-two length (matching @johnhenry/math's own contract)", () => {
  assert.throws(() => fft(new Float64Array([1, 2, 3])), /power of two/);
});

test("fftPadded: zero-pads to the next power of two, no length restriction", () => {
  const real = new Float64Array([1, 2, 3]);
  const { real: outReal } = fftPadded(real);
  assert.equal(outReal.length, 4); // next power of two above 3
});

test("realSignal: wraps a plain real array with a zero imaginary part", () => {
  const s = realSignal(new Float64Array([1, 2, 3]));
  assert.deepEqual([...s.real], [1, 2, 3]);
  assert.deepEqual([...s.imag], [0, 0, 0]);
});

test("convolve: linear convolution of two real signals matches direct computation", () => {
  const a = new Float64Array([1, 2, 3]);
  const b = new Float64Array([0, 1, 0.5]);
  const { real: out } = convolve(realSignal(a), realSignal(b));

  // Direct O(n*m) convolution as the reference.
  const expected = new Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      expected[i + j] += (a[i] as number) * (b[j] as number);
    }
  }
  closeArrays(out.slice(0, expected.length), expected, 1e-9);
});
