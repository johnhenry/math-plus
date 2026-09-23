/**
 * Differential test (matching the erf cross-check's #34 precedent):
 * @johnhenry/math is a devDependency ONLY here (@johnhenry/math-plus-fft's own shipped
 * runtime dependency graph is unchanged) -- used purely to build confidence
 * that this fresh, tensor-shaped Cooley-Tukey implementation agrees with an
 * independently-sourced reference.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { FFT } from "@johnhenry/math";
import { ComplexNumber } from "@johnhenry/math-plus-scalar-types";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { ComplexTensor, fft, fftPadded, ifft, irfft, rfft } from "../src/index.ts";

function closeArrays(a: readonly number[], b: readonly number[], tol: number): void {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs((a[i] as number) - (b[i] as number)) < tol, `index ${i}: ${a[i]} vs ${b[i]}`);
  }
}

test("fft matches @johnhenry/math's own FFT.fft directly (bridge doesn't introduce its own numerical bug)", () => {
  const real = [1, 2, 3, 4, 5, 6, 7, 8];
  const input = ComplexTensor.fromComplexArray(real.map((r) => new ComplexNumber(r, 0)));
  const result = fft(input).toComplexArray();

  const reference = FFT.fft(real);
  closeArrays(
    result.map((z) => z.re),
    reference.map((z) => z.re),
    1e-9,
  );
  closeArrays(
    result.map((z) => z.im),
    reference.map((z) => z.im),
    1e-9,
  );
});

test("fft -> ifft round-trips a complex signal", () => {
  const values = [
    new ComplexNumber(1, 0.5),
    new ComplexNumber(-2.5, 1),
    new ComplexNumber(3, -2),
    new ComplexNumber(0, 7),
  ];
  const input = ComplexTensor.fromComplexArray(values);
  const back = ifft(fft(input)).toComplexArray();
  for (let i = 0; i < values.length; i++) {
    assert.ok(Math.abs((values[i] as ComplexNumber).re - (back[i] as ComplexNumber).re) < 1e-9);
    assert.ok(Math.abs((values[i] as ComplexNumber).im - (back[i] as ComplexNumber).im) < 1e-9);
  }
});

test("fft: cross-checked against @johnhenry/math's own O(n^2) dft reference for a small input", () => {
  const real = [0, 1, 2, 3];
  const input = ComplexTensor.fromComplexArray(real.map((r) => new ComplexNumber(r, 0)));
  const fast = fft(input).toComplexArray();
  const reference = FFT.dft(real);
  closeArrays(
    fast.map((z) => z.re),
    reference.map((z) => z.re),
    1e-9,
  );
  closeArrays(
    fast.map((z) => z.im),
    reference.map((z) => z.im),
    1e-9,
  );
});

test("fft throws a clear error for a non-power-of-two length", () => {
  const input = ComplexTensor.fromComplexArray([1, 2, 3].map((r) => new ComplexNumber(r, 0)));
  assert.throws(() => fft(input), /power of two/);
});

test("fftPadded zero-pads to the next power of two, no length restriction", () => {
  const input = ComplexTensor.fromComplexArray([1, 2, 3].map((r) => new ComplexNumber(r, 0)));
  const out = fftPadded(input);
  assert.equal(out.size, 4); // next power of two above 3
});

test("rfft/irfft round-trip a real signal", () => {
  const real = Tensor.from([1, -2.5, 3, 0, 4.25, -1, 2, 7], { dtype: "f64" });
  const spectrum = rfft(real);
  const back = irfft(spectrum);
  closeArrays(back.toArray() as number[], real.toArray() as number[], 1e-9);
});

test("rfft matches fft on a zero-imaginary ComplexTensor of the same real values", () => {
  const real = Tensor.from([2, 4, 6, 8], { dtype: "f64" });
  const viaRfft = rfft(real).toComplexArray();
  const viaFft = fft(ComplexTensor.fromReal(real)).toComplexArray();
  for (let i = 0; i < viaRfft.length; i++) {
    assert.ok(Math.abs((viaRfft[i] as ComplexNumber).re - (viaFft[i] as ComplexNumber).re) < 1e-9);
    assert.ok(Math.abs((viaRfft[i] as ComplexNumber).im - (viaFft[i] as ComplexNumber).im) < 1e-9);
  }
});
