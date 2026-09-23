/**
 * correlate/freqz/welch (issue #70) — three half-pairs filled in: convolve
 * had no cross-correlation dual, butter had no way to inspect the filter
 * it designed, and stft had no aggregate (power spectral density).
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { butter, convolve1D, correlate, correlate1D, freqz, welch } from "../src/index.ts";
import { runScipyOracle, SCIPY_SKIP_REASON } from "./helpers.ts";

function close(a: ArrayLike<number>, b: ArrayLike<number>, tol: number): void {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs((a[i] as number) - (b[i] as number)) < tol, `index ${i}: ${a[i]} vs ${b[i]}`);
}

// ---- correlate --------------------------------------------------------------

test("correlate1D matches scipy.signal.correlate, all three modes", { skip: SCIPY_SKIP_REASON }, () => {
  const a = [1, 2, 3, 4, 5, 6, 7];
  const b = [1, 0, -1, 2];
  for (const mode of ["full", "same", "valid"] as const) {
    const got = correlate1D(Float64Array.from(a), Float64Array.from(b), mode);
    const { y } = runScipyOracle<{ y: number[] }>({ op: "correlate", a, b, mode });
    close(got, y, 1e-9);
  }
});

test("correlate is the direct dual of convolve: correlate(a,b) === convolve(a, reverse(b)), all three modes", () => {
  const a = Float64Array.from([2, -1, 3, 0, 5]);
  const b = Float64Array.from([1, 2, 3]);
  const reversedB = Float64Array.from([...b].reverse());
  for (const mode of ["full", "same", "valid"] as const) {
    const viaCorrelate = correlate1D(a, b, mode);
    const viaConvolve = convolve1D(a, reversedB, mode);
    close(viaCorrelate, viaConvolve, 1e-12);
  }
});

test("correlate (Tensor) agrees with correlate1D on a 1-D input", () => {
  const a = Tensor.from([1, 2, 3, 4, 5], { dtype: "f64" });
  const b = Tensor.from([1, 0, -1], { dtype: "f64" });
  const got = correlate(a, b).contiguous().data as Float64Array;
  const want = correlate1D(Float64Array.from([1, 2, 3, 4, 5]), Float64Array.from([1, 0, -1]));
  close(got, want, 1e-12);
});

test("correlate (Tensor) batches over a 2-D [N,T] input along the time axis, same contract as convolve", () => {
  const input = Tensor.from([1, 2, 3, 4, 0, 1, 0, -1], { dtype: "f64" }).reshape([2, 4]);
  const kernel = Tensor.from([1, -1], { dtype: "f64" });
  const got = correlate(input, kernel, { mode: "valid" }).toArray() as number[][];
  const row0 = correlate1D(Float64Array.from([1, 2, 3, 4]), Float64Array.from([1, -1]), "valid");
  const row1 = correlate1D(Float64Array.from([0, 1, 0, -1]), Float64Array.from([1, -1]), "valid");
  close(got[0] as number[], row0, 1e-12);
  close(got[1] as number[], row1, 1e-12);
});

// ---- freqz --------------------------------------------------------------

test("freqz matches scipy.signal.sosfreqz for a lowpass butter filter (single-section SOS, no grouping ambiguity)", { skip: SCIPY_SKIP_REASON }, () => {
  const sos = butter(2, 0.3, { btype: "lowpass" });
  const worN = 16;
  const got = freqz(sos, { worN });
  const oracle = runScipyOracle<{ frequencies: number[]; real: number[]; imag: number[] }>({
    op: "sosfreqz",
    sos: sos.map((s) => [...s]),
    worN,
  });
  close(got.frequencies, oracle.frequencies, 1e-9);
  close(got.response.map((z) => z.re), oracle.real, 1e-6);
  close(got.response.map((z) => z.im), oracle.imag, 1e-6);
});

test("freqz matches scipy.signal.sosfreqz for a highpass butter filter", { skip: SCIPY_SKIP_REASON }, () => {
  const sos = butter(2, 0.4, { btype: "highpass" });
  const worN = 16;
  const got = freqz(sos, { worN });
  const oracle = runScipyOracle<{ frequencies: number[]; real: number[]; imag: number[] }>({
    op: "sosfreqz",
    sos: sos.map((s) => [...s]),
    worN,
  });
  close(got.response.map((z) => z.re), oracle.real, 1e-6);
  close(got.response.map((z) => z.im), oracle.imag, 1e-6);
});

test("freqz: a lowpass filter's DC response has magnitude ~1 and its Nyquist-adjacent response is strongly attenuated", () => {
  const sos = butter(4, 0.2, { btype: "lowpass" });
  const { frequencies, response } = freqz(sos, { worN: 64 });
  assert.ok(Math.abs(frequencies[0] as number) < 1e-12);
  assert.ok((response[0] as { magnitude(): number }).magnitude() > 0.95, "DC gain should be near 1 for a lowpass filter");
  const nearNyquist = response[response.length - 1] as { magnitude(): number };
  assert.ok(nearNyquist.magnitude() < 0.1, "near-Nyquist gain should be strongly attenuated for a lowpass filter");
});

test("freqz rejects an empty sos", () => {
  assert.throws(() => freqz([]), RangeError);
});

// ---- welch --------------------------------------------------------------

test("welch matches scipy.signal.welch (return_onesided=False, scaling='density', detrend=False), the documented v1 scope", { skip: SCIPY_SKIP_REASON }, () => {
  const n = 1024;
  const nperseg = 256;
  const noverlap = 128;
  // Deterministic pseudo-random signal (not Math.random) so this test is reproducible.
  let seed = 42;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed / 0x7fffffff) * 2 - 1;
  };
  const x = Array.from({ length: n }, () => rng());
  const signal = Tensor.from(x, { dtype: "f64" });
  const got = welch(signal, { nperseg, noverlap });

  const window = new Array(nperseg);
  for (let i = 0; i < nperseg; i++) window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nperseg); // periodic Hann, matching hannWindow's default

  const oracle = runScipyOracle<{ frequencies: number[]; psd: number[] }>({
    op: "welch",
    x,
    window,
    nperseg,
    noverlap,
  });
  close(got.frequencies, oracle.frequencies, 1e-9);
  close(got.psd.contiguous().data as Float64Array, oracle.psd, 1e-6);
});

test("welch: a pure sinusoid's PSD peaks at (or symmetrically around) its own frequency bin", () => {
  const n = 2048;
  const nperseg = 256;
  const freqBin = 20; // target: energy concentrated near bin 20 or its mirror (nperseg - 20)
  const cyclesPerSample = freqBin / nperseg;
  const x = Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * cyclesPerSample * i));
  const { psd } = welch(Tensor.from(x, { dtype: "f64" }), { nperseg });
  const data = psd.contiguous().data as Float64Array;
  let peakBin = 0;
  for (let i = 1; i < data.length; i++) if ((data[i] as number) > (data[peakBin] as number)) peakBin = i;
  const mirror = nperseg - freqBin;
  assert.ok(
    Math.abs(peakBin - freqBin) <= 1 || Math.abs(peakBin - mirror) <= 1,
    `expected the PSD peak near bin ${freqBin} or its mirror ${mirror}, got ${peakBin}`,
  );
});

test("welch rejects a signal too short for the requested nperseg", () => {
  const signal = Tensor.from([1, 2, 3], { dtype: "f64" });
  assert.throws(() => welch(signal, { nperseg: 256 }), RangeError);
});
