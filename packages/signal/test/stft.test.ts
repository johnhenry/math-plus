import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { hannWindow, istft, stft } from "../src/index.ts";
import { runScipyOracle, SCIPY_SKIP_REASON } from "./helpers.ts";

// The SciPy-oracle comparisons for stft/istft live in spectral-options.test.ts.

test("stft: Zxx shape is [freqs, frames] (SciPy orientation); with no boundary or padding it is [nperseg/2+1, floor((n-nperseg)/hop)+1]", () => {
  const signal = Tensor.from(
    Array.from({ length: 1024 }, (_, i) => Math.sin(i * 0.1)),
    { dtype: "f64" },
  );
  const { Zxx, freqs, times } = stft(signal, { nperseg: 64, boundary: null, padded: false });
  assert.equal(Zxx.shape.length, 2);
  const [nFreqs, numFrames] = Zxx.shape as [number, number];
  assert.equal(nFreqs, 33);
  assert.equal(freqs.length, 33);
  assert.equal(numFrames, Math.floor((1024 - 64) / 32) + 1);
  assert.equal(times.length, numFrames);
});

test("stft: each frame's spectrum matches an independently-computed windowed-frame FFT (numpy oracle)", { skip: SCIPY_SKIP_REASON }, () => {
  const nperseg = 32;
  const signal = Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.2) + 0.4 * Math.cos(i * 0.7));
  const { Zxx } = stft(Tensor.from(signal, { dtype: "f64" }), {
    nperseg,
    onesided: false,
    boundary: null,
    padded: false,
    scaling: "none",
  });

  const window = hannWindow(nperseg);
  const oracle = runScipyOracle<{ real: number[]; imag: number[] }>({
    op: "windowed_frame_fft",
    x: signal.slice(0, nperseg),
    window: Array.from(window),
  });
  for (let k = 0; k < nperseg; k++) {
    assert.ok(Math.abs((Zxx.real.at(k, 0) as number) - (oracle.real[k] as number)) < 1e-9, `real[${k}]`);
    assert.ok(Math.abs((Zxx.imag.at(k, 0) as number) - (oracle.imag[k] as number)) < 1e-9, `imag[${k}]`);
  }
});

test("stft -> istft round-trips for a Hann window at 50% overlap with explicit nperseg", () => {
  const n = 512;
  const signal = Array.from({ length: n }, (_, i) => Math.sin(i * 0.15) + 0.3 * Math.sin(i * 0.6) + 0.1 * Math.sin(i * 1.9));
  const { Zxx } = stft(Tensor.from(signal, { dtype: "f64" }), { nperseg: 64 });
  const { x } = istft(Zxx, { nperseg: 64 });
  const reconstructed = Array.from(x.contiguous().data as Float64Array);
  // boundary 'zeros' + padded: the whole signal is reconstructed, edges included
  for (let i = 0; i < n; i++) {
    assert.ok(Math.abs((signal[i] as number) - (reconstructed[i] as number)) < 1e-9, `sample ${i}`);
  }
});

test("istft: throws when the window length doesn't match nperseg, and on a non-2-D input", () => {
  const signal = Tensor.from(Array.from({ length: 256 }, (_, i) => Math.sin(i * 0.1)), { dtype: "f64" });
  const { Zxx } = stft(signal, { nperseg: 64 });
  assert.throws(() => istft(Zxx, { nperseg: 64, window: new Float64Array(32) }), RangeError);
});

test("stft: a non-power-of-two nfft works (Bluestein) and sets the number of bins", () => {
  const signal = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" });
  assert.deepEqual(stft(signal, { nperseg: 6, boundary: null, padded: false }).Zxx.shape, [4, 1]);
  assert.deepEqual(stft(signal, { nperseg: 6, nfft: 10, boundary: null, padded: false }).Zxx.shape, [6, 1]);
  assert.deepEqual(stft(signal, { nperseg: 6, nfft: 9, boundary: null, padded: false }).Zxx.shape, [5, 1]);
});

test("stft: rejects a >1-D input", () => {
  const bad2d = Tensor.zeros([4, 4], { dtype: "f64" });
  assert.throws(() => stft(bad2d), RangeError);
});
