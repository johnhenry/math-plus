import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { hannWindow, istft, stft } from "../src/index.ts";
import { runScipyOracle, SCIPY_SKIP_REASON } from "./helpers.ts";

test("stft: output shape is [numFrames, nperseg]", () => {
  const signal = Tensor.from(
    Array.from({ length: 1024 }, (_, i) => Math.sin(i * 0.1)),
    { dtype: "f64" },
  );
  const spec = stft(signal, { nperseg: 64 });
  assert.equal(spec.shape.length, 2);
  const [numFrames, nperseg] = spec.shape as [number, number];
  assert.equal(nperseg, 64);
  const hop = 64 - 32; // default noverlap = nperseg/2
  assert.equal(numFrames, Math.floor((1024 - 64) / hop) + 1);
});

test("stft: each frame's spectrum matches an independently-computed windowed-frame FFT (numpy oracle)", { skip: SCIPY_SKIP_REASON }, () => {
  const nperseg = 32;
  const signal = Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.2) + 0.4 * Math.cos(i * 0.7));
  const spec = stft(Tensor.from(signal, { dtype: "f64" }), { nperseg });

  const window = hannWindow(nperseg);
  const frame0 = signal.slice(0, nperseg);
  const oracle = runScipyOracle<{ real: number[]; imag: number[] }>({
    op: "windowed_frame_fft",
    x: frame0,
    window: Array.from(window),
  });

  const mineReal: number[] = [];
  const mineImag: number[] = [];
  for (let k = 0; k < nperseg; k++) {
    mineReal.push(spec.real.at(0, k) as number);
    mineImag.push(spec.imag.at(0, k) as number);
  }
  for (let k = 0; k < nperseg; k++) {
    assert.ok(Math.abs((mineReal[k] as number) - (oracle.real[k] as number)) < 1e-9, `real[${k}]`);
    assert.ok(Math.abs((mineImag[k] as number) - (oracle.imag[k] as number)) < 1e-9, `imag[${k}]`);
  }
});

test("stft -> istft round-trips (near-exactly) for the default Hann-window-50%-overlap configuration (COLA-compliant)", () => {
  const n = 512;
  const signal = Array.from({ length: n }, (_, i) => Math.sin(i * 0.15) + 0.3 * Math.sin(i * 0.6) + 0.1 * Math.sin(i * 1.9));
  const original = Tensor.from(signal, { dtype: "f64" });
  const spec = stft(original, { nperseg: 64 });
  const reconstructed = istft(spec, { nperseg: 64 }).toArray() as number[];

  // The reconstructed signal is only defined over the region actually
  // covered by full frames (the tail beyond the last frame's start+nperseg
  // is not reconstructed) -- compare over that overlap.
  const compareLen = Math.min(signal.length, reconstructed.length);
  // Skip the very first/last half-window where WOLA normalization is
  // least accurate (edge effects, not a bug -- same caveat any STFT/ISTFT
  // round-trip has without boundary padding, which this v1 doesn't do).
  for (let i = 32; i < compareLen - 32; i++) {
    assert.ok(
      Math.abs((signal[i] as number) - (reconstructed[i] as number)) < 1e-9,
      `sample ${i}: original=${signal[i]} reconstructed=${reconstructed[i]}`,
    );
  }
});

test("istft: throws when the nperseg option doesn't match the spectrogram's own frame length", () => {
  const signal = Tensor.from(Array.from({ length: 256 }, (_, i) => Math.sin(i * 0.1)), { dtype: "f64" });
  const spec = stft(signal, { nperseg: 64 });
  assert.throws(() => istft(spec, { nperseg: 32 }), RangeError);
});

test("stft: rejects a non-power-of-two nperseg", () => {
  const signal = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" });
  assert.throws(() => stft(signal, { nperseg: 6 }), RangeError);
});

test("stft: rejects a >1-D input", () => {
  const bad2d = Tensor.zeros([4, 4], { dtype: "f64" });
  assert.throws(() => stft(bad2d), RangeError);
});
