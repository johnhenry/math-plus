import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { getWindow, hannWindow, istft, stft, welch } from "../src/index.ts";
import { runScipyOracle, SCIPY_SKIP_REASON } from "./helpers.ts";

// Issue #178: welch/stft options (fs, onesided, detrend, scaling, any
// nperseg + nfft, more windows), all against scipy.signal.

const REL = 1e-9;

function signalOf(n: number, seed: number): number[] {
  let s = seed >>> 0;
  const rng = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
  // a trend and an offset, so detrend choices actually differ
  return Array.from({ length: n }, (_, i) => 3 + 0.01 * i + Math.sin(i / 7) + 0.5 * Math.sin(i / 2.3) + rng());
}
const tensorOf = (x: number[]) => Tensor.from(x, { dtype: "f64" });
const arr = (t: Tensor) => Array.from(t.contiguous().data as Float64Array);

function closeRel(a: ArrayLike<number>, b: ArrayLike<number>, what: string): void {
  assert.equal(a.length, b.length, `${what}: lengths ${a.length} vs ${b.length}`);
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    worst = Math.max(worst, Math.abs((a[i] as number) - (b[i] as number)) / Math.max(1e-30, Math.abs(b[i] as number), 1e-12));
  }
  // relative to the spectrum's own scale, so near-zero bins don't dominate
  let scale = 0;
  for (let i = 0; i < b.length; i++) scale = Math.max(scale, Math.abs(b[i] as number));
  let worstScaled = 0;
  for (let i = 0; i < a.length; i++) worstScaled = Math.max(worstScaled, Math.abs((a[i] as number) - (b[i] as number)) / Math.max(scale, 1e-300));
  assert.ok(worstScaled <= REL, `${what}: differs by ${worstScaled} (rel to max), worst per-bin ${worst}`);
}

// ---- windows ------------------------------------------------------------

test("getWindow: periodic hann/hamming/blackman/boxcar/rectangular match closed forms", () => {
  const n = 10;
  const hamming = getWindow("hamming", n);
  const blackman = getWindow("blackman", n);
  for (let i = 0; i < n; i++) {
    const x = (2 * Math.PI * i) / n;
    assert.ok(Math.abs((hamming[i] as number) - (0.54 - 0.46 * Math.cos(x))) < 1e-15);
    assert.ok(Math.abs((blackman[i] as number) - (0.42 - 0.5 * Math.cos(x) + 0.08 * Math.cos(2 * x))) < 1e-15);
  }
  assert.deepEqual(Array.from(getWindow("boxcar", 5)), [1, 1, 1, 1, 1]);
  assert.deepEqual(Array.from(getWindow("rectangular", 5)), [1, 1, 1, 1, 1]);
  assert.deepEqual(Array.from(getWindow("hann", 8)), Array.from(hannWindow(8)));
});

test("getWindow: unknown names, bad lengths and wrong-length arrays throw", () => {
  assert.throws(() => getWindow("kaiser" as never, 8), RangeError);
  assert.throws(() => getWindow("hann", 0), RangeError);
  assert.throws(() => getWindow(new Float64Array(7), 8), RangeError);
  const mine = Float64Array.from([1, 2, 3]);
  assert.deepEqual(Array.from(getWindow(mine, 3)), [1, 2, 3]);
});

// ---- welch --------------------------------------------------------------

test("welch: defaults are unchanged (fs 1, hann, two-sided, density, no detrend, power-of-two nfft = nperseg)", () => {
  const x = signalOf(1024, 1);
  const base = welch(tensorOf(x), { nperseg: 128 });
  const explicit = welch(tensorOf(x), {
    nperseg: 128,
    fs: 1,
    window: "hann",
    onesided: false,
    detrend: false,
    scaling: "density",
    nfft: 128,
  });
  assert.deepEqual(base.frequencies, explicit.frequencies);
  assert.deepEqual(arr(base.psd), arr(explicit.psd));
  assert.equal(base.frequencies.length, 128);
});

test("welch: onesided gives nfft/2+1 bins with frequencies in hertz", () => {
  const x = signalOf(1000, 2);
  const r = welch(tensorOf(x), { fs: 250, nperseg: 100, onesided: true });
  assert.equal(r.frequencies.length, 65); // nfft 128
  assert.equal(r.frequencies[0], 0);
  assert.equal(r.frequencies[64], 125);
  assert.equal(r.psd.shape[0], 65);
});

test("welch: a sine of amplitude A peaks at A*A/2 with scaling 'spectrum' (one-sided, bin-centred)", () => {
  const n = 4096;
  const nperseg = 256;
  const bin = 32;
  const x = Array.from({ length: n }, (_, i) => 2 * Math.sin((2 * Math.PI * bin * i) / nperseg));
  const r = welch(tensorOf(x), { nperseg, onesided: true, scaling: "spectrum", window: "boxcar" });
  const p = arr(r.psd);
  assert.ok(Math.abs((p[bin] as number) - 2) < 1e-9, `peak ${p[bin]}`);
});

test("welch: rejects bad options", () => {
  const t = tensorOf(signalOf(256, 3));
  assert.throws(() => welch(t, { fs: 0 }), RangeError);
  assert.throws(() => welch(t, { nperseg: 64, nfft: 32 }), RangeError);
  assert.throws(() => welch(t, { nperseg: 64, nfft: 96 }), RangeError); // fft needs a power of two
  assert.throws(() => welch(t, { detrend: "quadratic" as never }), RangeError);
  assert.throws(() => welch(t, { scaling: "none" as never }), RangeError);
  assert.throws(() => welch(t, { nperseg: 64, window: new Float64Array(10) }), RangeError);
});

const WINDOWS = ["hann", "hamming", "blackman", "boxcar", "rectangular"] as const;
const DETRENDS = [false, "constant", "linear"] as const;
const SCALINGS = ["density", "spectrum"] as const;

test("welch matches scipy.signal.welch for every combination of fs, onesided, detrend, scaling, window, nperseg and nfft", { skip: SCIPY_SKIP_REASON }, () => {
  const x = signalOf(900, 4);
  let cases = 0;
  const checks: Array<(want: { frequencies: number[]; psd: number[] }) => void> = [];
  const jobs: Record<string, unknown>[] = [];
  for (const fs of [1, 250]) {
    for (const [nperseg, nfft] of [[64, undefined], [100, undefined], [37, 64], [50, 128]] as const) {
      for (const onesided of [false, true]) {
        for (const detrend of DETRENDS) {
          for (const scaling of SCALINGS) {
            for (const window of WINDOWS) {
              // one overlap per case, varied deterministically to cover several
              const noverlap = [undefined, 0, Math.floor(nperseg * 0.75)][cases % 3];
              const got = welch(tensorOf(x), { fs, nperseg, nfft, noverlap, onesided, detrend, scaling, window });
              const nf = nfft ?? 2 ** Math.ceil(Math.log2(nperseg));
              jobs.push({
                op: "welch_opts",
                x,
                fs,
                window: window === "rectangular" ? "boxcar" : window,
                nperseg,
                noverlap: noverlap ?? Math.floor(nperseg / 2),
                nfft: nf,
                detrend,
                onesided,
                scaling,
              });
              const what = JSON.stringify({ fs, nperseg, nfft, noverlap, onesided, detrend, scaling, window });
              checks.push((want) => {
                closeRel(got.frequencies, want.frequencies, `frequencies ${what}`);
                closeRel(arr(got.psd), want.psd, `psd ${what}`);
              });
              cases++;
            }
          }
        }
      }
    }
  }
  assert.ok(cases >= 400);
  const wants = runScipyOracle<Array<{ frequencies: number[]; psd: number[] }>>({ op: "batch", jobs });
  checks.forEach((check, i) => check(wants[i] as never));
});

test("welch matches scipy.signal.welch with a custom Float64Array window", { skip: SCIPY_SKIP_REASON }, () => {
  const x = signalOf(500, 5);
  const nperseg = 45;
  const window = Float64Array.from({ length: nperseg }, (_, i) => 1 + Math.sin(i / 5) ** 2);
  for (const onesided of [false, true]) {
    const got = welch(tensorOf(x), { fs: 8, nperseg, window, onesided, detrend: "linear" });
    const want = runScipyOracle<{ frequencies: number[]; psd: number[] }>({
      op: "welch_opts",
      x,
      fs: 8,
      window: Array.from(window),
      nperseg,
      noverlap: 22,
      nfft: 64,
      detrend: "linear",
      onesided,
      scaling: "density",
    });
    closeRel(got.frequencies, want.frequencies, "frequencies");
    closeRel(arr(got.psd), want.psd, "psd");
  }
});

// ---- stft ---------------------------------------------------------------

test("stft: still returns a ComplexTensor [numFrames, nperseg] by default, now with freqs, times and frames", () => {
  const x = signalOf(512, 6);
  const s = stft(tensorOf(x), { nperseg: 64 });
  assert.deepEqual(s.shape, [15, 64]);
  assert.equal(s.frames, s);
  assert.equal(s.freqs.length, 64);
  assert.equal(s.times.length, 15);
  assert.equal(s.times[0], 32); // (0 + 64/2) / 1
  assert.equal(s.times[1], 64); // hop 32
  assert.equal(s.freqs[1], 1 / 64);
});

test("stft: onesided, fs and nfft shape the output; times are segment centres in seconds", () => {
  const x = signalOf(1000, 7);
  const s = stft(tensorOf(x), { fs: 100, nperseg: 50, noverlap: 25, nfft: 64, onesided: true });
  assert.deepEqual(s.shape, [Math.floor((1000 - 50) / 25) + 1, 33]);
  assert.equal(s.freqs[32], 50);
  assert.equal(s.times[0], 0.25);
  assert.ok(Math.abs((s.times[2] as number) - (50 + 25) / 100) < 1e-15);
});

test("stft -> istft still round-trips with the default (raw, two-sided) configuration", () => {
  const x = signalOf(512, 8).map((v) => v - 3);
  const spec = stft(tensorOf(x), { nperseg: 64 });
  const back = arr(istft(spec, { nperseg: 64 }));
  for (let i = 64; i < back.length - 64; i++) assert.ok(Math.abs((back[i] as number) - (x[i] as number)) < 1e-9);
});

test("stft: rejects bad options (and istft rejects a zero-padded or one-sided spectrogram)", () => {
  const t = tensorOf(signalOf(256, 9));
  assert.throws(() => stft(t, { fs: -1 }), RangeError);
  assert.throws(() => stft(t, { nperseg: 64, nfft: 96 }), RangeError);
  assert.throws(() => stft(t, { scaling: "bogus" as never }), RangeError);
  assert.throws(() => istft(stft(t, { nperseg: 64, nfft: 128 }), { nperseg: 64 }), RangeError);
});

test("stft matches scipy.signal.stft (no boundary, no padding) for every combination of the options", { skip: SCIPY_SKIP_REASON }, () => {
  const x = signalOf(700, 10);
  let cases = 0;
  const checks: Array<(want: { freqs: number[]; times: number[]; real: number[][]; imag: number[][] }) => void> = [];
  const jobs: Record<string, unknown>[] = [];
  for (const fs of [1, 250]) {
    for (const [nperseg, nfft] of [[64, undefined], [100, undefined], [37, 64]] as const) {
      for (const onesided of [false, true]) {
        for (const detrend of DETRENDS) {
          for (const scaling of ["none", "density", "spectrum"] as const) {
            for (const window of WINDOWS) {
              const noverlap = [undefined, 0, Math.floor(nperseg * 0.75)][cases % 3];
              const got = stft(tensorOf(x), { fs, nperseg, nfft, noverlap, onesided, detrend, scaling, window });
              const nf = nfft ?? 2 ** Math.ceil(Math.log2(nperseg));
              jobs.push({
                op: "stft_opts",
                x,
                fs,
                window: window === "rectangular" ? "boxcar" : window,
                nperseg,
                noverlap: noverlap ?? Math.floor(nperseg / 2),
                nfft: nf,
                detrend,
                onesided,
                scaling,
              });
              const what = JSON.stringify({ fs, nperseg, nfft, noverlap, onesided, detrend, scaling, window });
              checks.push((want) => {
                closeRel(got.freqs, want.freqs, `freqs ${what}`);
                closeRel(got.times, want.times, `times ${what}`);
                assert.equal(got.shape[0], want.real.length, `frames ${what}`);
                closeRel(arr(got.real), want.real.flat(), `real ${what}`);
                closeRel(arr(got.imag), want.imag.flat(), `imag ${what}`);
              });
              cases++;
            }
          }
        }
      }
    }
  }
  assert.ok(cases >= 300);
  const wants = runScipyOracle<Array<{ freqs: number[]; times: number[]; real: number[][]; imag: number[][] }>>({ op: "batch", jobs });
  checks.forEach((check, i) => check(wants[i] as never));
});
