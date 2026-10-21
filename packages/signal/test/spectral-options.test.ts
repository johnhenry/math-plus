import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { ComplexTensor } from "@johnhenry/math-plus-fft";
import { getWindow, hannWindow, istft, stft, welch, type StftOptions, type WelchOptions } from "../src/index.ts";
import {
  checkStft,
  checkWelch,
  closeScaled,
  flat,
  flatten,
  type Kw,
  maxAbs,
  pyKw,
  runScipyOracle,
  SCIPY_SKIP_REASON,
  signalOf,
  type StftWant,
  tensorOf,
  type WelchWant,
  zParts,
} from "./helpers.ts";

// Issue #178 + the SciPy-parity release: welch/stft/istft take scipy.signal's
// defaults and options. Every comparison below is against scipy.signal itself;
// the "no options" cases call SciPy with no keyword arguments at all.

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
  closeScaled(getWindow("hann", 8), hannWindow(8), 1, "hann vs hannWindow", 1e-15);
});

test("getWindow: unknown names, bad lengths and wrong-length arrays throw", () => {
  assert.throws(() => getWindow("kaiser" as never, 8), RangeError);
  assert.throws(() => getWindow("hann", -1), RangeError);
  assert.equal(getWindow("hann", 0).length, 0, "length 0 is an empty window, as in SciPy");
  assert.throws(() => getWindow(new Float64Array(7), 8), RangeError);
  const mine = Float64Array.from([1, 2, 3]);
  assert.deepEqual(Array.from(getWindow(mine, 3)), [1, 2, 3]);
});

const LENGTHS = [8, 31, 50, 100, 255, 256, 257, 300, 513, 777, 1000, 1024, 2500];

// ---- welch --------------------------------------------------------------

test("welch with NO options matches scipy.signal.welch(x) with no options, on many lengths", { skip: SCIPY_SKIP_REASON }, () => {
  const jobs = LENGTHS.map((n) => ({ op: "welch_kw", x: signalOf(n, n), kwargs: {} }));
  const wants = runScipyOracle<WelchWant[]>({ op: "batch", jobs });
  LENGTHS.forEach((n, i) => checkWelch(welch(tensorOf(signalOf(n, n))), wants[i] as WelchWant, `n=${n}`));
});

test("welch: the defaults are SciPy's (one-sided, constant detrend, density, hann, nperseg=min(256,n), nfft=nperseg)", () => {
  const x = signalOf(1000, 1);
  const base = welch(tensorOf(x));
  const explicit = welch(tensorOf(x), {
    fs: 1,
    window: "hann",
    nperseg: 256,
    noverlap: 128,
    nfft: 256,
    detrend: "constant",
    onesided: true,
    scaling: "density",
    average: "mean",
  });
  assert.deepEqual(base.freqs, explicit.freqs);
  assert.deepEqual(flat(base.Pxx), flat(explicit.Pxx));
  assert.equal(base.freqs.length, 129);
  // nfft follows nperseg, NOT the next power of two
  assert.equal(welch(tensorOf(x), { nperseg: 100 }).freqs.length, 51);
  assert.equal(welch(tensorOf(x), { nperseg: 101 }).freqs.length, 51);
  // short input: nperseg = length
  assert.equal(welch(tensorOf(signalOf(50, 2))).freqs.length, 26);
});

test("welch: onesided gives floor(nfft/2)+1 bins in hertz; a sine of amplitude A peaks at A*A/2 with scaling 'spectrum'", () => {
  const r = welch(tensorOf(signalOf(1000, 2)), { fs: 250, nperseg: 100 });
  assert.equal(r.freqs.length, 51);
  assert.equal(r.freqs[50], 125);
  const n = 4096;
  const nperseg = 256;
  const bin = 32;
  const x = Array.from({ length: n }, (_, i) => 2 * Math.sin((2 * Math.PI * bin * i) / nperseg));
  const p = flat(welch(tensorOf(x), { nperseg, scaling: "spectrum", window: "boxcar" }).Pxx);
  assert.ok(Math.abs((p[bin] as number) - 2) < 1e-9, `peak ${p[bin]}`);
});

test("welch: rejects bad options", () => {
  const t = tensorOf(signalOf(256, 3));
  assert.throws(() => welch(t, { fs: 0 }), RangeError);
  assert.throws(() => welch(t, { nperseg: 64, nfft: 32 }), RangeError);
  assert.throws(() => welch(t, { detrend: "quadratic" as never }), RangeError);
  assert.throws(() => welch(t, { scaling: "none" as never }), RangeError);
  assert.throws(() => welch(t, { nperseg: 64, average: "max" as never }), RangeError);
  assert.doesNotThrow(() => welch(t, { average: "max" as never }), "SciPy only checks `average` when there is more than one segment");
  assert.throws(() => welch(t, { nperseg: 64, window: new Float64Array(10) }), RangeError);
  assert.throws(() => welch(t, { nperseg: 64, noverlap: 64 }), RangeError);
  assert.doesNotThrow(() => welch(t, { nperseg: 64, nfft: 96 })); // non-power-of-two nfft is fine now
});

const WINDOWS = ["hann", "hamming", "blackman", "boxcar", "rectangular"] as const;
const DETRENDS = [false, "constant", "linear"] as const;

test("welch matches scipy.signal.welch for combinations of fs, onesided, detrend, scaling, average, window, nperseg and nfft (incl. non-power-of-two nfft)", { skip: SCIPY_SKIP_REASON }, () => {
  const x = signalOf(900, 4);
  let cases = 0;
  const checks: Array<(want: WelchWant) => void> = [];
  const jobs: Kw[] = [];
  const shapes: Array<[number, number | undefined]> = [
    [64, undefined],
    [100, undefined],
    [37, 37],
    [37, 64],
    [50, 96],
    [64, 100],
    [100, 250],
    [31, 97],
  ];
  for (const fs of [1, 250]) {
    for (const [nperseg, nfft] of shapes) {
      for (const onesided of [false, true]) {
        for (const detrend of DETRENDS) {
          for (const scaling of ["density", "spectrum"] as const) {
            for (const window of WINDOWS) {
              const noverlap = [undefined, 0, Math.floor(nperseg * 0.75)][cases % 3];
              const average = (["mean", "median"] as const)[cases % 2];
              const opts: WelchOptions = { fs, nperseg, nfft, noverlap, onesided, detrend, scaling, window, average };
              const got = welch(tensorOf(x), opts);
              jobs.push({ op: "welch_kw", x, kwargs: pyKw({ ...opts }, false) });
              const what = JSON.stringify({ ...opts, window: String(window) });
              checks.push((want) => checkWelch(got, want, what));
              cases++;
            }
          }
        }
      }
    }
  }
  assert.ok(cases >= 700);
  const wants = runScipyOracle<WelchWant[]>({ op: "batch", jobs });
  checks.forEach((check, i) => check(wants[i] as WelchWant));
});

test("welch matches scipy.signal.welch with a custom Float64Array window and an odd nfft", { skip: SCIPY_SKIP_REASON }, () => {
  const x = signalOf(500, 5);
  const nperseg = 45;
  const window = Float64Array.from({ length: nperseg }, (_, i) => 1 + Math.sin(i / 5) ** 2);
  for (const onesided of [false, true]) {
    for (const nfft of [undefined, 45, 63]) {
      const opts: WelchOptions = { fs: 8, nperseg, window, nfft, onesided, detrend: "linear" };
      const got = welch(tensorOf(x), opts);
      const want = runScipyOracle<WelchWant>({ op: "welch_kw", x, kwargs: pyKw({ ...opts }, false) });
      checkWelch(got, want, `onesided=${onesided} nfft=${nfft}`);
    }
  }
});

// ---- stft ---------------------------------------------------------------

test("stft with NO options matches scipy.signal.stft(x) with no options, on many lengths", { skip: SCIPY_SKIP_REASON }, () => {
  const jobs = LENGTHS.map((n) => ({ op: "stft_kw", x: signalOf(n, n + 1), kwargs: {} }));
  const wants = runScipyOracle<StftWant[]>({ op: "batch", jobs });
  LENGTHS.forEach((n, i) => checkStft(stft(tensorOf(signalOf(n, n + 1))), wants[i] as StftWant, `n=${n}`));
});

test("stft: returns { freqs, times, Zxx } with Zxx shaped [freqs, times] and SciPy's defaults", () => {
  const x = signalOf(512, 6);
  const s = stft(tensorOf(x));
  // nperseg 256, hop 128, boundary 'zeros' adds 128 each side -> 768, padded: (768-256) % 128 == 0 -> 5 frames
  assert.deepEqual(s.Zxx.shape, [129, 5]);
  assert.equal(s.freqs.length, 129);
  assert.equal(s.times.length, 5);
  assert.equal(s.times[0], 0); // boundary shifts the first centre to t = 0
  assert.equal(s.times[1], 128);
  assert.equal(s.freqs[1], 1 / 256);
  const none = stft(tensorOf(x), { nperseg: 64, boundary: null, padded: false });
  assert.deepEqual(none.Zxx.shape, [33, 15]);
  assert.equal(none.times[0], 32); // no boundary: first centre is nperseg/2
  assert.ok(!("frames" in none));
});

test("stft/istft: scaling is SciPy's 'spectrum' or 'psd' only ('density' and 'none' are errors, as in SciPy)", () => {
  const t = tensorOf(signalOf(300, 11));
  const spec = zParts(stft(t, { nperseg: 64 }).Zxx);
  const psd = zParts(stft(t, { nperseg: 64, scaling: "psd" }).Zxx);
  const w = getWindow("hann", 64) as Float64Array;
  const sumW = w.reduce((a, b) => a + b, 0);
  const sumW2 = w.reduce((a, b) => a + b * b, 0);
  closeScaled(psd.re, spec.re.map((v) => (v * sumW) / Math.sqrt(sumW2)), maxAbs(psd.re), "psd vs spectrum");
  for (const bad of ["density", "none"] as const) {
    assert.throws(() => stft(t, { nperseg: 64, scaling: bad as never }), RangeError);
    assert.throws(() => istft(stft(t, { nperseg: 64 }).Zxx, { nperseg: 64, scaling: bad as never }), RangeError);
  }
});

test("stft: rejects bad options", () => {
  const t = tensorOf(signalOf(256, 9));
  assert.throws(() => stft(t, { fs: -1 }), RangeError);
  assert.throws(() => stft(t, { nperseg: 64, nfft: 32 }), RangeError);
  assert.throws(() => stft(t, { scaling: "bogus" as never }), RangeError);
  assert.throws(() => stft(t, { boundary: "wrap" as never }), RangeError);
  assert.throws(() => stft(t, { nperseg: 64, noverlap: 64 }), RangeError);
});

test("stft matches scipy.signal.stft for combinations of fs, onesided, detrend, scaling, boundary, padded, window, nperseg and nfft (incl. non-power-of-two nfft)", { skip: SCIPY_SKIP_REASON }, () => {
  const x = signalOf(700, 10);
  let cases = 0;
  const checks: Array<(want: StftWant) => void> = [];
  const jobs: Kw[] = [];
  const shapes: Array<[number, number | undefined]> = [
    [64, undefined],
    [100, undefined],
    [37, 37],
    [37, 64],
    [50, 96],
    [64, 100],
    [100, 250],
  ];
  const boundaries = ["zeros", null, "even", "odd", "constant"] as const;
  for (const fs of [1, 250]) {
    for (const [nperseg, nfft] of shapes) {
      for (const onesided of [false, true]) {
        for (const detrend of DETRENDS) {
          for (const scaling of ["spectrum", "psd"] as const) {
            for (const window of WINDOWS) {
              const noverlap = [undefined, 0, Math.floor(nperseg * 0.75)][cases % 3];
              const boundary = boundaries[cases % boundaries.length];
              const padded = cases % 4 < 2;
              const opts: StftOptions = { fs, nperseg, nfft, noverlap, onesided, detrend, scaling, window, boundary, padded };
              const got = stft(tensorOf(x), opts);
              jobs.push({ op: "stft_kw", x, kwargs: pyKw({ ...opts }, true) });
              const what = JSON.stringify({ ...opts, window: String(window) });
              checks.push((want) => checkStft(got, want, what));
              cases++;
            }
          }
        }
      }
    }
  }
  assert.ok(cases >= 700);
  const wants = runScipyOracle<StftWant[]>({ op: "batch", jobs });
  checks.forEach((check, i) => check(wants[i] as StftWant));
});

// ---- istft --------------------------------------------------------------

const toZxx = (w: { shape: number[]; real: unknown; imag: unknown }): ComplexTensor =>
  ComplexTensor.fromParts(
    Tensor.fromTypedArray(Float64Array.from(flatten(w.real)), w.shape, { dtype: "f64" }),
    Tensor.fromTypedArray(Float64Array.from(flatten(w.imag)), w.shape, { dtype: "f64" }),
  );

/** The real part of an istft result (a Tensor, or a ComplexTensor for a two-sided input). */
const realOf = (x: Tensor | ComplexTensor): number[] => flat("real" in x ? x.real : x);

test("istft(stft(x)) with all-default options reconstructs x (many lengths, incl. under 256 and non-power-of-two)", () => {
  for (const n of LENGTHS) {
    const x = signalOf(n, n + 2);
    // SciPy's istft infers an EVEN nperseg (2 * (bins - 1)); an odd one (here n=31) must be passed.
    const nperseg = Math.min(256, n);
    const { times, x: y } = istft(stft(tensorOf(x)).Zxx, nperseg % 2 ? { nperseg } : {});
    const back = flat(y);
    assert.ok(back.length >= n, `n=${n}: reconstructed ${back.length}`);
    assert.equal(times.length, back.length);
    closeScaled(back.slice(0, n), x, maxAbs(x), `round trip n=${n}`, 1e-9);
  }
});

test("istft round-trips with scipy: scipy.istft(our Zxx), our istft(scipy's Zxx) and our istft(our Zxx) all agree with scipy.istft and with x", { skip: SCIPY_SKIP_REASON }, () => {
  const configs: Array<{ n: number; opts: StftOptions }> = [
    { n: 1000, opts: {} },
    { n: 100, opts: {} },
    { n: 777, opts: { nperseg: 100 } },
    { n: 777, opts: { nperseg: 101 } },
    { n: 600, opts: { nperseg: 64, nfft: 100 } },
    { n: 600, opts: { nperseg: 64, nfft: 128, window: "hamming", noverlap: 48 } },
    { n: 600, opts: { nperseg: 50, onesided: false } },
    { n: 600, opts: { nperseg: 64, scaling: "psd", fs: 100 } },
  ];
  const xs = configs.map(({ n }) => signalOf(n, 77));
  const toKw = (opts: StftOptions) => pyKw({ ...opts, inputOnesided: opts.onesided ?? true, onesided: undefined }, true);
  const inv = (opts: StftOptions) => ({ ...opts, inputOnesided: opts.onesided ?? true, onesided: undefined }) as never;
  const rowsOf = (a: number[], nf: number, nt: number) => Array.from({ length: nf }, (_, r) => a.slice(r * nt, (r + 1) * nt));
  // one Python start for everything: scipy.istft of OUR spectrogram, and scipy.stft of x
  const mines = configs.map(({ opts }, i) => stft(tensorOf(xs[i] as number[]), opts));
  const jobs: Kw[] = [];
  configs.forEach(({ opts }, i) => {
    const mine = mines[i] as ReturnType<typeof stft>;
    const ours = zParts(mine.Zxx);
    const [nf, nt] = mine.Zxx.shape as [number, number];
    jobs.push({ op: "istft_kw", real: rowsOf(ours.re, nf, nt), imag: rowsOf(ours.im, nf, nt), kwargs: toKw(opts) });
    jobs.push({ op: "stft_kw", x: xs[i], kwargs: pyKw({ ...opts }, true) });
  });
  const wants = runScipyOracle<unknown[]>({ op: "batch", jobs });
  configs.forEach(({ n, opts }, i) => {
    const x = xs[i] as number[];
    const what = JSON.stringify({ n, ...opts });
    const wantFromOurs = wants[2 * i] as { times: number[]; real: number[] };
    const sc = wants[2 * i + 1] as StftWant;
    // scipy's istft of our spectrogram reconstructs the original ...
    closeScaled(wantFromOurs.real.slice(0, n), x, maxAbs(x), `scipy.istft(ours) ~ x ${what}`, 1e-9);
    // ... and so does ours, identically to scipy's
    const myIstft = istft((mines[i] as ReturnType<typeof stft>).Zxx, inv(opts));
    closeScaled(realOf(myIstft.x), wantFromOurs.real, maxAbs(x), `istft(ours) == scipy.istft(ours) ${what}`, 1e-9);
    closeScaled(myIstft.times, wantFromOurs.times, maxAbs(wantFromOurs.times), `times ${what}`);
    // our istft of scipy's own spectrogram
    const viaScipy = istft(toZxx(sc), inv(opts));
    closeScaled(realOf(viaScipy.x).slice(0, n), x, maxAbs(x), `istft(scipy's) ~ x ${what}`, 1e-9);
  });
});

test("istft with no options matches scipy.signal.istft on a spectrogram it did not produce", { skip: SCIPY_SKIP_REASON }, () => {
  const x = signalOf(800, 12);
  const sc = runScipyOracle<StftWant>({ op: "stft_kw", x, kwargs: {} });
  // perturb it so it is not a valid STFT any more
  const bent = { ...sc, real: (sc.real as number[][]).map((r, i) => r.map((v, j) => v * (1 + 0.1 * Math.sin(i + j)))) };
  const want = runScipyOracle<{ times: number[]; real: number[] }>({ op: "istft_kw", real: bent.real, imag: bent.imag, kwargs: {} });
  const got = istft(toZxx(bent));
  closeScaled(flat(got.x), want.real, maxAbs(want.real), "istft no options");
  closeScaled(got.times, want.times, maxAbs(want.times), "istft times");
});

test("istft: boundary:false and a mismatched window length are handled / rejected", () => {
  const t = tensorOf(signalOf(512, 13));
  const s = stft(t, { nperseg: 64, boundary: null, padded: false });
  const { x } = istft(s.Zxx, { nperseg: 64, boundary: false, onWarning: () => {} });
  const back = flat(x);
  const orig = signalOf(512, 13);
  for (let i = 32; i < back.length - 32; i++) assert.ok(Math.abs((back[i] as number) - (orig[i] as number)) < 1e-9);
  assert.throws(() => istft(s.Zxx, { nperseg: 64, window: new Float64Array(10) }), RangeError);
  assert.throws(() => istft(Tensor.zeros([4], { dtype: "f64" }) as never), RangeError);
});
