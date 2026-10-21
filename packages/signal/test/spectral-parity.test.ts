import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { ComplexTensor } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import {
  checkCOLA,
  checkNOLA,
  getWindow,
  istft,
  stft,
  welch,
  type DetrendFunction,
  type IstftOptions,
  type StftOptions,
  type WelchOptions,
  type WindowSpec,
} from "../src/index.ts";
import {
  checkStft,
  checkWelch,
  closeScaled,
  flat,
  flatten,
  type IstftWant,
  maxAbs,
  pyKw,
  pyWindow,
  runScipyOracle,
  SCIPY_SKIP_REASON,
  signalOf,
  type StftWant,
  type WelchWant,
} from "./helpers.ts";

// The gaps closed after the SciPy-parity defaults release: N-D input and axes, complex input, callable detrend,
// (name, param) windows, check_NOLA / check_COLA, istft's NOLA warning and complex output, and the welch field
// names. Every expectation comes from scipy.signal itself (see scripts/scipy_oracle.py).

type Kw = Record<string, unknown>;

// ---- helpers ------------------------------------------------------------------------------------------------------

const prod = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1);

/** Row-major nested lists of `flatValues` with `shape`. */
function nest(flatValues: readonly number[], shape: readonly number[]): unknown {
  if (shape.length === 1) return flatValues.slice();
  const inner = prod(shape.slice(1));
  return Array.from({ length: shape[0] as number }, (_, i) => nest(flatValues.slice(i * inner, (i + 1) * inner), shape.slice(1)));
}

const ndSignal = (shape: readonly number[], seed: number): number[] => signalOf(prod(shape), seed);
const ndTensor = (values: number[], shape: readonly number[]) => Tensor.fromTypedArray(Float64Array.from(values), [...shape], { dtype: "f64" });
const complexTensor = (re: number[], im: number[], shape: readonly number[]) => ComplexTensor.fromParts(ndTensor(re, shape), ndTensor(im, shape));
/** A second, different signal for the imaginary part. */
const imagOf = (n: number, seed: number) => signalOf(n, seed + 1000).map((v) => 0.7 * v - 1);

/** The warnings a call emitted through `onWarning`. */
function collect<T>(run: (onWarning: (m: string) => void) => T): { value: T; warnings: string[] } {
  const warnings: string[] = [];
  const value = run((m) => warnings.push(m));
  return { value, warnings };
}

const realOf = (x: Tensor | ComplexTensor): number[] => flat("real" in x ? x.real : x);

// ---- renamed fields -------------------------------------------------------------------------------------------------

test("naming: welch returns { freqs, Pxx }, stft { freqs, times, Zxx }, istft { times, x } -- and nothing else", () => {
  const t = Tensor.from(signalOf(300, 1), { dtype: "f64" });
  const w = welch(t);
  assert.deepEqual(Object.keys(w).sort(), ["Pxx", "freqs"]);
  assert.ok(!("frequencies" in w) && !("psd" in w));
  const s = stft(t);
  assert.deepEqual(Object.keys(s).sort(), ["Zxx", "freqs", "times"]);
  assert.deepEqual(Object.keys(istft(s.Zxx)).sort(), ["times", "x"]);
});

// ---- windows ------------------------------------------------------------------------------------------------------

const NAMED: WindowSpec[] = [
  "hann", "hamming", "blackman", "boxcar", "rectangular", "bartlett", "flattop", "parzen", "bohman", "nuttall",
  "blackmanharris", "barthann", "cosine", "triang", "tukey", "exponential", "taylor",
];
const TUPLES: WindowSpec[] = [
  ["kaiser", 0], ["kaiser", 5.5], ["kaiser", 14], ["gaussian", 3], ["gaussian", 0.7], ["general_gaussian", 1.5, 7],
  ["general_gaussian", 0.5, 3.1], ["tukey"], ["tukey", 0], ["tukey", 0.25], ["tukey", 0.5], ["tukey", 0.9], ["tukey", 1],
  ["exponential"], ["exponential", null, 2.5], ["general_hamming", 0.7], ["general_hamming", 0.5], ["general_cosine", [0.3, 0.5, 0.2]],
  ["taylor"], ["taylor", 6, 40], ["taylor", 3, 25, false], ["taylor", 1], ["hann"], ["triang"],
];
const WINDOW_LENGTHS = [1, 2, 3, 4, 5, 8, 15, 16, 31, 64, 100, 255];

test("getWindow: every supported name and (name, param) tuple matches scipy.signal.get_window, periodic and symmetric, at 1e-12", { skip: SCIPY_SKIP_REASON }, () => {
  const jobs: Kw[] = [];
  const cases: Array<{ spec: WindowSpec | number; n: number; fftbins: boolean }> = [];
  const all: Array<WindowSpec | number> = [...NAMED, ...TUPLES, 4.5, 0];
  for (const spec of all) {
    for (const n of WINDOW_LENGTHS) {
      for (const fftbins of [true, false]) {
        cases.push({ spec, n, fftbins });
        jobs.push({ op: "get_window", window: pyWindow(spec === "rectangular" ? "boxcar" : spec), n, fftbins });
      }
    }
  }
  // an explicit exponential centre is only legal for a periodic window
  for (const n of WINDOW_LENGTHS) {
    cases.push({ spec: ["exponential", 2.5, 3], n, fftbins: true });
    jobs.push({ op: "get_window", window: pyWindow(["exponential", 2.5, 3]), n, fftbins: true });
  }
  assert.ok(cases.length >= 900);
  const wants = runScipyOracle<Array<{ w: number[] }>>({ op: "batch", jobs });
  cases.forEach(({ spec, n, fftbins }, i) => {
    const got = getWindow(spec, n, fftbins);
    closeScaled(got, (wants[i] as { w: number[] }).w, 1, `getWindow ${JSON.stringify(spec)} n=${n} fftbins=${fftbins}`, 1e-12);
  });
});

test("getWindow: periodic is the default, a 1-sample window is [1], arrays pass through, bad specs throw like SciPy", { skip: SCIPY_SKIP_REASON }, () => {
  assert.deepEqual(Array.from(getWindow(["kaiser", 6], 1)), [1]);
  assert.deepEqual(Array.from(getWindow("bohman", 8)), Array.from(getWindow("bohman", 8, true)));
  const mine = Float64Array.from([1, 2, 3]);
  assert.equal(getWindow(mine, 3), mine);
  const bad: Array<{ spec: unknown; n?: number; fftbins?: boolean }> = [
    { spec: "kaiser" },
    { spec: "gaussian" },
    { spec: "general_hamming" },
    { spec: "nonesuch" },
    { spec: ["kaiser"] },
    { spec: ["kaiser", 1, 2] },
    { spec: ["hann", 3] },
    { spec: ["exponential", 2, 3], fftbins: false },
    { spec: ["taylor", 0] },
  ];
  const wants = runScipyOracle<Array<{ error?: string; error_type?: string }>>({
    op: "batch",
    jobs: bad.map((b) => ({ op: "get_window", expect_error: true, window: pyWindow(b.spec), n: b.n ?? 8, fftbins: b.fftbins ?? true })),
  });
  bad.forEach((b, i) => {
    assert.ok((wants[i] as { error?: string }).error, `SciPy accepts ${JSON.stringify(b.spec)} (the test expects it to reject it)`);
    // SciPy's ValueError is a RangeError here; its TypeError (wrong parameter count) is a TypeError
    const cls = (wants[i] as { error_type?: string }).error_type === "TypeError" ? TypeError : RangeError;
    assert.throws(() => getWindow(b.spec as never, b.n ?? 8, b.fftbins ?? true), cls, JSON.stringify(b.spec));
  });
  assert.throws(() => getWindow("hann", -1), RangeError);
  assert.throws(() => getWindow(new Float64Array(3), 4), RangeError);
});

test("welch/stft/istft take window tuples and match SciPy", { skip: SCIPY_SKIP_REASON }, () => {
  const x = signalOf(500, 3);
  const windows: WindowSpec[] = [["kaiser", 8], ["gaussian", 6], ["tukey", 0.3], ["exponential", null, 12], ["general_hamming", 0.6], "bartlett", "flattop", "parzen", "bohman", "nuttall", "blackmanharris", "barthann", "cosine", "triang", "taylor"];
  const jobs: Kw[] = [];
  const checks: Array<(want: unknown) => void> = [];
  for (const window of windows) {
    const what = JSON.stringify(window);
    const wo: WelchOptions = { window, nperseg: 64, noverlap: 40 };
    const gotW = welch(Tensor.from(x, { dtype: "f64" }), wo);
    jobs.push({ op: "welch_kw", x, kwargs: pyKw({ ...wo }, false) });
    checks.push((w) => checkWelch(gotW, w as WelchWant, `welch ${what}`));
    const so: StftOptions = { window, nperseg: 50, noverlap: 30 };
    const gotS = stft(Tensor.from(x, { dtype: "f64" }), so);
    jobs.push({ op: "stft_kw", x, kwargs: pyKw({ ...so }, true) });
    checks.push((w) => checkStft(gotS, w as StftWant, `stft ${what}`));
    // istft of our own spectrogram with the same window, against SciPy's istft of the same input
    const io: IstftOptions = { window, nperseg: 50, noverlap: 30, onWarning: () => {} };
    const { re, im } = { re: nest(flat(gotS.Zxx.real), gotS.Zxx.shape), im: nest(flat(gotS.Zxx.imag), gotS.Zxx.shape) };
    jobs.push({ op: "istft_kw", real: re, imag: im, kwargs: pyKw({ ...io }, true) });
    const gotI = istft(gotS.Zxx, io);
    checks.push((w) => {
      const want = w as IstftWant;
      // an unusual window need not satisfy NOLA at this overlap: SciPy and we must then agree on the (garbage) output too
      closeScaled(realOf(gotI.x), flatten(want.real), Math.max(maxAbs(flatten(want.real)), 1), `istft ${what}`, 1e-9);
    });
  }
  const wants = runScipyOracle<unknown[]>({ op: "batch", jobs });
  checks.forEach((c, i) => c(wants[i]));
});

// ---- check_NOLA / check_COLA ----------------------------------------------------------------------------------------

test("checkNOLA and checkCOLA agree with scipy.signal.check_NOLA / check_COLA over windows, lengths, overlaps and tolerances", { skip: SCIPY_SKIP_REASON }, () => {
  const windows: WindowSpec[] = ["hann", "hamming", "boxcar", "blackman", "bartlett", "triang", "flattop", "parzen", "bohman", "nuttall", "cosine", ["kaiser", 8], ["tukey", 0.25], ["gaussian", 4], ["exponential", null, 5]];
  const jobs: Kw[] = [];
  const cases: Array<{ window: WindowSpec; nperseg: number; noverlap: number; tol?: number }> = [];
  for (const window of windows) {
    for (const nperseg of [8, 16, 33, 64]) {
      const overlaps = new Set([0, 1, Math.floor(nperseg / 4), Math.floor(nperseg / 3), nperseg >> 1, Math.floor((3 * nperseg) / 4), nperseg - 1]);
      for (const noverlap of overlaps) {
        for (const tol of [undefined, 1e-3, 1e-14]) {
          cases.push({ window, nperseg, noverlap, tol });
        }
      }
    }
  }
  // an array window
  const arrayWindow = Float64Array.from({ length: 24 }, (_, i) => 1 + 0.5 * Math.sin(i));
  for (const noverlap of [0, 6, 12, 18]) cases.push({ window: arrayWindow, nperseg: 24, noverlap });
  for (const c of cases) {
    const base = { window: pyWindow(c.window), nperseg: c.nperseg, noverlap: c.noverlap, ...(c.tol === undefined ? {} : { tol: c.tol }) };
    jobs.push({ op: "check_nola", ...base }, { op: "check_cola", ...base });
  }
  const wants = runScipyOracle<Array<{ ok: boolean }>>({ op: "batch", jobs });
  let nolaTrue = 0;
  let nolaFalse = 0;
  let colaTrue = 0;
  let colaFalse = 0;
  cases.forEach((c, i) => {
    const what = `${JSON.stringify(c.window instanceof Float64Array ? "array" : c.window)} nperseg=${c.nperseg} noverlap=${c.noverlap} tol=${c.tol}`;
    const nola = checkNOLA(c.window, c.nperseg, c.noverlap, c.tol);
    const cola = checkCOLA(c.window, c.nperseg, c.noverlap, c.tol);
    assert.equal(nola, (wants[2 * i] as { ok: boolean }).ok, `checkNOLA ${what}`);
    assert.equal(cola, (wants[2 * i + 1] as { ok: boolean }).ok, `checkCOLA ${what}`);
    if (nola) nolaTrue++;
    else nolaFalse++;
    if (cola) colaTrue++;
    else colaFalse++;
  });
  // the grid is not degenerate: both answers occur for both functions
  assert.ok(nolaTrue > 20 && nolaFalse > 20 && colaTrue > 20 && colaFalse > 20, `${nolaTrue}/${nolaFalse}/${colaTrue}/${colaFalse}`);
  assert.ok(cases.length >= 800);
});

test("checkNOLA / checkCOLA: SciPy's documented examples and argument validation", () => {
  assert.equal(checkNOLA(getWindow("hann", 64, false), 64, 0), false);
  assert.equal(checkNOLA(getWindow("hann", 64, false), 64, 1), false);
  assert.equal(checkNOLA(getWindow("hann", 64, false), 64, 2), true);
  assert.equal(checkCOLA("boxcar", 100, 75), true);
  assert.equal(checkCOLA("boxcar", 100, 25), false);
  assert.equal(checkCOLA("hann", 120, 60), true);
  assert.throws(() => checkNOLA("hann", 0, 0), RangeError);
  assert.throws(() => checkNOLA("hann", 16, 16), RangeError);
  assert.throws(() => checkNOLA("hann", 16, -1), RangeError);
  assert.throws(() => checkCOLA("hann", 16, 16), RangeError);
  assert.throws(() => checkNOLA(new Float64Array(8), 16, 4), RangeError);
  assert.throws(() => checkNOLA("nonesuch" as never, 16, 4), RangeError);
});

// ---- N-D input and axes --------------------------------------------------------------------------------------------

const WELCH_VARIANTS: WelchOptions[] = [
  {},
  { nperseg: 12, nfft: 20, onesided: false, detrend: "linear", fs: 5 },
  { average: "median", scaling: "spectrum", window: ["kaiser", 6], noverlap: 2 },
];
const STFT_VARIANTS: StftOptions[] = [
  {},
  { nperseg: 10, boundary: "even", padded: false, detrend: "constant" },
  { nperseg: 7, nfft: 11, onesided: false, scaling: "psd", fs: 3, boundary: "odd" },
];

const ND_SHAPES: ReadonlyArray<{ shape: number[]; axes: number[] }> = [
  { shape: [4, 60], axes: [0, 1, -1, -2] },
  { shape: [18, 20, 26], axes: [0, 1, 2, -1, -2, -3] },
];

test("welch: 2-D and 3-D input along every axis (positive and negative) matches scipy.signal.welch in values and shape", { skip: SCIPY_SKIP_REASON }, () => {
  const jobs: Kw[] = [];
  const checks: Array<(want: WelchWant) => void> = [];
  ND_SHAPES.forEach(({ shape, axes }, si) => {
    const x = ndSignal(shape, 10 + si);
    const nested = nest(x, shape);
    for (const axis of axes) {
      for (const variant of WELCH_VARIANTS) {
        const opts: WelchOptions = { nperseg: 8, ...variant, axis };
        const got = welch(ndTensor(x, shape), opts);
        jobs.push({ op: "welch_kw", x: nested, kwargs: pyKw({ ...opts }, false) });
        const what = `shape=${shape} ${JSON.stringify({ ...opts, window: String(opts.window) })}`;
        checks.push((want) => checkWelch(got, want, what));
      }
    }
  });
  assert.ok(jobs.length >= 30);
  const wants = runScipyOracle<WelchWant[]>({ op: "batch", jobs });
  checks.forEach((c, i) => c(wants[i] as WelchWant));
});

test("welch: axis defaults to -1, and a transposed input along axis 0 equals the original along axis -1", () => {
  const shape = [5, 80];
  const x = ndSignal(shape, 21);
  const t = ndTensor(x, shape);
  const a = welch(t, { nperseg: 16 });
  const b = welch(t, { nperseg: 16, axis: -1 });
  assert.deepEqual(flat(a.Pxx), flat(b.Pxx));
  assert.deepEqual(a.Pxx.shape, [5, 9]);
  const transposed = Float64Array.from({ length: 400 }, (_, k) => x[(k % 5) * 80 + Math.floor(k / 5)] as number);
  const c = welch(ndTensor(Array.from(transposed), [80, 5]), { nperseg: 16, axis: 0 });
  assert.deepEqual(c.Pxx.shape, [9, 5]);
  const ac = flat(a.Pxx);
  const cc = flat(c.Pxx);
  for (let r = 0; r < 5; r++) for (let k = 0; k < 9; k++) assert.ok(Math.abs((ac[r * 9 + k] as number) - (cc[k * 5 + r] as number)) < 1e-12);
});

test("stft: 2-D and 3-D input along every axis matches scipy.signal.stft (frequency axis where the data axis was, time last)", { skip: SCIPY_SKIP_REASON }, () => {
  const jobs: Kw[] = [];
  const checks: Array<(want: StftWant) => void> = [];
  ND_SHAPES.forEach(({ shape, axes }, si) => {
    const x = ndSignal(shape, 30 + si);
    const nested = nest(x, shape);
    for (const axis of axes) {
      for (const variant of STFT_VARIANTS) {
        const opts: StftOptions = { nperseg: 8, ...variant, axis };
        const got = stft(ndTensor(x, shape), opts);
        jobs.push({ op: "stft_kw", x: nested, kwargs: pyKw({ ...opts }, true) });
        const what = `shape=${shape} ${JSON.stringify(opts)}`;
        checks.push((want) => checkStft(got, want, what));
      }
    }
  });
  assert.ok(jobs.length >= 30);
  const wants = runScipyOracle<StftWant[]>({ op: "batch", jobs });
  checks.forEach((c, i) => c(wants[i] as StftWant));
});

/** A random complex spectrogram of `shape` (not a valid STFT: istft is linear, so SciPy's output is still the reference). */
function randomZ(shape: number[], seed: number): { re: number[]; im: number[] } {
  const n = prod(shape);
  return { re: signalOf(n, seed).map((v) => v - 3), im: signalOf(n, seed + 5).map((v) => 0.5 * v) };
}

test("istft: freqAxis / timeAxis in every position of 3-D and 4-D input match scipy.signal.istft (shape, axis placement, values)", { skip: SCIPY_SKIP_REASON }, () => {
  const jobs: Kw[] = [];
  const checks: Array<(want: IstftWant) => void> = [];
  const cases: Array<{ shape: number[]; fa: number; ta: number; extra: IstftOptions }> = [];
  // bins=9 -> nperseg 16 one-sided; the time axis has 5 segments
  const place = (shape3: [number, number, number]) => {
    // shape3 holds [others, bins, segs] placed in every permutation
    const dims = { o: shape3[0], f: 9, t: shape3[2] };
    const perms: Array<Array<"o" | "f" | "t">> = [["o", "f", "t"], ["o", "t", "f"], ["f", "o", "t"], ["f", "t", "o"], ["t", "o", "f"], ["t", "f", "o"]];
    for (const perm of perms) {
      const shape = perm.map((k) => dims[k]);
      cases.push({ shape, fa: perm.indexOf("f"), ta: perm.indexOf("t"), extra: {} });
    }
  };
  place([3, 9, 5]);
  // negative axes, and the defaults on a 3-D input (-2 / -1)
  cases.push({ shape: [3, 9, 5], fa: -2, ta: -1, extra: {} });
  cases.push({ shape: [9, 5, 3], fa: -3, ta: -2, extra: {} });
  // 4-D
  cases.push({ shape: [2, 9, 3, 6], fa: 1, ta: 3, extra: {} });
  cases.push({ shape: [6, 2, 3, 9], fa: 3, ta: 0, extra: {} });
  cases.push({ shape: [2, 6, 9, 3], fa: 2, ta: 1, extra: { nperseg: 16, noverlap: 12, window: "hamming" } });
  // two-sided input (nperseg = bins): complex output
  cases.push({ shape: [3, 16, 5], fa: 1, ta: 2, extra: { inputOnesided: false } });
  cases.push({ shape: [16, 3, 5], fa: 0, ta: 2, extra: { inputOnesided: false, window: ["kaiser", 5] } });
  cases.push({ shape: [5, 3, 16], fa: 2, ta: 0, extra: { inputOnesided: false, fs: 7 } });
  cases.forEach(({ shape, fa, ta, extra }, ci) => {
    const { re, im } = randomZ(shape, 40 + ci);
    const Z = complexTensor(re, im, shape);
    const opts: IstftOptions = { freqAxis: fa, timeAxis: ta, onWarning: () => {}, ...extra };
    const got = istft(Z, opts);
    jobs.push({ op: "istft_kw", real: nest(re, shape), imag: nest(im, shape), kwargs: pyKw({ ...opts }, true) });
    const what = `shape=${shape} ${JSON.stringify(opts)}`;
    checks.push((want) => {
      assert.deepEqual(got.x.shape, want.shape, `x shape ${what}`);
      assert.equal(want.complex, extra.inputOnesided === false, `complex-ness ${what}`);
      assert.equal("imag" in got.x, want.complex, `x is ${want.complex ? "a ComplexTensor" : "a Tensor"} ${what}`);
      const wr = flatten(want.real);
      const scale = Math.max(maxAbs(wr), maxAbs(flatten(want.imag)), 1e-300);
      closeScaled(realOf(got.x), wr, scale, `x ${what}`);
      if ("imag" in got.x) closeScaled(flat(got.x.imag), flatten(want.imag), scale, `x.imag ${what}`);
      // times: one sample per output sample (SciPy 1.16 sizes it from x.shape[0], which is only right when the time axis comes first)
      const faPos = fa < 0 ? fa + shape.length : fa;
      const nperseg = (extra.nperseg ?? (extra.inputOnesided === false ? shape[faPos] : 2 * ((shape[faPos] as number) - 1))) as number;
      const hop = nperseg - (extra.noverlap ?? nperseg >> 1);
      const nseg = shape[ta < 0 ? ta + shape.length : ta] as number;
      assert.equal(got.times.length, nperseg + (nseg - 1) * hop - 2 * (nperseg >> 1), `times length ${what}`);
      assert.ok(got.times.every((t, i) => t === i / (extra.fs ?? 1)), `times values ${what}`);
    });
  });
  const wants = runScipyOracle<IstftWant[]>({ op: "batch", jobs });
  checks.forEach((c, i) => c(wants[i] as IstftWant));
});

test("stft -> istft round-trips N-D input along a middle axis, restoring the signal at the time position", () => {
  const shape = [3, 90, 2];
  const x = ndSignal(shape, 50);
  const { Zxx } = stft(ndTensor(x, shape), { nperseg: 16, axis: 1 });
  assert.deepEqual(Zxx.shape, [3, 9, 2, 13]); // [3, freqs, 2, times]
  const { x: y } = istft(Zxx, { freqAxis: 1, timeAxis: 3, nperseg: 16 });
  assert.deepEqual(y.shape, [3, 2, 96]); // outer axes, then the signal where the time axis was (padded to a whole number of segments)
  const back = flat(y as Tensor);
  for (let a = 0; a < 3; a++) for (let b = 0; b < 2; b++) for (let i = 0; i < 90; i++) {
    assert.ok(Math.abs((back[(a * 2 + b) * 96 + i] as number) - (x[(a * 90 + i) * 2 + b] as number)) < 1e-9, `[${a}][${i}][${b}]`);
  }
});

test("N-D: validation of axes and dimensions", () => {
  const t = ndTensor(ndSignal([4, 40], 1), [4, 40]);
  assert.throws(() => welch(t, { axis: 2 }), RangeError);
  assert.throws(() => welch(t, { axis: -3 }), RangeError);
  assert.throws(() => stft(t, { axis: 5 }), RangeError);
  assert.doesNotThrow(() => welch(t, { axis: 1.5 }), "SciPy does int(axis)");
  const Z = stft(t).Zxx;
  assert.throws(() => istft(Z, { freqAxis: 1, timeAxis: 1 }), RangeError);
  assert.throws(() => istft(Z, { freqAxis: -1, timeAxis: 2 }), RangeError); // the same axis, written two ways
  assert.throws(() => istft(Z, { freqAxis: 3 }), RangeError);
  assert.throws(() => istft(ndTensor([1, 2, 3], [3]) as never), RangeError);
});

// ---- complex input ------------------------------------------------------------------------------------------------

const COMPLEX_WELCH: WelchOptions[] = [
  {},
  { onesided: false, nperseg: 32, nfft: 50, detrend: "linear" },
  { scaling: "spectrum", average: "median", window: ["tukey", 0.4], noverlap: 8, nperseg: 24, fs: 10 },
  { detrend: false, nperseg: 21, noverlap: 3, window: "blackmanharris" },
];
const COMPLEX_STFT: StftOptions[] = [
  {},
  { onesided: false, nperseg: 32, nfft: 40, detrend: "linear", boundary: "even" },
  { scaling: "psd", window: ["kaiser", 4], noverlap: 8, nperseg: 24, fs: 10, boundary: "odd", padded: false },
  { detrend: "constant", nperseg: 21, boundary: null, window: "bohman" },
  { nperseg: 16, boundary: "constant", scaling: "spectrum" },
];

test("welch: complex input (1-D, 2-D along each axis) is two-sided in FFT order and matches scipy.signal.welch; no warning, as in SciPy", { skip: SCIPY_SKIP_REASON }, () => {
  const jobs: Kw[] = [];
  const checks: Array<(want: WelchWant) => void> = [];
  const specs: Array<{ shape: number[]; axes: number[] }> = [{ shape: [200], axes: [-1] }, { shape: [3, 90], axes: [0, 1, -1] }, { shape: [40, 3, 5], axes: [0, 1, 2] }];
  specs.forEach(({ shape, axes }, si) => {
    const re = ndSignal(shape, 60 + si);
    const im = imagOf(prod(shape), 60 + si);
    for (const axis of axes) {
      for (const variant of COMPLEX_WELCH) {
        const opts: WelchOptions = { nperseg: 20, ...variant, axis };
        if ((shape[axis < 0 ? axis + shape.length : axis] as number) < (opts.nperseg as number)) continue;
        const { value: got, warnings } = collect((onWarning) => welch(complexTensor(re, im, shape), { ...opts, onWarning }));
        assert.deepEqual(warnings, [], "welch must not warn for complex input (SciPy's does not)");
        jobs.push({ op: "welch_kw", x: nest(re, shape), x_imag: nest(im, shape), kwargs: pyKw({ ...opts }, false) });
        const what = `complex shape=${shape} ${JSON.stringify({ ...opts, window: String(opts.window) })}`;
        checks.push((want) => {
          assert.ok(!want.warnings.some((m) => /complex/i.test(m)), `scipy.welch warned: ${want.warnings}`);
          assert.equal(got.freqs.length, got.Pxx.shape[axis < 0 ? axis + shape.length : axis]);
          assert.ok(got.freqs.length === (opts.nfft ?? opts.nperseg), "complex input is two-sided: nfft bins");
          checkWelch(got, want, what);
        });
      }
    }
  });
  const wants = runScipyOracle<WelchWant[]>({ op: "batch", jobs });
  checks.forEach((c, i) => c(wants[i] as WelchWant));
});

test("stft: complex input (1-D, 2-D, 3-D along each axis) is two-sided in FFT order and matches scipy.signal.stft", { skip: SCIPY_SKIP_REASON }, () => {
  const jobs: Kw[] = [];
  const checks: Array<(want: StftWant) => void> = [];
  const specs: Array<{ shape: number[]; axes: number[] }> = [{ shape: [200], axes: [-1] }, { shape: [3, 90], axes: [0, 1] }, { shape: [40, 3, 5], axes: [0, 1, 2, -3] }];
  specs.forEach(({ shape, axes }, si) => {
    const re = ndSignal(shape, 70 + si);
    const im = imagOf(prod(shape), 70 + si);
    for (const axis of axes) {
      for (const variant of COMPLEX_STFT) {
        const opts: StftOptions = { nperseg: 20, ...variant, axis };
        if ((shape[axis < 0 ? axis + shape.length : axis] as number) < (opts.nperseg as number)) continue;
        const { value: got, warnings } = collect((onWarning) => stft(complexTensor(re, im, shape), { ...opts, onWarning }));
        jobs.push({ op: "stft_kw", x: nest(re, shape), x_imag: nest(im, shape), kwargs: pyKw({ ...opts }, true) });
        const what = `complex shape=${shape} ${JSON.stringify(opts)}`;
        checks.push((want) => {
          // SciPy warns exactly when onesided was left on; so do we, once, through onWarning
          const scipyWarned = want.warnings.some((m) => /complex/i.test(m));
          assert.equal(scipyWarned, opts.onesided !== false, `scipy warning ${what}`);
          assert.equal(warnings.length, scipyWarned ? 1 : 0, `our warning ${what}: ${warnings}`);
          if (scipyWarned) assert.match(warnings[0] as string, /complex/i);
          assert.equal(got.freqs.length, opts.nfft ?? opts.nperseg);
          checkStft(got, want, what);
        });
      }
    }
  });
  const wants = runScipyOracle<StftWant[]>({ op: "batch", jobs });
  checks.forEach((c, i) => c(wants[i] as StftWant));
});

test("istft: complex round trip (two-sided stft of a complex signal -> complex x), and inputOnesided: false returns a ComplexTensor as SciPy does", { skip: SCIPY_SKIP_REASON }, () => {
  const jobs: Kw[] = [];
  const checks: Array<() => void> = [];
  const make = (shape: number[], seed: number, sopts: StftOptions, axis: number) => {
    const re = ndSignal(shape, seed);
    const im = imagOf(prod(shape), seed);
    const opts: StftOptions = { ...sopts, axis, onesided: false, onWarning: () => {} };
    const { Zxx } = stft(complexTensor(re, im, shape), opts);
    const iopts: IstftOptions = { inputOnesided: false, freqAxis: axis < 0 ? axis + shape.length : axis, timeAxis: shape.length, nperseg: sopts.nperseg, noverlap: sopts.noverlap, nfft: sopts.nfft, window: sopts.window, scaling: sopts.scaling, fs: sopts.fs };
    const got = istft(Zxx, iopts);
    jobs.push({ op: "istft_kw", real: nest(flat(Zxx.real), Zxx.shape), imag: nest(flat(Zxx.imag), Zxx.shape), kwargs: pyKw({ ...iopts }, true) });
    checks.push(() => {
      assert.ok("imag" in got.x, "complex output");
    });
    return { re, im, shape, axis, got };
  };
  const cases = [
    make([150], 80, { nperseg: 20 }, -1),
    make([150], 81, { nperseg: 21, nfft: 32, window: ["kaiser", 5], noverlap: 15 }, -1),
    make([2, 90], 82, { nperseg: 16, scaling: "psd", fs: 4 }, 1),
    make([90, 2], 83, { nperseg: 16, window: "hamming" }, 0),
  ];
  const wants = runScipyOracle<IstftWant[]>({ op: "batch", jobs });
  cases.forEach(({ re, im, shape, axis, got }, i) => {
    const want = wants[i] as IstftWant;
    const wr = flatten(want.real);
    const scale = Math.max(maxAbs(wr), maxAbs(flatten(want.imag)));
    closeScaled(realOf(got.x), wr, scale, `real vs scipy ${i}`);
    closeScaled(flat((got.x as ComplexTensor).imag), flatten(want.imag), scale, `imag vs scipy ${i}`);
    // and it is the original signal back (the time axis is last: [outer..., signal])
    const ax = axis < 0 ? axis + shape.length : axis;
    const outer = shape.filter((_, d) => d !== ax);
    const L = shape[ax] as number;
    const rows = prod(outer);
    assert.deepEqual(got.x.shape, [...outer, got.x.shape[got.x.shape.length - 1]]);
    const gr = realOf(got.x);
    const gi = flat((got.x as ComplexTensor).imag);
    const stride = got.x.shape[got.x.shape.length - 1] as number;
    const inner = prod(shape.slice(ax + 1));
    for (let r = 0; r < rows; r++) {
      const p = Math.floor(r / inner);
      const q = r % inner;
      for (let j = 0; j < L; j++) {
        const src = (p * L + j) * inner + q;
        assert.ok(Math.abs((gr[r * stride + j] as number) - (re[src] as number)) < 1e-9, `re ${i} ${r} ${j}`);
        assert.ok(Math.abs((gi[r * stride + j] as number) - (im[src] as number)) < 1e-9, `im ${i} ${r} ${j}`);
      }
    }
  });
  checks.forEach((c) => c());
});

test("istft: a real signal's two-sided spectrogram gives a ComplexTensor with ~0 imaginary part (SciPy), a one-sided one a Tensor", () => {
  const x = signalOf(200, 90);
  const t = Tensor.from(x, { dtype: "f64" });
  const two = istft(stft(t, { onesided: false, nperseg: 32 }).Zxx, { inputOnesided: false });
  assert.ok("imag" in two.x);
  assert.ok(maxAbs(flat((two.x as ComplexTensor).imag)) < 1e-12);
  closeScaled(realOf(two.x).slice(0, 200), x, maxAbs(x), "real part is the signal", 1e-9);
  const one = istft(stft(t, { nperseg: 32 }).Zxx);
  assert.ok(!("imag" in one.x));
  assert.ok(one.x instanceof Tensor);
  // a real Tensor is accepted as a spectrogram too
  assert.deepEqual(istft(Tensor.zeros([9, 4], { dtype: "f64" })).x.shape, [24]); // 16 + 3 * 8 samples, minus 8 at each end
});

test("complex input: onWarning receives the stft message; with no onWarning it goes to console.warn; welch stays silent", () => {
  const re = signalOf(64, 1);
  const z = complexTensor(re, imagOf(64, 1), [64]);
  const original = console.warn;
  const logged: string[] = [];
  console.warn = (m: unknown) => void logged.push(String(m));
  try {
    stft(z, { nperseg: 16 });
    assert.equal(logged.length, 1);
    assert.match(logged[0] as string, /complex/i);
    welch(z, { nperseg: 16 });
    stft(z, { nperseg: 16, onesided: false });
    assert.equal(logged.length, 1, "welch and an explicit onesided:false do not warn");
    stft(z, { nperseg: 16, onWarning: () => {} });
    assert.equal(logged.length, 1, "onWarning replaces console.warn");
  } finally {
    console.warn = original;
  }
});

// ---- callable detrend -----------------------------------------------------------------------------------------------

/** A detrend function built from a per-part function, for real (Tensor) and complex (ComplexTensor) segments. */
function detrendOf(f: (v: Float64Array) => Float64Array): DetrendFunction {
  const part = (t: Tensor) => Tensor.fromTypedArray(f(Float64Array.from(t.contiguous().data as Float64Array)), [t.shape[0] as number], { dtype: "f64" });
  return (seg) => ("imag" in seg ? ComplexTensor.fromParts(part(seg.real), part(seg.imag)) : part(seg));
}
const first3 = detrendOf((v) => {
  const m = (v[0]! + v[1]! + v[2]!) / 3;
  return v.map((a) => a - m);
});
const ends = detrendOf((v) => {
  const n = v.length;
  return v.map((a, i) => a - (v[0]! + (v[n - 1]! - v[0]!) * (i / (n - 1))));
});
const DETRENDS: Array<{ name: string; fn: DetrendFunction }> = [{ name: "first3", fn: first3 }, { name: "ends", fn: ends }];

test("callable detrend: applied per segment, matches scipy.signal.welch / stft with the equivalent function (real and complex, 1-D and 2-D)", { skip: SCIPY_SKIP_REASON }, () => {
  const jobs: Kw[] = [];
  const checks: Array<(want: unknown) => void> = [];
  const inputs: Array<{ shape: number[]; complex: boolean; axis: number }> = [
    { shape: [300], complex: false, axis: -1 },
    { shape: [3, 120], complex: false, axis: -1 },
    { shape: [120, 3], complex: false, axis: 0 },
    { shape: [300], complex: true, axis: -1 },
    { shape: [120, 3], complex: true, axis: 0 },
  ];
  inputs.forEach(({ shape, complex, axis }, ii) => {
    const re = ndSignal(shape, 100 + ii);
    const im = complex ? imagOf(prod(shape), 100 + ii) : null;
    const t = im ? complexTensor(re, im, shape) : ndTensor(re, shape);
    const feed = { x: nest(re, shape), ...(im ? { x_imag: nest(im, shape) } : {}) };
    for (const { name, fn } of DETRENDS) {
      const wo: WelchOptions = { nperseg: 24, noverlap: 10, detrend: fn, axis, onesided: false };
      const gotW = welch(t, wo);
      jobs.push({ op: "welch_kw", ...feed, kwargs: { ...pyKw({ ...wo }, false), detrend: { fn: name } } });
      checks.push((w) => checkWelch(gotW, w as WelchWant, `welch ${name} ${JSON.stringify(shape)} complex=${complex}`));
      // stft: SciPy hands the function a differently shaped array, so only the default axis is compared
      if (axis === -1) {
        const so: StftOptions = { nperseg: 24, detrend: fn, axis, onWarning: () => {} };
        const gotS = stft(t, so);
        jobs.push({ op: "stft_kw", ...feed, kwargs: { ...pyKw({ ...so }, true), detrend: { fn: name } } });
        checks.push((w) => checkStft(gotS, w as StftWant, `stft ${name} ${JSON.stringify(shape)} complex=${complex}`));
      }
    }
  });
  // it is a real detrend: differs from the string options
  const x = ndTensor(signalOf(200, 5), [200]);
  assert.notDeepEqual(flat(welch(x, { nperseg: 32, detrend: first3 }).Pxx), flat(welch(x, { nperseg: 32 }).Pxx));
  const wants = runScipyOracle<unknown[]>({ op: "batch", jobs });
  checks.forEach((c, i) => c(wants[i]));
});

test("callable detrend: called once per segment with a 1-D nperseg-sample Tensor (ComplexTensor for complex input); bad returns throw", () => {
  const seen: number[][] = [];
  const x = Tensor.from(signalOf(40, 7), { dtype: "f64" });
  welch(x, {
    nperseg: 16,
    detrend: (seg) => {
      assert.ok(seg instanceof Tensor);
      seen.push([...seg.shape]);
      return seg;
    },
  });
  assert.equal(seen.length, 4); // (40 - 16) / 8 + 1 segments
  assert.ok(seen.every((s) => s.length === 1 && s[0] === 16));
  let complexSeen = 0;
  welch(complexTensor(signalOf(40, 7), imagOf(40, 7), [40]), {
    nperseg: 16,
    detrend: (seg) => {
      assert.ok("imag" in seg);
      complexSeen++;
      return seg;
    },
  });
  assert.equal(complexSeen, 4);
  // a Float64Array is fine for real input
  assert.doesNotThrow(() => welch(x, { nperseg: 16, detrend: (seg) => Float64Array.from((seg as Tensor).contiguous().data as Float64Array) }));
  assert.throws(() => welch(x, { nperseg: 16, detrend: () => new Float64Array(3) }), RangeError);
  assert.throws(() => welch(x, { nperseg: 16, detrend: (seg) => ComplexTensor.fromReal(seg as Tensor) }), RangeError);
  assert.throws(() => welch(complexTensor(signalOf(40, 7), imagOf(40, 7), [40]), { nperseg: 16, detrend: () => new Float64Array(16) }), RangeError);
  assert.throws(() => welch(x, { detrend: "quadratic" as never }), RangeError);
});

// ---- the NOLA warning -----------------------------------------------------------------------------------------------

test("istft: reports a failed NOLA check through onWarning, exactly when SciPy warns", { skip: SCIPY_SKIP_REASON }, () => {
  const x = signalOf(160, 120);
  const cases: Array<{ window: WindowSpec; nperseg: number; noverlap: number; boundary: boolean }> = [
    { window: "hann", nperseg: 16, noverlap: 8, boundary: true }, // fine
    { window: "hann", nperseg: 16, noverlap: 0, boundary: true }, // hann with no overlap has zeros: fails
    { window: "hann", nperseg: 16, noverlap: 8, boundary: false }, // fine in the middle, but the edge sample's window power is ~0: fails
    { window: "boxcar", nperseg: 16, noverlap: 0, boundary: false }, // fine
    { window: "hamming", nperseg: 20, noverlap: 5, boundary: true }, // fine
    { window: "blackman", nperseg: 16, noverlap: 2, boundary: true },
    { window: ["tukey", 0.5], nperseg: 16, noverlap: 8, boundary: true },
    { window: "bartlett", nperseg: 16, noverlap: 12, boundary: false },
  ];
  const jobs: Kw[] = [];
  const mine: string[][] = [];
  for (const c of cases) {
    const so: StftOptions = { window: c.window, nperseg: c.nperseg, noverlap: c.noverlap, boundary: c.boundary ? "zeros" : null, padded: false };
    const { Zxx } = stft(Tensor.from(x, { dtype: "f64" }), so);
    const io: IstftOptions = { window: c.window, nperseg: c.nperseg, noverlap: c.noverlap, boundary: c.boundary };
    jobs.push({ op: "istft_kw", real: nest(flat(Zxx.real), Zxx.shape), imag: nest(flat(Zxx.imag), Zxx.shape), kwargs: pyKw({ ...io }, true) });
    mine.push(collect((onWarning) => istft(Zxx, { ...io, onWarning })).warnings);
  }
  const wants = runScipyOracle<IstftWant[]>({ op: "batch", jobs });
  let failing = 0;
  cases.forEach((c, i) => {
    const scipyWarned = (wants[i] as IstftWant).warnings.some((m) => /NOLA/.test(m));
    assert.equal(mine[i]!.length, scipyWarned ? 1 : 0, `${JSON.stringify(c)}: ${mine[i]}`);
    if (scipyWarned) {
      failing++;
      assert.match(mine[i]![0] as string, /NOLA/);
    }
    if (!c.boundary && scipyWarned) assert.match(mine[i]![0] as string, /boundary/, "mentions the missing boundary, as SciPy does");
  });
  assert.ok(failing >= 2 && failing < cases.length);
});

test("istft: with no onWarning a failed NOLA check goes to console.warn; a passing one is silent; checkNOLA predicts the interior failure", () => {
  const t = Tensor.from(signalOf(160, 121), { dtype: "f64" });
  const { Zxx } = stft(t, { nperseg: 16, noverlap: 0, boundary: null, padded: false });
  const original = console.warn;
  const logged: string[] = [];
  console.warn = (m: unknown) => void logged.push(String(m));
  try {
    istft(Zxx, { nperseg: 16, noverlap: 0, boundary: false });
    assert.equal(logged.length, 1);
    assert.match(logged[0] as string, /NOLA/);
    istft(stft(t, { onWarning: () => {} }).Zxx);
    assert.equal(logged.length, 1, "the default round trip satisfies NOLA");
  } finally {
    console.warn = original;
  }
  assert.equal(checkNOLA("hann", 16, 0), false);
  assert.equal(checkNOLA("hann", 16, 8), true);
});
