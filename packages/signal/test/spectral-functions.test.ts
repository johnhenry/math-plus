import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { ComplexTensor } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { coherence, csd, type DetrendFunction, getWindow, istft, periodogram, type SignalInput, spectrogram, stft, welch } from "../src/index.ts";
import { closeScaled, flat, flatten, type Kw, maxAbs, pyKw, runScipyOracle, SCIPY_17_PLUS, SCIPY_SKIP_REASON, signalOf } from "./helpers.ts";

// periodogram, csd, spectrogram and coherence against scipy.signal, and the empty-input and warning behaviour of every
// spectral function (welch, csd, periodogram, coherence, spectrogram, stft, istft). Every expectation is SciPy's own
// output: values at 1e-9 of the reference's peak, shapes exactly, the warnings' text exactly, and the error CLASS
// (SciPy's ValueError is a RangeError here, its TypeError a TypeError).

// ---- inputs -------------------------------------------------------------------------------------------------------

interface In {
  readonly shape: number[];
  readonly seed: number;
  readonly imag?: boolean;
}
const prod = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1);
function nest(values: readonly number[], shape: readonly number[]): unknown {
  if (shape.length === 1) return values.slice();
  const inner = prod(shape.slice(1));
  return Array.from({ length: shape[0] as number }, (_, i) => nest(values.slice(i * inner, (i + 1) * inner), shape.slice(1)));
}
const tensorOf = (values: number[] | Float64Array, shape: readonly number[]) => Tensor.fromTypedArray(Float64Array.from(values), [...shape], { dtype: "f64" });
const imagOf = (n: number, seed: number) => signalOf(n, seed + 1000).map((v) => 0.7 * v - 1);

/** A fresh input (a new object every call) and the JSON fields that give the oracle the same data. */
function build(i: In, key: string): { json: Kw; tensor: SignalInput } {
  const n = prod(i.shape);
  if (n === 0) return { json: { [`${key}_shape`]: i.shape }, tensor: Tensor.fromTypedArray(new Float64Array(0), [...i.shape], { dtype: "f64" }) };
  const re = signalOf(n, i.seed);
  if (!i.imag) return { json: { [key]: nest(re, i.shape) }, tensor: tensorOf(re, i.shape) };
  const im = imagOf(n, i.seed);
  return { json: { [key]: nest(re, i.shape), [`${key}_imag`]: nest(im, i.shape) }, tensor: ComplexTensor.fromParts(tensorOf(re, i.shape), tensorOf(im, i.shape)) };
}

type FnName = "welch" | "csd" | "periodogram" | "coherence" | "spectrogram" | "stft";
interface Scn {
  readonly fn: FnName;
  readonly x: In;
  /** The second input of `csd` / `coherence`: another input, or `"same"` for the very same object. */
  readonly y?: In | "same";
  readonly opts?: Kw;
  /** Extra SciPy keyword arguments (a callable detrend, say). */
  readonly py?: Kw;
}
const OPS: Record<FnName, string> = { welch: "welch_kw", csd: "csd_kw", periodogram: "periodogram_kw", coherence: "coherence_kw", spectrogram: "spectrogram_kw", stft: "stft_kw" };
const shapeOfIn = (i: In) => JSON.stringify(i.shape) + (i.imag ? "c" : "");
const name = (s: Scn) => `${s.fn} x${shapeOfIn(s.x)}${s.y === undefined ? "" : s.y === "same" ? " y=x" : ` y${shapeOfIn(s.y)}`} ${JSON.stringify(s.opts ?? {}, (_, v) => (v instanceof Float64Array ? `Float64Array(${v.length})` : typeof v === "function" ? "fn" : v))}`;

interface Want {
  freqs: unknown;
  times?: number[];
  shape: number[];
  real?: unknown;
  imag?: unknown;
  Pxx?: unknown;
  Cxy?: unknown;
  complex?: boolean;
  warnings: string[];
  error?: string;
  error_type?: string;
}

/** Run our function on a scenario. `inputs` are the objects built for it, so `y === x` can be passed through. */
function execute(s: Scn, x: SignalInput, y: SignalInput | undefined, onWarning: (m: string) => void): { freqs: number[]; times?: number[]; out: Tensor | ComplexTensor } {
  const o = { ...(s.opts ?? {}), onWarning } as never;
  switch (s.fn) {
    case "welch": {
      const r = welch(x, o);
      return { freqs: r.freqs, out: r.Pxx };
    }
    case "periodogram": {
      const r = periodogram(x, o);
      return { freqs: r.freqs, out: r.Pxx };
    }
    case "csd": {
      const r = csd(x, y as SignalInput, o);
      return { freqs: r.freqs, out: r.Pxy };
    }
    case "coherence": {
      const r = coherence(x, y as SignalInput, o);
      return { freqs: r.freqs, out: r.Cxy };
    }
    case "spectrogram": {
      const r = spectrogram(x, o);
      return { freqs: r.freqs, times: r.times, out: r.Sxx };
    }
    case "stft": {
      const r = stft(x, o);
      return { freqs: r.freqs, times: r.times, out: r.Zxx };
    }
  }
}

function compare(what: string, s: Scn, got: { freqs: number[]; times?: number[]; out: Tensor | ComplexTensor }, want: Want): void {
  const wf = flatten(want.freqs);
  closeScaled(got.freqs, wf, Math.max(maxAbs(wf), 1e-300), `freqs ${what}`);
  if (got.times !== undefined) {
    const wt = flatten(want.times); // (an empty result's times is an empty array of the input's shape)
    closeScaled(got.times, wt, Math.max(maxAbs(wt), 1e-300), `times ${what}`);
  }
  const complexOut = got.out instanceof ComplexTensor;
  const shape = (got.out instanceof ComplexTensor ? got.out.real : got.out).shape;
  assert.deepEqual([...shape], want.shape, `shape ${what}`);
  const reOf = (t: Tensor) => flat(t);
  const wantRe = flatten(want.Pxx ?? want.Cxy ?? want.real);
  const wantIm = want.imag === undefined ? [] : flatten(want.imag);
  const scale = Math.max(maxAbs(wantRe), maxAbs(wantIm), 1e-18); // below that both sides are rounding noise
  // (SciPy returns a real empty array for an empty complex result; here it is an empty ComplexTensor, so the type holds)
  const wantComplex = s.fn === "stft" || (s.fn === "spectrogram" && s.opts?.mode === "complex") || want.complex === true;
  assert.equal(complexOut, wantComplex, `complex result ${what}`);
  const mode = s.fn === "spectrogram" ? s.opts?.mode : undefined;
  if (mode === "angle" || mode === "phase") {
    // The angle of a bin that is exactly real and negative (DC, Nyquist) is +pi or -pi depending on the sign of a zero,
    // which differs even between SciPy's own one- and two-sided paths. So compare angles modulo 2*pi, and a
    // phase unwrapped along the frequency axis up to its value at the first frequency.
    const g = reOf(got.out as Tensor);
    const fAxis = Math.min(Math.max((s.opts?.axis as number | undefined) ?? -1, -shape.length), shape.length - 1);
    const fa = fAxis < 0 ? fAxis + shape.length - 1 : fAxis; // the time axis is the last one
    let inner = 1;
    for (let d = fa + 1; d < shape.length; d++) inner *= shape[d] as number;
    const len = shape[fa] as number;
    const twoPi = 2 * Math.PI;
    for (let i = 0; i < g.length; i++) {
      const base = i - (Math.floor(i / inner) % len) * inner; // the same line at frequency 0
      const diff = mode === "angle" ? (g[i] as number) - (wantRe[i] as number) : (g[i] as number) - (g[base] as number) - ((wantRe[i] as number) - (wantRe[base] as number));
      const wrapped = mode === "angle" ? diff - twoPi * Math.round(diff / twoPi) : diff;
      assert.ok(Math.abs(wrapped) < 1e-9, `${mode} ${what} at ${i}: ${g[i]} vs ${wantRe[i]}`);
    }
    return;
  }
  if (got.out instanceof ComplexTensor) {
    closeScaled(reOf(got.out.real), wantRe, scale, `real ${what}`);
    closeScaled(reOf(got.out.imag), wantIm, scale, `imag ${what}`);
  } else {
    closeScaled(reOf(got.out as Tensor), wantRe, scale, `values ${what}`);
  }
}

/** What SciPy >= 1.17's get_window says about the two things 1.16 accepts and this package follows. */
const KNOWN_NEWER_ERRORS = /Parameter Nx=0 is not a positive integer|Invalid window name '[\s+\-.\d]*'/;

/** All scenarios against SciPy in one oracle call. */
function check(scenarios: readonly Scn[]): void {
  const built = scenarios.map((s) => {
    const x = build(s.x, "x");
    const y = s.y === undefined || s.y === "same" ? undefined : build(s.y, "y");
    return { x, y };
  });
  const jobs = scenarios.map((s, i) => ({
    op: OPS[s.fn],
    ...(built[i] as { x: { json: Kw } }).x.json,
    ...((built[i] as { y?: { json: Kw } }).y?.json ?? {}),
    ...(s.y === "same" ? { same: true } : {}),
    kwargs: { ...pyKw(s.opts ?? {}, s.fn === "stft"), ...(s.py ?? {}) },
    expect_error: true,
  }));
  const wants = runScipyOracle<Want[]>({ op: "batch", jobs });
  scenarios.forEach((s, i) => {
    const b = built[i] as { x: { tensor: SignalInput }; y?: { tensor: SignalInput } };
    const want = wants[i] as Want;
    const what = name(s);
    const yTensor = s.y === "same" ? b.x.tensor : b.y?.tensor;
    const warned: string[] = [];
    if (want.error !== undefined) {
      if (SCIPY_17_PLUS && KNOWN_NEWER_ERRORS.test(want.error)) {
        // SciPy >= 1.17's get_window refuses a length of 0 and numeric strings, which 1.16 (this package) accepts (see helpers.ts)
        try {
          execute(s, b.x.tensor, yTensor, (m) => warned.push(m));
        } catch {
          // (an error is fine too)
        }
        return;
      }
      const cls = /TypeError$/.test(want.error_type ?? "") ? TypeError : RangeError;
      // (SciPy >= 1.17 raises ValueError where 1.16 raises TypeError for a wrong window parameter count)
      if (SCIPY_17_PLUS) assert.throws(() => execute(s, b.x.tensor, yTensor, (m) => warned.push(m)), (e: unknown) => e instanceof RangeError || e instanceof TypeError, `${what}: SciPy raised ${want.error}`);
      else assert.throws(() => execute(s, b.x.tensor, yTensor, (m) => warned.push(m)), cls, `${what}: SciPy raised ${want.error}`);
      return;
    }
    const got = execute(s, b.x.tensor, yTensor, (m) => warned.push(m));
    compare(what, s, got, want);
    assert.deepEqual(warned, want.warnings, `warnings ${what}`);
  });
}

const R = (shape: number[], seed = 1): In => ({ shape, seed });
const C = (shape: number[], seed = 1): In => ({ shape, seed, imag: true });

// ---- option sets ------------------------------------------------------------------------------------------------

const hannArray = (n: number) => Float64Array.from({ length: n }, (_, i) => 0.3 + Math.sin((Math.PI * (i + 0.5)) / n) ** 2);
const SEGMENT_OPTS: Kw[] = [
  {},
  { nperseg: 64 },
  { nperseg: 64, noverlap: 40, nfft: 100 },
  { fs: 100, window: "hamming", nperseg: 50 },
  { window: ["kaiser", 6], nperseg: 40, noverlap: 10, detrend: "linear" },
  { onesided: false, nperseg: 30, nfft: 45, detrend: false },
  { scaling: "spectrum", nperseg: 31, window: "blackman" },
  { nperseg: 48, window: hannArray(48), noverlap: 47 },
  { window: ["chebwin", 60], nperseg: 33, nfft: 64 },
  { window: ["dpss", 2.5], nperseg: 40 },
  { window: "lanczos", nperseg: 35, scaling: "spectrum" },
  { window: "tuk", nperseg: 100, noverlap: 99 },
];
const first3 = ((seg) => {
  const part = (t: Tensor) => {
    const v = Float64Array.from(t.contiguous().data as Float64Array);
    const m = (v[0]! + v[1]! + v[2]!) / 3;
    return Tensor.fromTypedArray(v.map((a) => a - m), [v.length], { dtype: "f64" });
  };
  return "imag" in seg ? ComplexTensor.fromParts(part(seg.real), part(seg.imag)) : part(seg);
}) as DetrendFunction;

// ---- periodogram ----------------------------------------------------------------------------------------------

test("periodogram matches scipy.signal.periodogram: defaults, every option, N-D along each axis, complex, nfft shorter and longer than the signal", { skip: SCIPY_SKIP_REASON }, () => {
  const n = 100;
  const opts: Kw[] = [
    {},
    { fs: 8 },
    { window: "hann" },
    { window: ["kaiser", 5], detrend: "linear" },
    { window: hannArray(n) },
    { nfft: 64 },
    { nfft: 256, window: "hamming" },
    { nfft: n },
    { nfft: 129, onesided: false },
    { detrend: false },
    { detrend: "linear", scaling: "spectrum" },
    { onesided: false, fs: 3 },
    { scaling: "spectrum", window: ["chebwin", 70] },
    { window: ["dpss", 3], nfft: 150 },
    { window: "lanczos", nfft: 40 },
  ];
  const scenarios: Scn[] = [];
  for (const o of opts) scenarios.push({ fn: "periodogram", x: R([n]), opts: o }, { fn: "periodogram", x: R([n + 1], 2), opts: { ...o, window: o.window instanceof Float64Array ? hannArray(n + 1) : o.window } });
  scenarios.push({ fn: "periodogram", x: R([100]), opts: { detrend: first3 }, py: { detrend: { fn: "first3" } } });
  for (const len of [1, 2, 3, 7]) scenarios.push({ fn: "periodogram", x: R([len]), opts: {} }, { fn: "periodogram", x: R([len]), opts: { nfft: 16 } });
  // N-D: the axis is replaced by the frequencies
  for (const [shape, axis] of [[[3, 60], -1], [[3, 60], 1], [[60, 3], 0], [[60, 3], -2], [[2, 3, 40], 1], [[2, 3, 40], -2], [[2, 3, 40], 0], [[2, 3, 40], 2]] as const) {
    scenarios.push({ fn: "periodogram", x: R([...shape]), opts: { axis } });
    scenarios.push({ fn: "periodogram", x: R([...shape]), opts: { axis, nfft: 24, scaling: "spectrum", detrend: "linear" } });
    scenarios.push({ fn: "periodogram", x: R([...shape]), opts: { axis, nfft: 90, window: "hamming" } });
  }
  // complex input: two-sided, silently
  for (const [shape, axis] of [[[50], -1], [[3, 50], -1], [[50, 3], 0]] as const) {
    scenarios.push({ fn: "periodogram", x: C([...shape]), opts: { axis } }, { fn: "periodogram", x: C([...shape]), opts: { axis, nfft: 70, scaling: "spectrum", window: "hann" } });
  }
  // errors
  scenarios.push(
    { fn: "periodogram", x: R([50]), opts: { window: hannArray(40) } },
    { fn: "periodogram", x: R([50]), opts: { window: hannArray(50), nfft: 20 } },
    { fn: "periodogram", x: R([50]), opts: { scaling: "power" } },
    { fn: "periodogram", x: R([50]), opts: { detrend: "quadratic" } },
    { fn: "periodogram", x: R([50]), opts: { window: "hanning" } },
    { fn: "periodogram", x: R([50]), opts: { window: "kaiser" } },
    { fn: "periodogram", x: R([3, 50]), opts: { axis: 2 } },
    { fn: "periodogram", x: R([3, 50]), opts: { axis: -3 } },
  );
  check(scenarios);
});

test("periodogram: one segment, so it equals welch with the whole signal and no overlap; names and defaults", () => {
  const x = tensorOf(signalOf(80, 3), [80]);
  const p = periodogram(x);
  const w = welch(x, { window: "boxcar", nperseg: 80, noverlap: 0 });
  assert.deepEqual(Object.keys(p).sort(), ["Pxx", "freqs"]);
  assert.deepEqual(flat(p.Pxx), flat(w.Pxx));
  assert.equal(p.freqs.length, 41);
  assert.deepEqual([...periodogram(x, { onesided: false }).Pxx.shape], [80]);
});

// ---- csd --------------------------------------------------------------------------------------------------------

test("csd matches scipy.signal.csd: every welch option, same object vs equal data, complex x and/or y, average median", { skip: SCIPY_SKIP_REASON }, () => {
  const scenarios: Scn[] = [];
  for (const o of SEGMENT_OPTS) {
    scenarios.push({ fn: "csd", x: R([300], 1), y: R([300], 2), opts: o });
    scenarios.push({ fn: "csd", x: R([300], 1), y: "same", opts: o }); // y is x: real
    scenarios.push({ fn: "csd", x: R([300], 1), y: R([300], 1), opts: o }); // equal data, other object: complex
    scenarios.push({ fn: "csd", x: R([301], 3), y: R([301], 4), opts: { ...o, average: "median", scaling: "spectrum", ...(o.window instanceof Float64Array ? {} : {}) } });
  }
  scenarios.push({ fn: "csd", x: R([300], 1), y: R([300], 2), opts: { detrend: first3, nperseg: 32 }, py: { detrend: { fn: "first3" } } });
  scenarios.push({ fn: "csd", x: R([300], 1), y: R([300], 2), opts: { average: "median", nperseg: 50, noverlap: 10 } });
  scenarios.push({ fn: "csd", x: R([300], 1), y: R([300], 2), opts: { average: "mean", scaling: "density", nperseg: 300 } }); // one segment
  // complex: x complex (y real or complex) is two-sided; a complex y with a real x needs onesided: false
  scenarios.push(
    { fn: "csd", x: C([200], 1), y: R([200], 2), opts: { nperseg: 40 } },
    { fn: "csd", x: C([200], 1), y: C([200], 2), opts: { nperseg: 40 } },
    { fn: "csd", x: C([200], 1), y: "same", opts: { nperseg: 40 } },
    { fn: "csd", x: R([200], 1), y: C([200], 2), opts: { nperseg: 40 } }, // SciPy raises
    { fn: "csd", x: R([200], 1), y: C([200], 2), opts: { nperseg: 40, onesided: false } },
    { fn: "csd", x: C([200], 1), y: C([200], 2), opts: { nperseg: 40, onesided: false, nfft: 63, detrend: "linear", average: "median" } },
  );
  check(scenarios);
});

test("csd: x and y of different lengths (the shorter is zero-padded), and the broadcasting of their other dimensions, as SciPy", { skip: SCIPY_SKIP_REASON }, () => {
  const o = { nperseg: 32 };
  const scenarios: Scn[] = [
    // lengths
    { fn: "csd", x: R([300], 1), y: R([250], 2), opts: o },
    { fn: "csd", x: R([100], 1), y: R([130], 2), opts: o },
    { fn: "csd", x: R([100], 1), y: R([130], 2) }, // default nperseg: clamped to the longer, with a warning
    { fn: "csd", x: R([100], 1), y: R([40], 2), opts: { nperseg: 64, scaling: "spectrum", average: "median" } },
    { fn: "csd", x: R([100], 1), y: R([1], 2), opts: o },
    { fn: "csd", x: C([100], 1), y: R([70], 2), opts: o },
    // outer dimensions broadcast
    { fn: "csd", x: R([3, 100], 1), y: R([100], 2), opts: o },
    { fn: "csd", x: R([100], 1), y: R([3, 100], 2), opts: o },
    { fn: "csd", x: R([3, 100], 1), y: R([1, 100], 2), opts: o },
    { fn: "csd", x: R([1, 100], 1), y: R([3, 100], 2), opts: o },
    { fn: "csd", x: R([3, 100], 1), y: R([3, 100], 2), opts: o },
    { fn: "csd", x: R([2, 3, 100], 1), y: R([3, 100], 2), opts: o },
    { fn: "csd", x: R([2, 1, 100], 1), y: R([1, 3, 100], 2), opts: o },
    { fn: "csd", x: R([2, 3, 100], 1), y: R([100], 2), opts: o },
    { fn: "csd", x: R([3, 100], 1), y: "same", opts: o },
    { fn: "csd", x: C([3, 100], 1), y: R([100], 2), opts: { ...o, scaling: "spectrum" } },
    { fn: "csd", x: R([3, 100], 1), y: R([3, 100], 2), opts: { ...o, average: "median", onesided: false } },
    // the axis, with equal ranks and with broadcasting
    { fn: "csd", x: R([100, 3], 1), y: R([100, 3], 2), opts: { ...o, axis: 0 } },
    { fn: "csd", x: R([100, 3], 1), y: R([100, 1], 2), opts: { ...o, axis: 0 } },
    { fn: "csd", x: R([100, 1], 1), y: R([100, 3], 2), opts: { ...o, axis: -2 } },
    { fn: "csd", x: R([2, 100, 3], 1), y: R([2, 100, 3], 2), opts: { ...o, axis: 1 } },
    { fn: "csd", x: R([2, 100, 3], 1), y: R([2, 100, 1], 2), opts: { ...o, axis: -2 } },
    { fn: "csd", x: R([2, 100, 3], 1), y: R([2, 70, 3], 2), opts: { ...o, axis: 1 } }, // lengths differ, other dims equal
    { fn: "csd", x: R([100, 3], 1), y: "same", opts: { ...o, axis: 0 } },
    // SciPy multiplies the segment spectra with the frequency axis still in place: a 1-D y against an N-D x along a
    // positive axis does not line up (an error, or, when the frequency count happens to equal the other dimension, a
    // broadcast along the wrong axis), and so it does here
    { fn: "csd", x: R([100, 3], 1), y: R([100], 2), opts: { ...o, axis: 0 } },
    { fn: "csd", x: R([100, 3], 1), y: R([100], 2), opts: { nperseg: 4, axis: 0 } },
    { fn: "csd", x: R([100], 1), y: R([100, 3], 2), opts: { nperseg: 4, axis: 0 } },
    { fn: "csd", x: R([100, 3], 1), y: R([100], 2), opts: { nperseg: 4, axis: 0, onesided: false } },
    // errors
    { fn: "csd", x: R([3, 100], 1), y: R([2, 100], 2), opts: o }, // outer dimensions do not broadcast
    { fn: "csd", x: R([3, 100], 1), y: R([100, 3], 2), opts: o },
    { fn: "csd", x: R([3, 100], 1), y: R([60], 2), opts: o }, // lengths differ and so do the other dimensions
    { fn: "csd", x: R([1, 100], 1), y: R([3, 60], 2), opts: o },
    { fn: "csd", x: R([100], 1), y: R([2, 3, 100], 2), opts: { ...o, axis: 1 } },
    { fn: "csd", x: R([100], 1), y: R([100], 2), opts: { axis: 1 } },
    { fn: "csd", x: R([100], 1), y: R([100], 2), opts: { nperseg: 32, scaling: "psd" } },
    { fn: "csd", x: R([100], 1), y: R([100], 2), opts: { nperseg: 32, average: "max" } },
    { fn: "csd", x: R([100], 1), y: R([100], 2), opts: { nperseg: 32, nfft: 31 } },
    { fn: "csd", x: R([100], 1), y: R([100], 2), opts: { nperseg: 32, noverlap: 32 } },
    { fn: "csd", x: R([100], 1), y: R([100], 2), opts: { nperseg: 0 } },
    { fn: "csd", x: R([100], 1), y: R([100], 2), opts: { nperseg: 20, window: hannArray(30) } },
    { fn: "csd", x: R([20], 1), y: R([20], 2), opts: { window: hannArray(30) } },
    { fn: "csd", x: R([100], 1), y: R([100], 2), opts: { fs: 0 } },
    { fn: "csd", x: R([100], 1), y: R([100], 2), opts: { fs: -1 } },
  ];
  check(scenarios);
});

test("csd: Pxy is conj(X) * Y, a ComplexTensor, real only when y is x; csd(x, x).Pxy equals welch(x).Pxx", () => {
  const re = signalOf(128, 1);
  const x = tensorOf(re, [128]);
  const y = tensorOf(re.map((v, i) => v * 0.5 + Math.sin(i)), [128]);
  const same = csd(x, x, { nperseg: 32 });
  assert.ok(same.Pxy instanceof Tensor && !(same.Pxy instanceof ComplexTensor));
  assert.deepEqual(flat(same.Pxy as Tensor), flat(welch(x, { nperseg: 32 }).Pxx));
  const cross = csd(x, y, { nperseg: 32 });
  assert.ok(cross.Pxy instanceof ComplexTensor);
  assert.deepEqual(Object.keys(cross).sort(), ["Pxy", "freqs"]);
  // csd(y, x) is the complex conjugate of csd(x, y)
  const back = csd(y, x, { nperseg: 32 });
  const a = cross.Pxy as ComplexTensor;
  const b = back.Pxy as ComplexTensor;
  closeScaled(flat(a.real), flat(b.real), maxAbs(flat(a.real)), "conj real");
  closeScaled(flat(a.imag), flat(b.imag).map((v) => -v), maxAbs(flat(a.imag)), "conj imag");
});

test("csd: a shared signal gives a real, non-negative auto-spectrum and the cross-spectrum of unrelated noise is small", () => {
  const x = tensorOf(signalOf(512, 5), [512]);
  const copy = tensorOf(signalOf(512, 5), [512]);
  const c = csd(x, copy, { nperseg: 64 });
  const re = flat((c.Pxy as ComplexTensor).real);
  const im = flat((c.Pxy as ComplexTensor).imag);
  assert.ok(maxAbs(im) < 1e-12 * maxAbs(re), "a copy has no phase difference");
  assert.ok(re.every((v) => v >= 0));
});

// ---- coherence ------------------------------------------------------------------------------------------------

test("coherence matches scipy.signal.coherence: defaults, options, N-D with broadcasting, complex, equal data, and the errors", { skip: SCIPY_SKIP_REASON }, () => {
  const scenarios: Scn[] = [];
  const opts: Kw[] = [
    {},
    { nperseg: 64 },
    { nperseg: 64, noverlap: 40, nfft: 100 },
    { fs: 100, window: "hamming", nperseg: 50 },
    { window: ["kaiser", 6], nperseg: 40, noverlap: 10, detrend: "linear" },
    { nperseg: 31, detrend: false, window: ["chebwin", 65] },
    { nperseg: 48, window: hannArray(48), noverlap: 47 },
    { window: ["dpss", 2.5], nperseg: 40 },
    { window: "sinc", nperseg: 35 },
  ];
  for (const o of opts) {
    scenarios.push({ fn: "coherence", x: R([300], 1), y: R([300], 2), opts: o });
    scenarios.push({ fn: "coherence", x: R([300], 1), y: "same", opts: o });
  }
  scenarios.push({ fn: "coherence", x: R([300], 1), y: R([300], 2), opts: { detrend: first3, nperseg: 32 }, py: { detrend: { fn: "first3" } } });
  scenarios.push(
    { fn: "coherence", x: C([200], 1), y: R([200], 2), opts: { nperseg: 40 } },
    { fn: "coherence", x: C([200], 1), y: C([200], 2), opts: { nperseg: 40 } },
    { fn: "coherence", x: R([200], 1), y: C([200], 2), opts: { nperseg: 40 } }, // SciPy raises: real x, complex y, one-sided
    // N-D and broadcasting
    { fn: "coherence", x: R([3, 200], 1), y: R([3, 200], 2), opts: { nperseg: 40 } },
    { fn: "coherence", x: R([3, 200], 1), y: R([200], 2), opts: { nperseg: 40 } },
    { fn: "coherence", x: R([200], 1), y: R([3, 200], 2), opts: { nperseg: 40 } },
    { fn: "coherence", x: R([2, 3, 200], 1), y: R([1, 3, 200], 2), opts: { nperseg: 40 } },
    { fn: "coherence", x: R([200, 3], 1), y: R([200, 3], 2), opts: { nperseg: 40, axis: 0 } },
    { fn: "coherence", x: R([2, 200, 3], 1), y: R([2, 200, 3], 2), opts: { nperseg: 40, axis: 1 } },
    { fn: "coherence", x: R([200, 3], 1), y: R([200, 1], 2), opts: { nperseg: 40, axis: 0 } },
    // different lengths: the spectra differ in size and cannot be divided (SciPy's error); with nperseg <= both they can
    { fn: "coherence", x: R([200], 1), y: R([150], 2) },
    { fn: "coherence", x: R([200], 1), y: R([150], 2), opts: { nperseg: 32 } },
    { fn: "coherence", x: R([200], 1), y: R([150], 2), opts: { nperseg: 32, nfft: 64 } },
    { fn: "coherence", x: R([3, 200], 1), y: R([2, 200], 2), opts: { nperseg: 32 } },
    // short signals: nperseg > n warns in each of the three estimates
    { fn: "coherence", x: R([50], 1), y: R([50], 2) },
    { fn: "coherence", x: R([100], 1), y: R([100], 2), opts: { nperseg: 128 } },
  );
  check(scenarios);
});

test("coherence is bounded by [0, 1], is 1 for a signal against itself, and is lower for noisy copies", () => {
  const x = signalOf(512, 1);
  const y = signalOf(512, 9);
  const self = coherence(tensorOf(x, [512]), tensorOf(x, [512]), { nperseg: 64 });
  for (const v of flat(self.Cxy)) assert.ok(Math.abs(v - 1) < 1e-9, String(v));
  const mixed = coherence(tensorOf(x, [512]), tensorOf(x.map((v, i) => v + 3 * (y[i] as number)), [512]), { nperseg: 64 });
  const cxy = flat(mixed.Cxy);
  assert.ok(cxy.every((v) => v >= 0 && v <= 1 + 1e-12));
  assert.ok(cxy.some((v) => v < 0.95));
  assert.deepEqual(Object.keys(mixed).sort(), ["Cxy", "freqs"]);
});

// ---- spectrogram ----------------------------------------------------------------------------------------------

const MODES = ["psd", "complex", "magnitude", "angle", "phase"] as const;
const SPECTROGRAM_OPTS: Kw[] = [
  {},
  { nperseg: 64 },
  { nperseg: 64, noverlap: 32, nfft: 100 },
  { fs: 100, window: "hann", nperseg: 50 },
  { window: ["kaiser", 5], nperseg: 40, noverlap: 10, detrend: "linear" },
  { onesided: false, nperseg: 30, nfft: 45 },
  { scaling: "spectrum", nperseg: 31 },
  { detrend: false, nperseg: 20, noverlap: 19 },
  { nperseg: 48, window: hannArray(48), noverlap: 12 },
  { window: ["dpss", 2], nperseg: 33, nfft: 64 },
  { window: ["chebwin", 80], nperseg: 32 },
  { window: "lanczos", nperseg: 25, scaling: "spectrum", fs: 7 },
  { nperseg: 17, noverlap: -3 },
];

test("spectrogram matches scipy.signal.spectrogram in every mode: defaults, every option, windows", { skip: SCIPY_SKIP_REASON }, () => {
  const scenarios: Scn[] = [];
  for (const mode of MODES) {
    for (const o of SPECTROGRAM_OPTS) scenarios.push({ fn: "spectrogram", x: R([400], 1), opts: { ...o, mode } });
    scenarios.push({ fn: "spectrogram", x: R([401], 2), opts: { mode } }, { fn: "spectrogram", x: R([100], 3), opts: { mode } });
  }
  scenarios.push({ fn: "spectrogram", x: R([300], 1), opts: { nperseg: 40, detrend: first3 }, py: { detrend: { fn: "first3" } } });
  check(scenarios);
});

test("spectrogram: N-D along every axis and complex input in every mode (phase is unwrapped along the frequency axis)", { skip: SCIPY_SKIP_REASON }, () => {
  const scenarios: Scn[] = [];
  for (const mode of MODES) {
    for (const [shape, axis] of [[[3, 120], -1], [[120, 3], 0], [[120, 3], -2], [[2, 120, 3], 1], [[2, 120, 3], -2], [[2, 3, 120], 2], [[2, 3, 120], 0]] as const) {
      scenarios.push({ fn: "spectrogram", x: R([...shape]), opts: { axis, nperseg: 30, mode } });
      scenarios.push({ fn: "spectrogram", x: R([...shape]), opts: { axis, nperseg: 24, noverlap: 6, nfft: 40, detrend: "linear", scaling: "spectrum", mode } });
    }
    for (const [shape, axis] of [[[200], -1], [[3, 100], -1], [[100, 3], 0]] as const) {
      scenarios.push({ fn: "spectrogram", x: C([...shape]), opts: { axis, nperseg: 25, mode } });
      scenarios.push({ fn: "spectrogram", x: C([...shape]), opts: { axis, nperseg: 25, onesided: false, nfft: 31, mode } });
    }
  }
  check(scenarios);
});

test("spectrogram: short signals clamp nperseg with SciPy's warning; errors match SciPy's classes; the mode is validated first", { skip: SCIPY_SKIP_REASON }, () => {
  const scenarios: Scn[] = [];
  for (const mode of MODES) scenarios.push({ fn: "spectrogram", x: R([5], 1), opts: { mode } }, { fn: "spectrogram", x: R([1], 1), opts: { mode } }, { fn: "spectrogram", x: R([40], 1), opts: { nperseg: 100, mode } });
  scenarios.push(
    { fn: "spectrogram", x: R([100], 1), opts: { mode: "power" } },
    { fn: "spectrogram", x: R([100], 1), opts: { window: hannArray(120) } }, // window longer than the input
    { fn: "spectrogram", x: R([100], 1), opts: { window: hannArray(40), nperseg: 50 } },
    { fn: "spectrogram", x: R([100], 1), opts: { nperseg: 40, nfft: 30 } },
    { fn: "spectrogram", x: R([100], 1), opts: { nperseg: 40, noverlap: 40 } },
    { fn: "spectrogram", x: R([100], 1), opts: { nperseg: 0 } },
    { fn: "spectrogram", x: R([100], 1), opts: { nperseg: 40, scaling: "psd" } },
    { fn: "spectrogram", x: R([100], 1), opts: { nperseg: 40, scaling: "none" } },
    { fn: "spectrogram", x: R([100], 1), opts: { nperseg: 40, detrend: "cubic" } },
    { fn: "spectrogram", x: R([100], 1), opts: { window: "hanning" } },
    { fn: "spectrogram", x: R([100], 1), opts: { window: ["kaiser"] } },
    { fn: "spectrogram", x: R([100], 1), opts: { axis: 3 } },
    { fn: "spectrogram", x: R([3, 100], 1), opts: { axis: -3 } },
  );
  check(scenarios);
});

test("spectrogram: result names, shapes and the default Tukey window", () => {
  const x = tensorOf(signalOf(256, 3), [256]);
  const s = spectrogram(x);
  assert.deepEqual(Object.keys(s).sort(), ["Sxx", "freqs", "times"]);
  assert.equal(s.Sxx instanceof Tensor, true);
  assert.deepEqual([...s.Sxx.shape], [s.freqs.length, s.times.length]);
  assert.equal(s.freqs.length, 129);
  assert.equal(s.times.length, 1); // 256 samples, nperseg 256: one segment
  const small = spectrogram(x, { nperseg: 64 });
  assert.equal(small.times.length, Math.floor((256 - 64) / (64 - 8)) + 1);
  const z = spectrogram(x, { nperseg: 64, mode: "complex" });
  assert.ok(z.Sxx instanceof ComplexTensor);
  assert.deepEqual([...z.Sxx.real.shape], [33, small.times.length]);
  // the default window is ["tukey", 0.25]
  const explicit = spectrogram(x, { nperseg: 64, window: ["tukey", 0.25], noverlap: 8 });
  assert.deepEqual(flat(explicit.Sxx), flat(small.Sxx));
  assert.deepEqual(explicit.times, small.times);
});

// ---- empty input ------------------------------------------------------------------------------------------------

test("empty input: every spectral function returns SciPy's empty arrays (or raises where SciPy does), whatever the shape", { skip: SCIPY_SKIP_REASON }, () => {
  const scenarios: Scn[] = [];
  const empties: Array<[number[], number]> = [[[0], -1], [[3, 0], -1], [[3, 0], 1], [[0, 5], -1], [[0, 5], 0], [[2, 0, 4], 1], [[2, 3, 0], -1], [[0, 0], 0]];
  for (const [shape, axis] of empties) {
    const e = R(shape);
    scenarios.push(
      { fn: "welch", x: e, opts: { axis } },
      { fn: "welch", x: e, opts: { axis, nperseg: 16, window: "kaiser", nfft: 3, scaling: "bogus" } }, // SciPy returns before looking at any of it
      { fn: "periodogram", x: e, opts: { axis } },
      { fn: "periodogram", x: e, opts: { axis, nfft: 5, window: hannArray(7) } },
      { fn: "csd", x: e, y: "same", opts: { axis } },
      { fn: "csd", x: e, y: R(shape, 2), opts: { axis } },
      { fn: "csd", x: e, y: R(shape.map((d) => d), 2), opts: { axis, nperseg: 4, window: "kaiser" } },
      { fn: "coherence", x: e, y: R(shape, 2), opts: { axis } },
      { fn: "coherence", x: e, y: "same", opts: { axis } },
      { fn: "spectrogram", x: e, opts: { axis } },
      { fn: "spectrogram", x: e, opts: { axis, mode: "complex" } },
      { fn: "spectrogram", x: e, opts: { axis, mode: "phase", nperseg: 8 } },
      { fn: "stft", x: e, opts: { axis } },
      { fn: "stft", x: e, opts: { axis, nperseg: 8, window: "kaiser", noverlap: 50 } }, // SciPy returns before the window and nperseg checks
    );
  }
  // csd / coherence with one side empty
  scenarios.push(
    { fn: "csd", x: R([0]), y: R([50], 2) },
    { fn: "csd", x: R([50]), y: R([0], 2) },
    { fn: "csd", x: R([3, 0]), y: R([3, 50], 2) },
    { fn: "csd", x: R([3, 0]), y: R([1, 50], 2) },
    { fn: "csd", x: R([3, 50]), y: R([0, 50], 2) }, // the empty one has 0 rows: outer dimensions 3 and 0 do not broadcast
    { fn: "csd", x: R([0, 3]), y: R([50, 3], 2), opts: { axis: 0 } },
    { fn: "csd", x: R([0, 3]), y: R([50, 1], 2), opts: { axis: 0 } },
    { fn: "csd", x: R([3, 0]), y: R([2, 0], 2) },
    { fn: "csd", x: R([0]), y: R([2, 50], 2) },
    { fn: "coherence", x: R([0]), y: R([50], 2) },
    { fn: "coherence", x: R([50]), y: R([0], 2) },
    { fn: "coherence", x: R([3, 0]), y: R([3, 50], 2) },
    { fn: "coherence", x: R([3, 0]), y: R([3, 0], 2), opts: { nperseg: 8 } },
    // spectrogram sets the window up before it notices the input is empty
    { fn: "spectrogram", x: R([0]), opts: { nperseg: 4 } },
    { fn: "spectrogram", x: R([0]), opts: { window: hannArray(8) } }, // window longer than the (empty) input
    { fn: "spectrogram", x: R([0]), opts: { window: hannArray(8), nperseg: 4 } },
    { fn: "spectrogram", x: R([0]), opts: { window: "kaiser" } },
    { fn: "spectrogram", x: R([0]), opts: { mode: "bogus" } },
    { fn: "spectrogram", x: R([2, 0]), opts: { nperseg: 0 } },
    // stft still checks scaling and boundary first
    { fn: "stft", x: R([0]), opts: { scaling: "bogus" } },
    { fn: "stft", x: R([0]), opts: { boundary: "wrap" } },
    { fn: "stft", x: R([0]), opts: { scaling: "psd", boundary: null } },
    { fn: "stft", x: R([0]), opts: { scaling: "density" }, py: { scaling: "density" } }, // SciPy's stft has no "density"
  );
  check(scenarios);
});

test("empty input: shapes of what is returned (Pxx / Pxy / Cxy / Sxx / Zxx keep the input's shape; freqs and times are [])", () => {
  const e1 = Tensor.fromTypedArray(new Float64Array(0), [0], { dtype: "f64" });
  const e2 = Tensor.fromTypedArray(new Float64Array(0), [3, 0], { dtype: "f64" });
  const w = welch(e2);
  assert.deepEqual(w.freqs, []);
  assert.deepEqual([...w.Pxx.shape], [3, 0]);
  assert.deepEqual([...periodogram(e1).Pxx.shape], [0]);
  assert.deepEqual([...(csd(e2, e2).Pxy as Tensor).shape], [3, 0]);
  assert.deepEqual([...coherence(e2, e2).Cxy.shape], [3, 0]);
  const sg = spectrogram(e2, { onWarning: () => {} });
  assert.deepEqual([sg.freqs, sg.times], [[], []]);
  assert.deepEqual([...(sg.Sxx as Tensor).shape], [3, 0]);
  const st = stft(e1);
  assert.deepEqual([st.freqs, st.times], [[], []]);
  assert.deepEqual([...st.Zxx.real.shape], [0]);
  // an empty input never warns, whatever nperseg is, except spectrogram's (which looks at nperseg first, as SciPy does)
  const logged: string[] = [];
  welch(e1, { nperseg: 999, onWarning: (m) => logged.push(m) });
  stft(e1, { nperseg: 999, onWarning: (m) => logged.push(m) });
  assert.deepEqual(logged, []);
  const second: string[] = [];
  spectrogram(e1, { onWarning: (m: string) => second.push(m) });
  assert.deepEqual(second, ["nperseg = 256 is greater than input length  = 0, using nperseg = 0"]);
});

test("istft of an empty spectrogram matches scipy.signal.istft (empty x, or SciPy's error)", { skip: SCIPY_SKIP_REASON }, () => {
  const cases: Array<{ shape: number[]; opts: Kw }> = [
    { shape: [3, 0], opts: {} },
    { shape: [3, 0], opts: { boundary: false } },
    { shape: [5, 0], opts: { nperseg: 8 } },
    { shape: [5, 0], opts: { nperseg: 8, boundary: false } },
    { shape: [2, 3, 0], opts: {} },
    { shape: [0, 3, 4], opts: {} },
    { shape: [3, 3, 0], opts: { inputOnesided: false, nperseg: 3 } },
    { shape: [3, 0], opts: { window: "boxcar", noverlap: 0 } },
    { shape: [3, 0], opts: { nperseg: 1 } }, // x[0:-0] is empty in SciPy
    { shape: [0, 3], opts: {} }, // nperseg would be -2
    { shape: [1, 3], opts: {} },
    { shape: [4], opts: {} },
    { shape: [3, 0], opts: { scaling: "bogus" } },
    { shape: [3, 0], opts: { scaling: "psd" } },
  ];
  const jobs = cases.map((c) => ({ op: "istft_kw", z_shape: c.shape, kwargs: pyKw(c.opts, true), expect_error: true }));
  const wants = runScipyOracle<Array<{ shape: number[]; complex: boolean; warnings: string[]; error?: string; error_type?: string }>>({ op: "batch", jobs });
  cases.forEach((c, i) => {
    const want = wants[i] as (typeof wants)[number];
    const zeros = Tensor.fromTypedArray(new Float64Array(prod(c.shape)), c.shape, { dtype: "f64" });
    const z = ComplexTensor.fromParts(zeros, zeros);
    const warned: string[] = [];
    const run = () => istft(c.shape.length < 2 ? zeros : z, { ...c.opts, onWarning: (m: string) => warned.push(m) } as never);
    const what = JSON.stringify(c);
    if (want.error !== undefined) {
      assert.throws(run, /TypeError$/.test(want.error_type ?? "") ? TypeError : RangeError, `${what}: SciPy raised ${want.error}`);
      return;
    }
    const got = run();
    const shape = got.x instanceof ComplexTensor ? got.x.real.shape : (got.x as Tensor).shape;
    assert.deepEqual([...shape], want.shape, `x shape ${what}`);
    assert.equal(got.x instanceof ComplexTensor, want.complex, `complex ${what}`);
    // times: SciPy's has x.shape[0] entries (for N-D input that is not the time axis, which this package does not copy)
    if (c.shape.length === 2) assert.equal(got.times.length, want.shape[0] as number, `times ${what}`);
    assert.deepEqual(warned, want.warnings, `warnings ${what}`);
  });
});

// ---- warnings and the options SciPy's welch / stft / csd treat differently from before --------------------------

test("nperseg longer than the signal warns with SciPy's text (welch, csd, periodogram's inputs, coherence, spectrogram, stft) and then clamps", { skip: SCIPY_SKIP_REASON }, () => {
  const scenarios: Scn[] = [];
  for (const len of [1, 2, 10, 50, 255, 256]) {
    scenarios.push(
      { fn: "welch", x: R([len]) },
      { fn: "welch", x: R([len]), opts: { nperseg: 300 } },
      { fn: "csd", x: R([len]), y: R([len], 2) },
      { fn: "csd", x: R([len]), y: R([len + 7], 2), opts: { nperseg: 300 } },
      ...(len > 1 ? [{ fn: "coherence" as const, x: R([len]), y: R([len], 2) }] : []), // (one sample has no power: 0 / 0)
      { fn: "spectrogram", x: R([len]) },
      { fn: "stft", x: R([len]) },
      { fn: "stft", x: R([len]), opts: { nperseg: 300, boundary: null, padded: false } },
    );
  }
  scenarios.push({ fn: "welch", x: R([3, 40]), opts: { nperseg: 60 } }, { fn: "stft", x: R([40, 3]), opts: { axis: 0, nperseg: 60 } }, { fn: "welch", x: R([40]), opts: { nperseg: 40 } }, { fn: "stft", x: R([40]), opts: { nperseg: 40 } });
  check(scenarios);
});

test("welch / stft details that follow SciPy 1.16: int() truncation of sizes, negative noverlap, average only checked with several segments, window checks", { skip: SCIPY_SKIP_REASON }, () => {
  const scenarios: Scn[] = [
    // sizes are truncated like Python's int()
    { fn: "welch", x: R([200]), opts: { nperseg: 32.9, noverlap: 10.5, nfft: 40.7 } },
    { fn: "csd", x: R([200]), y: R([200], 2), opts: { nperseg: 32.9, noverlap: 10.5, nfft: 40.7 } },
    { fn: "stft", x: R([200]), opts: { nperseg: 32.9, noverlap: 10.5, nfft: 40.7 } },
    { fn: "spectrogram", x: R([200]), opts: { nperseg: 32.9, noverlap: 10.5, nfft: 40.7 } },
    { fn: "welch", x: R([200]), opts: { nperseg: 0.5 } },
    { fn: "welch", x: R([200]), opts: { nperseg: 32, nfft: 31.9 } },
    // negative overlap leaves gaps between the segments
    { fn: "welch", x: R([200]), opts: { nperseg: 20, noverlap: -5 } },
    { fn: "csd", x: R([200]), y: R([200], 2), opts: { nperseg: 20, noverlap: -5 } },
    { fn: "stft", x: R([200]), opts: { nperseg: 20, noverlap: -5 } },
    { fn: "stft", x: R([200]), opts: { nperseg: 20, noverlap: -5, boundary: null, padded: false } },
    { fn: "spectrogram", x: R([200]), opts: { nperseg: 20, noverlap: -5 } },
    // `average` is only looked at when there are several segments
    { fn: "welch", x: R([200]), opts: { average: "max" } },
    { fn: "welch", x: R([200]), opts: { nperseg: 50, average: "max" } },
    { fn: "csd", x: R([200]), y: R([200], 2), opts: { average: "max" } },
    // windows: an array longer than the signal, a window given as a bare number, a stft with an array window and no nperseg
    { fn: "welch", x: R([40]), opts: { window: hannArray(64) } },
    { fn: "welch", x: R([40]), opts: { window: hannArray(40) } },
    { fn: "welch", x: R([40]), opts: { window: hannArray(41) } },
    { fn: "welch", x: R([40]), opts: { window: 4 } },
    { fn: "welch", x: R([40]), opts: { window: "4.0", nperseg: 20 } },
    { fn: "stft", x: R([400]), opts: { window: hannArray(64) } }, // nperseg defaults to 256 whatever the window is
    { fn: "stft", x: R([400]), opts: { window: hannArray(256) } },
    { fn: "stft", x: R([400]), opts: { window: hannArray(64), nperseg: 64 } },
    { fn: "stft", x: R([40]), opts: { window: hannArray(64), nperseg: 64 } },
    { fn: "spectrogram", x: R([400]), opts: { window: hannArray(64) } }, // here nperseg follows the window
    { fn: "spectrogram", x: R([400]), opts: { window: 4 } },
    // new windows through every function
    { fn: "welch", x: R([300]), opts: { window: ["chebwin", 60], nperseg: 64 } },
    { fn: "welch", x: R([300]), opts: { window: ["chebwin", 30], nperseg: 64 } }, // warns
    { fn: "stft", x: R([300]), opts: { window: ["dpss", 3], nperseg: 64 } },
    { fn: "stft", x: R([300]), opts: { window: ["kbd", 5], nperseg: 64 } }, // periodic: SciPy raises
    { fn: "welch", x: R([300]), opts: { window: "lanczos", nperseg: 63 } },
    { fn: "welch", x: R([300]), opts: { window: "hanning" } },
    { fn: "welch", x: R([300]), opts: { window: ["hann", 3] } },
  ];
  check(scenarios);
});

test("welch/stft: empty-input and clamp behaviour keep the package's own console.warn default and onWarning override", () => {
  const original = console.warn;
  const logged: string[] = [];
  console.warn = (m: unknown) => void logged.push(String(m));
  try {
    const x = tensorOf(signalOf(40, 1), [40]);
    welch(x);
    assert.deepEqual(logged, ["nperseg=256 is greater than signal length max(len(x), len(y)) = 40, using nperseg = 40"]);
    welch(x, { onWarning: () => {} });
    assert.equal(logged.length, 1, "onWarning replaces console.warn");
    welch(x, { nperseg: 40 });
    stft(x, { nperseg: 40 });
    assert.equal(logged.length, 1, "no warning when nperseg fits");
    stft(x);
    assert.deepEqual(logged.slice(1), ["nperseg = 256 is greater than input length  = 40, using nperseg = 40"]);
  } finally {
    console.warn = original;
  }
});

test("error classes: SciPy's ValueError is a RangeError, a bad window parameter list is a TypeError; a window must be a name, tuple or Float64Array", () => {
  const x = tensorOf(signalOf(100, 1), [100]);
  assert.throws(() => welch(x, { window: "hanning" as never }), RangeError);
  assert.throws(() => welch(x, { window: ["hann", 1] as never }), TypeError);
  assert.throws(() => welch(x, { window: ["kaiser"] as never }), TypeError);
  assert.throws(() => welch(x, { window: 4 as never }), TypeError);
  assert.throws(() => welch(x, { window: "kaiser" as never }), RangeError);
  assert.throws(() => csd(x, tensorOf(signalOf(100, 2), [100]), { nperseg: 0 }), RangeError);
  assert.throws(() => spectrogram(x, { mode: "bogus" as never }), RangeError);
  assert.throws(() => periodogram(x, { window: new Float64Array(5) }), RangeError);
  assert.doesNotThrow(() => getWindow(["dpss", 3], 20, true));
});

// ---- istft options -----------------------------------------------------------------------------------------------

test("istft takes SciPy's option semantics: int() truncation, negative noverlap, nperseg from the bins (not from an array window), window errors and warnings", { skip: SCIPY_SKIP_REASON }, () => {
  const make = (bins: number, frames: number, seed: number) => {
    const re = signalOf(bins * frames, seed);
    const im = imagOf(bins * frames, seed);
    return { shape: [bins, frames], re, im };
  };
  const cases: Array<{ z: { shape: number[]; re: number[]; im: number[] }; opts: Kw }> = [
    { z: make(33, 9, 1), opts: {} },
    { z: make(33, 9, 2), opts: { nperseg: 64.9, noverlap: 30.5, nfft: 70.2 } },
    { z: make(33, 9, 3), opts: { nperseg: 40, noverlap: -5 } },
    { z: make(33, 9, 4), opts: { nperseg: 64, noverlap: -3, boundary: false } },
    { z: make(33, 9, 5), opts: { window: hannArray(64) } },
    { z: make(33, 9, 6), opts: { window: hannArray(30) } }, // window must be nperseg long: SciPy raises
    { z: make(33, 9, 7), opts: { window: hannArray(30), nperseg: 30 } },
    { z: make(33, 9, 8), opts: { window: 4 } }, // not 1-D
    { z: make(33, 9, 9), opts: { window: ["chebwin", 30] } }, // warns (and the NOLA check may too)
    { z: make(33, 9, 10), opts: { window: ["dpss", 3], scaling: "psd", fs: 4 } },
    { z: make(33, 9, 11), opts: { window: ["kbd", 4] } }, // periodic: SciPy raises
    { z: make(33, 9, 12), opts: { window: "lanczos", noverlap: 40 } },
    { z: make(33, 9, 13), opts: { nperseg: 0 } },
    { z: make(33, 9, 14), opts: { nperseg: 64, nfft: 63 } },
    { z: make(33, 9, 15), opts: { nperseg: 64, noverlap: 64 } },
    { z: make(33, 9, 16), opts: { scaling: "density" } },
    { z: make(33, 9, 17), opts: { scaling: "none" } },
  ];
  const jobs = cases.map((c) => ({ op: "istft_kw", real: nest(c.z.re, c.z.shape), imag: nest(c.z.im, c.z.shape), kwargs: pyKw(c.opts, false), expect_error: true }));
  const wants = runScipyOracle<Array<{ shape: number[]; real: unknown; imag: unknown; complex: boolean; warnings: string[]; error?: string; error_type?: string }>>({ op: "batch", jobs });
  cases.forEach((c, i) => {
    const want = wants[i] as (typeof wants)[number];
    const z = ComplexTensor.fromParts(tensorOf(c.z.re, c.z.shape), tensorOf(c.z.im, c.z.shape));
    const warned: string[] = [];
    const run = () => istft(z, { ...c.opts, onWarning: (m: string) => warned.push(m) } as never);
    const what = JSON.stringify(c.opts, (_, v) => (v instanceof Float64Array ? `Float64Array(${v.length})` : v));
    if (want.error !== undefined) {
      if (SCIPY_17_PLUS) assert.throws(run, (e: unknown) => e instanceof RangeError || e instanceof TypeError, `${what}: SciPy raised ${want.error}`);
      else assert.throws(run, /TypeError$/.test(want.error_type ?? "") ? TypeError : RangeError, `${what}: SciPy raised ${want.error}`);
      return;
    }
    const got = run();
    const gotRe = got.x instanceof ComplexTensor ? got.x.real : got.x;
    assert.deepEqual([...gotRe.shape], want.shape, `shape ${what}`);
    const wr = flatten(want.real);
    closeScaled(flat(gotRe), wr, Math.max(maxAbs(wr), 1e-300), `x ${what}`);
    assert.deepEqual(warned, want.warnings, `warnings ${what}`);
  });
});
