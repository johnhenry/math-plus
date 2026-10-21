/**
 * Shared machinery for `welch` and `stft` (issues #178, SciPy parity): option
 * resolution with `scipy.signal`'s meanings and defaults, boundary extension
 * and padding, per-segment detrend and windowing, and the zero-padded DFT of
 * any `nfft` (see dft.ts). One implementation, so the two cannot drift apart.
 */
import { ComplexTensor } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { type DftPlan, planDft } from "./dft.ts";
import { isComplexTensor, type Rows } from "./nd.ts";
import { emitWarning, type WarningOptions } from "./warn.ts";
import { getWindow, type WindowSpec } from "./window.ts";

/**
 * A detrend function, applied to each 1-D segment on its own (`detrend(segment) -> segment`). It gets a 1-D `Tensor` of
 * `nperseg` samples (a 1-D `ComplexTensor` for complex input) and returns the detrended segment of the same length: a
 * `Tensor` / `ComplexTensor` of that kind, or a `Float64Array` for real input. SciPy hands its function every segment
 * (and every other row) at once; a function that acts on the segment axis independently, as `scipy.signal.detrend`
 * does, gives the same result either way.
 */
export type DetrendFunction = (segment: Tensor | ComplexTensor) => Tensor | ComplexTensor | Float64Array;
export type Detrend = false | "constant" | "linear" | DetrendFunction;

export { emitWarning, type WarningOptions };
/** How `stft` extends the signal by `floor(nperseg / 2)` samples at both ends (`scipy.signal.stft`'s `boundary`); `null` = no extension. */
export type Boundary = "zeros" | "even" | "odd" | "constant" | null;

export interface SegmentOptions extends WarningOptions {
  /** Sample rate; default 1. Frequencies come out in the same unit (hertz for samples per second). */
  readonly fs?: number;
  /** Window: a name (any of `get_window`'s, see `WINDOW_NAMES`), a `(name, param, ...)` tuple (both periodic, as `scipy.signal.get_window`) or a `Float64Array` (its length then sets `nperseg`). A bare number is not a window here (SciPy rejects it too). Default `"hann"`. */
  readonly window?: WindowSpec;
  /** Segment length, truncated to an integer as Python's `int()` does; default `min(256, signal.length)` (a longer request is clamped with SciPy's warning). */
  readonly nperseg?: number;
  /** Samples shared by consecutive segments, below `nperseg` (negative leaves gaps); truncated to an integer; default `floor(nperseg / 2)`. */
  readonly noverlap?: number;
  /** DFT length, any integer `>= nperseg` (not just powers of two; truncated to an integer); each segment is zero-padded to it. Default: `nperseg`. */
  readonly nfft?: number;
  /** Removed from each segment before windowing: `"constant"` (mean), `"linear"` (least-squares line), a function `(segment) -> segment` ({@link DetrendFunction}) or `false`. Default: `"constant"` for `welch`, `false` for `stft` (SciPy's own defaults). */
  readonly detrend?: Detrend;
  /** Keep only the `floor(nfft / 2) + 1` bins from DC up (real input). Default `true`; `false` keeps all `nfft` bins. Complex input is always two-sided (`stft` and `spectrogram` warn when you left this on, as SciPy does; `welch`, `csd`, `periodogram` and `coherence` switch silently, as SciPy does). */
  readonly onesided?: boolean;
}

export interface ResolvedSegments {
  readonly fs: number;
  /** Length of the signal after boundary extension and padding (the input's own when neither applies). */
  readonly extLength: number;
  readonly window: Float64Array;
  readonly nperseg: number;
  readonly noverlap: number;
  readonly nfft: number;
  readonly hop: number;
  readonly numFrames: number;
  readonly detrend: Detrend;
  readonly complex: boolean;
  readonly onesided: boolean;
  /** Number of output bins: `floor(nfft / 2) + 1` one-sided, else `nfft`. */
  readonly bins: number;
  /** Whether the signal was boundary-extended (shifts `times` back by `nperseg / 2`). */
  readonly extended: boolean;
  /** Σw */
  readonly sumW: number;
  /** Σw² */
  readonly sumW2: number;
  readonly plan: DftPlan;
  readonly boundary: Boundary;
  readonly padded: boolean;
}

export interface ResolveExtras {
  readonly defaultDetrend: Detrend;
  readonly boundary?: Boundary;
  readonly padded?: boolean;
  /** Emit SciPy's "Input data is complex, switching to return_onesided=False" warning (`stft` and `spectrogram` do; `welch`, `csd` do not). */
  readonly warnComplexOnesided?: boolean;
  /**
   * Which SciPy code path's rules apply. `"csd"` (`welch`, `csd`, `periodogram`, `coherence`): `nperseg` defaults to 256
   * for a named window and to the window's length for an array; a longer `nperseg` warns and clamps for a named window.
   * `"helper"` (`stft`, `spectrogram`): `nperseg` given is compared with an array window's length, and an array window
   * longer than the signal is an error. Default `"csd"`.
   */
  readonly family?: "csd" | "helper";
  /** `noverlap` when not given, from the resolved `nperseg`; default `floor(nperseg / 2)` (`spectrogram` uses `floor(nperseg / 8)`). */
  readonly defaultNoverlap?: (nperseg: number) => number;
}

const DEFAULT_NPERSEG = 256;
const BOUNDARIES = ["zeros", "even", "odd", "constant"];

/** Python's `int(v)` for the numbers SciPy takes as sizes: truncation toward zero; NaN and infinities are errors. */
export function toInt(who: string, name: string, v: number): number {
  if (typeof v !== "number" || !Number.isFinite(v)) throw new RangeError(`${who}: ${name} must be a finite number, got ${String(v)}`);
  return Math.trunc(v);
}

/** Re-throw an error from `getWindow` as `who`'s, keeping its class (`TypeError` for a wrong parameter count or type, else `RangeError`). */
export function rethrowWindowError(who: string, e: unknown): never {
  const message = `${who}: ${(e as Error).message.replace(/^getWindow: /, "")}`;
  throw e instanceof TypeError ? new TypeError(message) : new RangeError(message);
}

/** `scipy.signal.windows`-style `window` argument that is neither a name nor a tuple nor an array. */
function checkWindowKind(who: string, spec: unknown, family: "csd" | "helper"): void {
  if (spec instanceof Float64Array || typeof spec === "string" || Array.isArray(spec)) return;
  if (family === "helper") throw new RangeError(`${who}: window must be 1-D`); // np.asarray(4.0).shape is ()
  throw new TypeError(`${who}: window must be a name, a (name, parameter, ...) tuple or a Float64Array, got ${typeof spec}`);
}

/**
 * The window and `nperseg` SciPy's rules give for an input of `length` samples (`csd`'s or `_triage_segments`'s,
 * per `family`), with SciPy's warning when a longer `nperseg` is clamped.
 */
export function triageSegments(who: string, length: number, options: SegmentOptions, family: "csd" | "helper"): { window: Float64Array; nperseg: number } {
  const spec = options.window ?? "hann";
  checkWindowKind(who, spec, family);
  let nperseg = options.nperseg === undefined ? undefined : family === "helper" ? toInt(who, "nperseg", options.nperseg) : options.nperseg;
  if (spec instanceof Float64Array) {
    if (family === "helper") {
      if (length < spec.length) throw new RangeError(`${who}: window is longer than input signal`);
      if (nperseg !== undefined && nperseg !== spec.length) throw new RangeError(`${who}: value specified for nperseg is different from length of window`);
    } else if (nperseg !== undefined && nperseg !== spec.length) {
      throw new RangeError(`${who}: nperseg=${nperseg} does not equal len(win)=${spec.length}`);
    }
    return { window: spec, nperseg: spec.length };
  }
  if (family === "csd") {
    nperseg = options.nperseg === undefined ? DEFAULT_NPERSEG : toInt(who, "nperseg", options.nperseg);
    if (nperseg < 1) throw new RangeError(`${who}: Parameter nperseg=${nperseg} is not a positive integer!`);
    if (length < nperseg) {
      emitWarning(options, `nperseg=${nperseg} is greater than signal length max(len(x), len(y)) = ${length}, using nperseg = ${length}`);
      nperseg = length;
    }
  } else {
    nperseg = nperseg ?? DEFAULT_NPERSEG;
    if (nperseg > length) {
      emitWarning(options, `nperseg = ${nperseg} is greater than input length  = ${length}, using nperseg = ${length}`);
      nperseg = length;
    }
  }
  try {
    return { window: getWindow(spec, nperseg, true, options), nperseg };
  } catch (e) {
    return rethrowWindowError(who, e);
  }
}

/** `scipy.signal`'s `even_ext`/`odd_ext`/`const_ext`/`zero_ext` by `n` samples at each end. */
function extend(who: string, x: Float64Array, n: number, kind: Exclude<Boundary, null>): Float64Array {
  if (n < 1) return x;
  const L = x.length;
  if ((kind === "even" || kind === "odd") && n > L - 1) {
    throw new RangeError(`${who}: boundary "${kind}" needs nperseg / 2 (${n}) <= signal length - 1 (${L - 1})`);
  }
  const out = new Float64Array(L + 2 * n);
  out.set(x, n);
  const first = x[0] as number;
  const last = x[L - 1] as number;
  for (let i = 0; i < n; i++) {
    let left = 0;
    let right = 0;
    if (kind === "even") {
      left = x[n - i] as number;
      right = x[L - 2 - i] as number;
    } else if (kind === "odd") {
      left = 2 * first - (x[n - i] as number);
      right = 2 * last - (x[L - 2 - i] as number);
    } else if (kind === "constant") {
      left = first;
      right = last;
    }
    out[i] = left;
    out[n + L + i] = right;
  }
  return out;
}

/** Resolve the options for rows of `length` samples (real or complex). Validation and defaults follow `scipy.signal`. */
export function resolveSegments(who: string, length: number, complex: boolean, options: SegmentOptions, extras: ResolveExtras): ResolvedSegments {
  const fs = options.fs ?? 1;
  if (!(fs > 0) || !Number.isFinite(fs)) throw new RangeError(`${who}: fs must be a finite number above 0, got ${fs}`);
  if (length < 1) throw new RangeError(`${who}: signal must not be empty`);
  const family = extras.family ?? "csd";
  // (`_spectral_helper` checks a given nperseg before it sets the window up; `spectrogram`'s own triage does not, hence not in triageSegments)
  if (family === "helper" && options.nperseg !== undefined && toInt(who, "nperseg", options.nperseg) < 1) throw new RangeError(`${who}: nperseg must be a positive integer`);
  const { window, nperseg } = triageSegments(who, length, options, family);
  if (!Number.isInteger(nperseg) || nperseg < 1) throw new RangeError(`${who}: nperseg must be a positive integer, got ${nperseg}`);
  const noverlap = options.noverlap === undefined ? (extras.defaultNoverlap ?? ((n: number) => Math.floor(n / 2)))(nperseg) : toInt(who, "noverlap", options.noverlap);
  if (noverlap >= nperseg) throw new RangeError(`${who}: noverlap=${noverlap} must be less than nperseg=${nperseg}!`);
  const nfft = options.nfft === undefined ? nperseg : toInt(who, "nfft", options.nfft);
  if (nfft < nperseg) throw new RangeError(`${who}: nfft=${nfft} must be greater than or equal to nperseg=${nperseg}!`);
  const detrend = options.detrend ?? extras.defaultDetrend;
  if (detrend !== false && detrend !== "constant" && detrend !== "linear" && typeof detrend !== "function") {
    throw new RangeError(`${who}: detrend must be false, "constant", "linear" or a function, got ${String(detrend)}`);
  }
  const boundary = extras.boundary ?? null;
  if (boundary !== null && !BOUNDARIES.includes(boundary)) {
    throw new RangeError(`${who}: boundary must be null, "zeros", "even", "odd" or "constant", got ${String(boundary)}`);
  }
  const hop = nperseg - noverlap;
  const padded = extras.padded ?? false;

  // Boundary extension first, padding after, so the extended signal ends in zeros (as SciPy does).
  let extLength = boundary === null ? length : length + 2 * Math.floor(nperseg / 2);
  if (padded) extLength += ((((-(extLength - nperseg)) % hop) + hop) % hop) % nperseg;
  const numFrames = Math.floor((extLength - nperseg) / hop) + 1;
  if (numFrames < 1 || extLength < nperseg) throw new RangeError(`${who}: signal too short (${length} samples) for nperseg=${nperseg}`);

  let sumW = 0;
  let sumW2 = 0;
  for (const w of window) {
    sumW += w;
    sumW2 += w * w;
  }
  let onesided = options.onesided ?? true;
  if (complex && onesided) {
    onesided = false;
    if (extras.warnComplexOnesided) emitWarning(options, "Input data is complex, switching to return_onesided=False");
  }
  return {
    fs,
    extLength,
    window,
    nperseg,
    noverlap,
    nfft,
    hop,
    numFrames,
    detrend,
    complex,
    onesided,
    bins: onesided ? Math.floor(nfft / 2) + 1 : nfft,
    extended: boundary !== null,
    sumW,
    sumW2,
    plan: planDft(nfft),
    boundary,
    padded,
  };
}

/** One row after boundary extension and padding. */
export interface PreparedRow {
  readonly re: Float64Array;
  readonly im: Float64Array | null;
}

/** Boundary-extend and pad a row (`re` / `im` parts) to `p.extLength`. */
export function prepareRow(who: string, p: ResolvedSegments, re: Float64Array, im: Float64Array | null): PreparedRow {
  const fix = (x: Float64Array): Float64Array => {
    let data = p.boundary === null ? x : extend(who, x, Math.floor(p.nperseg / 2), p.boundary);
    if (data.length < p.extLength) {
      const padded = new Float64Array(p.extLength);
      padded.set(data);
      data = padded;
    }
    return data;
  };
  return { re: fix(re), im: im === null ? null : fix(im) };
}

/** Frequencies of the output bins: `rfftfreq` one-sided (`0 … fs/2`), else `fftfreq` order (Nyquist negative for even `nfft`). */
export function binFrequencies(nfft: number, fs: number, onesided: boolean): number[] {
  if (onesided) return Array.from({ length: Math.floor(nfft / 2) + 1 }, (_, k) => (k * fs) / nfft);
  const half = Math.floor((nfft - 1) / 2) + 1; // fftfreq: non-negative bins are 0 … (nfft-1)//2
  return Array.from({ length: nfft }, (_, k) => ((k < half ? k : k - nfft) * fs) / nfft);
}

/** Remove the mean or the least-squares line from `f` in place (`scipy.signal.detrend`). */
export function detrendInPlace(f: Float64Array, kind: "constant" | "linear"): void {
  const n = f.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += f[i] as number;
  mean /= n;
  if (kind === "constant" || n === 1) {
    // (a single point has a perfect "line": its detrended value is 0 either way)
    for (let i = 0; i < n; i++) f[i] = (f[i] as number) - mean;
    return;
  }
  const tm = (n - 1) / 2;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (i - tm) * ((f[i] as number) - mean);
    den += (i - tm) * (i - tm);
  }
  const slope = num / den;
  for (let i = 0; i < n; i++) f[i] = (f[i] as number) - mean - slope * (i - tm);
}

/** Run a user detrend function on one segment (`sr` / `si` are replaced in place by its result). */
function detrendWith(who: string, fn: DetrendFunction, sr: Float64Array, si: Float64Array | null): void {
  const n = sr.length;
  const arg = si === null
    ? Tensor.fromTypedArray(Float64Array.from(sr), [n], { dtype: "f64" })
    : ComplexTensor.fromParts(Tensor.fromTypedArray(Float64Array.from(sr), [n], { dtype: "f64" }), Tensor.fromTypedArray(Float64Array.from(si), [n], { dtype: "f64" }));
  const out = fn(arg);
  const parts = out instanceof Float64Array ? { re: out, im: null } : isComplexTensor(out) ? { re: out.real, im: out.imag } : { re: out, im: null };
  const read = (v: Float64Array | Tensor): Float64Array => (v instanceof Float64Array ? v : (v.contiguous().data as Float64Array));
  const re = read(parts.re);
  if (re.length !== n) throw new RangeError(`${who}: detrend must return a segment of ${n} samples, got ${re.length}`);
  if (si === null) {
    if (parts.im !== null) throw new RangeError(`${who}: detrend returned a complex segment for real input`);
    sr.set(re);
    return;
  }
  if (parts.im === null) throw new RangeError(`${who}: detrend must return a ComplexTensor for complex input`);
  const im = read(parts.im);
  if (im.length !== n) throw new RangeError(`${who}: detrend must return a segment of ${n} samples, got ${im.length}`);
  sr.set(re);
  si.set(im);
}

/** Segment `start…start+nperseg` of a prepared row: detrended, windowed, zero-padded to `nfft`, transformed. Returns the full `nfft`-bin spectrum. */
export function segmentSpectrum(who: string, start: number, p: ResolvedSegments, row: PreparedRow): { re: Float64Array; im: Float64Array } {
  const sr = row.re.slice(start, start + p.nperseg);
  const si = row.im === null ? null : row.im.slice(start, start + p.nperseg);
  if (typeof p.detrend === "function") {
    detrendWith(who, p.detrend, sr, si);
  } else if (p.detrend !== false) {
    detrendInPlace(sr, p.detrend);
    if (si !== null) detrendInPlace(si, p.detrend);
  }
  const fr = new Float64Array(p.nfft);
  const fi = new Float64Array(p.nfft);
  for (let i = 0; i < p.nperseg; i++) {
    const w = p.window[i] as number;
    fr[i] = (sr[i] as number) * w;
    if (si !== null) fi[i] = (si[i] as number) * w;
  }
  const out = p.plan.transform(fr, fi, false);
  if (si === null) {
    // The DFT of real data is exactly real at DC and (even nfft) at Nyquist: set the rounding residue to zero, as an
    // rfft does, so that angle() of those bins is 0 or pi like SciPy's, not a coin toss.
    out.im[0] = 0;
    if (p.nfft % 2 === 0) out.im[p.nfft / 2] = 0;
  }
  return out;
}

/** The per-frame spectra of every row, in SciPy's output layout: the input's shape with `axis` replaced by the bins, then the frames last. */
export interface FrameSpectra {
  readonly re: Float64Array;
  /** `null` when only the power was asked for. */
  readonly im: Float64Array | null;
  readonly shape: number[];
}

/**
 * One windowed `nfft`-point DFT per frame of every row (what `stft`, `spectrogram`, `csd` and `welch` are made of):
 * the complex spectra times `scale`, or, with `power`, their squared magnitudes `|scale · X|²` (then `im` is `null`).
 */
export function frameSpectra(who: string, rows: Rows, p: ResolvedSegments, scale: number, power: boolean): FrameSpectra {
  const { bins, numFrames } = p;
  const total = rows.pre * bins * rows.inner * numFrames;
  const re = new Float64Array(total);
  const im = power ? null : new Float64Array(total);
  for (let r = 0; r < rows.rows; r++) {
    const row = prepareRow(who, p, rows.re.subarray(r * rows.n, (r + 1) * rows.n), rows.im === null ? null : rows.im.subarray(r * rows.n, (r + 1) * rows.n));
    const pi = Math.floor(r / rows.inner);
    const ii = r % rows.inner;
    for (let f = 0; f < numFrames; f++) {
      const spec = segmentSpectrum(who, f * p.hop, p, row);
      for (let k = 0; k < bins; k++) {
        const at = ((pi * bins + k) * rows.inner + ii) * numFrames + f;
        const a = (spec.re[k] as number) * scale;
        const b = (spec.im[k] as number) * scale;
        if (im === null) re[at] = a * a + b * b;
        else {
          re[at] = a;
          im[at] = b;
        }
      }
    }
  }
  const shape = [...rows.shape];
  shape[rows.axis] = bins;
  shape.push(numFrames);
  return { re, im, shape };
}

/** `np.mod` for floats: the result takes the sign of the divisor. */
function pyMod(a: number, b: number): number {
  const m = a % b;
  return m !== 0 && m < 0 !== b < 0 ? m + b : m;
}

/** `np.unwrap(values, axis=axis)` (`discont` and `period` at their defaults) in place, for a row-major array of `shape`. */
export function unwrapAlong(values: Float64Array, shape: readonly number[], axis: number): void {
  const n = shape[axis] as number;
  if (n < 2) return;
  let inner = 1;
  for (let d = axis + 1; d < shape.length; d++) inner *= shape[d] as number;
  const pre = values.length / (n * inner);
  const period = 2 * Math.PI;
  const half = period / 2;
  for (let a = 0; a < pre; a++) {
    for (let b = 0; b < inner; b++) {
      const base = a * n * inner + b;
      let correction = 0;
      let previous = values[base] as number;
      for (let k = 1; k < n; k++) {
        const at = base + k * inner;
        const current = values[at] as number;
        const dd = current - previous;
        let ddmod = pyMod(dd + half, period) - half;
        if (ddmod === -half && dd > 0) ddmod = half;
        let fix = ddmod - dd;
        if (Math.abs(dd) < half) fix = 0;
        correction += fix;
        values[at] = current + correction;
        previous = current; // the differences are those of the original values
      }
    }
  }
}

/** Segment-centre times in `1/fs` units (`np.arange(nperseg / 2, n - nperseg / 2 + 1, hop) / fs`, shifted back by `nperseg / 2` when the signal was boundary-extended). */
export function frameTimes(p: ResolvedSegments): number[] {
  const shift = p.extended ? p.nperseg / 2 : 0;
  const times: number[] = [];
  for (let f = 0; f < p.numFrames; f++) times.push((f * p.hop + p.nperseg / 2 - shift) / p.fs);
  return times;
}
