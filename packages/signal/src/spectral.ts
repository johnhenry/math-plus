/**
 * Shared machinery for `welch` and `stft` (issues #178, SciPy parity): option
 * resolution with `scipy.signal`'s meanings and defaults, boundary extension
 * and padding, per-segment detrend and windowing, and the zero-padded DFT of
 * any `nfft` (see dft.ts). One implementation, so the two cannot drift apart.
 */
import { ComplexTensor } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { type DftPlan, planDft } from "./dft.ts";
import { isComplexTensor } from "./nd.ts";
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

/**
 * Where warnings go. SciPy's `warnings.warn` becomes `onWarning(message)`; without a callback the message goes to
 * `console.warn`. Pass `() => {}` to silence them.
 */
export interface WarningOptions {
  readonly onWarning?: (message: string) => void;
}

export function emitWarning(options: WarningOptions, message: string): void {
  if (options.onWarning) options.onWarning(message);
  else console.warn(message);
}
/** How `stft` extends the signal by `floor(nperseg / 2)` samples at both ends (`scipy.signal.stft`'s `boundary`); `null` = no extension. */
export type Boundary = "zeros" | "even" | "odd" | "constant" | null;

export interface SegmentOptions extends WarningOptions {
  /** Sample rate; default 1. Frequencies come out in the same unit (hertz for samples per second). */
  readonly fs?: number;
  /** Window: a name, a `(name, param)` tuple or a number (all periodic, as `scipy.signal.get_window`) or a `Float64Array` (its length then sets `nperseg`). Default `"hann"`. */
  readonly window?: WindowSpec;
  /** Segment length, any positive integer; default `min(256, signal.length)` (a longer request is clamped, as SciPy does). */
  readonly nperseg?: number;
  /** Samples shared by consecutive segments, `[0, nperseg)`; default `floor(nperseg / 2)`. */
  readonly noverlap?: number;
  /** DFT length, any integer `>= nperseg` (not just powers of two); each segment is zero-padded to it. Default: `nperseg`. */
  readonly nfft?: number;
  /** Removed from each segment before windowing: `"constant"` (mean), `"linear"` (least-squares line), a function `(segment) -> segment` ({@link DetrendFunction}) or `false`. Default: `"constant"` for `welch`, `false` for `stft` (SciPy's own defaults). */
  readonly detrend?: Detrend;
  /** Keep only the `floor(nfft / 2) + 1` bins from DC up (real input). Default `true`; `false` keeps all `nfft` bins. Complex input is always two-sided (`stft` warns when you left this on, as SciPy does; `welch` switches silently, as SciPy does). */
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
  /** Emit SciPy's "Input data is complex, switching to onesided=false" warning (`stft` does, `welch` does not). */
  readonly warnComplexOnesided?: boolean;
}

const DEFAULT_NPERSEG = 256;
const BOUNDARIES = ["zeros", "even", "odd", "constant"];

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
  const spec = options.window ?? "hann";
  if (length < 1) throw new RangeError(`${who}: signal must not be empty`);

  let nperseg: number;
  if (spec instanceof Float64Array) {
    nperseg = options.nperseg ?? spec.length;
    if (nperseg !== spec.length) throw new RangeError(`${who}: nperseg (${nperseg}) differs from the window's length (${spec.length})`);
    if (length < nperseg) throw new RangeError(`${who}: window is longer than the signal (${nperseg} > ${length})`);
  } else {
    nperseg = options.nperseg ?? DEFAULT_NPERSEG;
    if (!Number.isInteger(nperseg) || nperseg < 1) throw new RangeError(`${who}: nperseg must be a positive integer, got ${nperseg}`);
    if (nperseg > length) nperseg = length; // SciPy warns and uses the whole signal
  }
  if (!Number.isInteger(nperseg) || nperseg < 1) throw new RangeError(`${who}: nperseg must be a positive integer, got ${nperseg}`);
  const noverlap = options.noverlap ?? Math.floor(nperseg / 2);
  if (!Number.isInteger(noverlap) || noverlap < 0 || noverlap >= nperseg) {
    throw new RangeError(`${who}: noverlap (${noverlap}) must be an integer in [0, nperseg) (nperseg=${nperseg})`);
  }
  const nfft = options.nfft ?? nperseg;
  if (!Number.isInteger(nfft) || nfft < nperseg) {
    throw new RangeError(`${who}: nfft must be an integer >= nperseg (${nperseg}), got ${nfft}`);
  }
  const detrend = options.detrend ?? extras.defaultDetrend;
  if (detrend !== false && detrend !== "constant" && detrend !== "linear" && typeof detrend !== "function") {
    throw new RangeError(`${who}: detrend must be false, "constant", "linear" or a function, got ${String(detrend)}`);
  }
  const boundary = extras.boundary ?? null;
  if (boundary !== null && !BOUNDARIES.includes(boundary)) {
    throw new RangeError(`${who}: boundary must be null, "zeros", "even", "odd" or "constant", got ${String(boundary)}`);
  }
  let window: Float64Array;
  try {
    window = getWindow(spec, nperseg);
  } catch (e) {
    throw new RangeError(`${who}: ${(e as Error).message}`);
  }
  const hop = nperseg - noverlap;
  const padded = extras.padded ?? false;

  // Boundary extension first, padding after, so the extended signal ends in zeros (as SciPy does).
  let extLength = boundary === null ? length : length + 2 * Math.floor(nperseg / 2);
  if (padded) extLength += ((((-(extLength - nperseg)) % hop) + hop) % hop) % nperseg;
  const numFrames = Math.floor((extLength - nperseg) / hop) + 1;
  if (numFrames < 1) throw new RangeError(`${who}: signal too short (${length} samples) for nperseg=${nperseg}`);

  let sumW = 0;
  let sumW2 = 0;
  for (const w of window) {
    sumW += w;
    sumW2 += w * w;
  }
  let onesided = options.onesided ?? true;
  if (complex && onesided) {
    onesided = false;
    if (extras.warnComplexOnesided) emitWarning(options, `${who}: input data is complex, switching to onesided=false`);
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
  return p.plan.transform(fr, fi, false);
}
