/**
 * Shared machinery for `welch` and `stft` (issues #178, SciPy parity): option
 * resolution with `scipy.signal`'s meanings and defaults, boundary extension
 * and padding, per-segment detrend and windowing, and the zero-padded DFT of
 * any `nfft` (see dft.ts). One implementation, so the two cannot drift apart.
 */
import { type DftPlan, planDft } from "./dft.ts";
import { getWindow, type WindowSpec } from "./window.ts";

export type Detrend = false | "constant" | "linear";
/** How `stft` extends the signal by `floor(nperseg / 2)` samples at both ends (`scipy.signal.stft`'s `boundary`); `null` = no extension. */
export type Boundary = "zeros" | "even" | "odd" | "constant" | null;

export interface SegmentOptions {
  /** Sample rate; default 1. Frequencies come out in the same unit (hertz for samples per second). */
  readonly fs?: number;
  /** Window: a name (periodic, as `scipy.signal.get_window`) or a `Float64Array` (its length then sets `nperseg`). Default `"hann"`. */
  readonly window?: WindowSpec;
  /** Segment length, any positive integer; default `min(256, signal.length)` (a longer request is clamped, as SciPy does). */
  readonly nperseg?: number;
  /** Samples shared by consecutive segments, `[0, nperseg)`; default `floor(nperseg / 2)`. */
  readonly noverlap?: number;
  /** DFT length, any integer `>= nperseg` (not just powers of two); each segment is zero-padded to it. Default: `nperseg`. */
  readonly nfft?: number;
  /** Removed from each segment before windowing: `"constant"` (mean), `"linear"` (least-squares line) or `false`. Default: `"constant"` for `welch`, `false` for `stft` (SciPy's own defaults). */
  readonly detrend?: Detrend;
  /** Keep only the `floor(nfft / 2) + 1` bins from DC up (real input). Default `true`; `false` keeps all `nfft` bins. */
  readonly onesided?: boolean;
}

export interface ResolvedSegments {
  readonly fs: number;
  /** The signal after boundary extension and padding (the input itself when neither applies). */
  readonly data: Float64Array;
  readonly window: Float64Array;
  readonly nperseg: number;
  readonly noverlap: number;
  readonly nfft: number;
  readonly hop: number;
  readonly numFrames: number;
  readonly detrend: Detrend;
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
}

export interface ResolveExtras {
  readonly defaultDetrend: Detrend;
  readonly boundary?: Boundary;
  readonly padded?: boolean;
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

export function resolveSegments(who: string, signal: Float64Array, options: SegmentOptions, extras: ResolveExtras): ResolvedSegments {
  const fs = options.fs ?? 1;
  if (!(fs > 0) || !Number.isFinite(fs)) throw new RangeError(`${who}: fs must be a finite number above 0, got ${fs}`);
  const spec = options.window ?? "hann";
  const length = signal.length;
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
  if (detrend !== false && detrend !== "constant" && detrend !== "linear") {
    throw new RangeError(`${who}: detrend must be false, "constant" or "linear", got ${String(detrend)}`);
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

  // Boundary extension first, padding after, so the extended signal ends in zeros (as SciPy does).
  let data = boundary === null ? signal : extend(who, signal, Math.floor(nperseg / 2), boundary);
  if (extras.padded) {
    const nadd = ((((-(data.length - nperseg)) % hop) + hop) % hop) % nperseg;
    if (nadd > 0) {
      const padded = new Float64Array(data.length + nadd);
      padded.set(data);
      data = padded;
    }
  }
  const numFrames = Math.floor((data.length - nperseg) / hop) + 1;
  if (numFrames < 1) throw new RangeError(`${who}: signal too short (${length} samples) for nperseg=${nperseg}`);

  let sumW = 0;
  let sumW2 = 0;
  for (const w of window) {
    sumW += w;
    sumW2 += w * w;
  }
  const onesided = options.onesided ?? true;
  return {
    fs,
    data,
    window,
    nperseg,
    noverlap,
    nfft,
    hop,
    numFrames,
    detrend,
    onesided,
    bins: onesided ? Math.floor(nfft / 2) + 1 : nfft,
    extended: boundary !== null,
    sumW,
    sumW2,
    plan: planDft(nfft),
  };
}

/** Frequencies of the output bins: `rfftfreq` one-sided (`0 … fs/2`), else `fftfreq` order (Nyquist negative for even `nfft`). */
export function binFrequencies(nfft: number, fs: number, onesided: boolean): number[] {
  if (onesided) return Array.from({ length: Math.floor(nfft / 2) + 1 }, (_, k) => (k * fs) / nfft);
  const half = Math.floor((nfft - 1) / 2) + 1; // fftfreq: non-negative bins are 0 … (nfft-1)//2
  return Array.from({ length: nfft }, (_, k) => ((k < half ? k : k - nfft) * fs) / nfft);
}

/** Remove the mean or the least-squares line from `f` in place (`scipy.signal.detrend`). */
export function detrendInPlace(f: Float64Array, kind: Detrend): void {
  if (kind === false) return;
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

/** Segment `start…start+nperseg` of `p.data`: detrended, windowed, zero-padded to `nfft`, transformed. Returns the full `nfft`-bin spectrum. */
export function segmentSpectrum(start: number, p: ResolvedSegments): { re: Float64Array; im: Float64Array } {
  const seg = p.data.slice(start, start + p.nperseg);
  detrendInPlace(seg, p.detrend);
  const frame = new Float64Array(p.nfft);
  for (let i = 0; i < p.nperseg; i++) frame[i] = (seg[i] as number) * (p.window[i] as number);
  return p.plan.transform(frame, new Float64Array(p.nfft), false);
}
