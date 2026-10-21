/**
 * Shared machinery for `welch` and `stft` (issue #178): option resolution
 * (SciPy's meanings), per-segment detrend, windowing and the zero-padded FFT.
 * One implementation, so the two cannot drift apart.
 */
import { ComplexTensor, fft } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { getWindow, type WindowSpec } from "./window.ts";

export type Detrend = false | "constant" | "linear";

export interface SegmentOptions {
  /** Sample rate; default 1. Frequencies come out in the same unit (hertz for samples per second). */
  readonly fs?: number;
  /** Window: a name (periodic, as `scipy.signal.get_window`) or a `Float64Array` of length `nperseg`. Default `"hann"`. */
  readonly window?: WindowSpec;
  /** Segment length, any positive integer; default `min(256, signal.length)`. */
  readonly nperseg?: number;
  /** Samples shared by consecutive segments, `[0, nperseg)`; default `floor(nperseg / 2)`. */
  readonly noverlap?: number;
  /** FFT length, a power of two `>= nperseg`; each segment is zero-padded to it. Default: the next power of two at or above `nperseg`. */
  readonly nfft?: number;
  /** Removed from each segment before windowing: `"constant"` (mean), `"linear"` (least-squares line) or `false` (default). */
  readonly detrend?: Detrend;
  /** Keep only the `nfft / 2 + 1` bins from DC to Nyquist. Default `false` (all `nfft` bins). */
  readonly onesided?: boolean;
}

export interface ResolvedSegments {
  readonly fs: number;
  readonly window: Float64Array;
  readonly nperseg: number;
  readonly noverlap: number;
  readonly nfft: number;
  readonly hop: number;
  readonly numFrames: number;
  readonly detrend: Detrend;
  readonly onesided: boolean;
  /** Σw */
  readonly sumW: number;
  /** Σw² */
  readonly sumW2: number;
}

const DEFAULT_NPERSEG = 256;

function isPowerOfTwo(n: number): boolean {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

function nextPowerOfTwo(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

export function resolveSegments(who: string, length: number, options: SegmentOptions): ResolvedSegments {
  const fs = options.fs ?? 1;
  if (!(fs > 0) || !Number.isFinite(fs)) throw new RangeError(`${who}: fs must be a finite number above 0, got ${fs}`);
  const nperseg = options.nperseg ?? Math.min(DEFAULT_NPERSEG, length);
  if (!Number.isInteger(nperseg) || nperseg < 1) throw new RangeError(`${who}: nperseg must be a positive integer, got ${nperseg}`);
  const noverlap = options.noverlap ?? Math.floor(nperseg / 2);
  if (!Number.isInteger(noverlap) || noverlap < 0 || noverlap >= nperseg) {
    throw new RangeError(`${who}: noverlap (${noverlap}) must be an integer in [0, nperseg) (nperseg=${nperseg})`);
  }
  const nfft = options.nfft ?? nextPowerOfTwo(nperseg);
  if (!isPowerOfTwo(nfft) || nfft < nperseg) {
    throw new RangeError(`${who}: nfft must be a power of two >= nperseg (${nperseg}), got ${nfft}`);
  }
  const detrend = options.detrend ?? false;
  if (detrend !== false && detrend !== "constant" && detrend !== "linear") {
    throw new RangeError(`${who}: detrend must be false, "constant" or "linear", got ${String(detrend)}`);
  }
  let window: Float64Array;
  try {
    window = getWindow(options.window ?? "hann", nperseg);
  } catch (e) {
    throw new RangeError(`${who}: ${(e as Error).message}`);
  }
  const hop = nperseg - noverlap;
  const numFrames = Math.floor((length - nperseg) / hop) + 1;
  if (numFrames < 1) throw new RangeError(`${who}: signal too short (${length} samples) for nperseg=${nperseg}`);
  let sumW = 0;
  let sumW2 = 0;
  for (const w of window) {
    sumW += w;
    sumW2 += w * w;
  }
  return { fs, window, nperseg, noverlap, nfft, hop, numFrames, detrend, onesided: options.onesided ?? false, sumW, sumW2 };
}

/** Frequencies of the `nfft` FFT bins (numpy `fftfreq` order, Nyquist negative for even `nfft`) or, one-sided, `0 … fs/2`. */
export function binFrequencies(nfft: number, fs: number, onesided: boolean): number[] {
  if (onesided) return Array.from({ length: nfft / 2 + 1 }, (_, k) => (k * fs) / nfft);
  return Array.from({ length: nfft }, (_, k) => ((k < nfft / 2 ? k : k - nfft) * fs) / nfft);
}

/** Remove the mean or the least-squares line from `f` in place (`scipy.signal.detrend`). */
export function detrendInPlace(f: Float64Array, kind: Detrend): void {
  if (kind === false) return;
  const n = f.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += f[i] as number;
  mean /= n;
  if (kind === "constant" || n < 3) {
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

/** Segment `start…start+nperseg` of `data`: detrended, windowed, zero-padded to `nfft`, transformed. Returns the full `nfft`-bin spectrum. */
export function segmentSpectrum(
  data: Float64Array,
  start: number,
  p: ResolvedSegments,
): { real: Float64Array; imag: Float64Array } {
  const seg = data.slice(start, start + p.nperseg);
  detrendInPlace(seg, p.detrend);
  const frame = new Float64Array(p.nfft);
  for (let i = 0; i < p.nperseg; i++) frame[i] = (seg[i] as number) * (p.window[i] as number);
  const spectrum = fft(ComplexTensor.fromReal(Tensor.fromTypedArray(frame, [p.nfft], { dtype: "f64" })));
  return {
    real: spectrum.real.contiguous().data as Float64Array,
    imag: spectrum.imag.contiguous().data as Float64Array,
  };
}
