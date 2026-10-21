/**
 * stft/istft (issues #44, #178). With no options `stft(x)` is
 * `scipy.signal.stft(x)` and `istft(Zxx)` is `scipy.signal.istft(Zxx)`:
 * fs 1, periodic Hann window, `nperseg = min(256, n)`, 50% overlap,
 * `nfft = nperseg`, `detrend=false`, `scaling="spectrum"` (`X / Σw`),
 * one-sided (`return_onesided=True`), `boundary="zeros"`, `padded=true`.
 *
 * Return shapes follow SciPy's `(f, t, Zxx)` / `(t, x)` as plain objects:
 *  - `stft(x)` -> `{ freqs: number[], times: number[], Zxx: ComplexTensor }`
 *    with `Zxx` shaped `[freqs.length, times.length]` (frequency-major, like
 *    SciPy; the `times` are segment centres, shifted so the first is 0 when
 *    `boundary` is set).
 *  - `istft(Zxx)` -> `{ times: number[], x: Tensor }`.
 *
 * `istft` is weighted overlap-add with window-power normalisation (SciPy's
 * algorithm), exact (to floating-point precision) whenever the window and
 * overlap satisfy the NOLA condition -- the Hann-at-50% default does, and with
 * the default `boundary`/`padded` the whole signal comes back, edges included.
 *
 * `scaling` accepts SciPy's `"spectrum"` and `"psd"` (`X / sqrt(fs · Σw²)`),
 * `"density"` as an alias of `"psd"`, and a non-SciPy `"none"` (the raw
 * windowed DFT). One-sided bins are NOT doubled (SciPy doesn't either).
 *
 * Disclosed limits (not implemented): 1-D real input only (no `axis` /
 * `time_axis` / `freq_axis`, no complex input); `istft` returns the real part
 * (SciPy returns complex for `input_onesided=false`) and does not emit
 * SciPy's NOLA warning; `window` is a name or an array, not a
 * `(name, param)` tuple; `detrend` is not a callable.
 */
import { ComplexTensor } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { planDft } from "./dft.ts";
import { binFrequencies, type Boundary, resolveSegments, segmentSpectrum, type SegmentOptions } from "./spectral.ts";
import { getWindow, type WindowSpec } from "./window.ts";

export type StftScaling = "spectrum" | "psd" | "density" | "none";

export interface StftOptions extends SegmentOptions {
  /** `"spectrum"` (default): `X / Σw`. `"psd"` / `"density"`: `X / sqrt(fs · Σw²)`. `"none"`: the raw windowed DFT (not in SciPy). */
  readonly scaling?: StftScaling;
  /** Extend the signal by `floor(nperseg / 2)` samples at both ends so the first and last samples sit at a window centre: `"zeros"` (default), `"even"`, `"odd"`, `"constant"`, or `null` for none. */
  readonly boundary?: Boundary;
  /** Zero-pad the (extended) signal to a whole number of segments so no tail is dropped. Default `true`. */
  readonly padded?: boolean;
}

export interface StftResult {
  /** Bin frequencies (`rfftfreq` one-sided, `fftfreq` order two-sided). */
  readonly freqs: number[];
  /** Segment-centre times in `1/fs` units. */
  readonly times: number[];
  /** The complex spectrogram, shape `[freqs.length, times.length]`. */
  readonly Zxx: ComplexTensor;
}

function checkScaling(who: string, scaling: unknown): StftScaling {
  if (scaling !== "none" && scaling !== "density" && scaling !== "psd" && scaling !== "spectrum") {
    throw new RangeError(`${who}: scaling must be "spectrum", "psd", "density" or "none", got ${String(scaling)}`);
  }
  return scaling;
}

/** `signal[T]` -> `{ freqs, times, Zxx }`, `Zxx` being `[bins, numFrames]` (one `nfft`-point DFT per windowed frame; `bins` is `floor(nfft/2)+1` one-sided, else `nfft`). */
export function stft(signal: Tensor, options: StftOptions = {}): StftResult {
  if (signal.shape.length !== 1) throw new RangeError("stft: 1-D Tensor only");
  const scaling = checkScaling("stft", options.scaling ?? "spectrum");
  // Read-only below, so no defensive copy is needed even when `.data`
  // aliases `signal`'s own storage.
  const data = signal.contiguous().data as Float64Array;
  const p = resolveSegments("stft", data, options, {
    defaultDetrend: false,
    boundary: options.boundary === undefined ? "zeros" : options.boundary,
    padded: options.padded ?? true,
  });
  const { bins, numFrames } = p;
  const scale = scaling === "none" ? 1 : scaling === "spectrum" ? 1 / p.sumW : 1 / Math.sqrt(p.fs * p.sumW2);

  const realOut = new Float64Array(bins * numFrames);
  const imagOut = new Float64Array(bins * numFrames);
  const times: number[] = [];
  const shift = p.extended ? p.nperseg / 2 : 0;
  for (let f = 0; f < numFrames; f++) {
    const start = f * p.hop;
    const { re, im } = segmentSpectrum(start, p);
    for (let k = 0; k < bins; k++) {
      realOut[k * numFrames + f] = (re[k] as number) * scale;
      imagOut[k * numFrames + f] = (im[k] as number) * scale;
    }
    times.push((start + p.nperseg / 2 - shift) / p.fs);
  }
  const Zxx = ComplexTensor.fromParts(
    Tensor.fromTypedArray(realOut, [bins, numFrames], { dtype: "f64" }),
    Tensor.fromTypedArray(imagOut, [bins, numFrames], { dtype: "f64" }),
  );
  return { freqs: binFrequencies(p.nfft, p.fs, p.onesided), times, Zxx };
}

export interface IstftOptions {
  /** Sample rate; default 1. */
  readonly fs?: number;
  /** Window used by `stft` (name, or a `Float64Array` of length `nperseg`). Default `"hann"`. */
  readonly window?: WindowSpec;
  /** Segment length used by `stft`. Default `2 * (bins - 1)` for a one-sided input, else `bins` (so pass it for odd segment lengths). */
  readonly nperseg?: number;
  /** Overlap used by `stft`; default `floor(nperseg / 2)`. */
  readonly noverlap?: number;
  /** DFT length used by `stft`; default as SciPy: `nperseg` for a one-sided odd-`nperseg` input, else the inferred segment length. */
  readonly nfft?: number;
  /** `Zxx` holds only the non-negative frequencies (`stft`'s `onesided`). Default `true`. */
  readonly inputOnesided?: boolean;
  /** `Zxx` was made with a boundary extension; drop it from the output. Default `true`. */
  readonly boundary?: boolean;
  /** The `scaling` used by `stft`: `"spectrum"` (default), `"psd"`/`"density"`, or `"none"`. */
  readonly scaling?: StftScaling;
}

/**
 * Inverse of `stft` (`scipy.signal.istft`): `Zxx` is `[bins, numFrames]`; returns
 * `{ times, x }` with `times[i] = i / fs`. With no options it inverts
 * `stft(x)`'s defaults; pass the same options you gave `stft` otherwise
 * (`onesided` there is `inputOnesided` here).
 */
export function istft(Zxx: ComplexTensor, options: IstftOptions = {}): { times: number[]; x: Tensor } {
  if (Zxx.shape.length !== 2) throw new RangeError("istft: expected a 2-D [bins, numFrames] ComplexTensor");
  const [nBins, nseg] = Zxx.shape as [number, number];
  const fs = options.fs ?? 1;
  if (!(fs > 0) || !Number.isFinite(fs)) throw new RangeError(`istft: fs must be a finite number above 0, got ${fs}`);
  const scaling = checkScaling("istft", options.scaling ?? "spectrum");
  const inputOnesided = options.inputOnesided ?? true;
  const nDefault = inputOnesided ? 2 * (nBins - 1) : nBins;

  const spec = options.window ?? "hann";
  const nperseg = options.nperseg ?? (spec instanceof Float64Array ? spec.length : nDefault);
  if (!Number.isInteger(nperseg) || nperseg < 1) throw new RangeError(`istft: nperseg must be a positive integer, got ${nperseg}`);
  const nfft = options.nfft ?? (inputOnesided && nperseg === nDefault + 1 ? nperseg : nDefault);
  if (!Number.isInteger(nfft) || nfft < nperseg) throw new RangeError(`istft: nfft (${nfft}) must be an integer >= nperseg (${nperseg})`);
  const noverlap = options.noverlap ?? Math.floor(nperseg / 2);
  if (!Number.isInteger(noverlap) || noverlap < 0 || noverlap >= nperseg) {
    throw new RangeError(`istft: noverlap (${noverlap}) must be an integer in [0, nperseg) (nperseg=${nperseg})`);
  }
  const hop = nperseg - noverlap;
  let window: Float64Array;
  try {
    window = getWindow(spec, nperseg);
  } catch (e) {
    throw new RangeError(`istft: ${(e as Error).message}`);
  }
  let sumW = 0;
  let sumW2 = 0;
  for (const w of window) {
    sumW += w;
    sumW2 += w * w;
  }
  const scale = scaling === "none" ? 1 : scaling === "spectrum" ? sumW : Math.sqrt(fs * sumW2);

  const zRe = Zxx.real.contiguous().data as Float64Array;
  const zIm = Zxx.imag.contiguous().data as Float64Array;
  const plan = planDft(nfft);
  const outLen = nperseg + (nseg - 1) * hop;
  const out = new Float64Array(outLen);
  const norm = new Float64Array(outLen);
  const fullRe = new Float64Array(nfft);
  const fullIm = new Float64Array(nfft);

  for (let t = 0; t < nseg; t++) {
    fullRe.fill(0);
    fullIm.fill(0);
    if (inputOnesided) {
      // irfft(n=nfft): bins 0 … floor(nfft/2) of the column (zero-padded or truncated), rebuilt Hermitian
      const top = Math.floor(nfft / 2);
      for (let k = 0; k <= top; k++) {
        const re = k < nBins ? (zRe[k * nseg + t] as number) : 0;
        let im = k < nBins ? (zIm[k * nseg + t] as number) : 0;
        if (k === 0 || (nfft % 2 === 0 && k === top)) im = 0; // C2R ignores the imaginary part of DC and Nyquist
        fullRe[k] = re;
        fullIm[k] = im;
        if (k > 0 && nfft - k !== k) {
          fullRe[nfft - k] = re;
          fullIm[nfft - k] = -im;
        }
      }
    } else {
      const keep = Math.min(nBins, nfft); // ifft(n=nfft): truncate or zero-pad the column
      for (let k = 0; k < keep; k++) {
        fullRe[k] = zRe[k * nseg + t] as number;
        fullIm[k] = zIm[k * nseg + t] as number;
      }
    }
    const sub = plan.transform(fullRe, fullIm, true).re;
    const start = t * hop;
    for (let i = 0; i < nperseg; i++) {
      const w = window[i] as number;
      out[start + i] = (out[start + i] as number) + (sub[i] as number) * scale * w;
      norm[start + i] = (norm[start + i] as number) + w * w;
    }
  }

  const trim = options.boundary ?? true ? Math.floor(nperseg / 2) : 0;
  const end = trim > 0 ? outLen - trim : outLen;
  const x = new Float64Array(Math.max(0, end - trim));
  for (let i = trim; i < end; i++) {
    const nrm = norm[i] as number;
    x[i - trim] = nrm > 1e-10 ? (out[i] as number) / nrm : (out[i] as number);
  }
  const times = Array.from({ length: x.length }, (_, i) => i / fs);
  return { times, x: Tensor.fromTypedArray(x, [x.length], { dtype: "f64" }) };
}
