/**
 * stft/istft (issues #44, #178). With no options `stft(x)` is
 * `scipy.signal.stft(x)` and `istft(Zxx)` is `scipy.signal.istft(Zxx)`:
 * fs 1, periodic Hann window, `nperseg = min(256, n)`, 50% overlap,
 * `nfft = nperseg`, `detrend=false`, `scaling="spectrum"` (`X / Σw`),
 * one-sided (`return_onesided=True`), `boundary="zeros"`, `padded=true`.
 *
 * Return shapes follow SciPy's `(f, t, Zxx)` / `(t, x)` as plain objects:
 *  - `stft(x, { axis })` -> `{ freqs: number[], times: number[], Zxx: ComplexTensor }`.
 *    For a 1-D `x`, `Zxx` is `[freqs.length, times.length]` (frequency-major,
 *    like SciPy; the `times` are segment centres, shifted so the first is 0
 *    when `boundary` is set). For N-D input the frequency axis takes the place
 *    of `axis` and the time axis is appended last, exactly as in SciPy:
 *    `[a, n, b]` with `axis: 1` gives `[a, freqs, b, times]`.
 *  - `istft(Zxx, { freqAxis, timeAxis })` -> `{ times: number[], x }`; `x` is
 *    a `Tensor` for a one-sided input and a `ComplexTensor` for
 *    `inputOnesided: false` (as SciPy returns complex there). The frequency
 *    axis is removed from `Zxx`'s shape and the signal takes the place of the
 *    time axis, as in SciPy.
 *
 * Complex input to `stft` (a `ComplexTensor`) is always two-sided, in FFT
 * order; like SciPy, `stft` warns when `onesided` was not turned off.
 *
 * `istft` is weighted overlap-add with window-power normalisation (SciPy's
 * algorithm), exact (to floating-point precision) whenever the window and
 * overlap satisfy the NOLA condition -- the Hann-at-50% default does, and with
 * the default `boundary`/`padded` the whole signal comes back, edges included.
 * When the NOLA condition fails for the given window and overlap (some
 * sample's window power sums to <= 1e-10), `istft` reports it as SciPy's
 * warning: it calls `options.onWarning(message)` if given, else
 * `console.warn(message)`. (SciPy's `stft` does not check NOLA, so neither does
 * this one; see `checkNOLA` to test parameters up front.)
 *
 * `scaling` is SciPy's `"spectrum"` or `"psd"` (`X / sqrt(fs · Σw²)`); anything
 * else is an error, as in SciPy. One-sided bins are NOT doubled (SciPy doesn't either).
 *
 * An empty input to `stft` gives an empty `Zxx` of the input's shape, `freqs: []`
 * and `times: []`, as SciPy does (before it looks at any other option but
 * `scaling` and `boundary`). An `nperseg` longer than the signal is clamped with
 * SciPy's warning (through `onWarning`, else `console.warn`). As in SciPy,
 * `nperseg` defaults to 256 whatever the window is, so an array `window` of
 * another length needs its `nperseg`. For
 * N-D `istft` input, `times` has one entry per output sample (`x`'s length
 * along its time axis); SciPy 1.16 computes it from `x.shape[0]`, which is the
 * wrong length for N-D input unless the time axis comes first.
 */
import { ComplexTensor } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { planDft } from "./dft.ts";
import { checkAxesDistinct, emptyTensor, isComplexTensor, normalizeAxis, rowsAlong, type SignalInput, shapeOf, sizeOf, valuesOf } from "./nd.ts";
import { binFrequencies, type Boundary, emitWarning, frameSpectra, frameTimes, resolveSegments, rethrowWindowError, type SegmentOptions, toInt, type WarningOptions } from "./spectral.ts";
import { getWindow, type WindowSpec } from "./window.ts";

export type StftScaling = "spectrum" | "psd";

export interface StftOptions extends SegmentOptions {
  /** `"spectrum"` (default): `X / Σw`. `"psd"`: `X / sqrt(fs · Σw²)`. (SciPy's two; its `"density"` for `spectrogram` is `spectrogram`'s own option.) */
  readonly scaling?: StftScaling;
  /** Extend the signal by `floor(nperseg / 2)` samples at both ends so the first and last samples sit at a window centre: `"zeros"` (default), `"even"`, `"odd"`, `"constant"`, or `null` for none. */
  readonly boundary?: Boundary;
  /** Zero-pad the (extended) signal to a whole number of segments so no tail is dropped. Default `true`. */
  readonly padded?: boolean;
  /** Axis along which the STFT is computed; default -1 (the last). */
  readonly axis?: number;
}

export interface StftResult {
  /** Bin frequencies (`rfftfreq` one-sided, `fftfreq` order two-sided). */
  readonly freqs: number[];
  /** Segment-centre times in `1/fs` units. */
  readonly times: number[];
  /** The complex spectrogram: the input's shape with `axis` replaced by the frequency axis (`freqs.length`) and the time axis (`times.length`) appended; `[freqs.length, times.length]` for a 1-D signal. */
  readonly Zxx: ComplexTensor;
}

function checkScaling(who: string, scaling: unknown): StftScaling {
  if (scaling !== "psd" && scaling !== "spectrum") {
    throw new RangeError(`${who}: Parameter scaling=${JSON.stringify(scaling)} not in ['spectrum', 'psd']!`);
  }
  return scaling;
}

/** `signal` -> `{ freqs, times, Zxx }`: one `nfft`-point DFT per windowed frame along `axis`; `bins` is `floor(nfft/2)+1` one-sided, else `nfft`. */
export function stft(signal: SignalInput, options: StftOptions = {}): StftResult {
  const scaling = checkScaling("stft", options.scaling ?? "spectrum");
  const boundary = options.boundary === undefined ? "zeros" : options.boundary;
  if (boundary !== null && !["zeros", "even", "odd", "constant"].includes(boundary)) {
    throw new RangeError(`stft: boundary must be null, "zeros", "even", "odd" or "constant", got ${String(boundary)}`);
  }
  const shape = shapeOf(signal);
  if (sizeOf(shape) === 0) {
    // SciPy returns here, before it looks at the window and segment options: three empty arrays shaped like the input
    return { freqs: [], times: [], Zxx: ComplexTensor.fromParts(emptyTensor(shape), emptyTensor(shape)) };
  }
  const rows = rowsAlong("stft", signal, options.axis ?? -1);
  const p = resolveSegments("stft", rows.n, rows.im !== null, { ...options, nperseg: options.nperseg ?? 256 }, {
    defaultDetrend: false,
    boundary,
    padded: options.padded ?? true,
    warnComplexOnesided: true,
    family: "helper",
  });
  const scale = scaling === "spectrum" ? 1 / p.sumW : 1 / Math.sqrt(p.fs * p.sumW2);
  const { re, im, shape: outShape } = frameSpectra("stft", rows, p, scale, false);
  return {
    freqs: binFrequencies(p.nfft, p.fs, p.onesided),
    times: frameTimes(p),
    Zxx: ComplexTensor.fromParts(
      Tensor.fromTypedArray(re, outShape, { dtype: "f64" }),
      Tensor.fromTypedArray(im as Float64Array, outShape, { dtype: "f64" }),
    ),
  };
}

export interface IstftOptions extends WarningOptions {
  /** Sample rate; default 1. */
  readonly fs?: number;
  /** Window used by `stft` (a name, a tuple or a `Float64Array` of length `nperseg`; a bare number is not one, as in SciPy). Default `"hann"`. */
  readonly window?: WindowSpec;
  /** Segment length used by `stft`. Default `2 * (bins - 1)` for a one-sided input, else `bins` (so pass it for odd segment lengths). */
  readonly nperseg?: number;
  /** Overlap used by `stft`; default `floor(nperseg / 2)`. */
  readonly noverlap?: number;
  /** DFT length used by `stft`; default as SciPy: `nperseg` for a one-sided odd-`nperseg` input, else the inferred segment length. */
  readonly nfft?: number;
  /** `Zxx` holds only the non-negative frequencies (`stft`'s `onesided`). Default `true`. When `false` the result `x` is complex, as in SciPy. */
  readonly inputOnesided?: boolean;
  /** `Zxx` was made with a boundary extension; drop it from the output. Default `true`. */
  readonly boundary?: boolean;
  /** The `scaling` used by `stft`: `"spectrum"` (default) or `"psd"`. */
  readonly scaling?: StftScaling;
  /** Axis of `Zxx` holding the frequencies; default -2. */
  readonly freqAxis?: number;
  /** Axis of `Zxx` holding the segment index (time); default -1. */
  readonly timeAxis?: number;
}

export interface IstftResult<X extends Tensor | ComplexTensor = Tensor | ComplexTensor> {
  /** Sample times `i / fs`, one per output sample. */
  readonly times: number[];
  /** The signal: `Zxx`'s shape without the frequency axis, the signal taking the place of the time axis. A `Tensor` for a one-sided input, a `ComplexTensor` for `inputOnesided: false`. */
  readonly x: X;
}

/**
 * Inverse of `stft` (`scipy.signal.istft`): `Zxx` is a `ComplexTensor` (or a real `Tensor`) of at least 2 dimensions,
 * `[bins, numFrames]` by default (`freqAxis` -2, `timeAxis` -1). With no options it inverts `stft(x)`'s defaults;
 * pass the same options you gave `stft` otherwise (`onesided` there is `inputOnesided` here).
 */
export function istft(Zxx: SignalInput, options: IstftOptions & { readonly inputOnesided: false }): IstftResult<ComplexTensor>;
export function istft(Zxx: SignalInput, options?: IstftOptions & { readonly inputOnesided?: true }): IstftResult<Tensor>;
export function istft(Zxx: SignalInput, options?: IstftOptions): IstftResult;
export function istft(Zxx: SignalInput, options: IstftOptions = {}): IstftResult {
  const complexIn = isComplexTensor(Zxx);
  const shape = (complexIn ? Zxx.real : Zxx).shape as readonly number[];
  const ndim = shape.length;
  if (ndim < 2) throw new RangeError("istft: expected a ComplexTensor (or real Tensor) spectrogram of at least 2 dimensions (frequency x time)");
  const fa = normalizeAxis("istft", options.freqAxis ?? -2, ndim, "freqAxis");
  const ta = normalizeAxis("istft", options.timeAxis ?? -1, ndim, "timeAxis");
  checkAxesDistinct("istft", fa, ta);
  const nBins = shape[fa] as number;
  const nseg = shape[ta] as number;
  const fs = options.fs ?? 1;
  if (!(fs > 0) || !Number.isFinite(fs)) throw new RangeError(`istft: fs must be a finite number above 0, got ${fs}`);
  const scaling = checkScaling("istft", options.scaling ?? "spectrum");
  const inputOnesided = options.inputOnesided ?? true;
  const nDefault = inputOnesided ? 2 * (nBins - 1) : nBins;

  const spec = options.window ?? "hann";
  const given = options.nperseg === undefined ? undefined : toInt("istft", "nperseg", options.nperseg);
  if (given !== undefined && given < 1) throw new RangeError("istft: nperseg must be a positive integer");
  const nperseg = given ?? nDefault;
  if (!Number.isInteger(nperseg) || nperseg < 1) throw new RangeError(`istft: nperseg must be a positive integer, got ${nperseg}`);
  const nfft = options.nfft === undefined ? (inputOnesided && nperseg === nDefault + 1 ? nperseg : nDefault) : toInt("istft", "nfft", options.nfft);
  if (nfft < nperseg) throw new RangeError(`istft: nfft=${nfft} must be greater than or equal to nperseg=${nperseg}.`);
  const noverlap = options.noverlap === undefined ? Math.floor(nperseg / 2) : toInt("istft", "noverlap", options.noverlap);
  if (noverlap >= nperseg) throw new RangeError("istft: noverlap must be less than nperseg.");
  const hop = nperseg - noverlap;
  if (typeof spec === "number") throw new RangeError("istft: window must be 1-D"); // np.asarray(4.0).shape is ()
  let window: Float64Array;
  try {
    window = getWindow(spec, nperseg, true, options);
  } catch (e) {
    return rethrowWindowError("istft", e);
  }
  let sumW = 0;
  let sumW2 = 0;
  for (const w of window) {
    sumW += w;
    sumW2 += w * w;
  }
  const scale = scaling === "spectrum" ? sumW : Math.sqrt(fs * sumW2);

  const zRe = valuesOf(complexIn ? Zxx.real : Zxx);
  const zIm = complexIn ? valuesOf(Zxx.imag) : null;
  const plan = planDft(nfft);
  const outLen = nperseg + (nseg - 1) * hop;

  // Window power: the same for every row, so the NOLA check and the divisor are computed once.
  const norm = new Float64Array(outLen);
  for (let t = 0; t < nseg; t++) {
    for (let i = 0; i < nperseg; i++) norm[t * hop + i] = (norm[t * hop + i] as number) + (window[i] as number) ** 2;
  }
  const boundaryOn = options.boundary ?? true;
  const trim = boundaryOn ? Math.floor(nperseg / 2) : 0;
  const end = boundaryOn && trim === 0 ? 0 : outLen - trim; // SciPy slices x[nperseg//2:-(nperseg//2)]: for nperseg = 1 that is x[0:-0], which is empty
  const L = Math.max(0, end - trim);
  let nolaOk = true;
  for (let i = trim; i < end; i++) if (!((norm[i] as number) > 1e-10)) nolaOk = false;
  if (!nolaOk) {
    emitWarning(options, `NOLA condition failed, STFT may not be invertible.${boundaryOn ? "" : " Possibly due to missing boundary"}`);
  }

  // Strides of Zxx (row-major), the outer axes in order, and the output layout (outer axes with the signal at `timeOut`).
  const strides = new Array<number>(ndim);
  let acc = 1;
  for (let d = ndim - 1; d >= 0; d--) {
    strides[d] = acc;
    acc *= shape[d] as number;
  }
  const outer = shape.map((_, d) => d).filter((d) => d !== fa && d !== ta);
  const outerDims = outer.map((d) => shape[d] as number);
  const timeOut = ta - (fa < ta ? 1 : 0);
  const outShape = outerDims.slice();
  outShape.splice(timeOut, 0, L);
  const outStrides = new Array<number>(outShape.length);
  acc = 1;
  for (let d = outShape.length - 1; d >= 0; d--) {
    outStrides[d] = acc;
    acc *= outShape[d] as number;
  }
  const outerOutStride = outer.map((_, k) => outStrides[k < timeOut ? k : k + 1] as number);
  const lStride = outStrides[timeOut] as number;
  const rows = outerDims.reduce((a, b) => a * b, 1);

  const outRe = new Float64Array(acc);
  const outIm = inputOnesided ? null : new Float64Array(acc);
  const sRe = new Float64Array(outLen);
  const sIm = new Float64Array(outLen);
  const fullRe = new Float64Array(nfft);
  const fullIm = new Float64Array(nfft);
  const idx = new Array<number>(outer.length).fill(0);

  for (let r = 0; r < rows; r++) {
    let inBase = 0;
    let outBase = 0;
    for (let k = 0; k < outer.length; k++) {
      inBase += (idx[k] as number) * (strides[outer[k] as number] as number);
      outBase += (idx[k] as number) * (outerOutStride[k] as number);
    }
    sRe.fill(0);
    sIm.fill(0);
    for (let t = 0; t < nseg; t++) {
      fullRe.fill(0);
      fullIm.fill(0);
      const col = inBase + t * (strides[ta] as number);
      const fStride = strides[fa] as number;
      if (inputOnesided) {
        // irfft(n=nfft): bins 0 … floor(nfft/2) of the column (zero-padded or truncated), rebuilt Hermitian
        const top = Math.floor(nfft / 2);
        for (let k = 0; k <= top; k++) {
          const re = k < nBins ? (zRe[col + k * fStride] as number) : 0;
          let im = k < nBins && zIm !== null ? (zIm[col + k * fStride] as number) : 0;
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
          fullRe[k] = zRe[col + k * fStride] as number;
          fullIm[k] = zIm === null ? 0 : (zIm[col + k * fStride] as number);
        }
      }
      const sub = plan.transform(fullRe, fullIm, true);
      const start = t * hop;
      for (let i = 0; i < nperseg; i++) {
        const w = (window[i] as number) * scale;
        sRe[start + i] = (sRe[start + i] as number) + (sub.re[i] as number) * w;
        if (!inputOnesided) sIm[start + i] = (sIm[start + i] as number) + (sub.im[i] as number) * w;
      }
    }
    for (let i = trim; i < end; i++) {
      const nrm = norm[i] as number;
      const div = nrm > 1e-10 ? nrm : 1;
      outRe[outBase + (i - trim) * lStride] = (sRe[i] as number) / div;
      if (outIm !== null) outIm[outBase + (i - trim) * lStride] = (sIm[i] as number) / div;
    }
    for (let k = outer.length - 1; k >= 0; k--) {
      idx[k] = (idx[k] as number) + 1;
      if ((idx[k] as number) < (outerDims[k] as number)) break;
      idx[k] = 0;
    }
  }
  const times = Array.from({ length: L }, (_, i) => i / fs);
  const re = Tensor.fromTypedArray(outRe, outShape, { dtype: "f64" });
  if (outIm === null) return { times, x: re };
  return { times, x: ComplexTensor.fromParts(re, Tensor.fromTypedArray(outIm, outShape, { dtype: "f64" })) };
}
