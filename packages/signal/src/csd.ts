/**
 * csd (SciPy parity): the cross power spectral density by Welch's method, and the one routine `welch`, `periodogram`
 * and `coherence` run on. As in SciPy 1.16 `welch(x)` is `csd(x, x).real`, and what is shared here is SciPy's own
 * order of work: option checks, zero-padding the shorter input, the windowed DFT of every segment of `x` and of `y`
 * (`frameSpectra`), `conj(X) * Y` with NumPy broadcasting, the one-sided doubling, then the mean (or median) over the
 * segments.
 *
 * Input shapes. `x` and `y` may be real or complex and of any dimension, with the transform axis `axis` of each. Their
 * other dimensions must broadcast together (`x` of `[3, n]` with `y` of `[n]`, or of `[1, n]`, works). If the lengths
 * along `axis` differ, the shorter is zero-padded to the longer (SciPy concatenates zeros shaped like the *other*
 * array, so then the other dimensions must match exactly, not just broadcast). The segment spectra are broadcast
 * against each other as NumPy would (SciPy multiplies them with their frequency and segment axes still in place), so
 * a negative `axis`, or `axis` the same in both, gives the result you expect; mixed-rank inputs with a positive `axis`
 * reproduce SciPy's own result or its broadcasting error, whichever it gives.
 *
 * `Pxy` is a `ComplexTensor` (`conj(X) * Y`, SciPy's convention) except when `y` is the very same object as `x`
 * (`csd(x, x)`, as `welch` does), where SciPy returns the real part and so does this.
 */
import { ComplexTensor } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import {
  broadcastMap,
  broadcastShapes,
  normalizeAxis,
  type Rows,
  rowsAlong,
  type SignalInput,
  shapeOf,
  sizeOf,
} from "./nd.ts";
import { binFrequencies, frameSpectra, resolveSegments, type SegmentOptions, toInt } from "./spectral.ts";

export interface CsdOptions extends SegmentOptions {
  /** `"density"` (default): power per unit frequency, `|X|² / (fs · Σw²)`. `"spectrum"`: power per bin, `|X|² / (Σw)²`. */
  readonly scaling?: "density" | "spectrum";
  /** How the segments' spectra are combined: `"mean"` (default) or `"median"` (bias-corrected, robust to outlier segments). Only checked when there is more than one segment, as in SciPy. */
  readonly average?: "mean" | "median";
  /** Axis along which the spectrum is computed; default -1 (the last). */
  readonly axis?: number;
}

export interface CsdResult {
  /** Bin frequencies (`rfftfreq` one-sided, `fftfreq` order two-sided), SciPy's `f`. Empty (`[]`) for an empty input, where SciPy returns an empty array shaped like the input. */
  readonly freqs: number[];
  /** The cross spectral density, SciPy's `Pxy`: the (broadcast) input shape with `axis` replaced by `freqs.length`. A `ComplexTensor`, or a real `Tensor` when `y` is `x` itself. */
  readonly Pxy: Tensor | ComplexTensor;
}

/** `scipy.signal.welch`'s `_median_bias`: the median of n exponential variables is biased low by this factor. */
export function medianBias(n: number): number {
  let bias = 1;
  for (let i = 1; i <= Math.floor((n - 1) / 2); i++) bias += 1 / (2 * i + 1) - 1 / (2 * i);
  return bias;
}

/** Pad every row of `rows` with zeros to `n` samples. */
function padRows(rows: Rows, n: number): Rows {
  const pad = (src: Float64Array): Float64Array => {
    const out = new Float64Array(rows.rows * n);
    for (let r = 0; r < rows.rows; r++) out.set(src.subarray(r * rows.n, (r + 1) * rows.n), r * n);
    return out;
  };
  const shape = rows.shape.slice();
  shape[rows.axis] = n;
  return { ...rows, re: pad(rows.re), im: rows.im === null ? null : pad(rows.im), n, shape };
}

const shapeText = (s: readonly number[]) => `(${s.join(",")}${s.length === 1 ? "," : ""})`;

export interface CsdCore {
  readonly freqs: number[];
  readonly shape: number[];
  readonly re: Float64Array;
  /** `null` for a real result (`x` is `y`). */
  readonly im: Float64Array | null;
}

/** The shared routine. `y` is `null` or `x` itself for the auto-spectrum (real result). */
export function csdCore(who: string, x: SignalInput, y: SignalInput | null, options: CsdOptions): CsdCore {
  const sameData = y === null || y === x;
  const axisArg = toInt(who, "axis", options.axis ?? -1);
  const xShape = shapeOf(x);
  const yShape = sameData ? xShape : shapeOf(y as SignalInput);
  const xAxis = normalizeAxis(who, axisArg, xShape.length);
  const yAxis = normalizeAxis(who, axisArg, yShape.length);
  const empty = (shape: readonly number[]): CsdCore => ({ freqs: [], shape: [...shape], re: new Float64Array(0), im: null });

  if (sameData) {
    if (sizeOf(xShape) === 0) return empty(xShape);
  } else {
    const xOuter = xShape.filter((_, d) => d !== xAxis);
    const yOuter = yShape.filter((_, d) => d !== yAxis);
    const outer = broadcastShapes(xOuter, yOuter);
    if (outer === null) throw new RangeError(`${who}: x and y cannot be broadcast together.`);
    if (sizeOf(xShape) === 0 || sizeOf(yShape) === 0) {
      const out = [...outer, Math.min(xShape[xAxis] as number, yShape[yAxis] as number)];
      const at = normalizeAxis(who, axisArg, out.length);
      const last = out.pop() as number;
      out.splice(at, 0, last); // np.moveaxis(empty, -1, axis)
      return empty(out);
    }
  }

  let rx = rowsAlong(who, x, axisArg);
  let ry = sameData ? null : rowsAlong(who, y as SignalInput, axisArg);
  const n = ry === null ? rx.n : Math.max(rx.n, ry.n);
  const p = resolveSegments(who, n, rx.im !== null, options, { defaultDetrend: "constant", family: "csd" });
  if (ry !== null && ry.im !== null && rx.im === null && p.onesided) {
    throw new RangeError(`${who}: Complex-valued \`x\` not allowed for self.fft_mode='onesided''! Set property \`fft_mode\` to 'twosided' or 'centered'.`);
  }
  if (ry !== null && rx.n !== ry.n) {
    const same = xShape.length === yShape.length && xShape.every((d, i) => i === xAxis || d === yShape[i]);
    if (!same) throw new RangeError(`${who}: all the input array dimensions except for the concatenation axis must match exactly`);
    if (rx.n < ry.n) rx = padRows(rx, ry.n);
    else ry = padRows(ry, rx.n);
  }
  const scaling = options.scaling ?? "density";
  if (scaling !== "density" && scaling !== "spectrum") {
    throw new RangeError(`${who}: Parameter scaling=${JSON.stringify(scaling)} not in {'spectrum': 'magnitude', 'density': 'psd'}!`);
  }
  // each side carries the square root of the power scale: 1 / sqrt(fs Σw²) for a density, 1 / Σw for a spectrum
  const amp = scaling === "density" ? 1 / Math.sqrt(p.fs * p.sumW2) : 1 / p.sumW;
  const sx = frameSpectra(who, rx, p, amp, sameData);
  let shape: number[];
  let re: Float64Array;
  let im: Float64Array | null;
  if (sameData) {
    shape = sx.shape;
    re = sx.re;
    im = null;
  } else {
    const sy = frameSpectra(who, ry as Rows, p, amp, false);
    // SciPy: spectrogram(y, x) -> stft(y) * conj(stft(x))
    const out = broadcastShapes(sy.shape, sx.shape);
    if (out === null) throw new RangeError(`${who}: operands could not be broadcast together with shapes ${shapeText(sy.shape)} ${shapeText(sx.shape)}`);
    const mapY = broadcastMap(out, sy.shape);
    const mapX = broadcastMap(out, sx.shape);
    const total = sizeOf(out);
    re = new Float64Array(total);
    im = new Float64Array(total);
    const yr = sy.re;
    const yi = sy.im as Float64Array;
    const xr = sx.re;
    const xi = sx.im as Float64Array;
    for (let i = 0; i < total; i++) {
      const a = yr[mapY[i] as number] as number;
      const b = yi[mapY[i] as number] as number;
      const c = xr[mapX[i] as number] as number;
      const d = -(xi[mapX[i] as number] as number);
      re[i] = a * c - b * d;
      im[i] = a * d + b * c;
    }
    shape = out;
  }

  // One-sided: fold the negative frequencies in -- every bin but DC (and Nyquist, for even nfft) counts twice.
  const nd = shape.length;
  if (p.onesided) {
    const fAxis = axisArg < 0 ? nd - 1 + axisArg : axisArg;
    if (fAxis < 0 || fAxis >= nd) throw new RangeError(`${who}: axis ${axisArg} is out of bounds for a ${nd}-D array`);
    const fLen = shape[fAxis] as number;
    const last = p.nfft % 2 === 0 ? fLen - 1 : fLen; // [1:-1] for even nfft, [1:] for odd
    let inner = 1;
    for (let d = fAxis + 1; d < nd; d++) inner *= shape[d] as number;
    for (let i = 0; i < re.length; i++) {
      const k = Math.floor(i / inner) % fLen;
      if (k >= 1 && k < last) {
        re[i] = (re[i] as number) * 2;
        if (im !== null) im[i] = (im[i] as number) * 2;
      }
    }
  }

  // Average over the segments (the last axis).
  const frames = shape[nd - 1] as number;
  const outShape = shape.slice(0, nd - 1);
  if (frames <= 1) return { freqs: binFrequencies(p.nfft, p.fs, p.onesided), shape: outShape, re, im };
  const average = options.average ?? "mean";
  if (average !== "mean" && average !== "median") throw new RangeError(`${who}: Parameter average=${JSON.stringify(average)} must be 'median' or 'mean'!`);
  const groups = re.length / frames;
  const reduce = (src: Float64Array): Float64Array => {
    const out = new Float64Array(groups);
    if (average === "mean") {
      for (let g = 0; g < groups; g++) {
        let s = 0;
        for (let f = 0; f < frames; f++) s += src[g * frames + f] as number;
        out[g] = s / frames;
      }
    } else {
      const bias = medianBias(frames);
      const column = new Float64Array(frames);
      const mid = frames >> 1;
      for (let g = 0; g < groups; g++) {
        column.set(src.subarray(g * frames, (g + 1) * frames));
        column.sort();
        const median = frames % 2 ? (column[mid] as number) : ((column[mid - 1] as number) + (column[mid] as number)) / 2;
        out[g] = median / bias;
      }
    }
    return out;
  };
  return { freqs: binFrequencies(p.nfft, p.fs, p.onesided), shape: outShape, re: reduce(re), im: im === null ? null : reduce(im) };
}

/** Wrap a `csdCore` result as a real `Tensor` (the imaginary part, if any, dropped). */
export function realResult(core: CsdCore): Tensor {
  return Tensor.fromTypedArray(core.re, core.shape, { dtype: "f64" });
}

/**
 * `scipy.signal.csd(x, y, ...)`: the cross spectral density `Pxy = mean over segments of conj(X) · Y`. Same options
 * as {@link welch}. See the module comment for the accepted shapes. `csd(x, x)` (the same object) is `welch(x)`
 * with a real `Pxy`.
 */
export function csd(x: SignalInput, y: SignalInput, options: CsdOptions = {}): CsdResult {
  const core = csdCore("csd", x, y, options);
  if (core.im === null) return { freqs: core.freqs, Pxy: realResult(core) };
  return {
    freqs: core.freqs,
    Pxy: ComplexTensor.fromParts(
      Tensor.fromTypedArray(core.re, core.shape, { dtype: "f64" }),
      Tensor.fromTypedArray(core.im, core.shape, { dtype: "f64" }),
    ),
  };
}
