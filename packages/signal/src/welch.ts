/**
 * welch (issues #70, #178) — power spectral density via Welch's method:
 * the average of `|X_i(f)|^2` over windowed, overlapping segments. With no
 * options `welch(x)` is `scipy.signal.welch(x)`: fs 1, periodic Hann window,
 * `nperseg = min(256, n)`, 50% overlap, `nfft = nperseg`, `detrend="constant"`,
 * `scaling="density"`, one-sided (`return_onesided=True`), `average="mean"`,
 * `axis=-1`.
 *
 * N-D input is handled along `axis` (default -1): the result has the input's
 * shape with that axis replaced by the frequency axis, as SciPy's does. Complex
 * input (a `ComplexTensor`) is always two-sided, in FFT order, and silently so
 * (SciPy's `welch` does not warn, unlike its `stft`).
 *
 * Disclosed limits: an empty input throws (SciPy returns empty arrays), and
 * SciPy's warning for an `nperseg` longer than the signal is not emitted (it is
 * clamped, silently).
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { rowsAlong, type SignalInput, withSize } from "./nd.ts";
import { binFrequencies, prepareRow, resolveSegments, segmentSpectrum, type SegmentOptions } from "./spectral.ts";

export interface WelchOptions extends SegmentOptions {
  /** `"density"` (default): power per unit frequency, `|X|² / (fs · Σw²)`. `"spectrum"`: power per bin, `|X|² / (Σw)²`. */
  readonly scaling?: "density" | "spectrum";
  /** How the segments' periodograms are combined: `"mean"` (default) or `"median"` (bias-corrected, robust to outlier segments). */
  readonly average?: "mean" | "median";
  /** Axis along which the spectrum is computed; default -1 (the last). */
  readonly axis?: number;
}

export interface WelchResult {
  /** Bin frequencies (`rfftfreq` one-sided, `fftfreq` order two-sided), SciPy's `f`. */
  readonly freqs: number[];
  /** The power spectral density (or spectrum), real, SciPy's `Pxx`: the input's shape with `axis` replaced by `freqs.length`. */
  readonly Pxx: Tensor;
}

/** `scipy.signal.welch`'s `_median_bias`: the median of n exponential variables is biased low by this factor. */
function medianBias(n: number): number {
  let bias = 1;
  for (let i = 1; i <= Math.floor((n - 1) / 2); i++) bias += 1 / (2 * i + 1) - 1 / (2 * i);
  return bias;
}

export function welch(signal: SignalInput, options: WelchOptions = {}): WelchResult {
  const scaling = options.scaling ?? "density";
  if (scaling !== "density" && scaling !== "spectrum") {
    throw new RangeError(`welch: scaling must be "density" or "spectrum", got ${String(scaling)}`);
  }
  const average = options.average ?? "mean";
  if (average !== "mean" && average !== "median") {
    throw new RangeError(`welch: average must be "mean" or "median", got ${String(average)}`);
  }
  const rows = rowsAlong("welch", signal, options.axis ?? -1);
  const p = resolveSegments("welch", rows.n, rows.im !== null, options, { defaultDetrend: "constant" });
  const { nfft, bins } = p;

  const scale = scaling === "density" ? 1 / (p.fs * p.sumW2) : 1 / (p.sumW * p.sumW);
  // One-sided: fold the negative frequencies in -- every bin but DC (and Nyquist, for even nfft) counts twice.
  const lastDoubled = nfft % 2 ? bins - 1 : bins - 2;
  const keepAll = average === "median"; // only the median needs every segment's periodogram; the mean streams
  const n = p.numFrames;
  const bias = keepAll && n > 1 ? medianBias(n) : 1;
  const out = new Float64Array(rows.pre * bins * rows.inner);
  const acc = new Float64Array(bins);
  const periodograms: Float64Array[] = [];
  const column = new Float64Array(n);

  for (let r = 0; r < rows.rows; r++) {
    const row = prepareRow("welch", p, rows.re.subarray(r * rows.n, (r + 1) * rows.n), rows.im === null ? null : rows.im.subarray(r * rows.n, (r + 1) * rows.n));
    acc.fill(0);
    periodograms.length = 0;
    for (let f = 0; f < n; f++) {
      const { re, im } = segmentSpectrum("welch", f * p.hop, p, row);
      const pw = keepAll ? new Float64Array(bins) : acc;
      for (let k = 0; k < bins; k++) {
        let v = ((re[k] as number) ** 2 + (im[k] as number) ** 2) * scale;
        if (p.onesided && k >= 1 && k <= lastDoubled) v *= 2;
        if (keepAll) pw[k] = v;
        else pw[k] = (pw[k] as number) + v;
      }
      if (keepAll) periodograms.push(pw);
    }
    const base = Math.floor(r / rows.inner) * bins * rows.inner + (r % rows.inner);
    for (let k = 0; k < bins; k++) {
      let value: number;
      if (!keepAll) {
        value = (acc[k] as number) / n;
      } else if (n === 1) {
        value = (periodograms[0] as Float64Array)[k] as number; // a single segment is returned as is (no bias correction)
      } else {
        for (let f = 0; f < n; f++) column[f] = (periodograms[f] as Float64Array)[k] as number;
        column.sort();
        const mid = n >> 1;
        const median = n % 2 ? (column[mid] as number) : ((column[mid - 1] as number) + (column[mid] as number)) / 2;
        value = median / bias;
      }
      out[base + k * rows.inner] = value;
    }
  }
  return {
    freqs: binFrequencies(nfft, p.fs, p.onesided),
    Pxx: Tensor.fromTypedArray(out, withSize(rows.shape, rows.axis, bins, true), { dtype: "f64" }),
  };
}
