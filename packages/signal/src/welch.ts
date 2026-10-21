/**
 * welch (issues #70, #178) — power spectral density via Welch's method:
 * the average of `|X_i(f)|^2` over windowed, overlapping segments. With no
 * options `welch(x)` is `scipy.signal.welch(x)`: fs 1, periodic Hann window,
 * `nperseg = min(256, n)`, 50% overlap, `nfft = nperseg`, `detrend="constant"`,
 * `scaling="density"`, one-sided (`return_onesided=True`), `average="mean"`.
 *
 * Disclosed limits (not implemented): 1-D real input only (no `axis`, no
 * complex input, so no complex-forced two-sided mode); `window` is a name or
 * an array, not a parameterised `(name, param)` tuple; `detrend` is
 * `false`/`"constant"`/`"linear"`, not a callable.
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { binFrequencies, resolveSegments, segmentSpectrum, type SegmentOptions } from "./spectral.ts";

export interface WelchOptions extends SegmentOptions {
  /** `"density"` (default): power per unit frequency, `|X|² / (fs · Σw²)`. `"spectrum"`: power per bin, `|X|² / (Σw)²`. */
  readonly scaling?: "density" | "spectrum";
  /** How the segments' periodograms are combined: `"mean"` (default) or `"median"` (bias-corrected, robust to outlier segments). */
  readonly average?: "mean" | "median";
}

/** `scipy.signal.welch`'s `_median_bias`: the median of n exponential variables is biased low by this factor. */
function medianBias(n: number): number {
  let bias = 1;
  for (let i = 1; i <= Math.floor((n - 1) / 2); i++) bias += 1 / (2 * i + 1) - 1 / (2 * i);
  return bias;
}

export function welch(signal: Tensor, options: WelchOptions = {}): { frequencies: number[]; psd: Tensor } {
  if (signal.shape.length !== 1) throw new RangeError("welch: 1-D Tensor only");
  const scaling = options.scaling ?? "density";
  if (scaling !== "density" && scaling !== "spectrum") {
    throw new RangeError(`welch: scaling must be "density" or "spectrum", got ${String(scaling)}`);
  }
  const average = options.average ?? "mean";
  if (average !== "mean" && average !== "median") {
    throw new RangeError(`welch: average must be "mean" or "median", got ${String(average)}`);
  }
  // Read-only below, so no defensive copy is needed even when `.data`
  // aliases `signal`'s own storage.
  const data = signal.contiguous().data as Float64Array;
  const p = resolveSegments("welch", data, options, { defaultDetrend: "constant" });
  const { nfft, bins } = p;

  const scale = scaling === "density" ? 1 / (p.fs * p.sumW2) : 1 / (p.sumW * p.sumW);
  // One-sided: fold the negative frequencies in -- every bin but DC (and Nyquist, for even nfft) counts twice.
  const lastDoubled = nfft % 2 ? bins - 1 : bins - 2;
  const keepAll = average === "median"; // only the median needs every segment's periodogram; the mean streams
  const periodograms: Float64Array[] = [];
  const psdData = new Float64Array(bins);
  for (let f = 0; f < p.numFrames; f++) {
    const { re, im } = segmentSpectrum(f * p.hop, p);
    const pw = new Float64Array(bins);
    for (let k = 0; k < bins; k++) {
      let v = ((re[k] as number) ** 2 + (im[k] as number) ** 2) * scale;
      if (p.onesided && k >= 1 && k <= lastDoubled) v *= 2;
      pw[k] = v;
    }
    if (keepAll) periodograms.push(pw);
    else for (let k = 0; k < bins; k++) psdData[k] = (psdData[k] as number) + (pw[k] as number);
  }

  const n = p.numFrames;
  if (!keepAll) {
    for (let k = 0; k < bins; k++) psdData[k] = (psdData[k] as number) / n;
  } else if (n === 1) {
    psdData.set(periodograms[0] as Float64Array); // a single segment is returned as is (no bias correction)
  } else {
    const bias = medianBias(n);
    const column = new Float64Array(n);
    for (let k = 0; k < bins; k++) {
      for (let f = 0; f < n; f++) column[f] = (periodograms[f] as Float64Array)[k] as number;
      column.sort();
      const mid = n >> 1;
      const median = n % 2 ? (column[mid] as number) : ((column[mid - 1] as number) + (column[mid] as number)) / 2;
      psdData[k] = median / bias;
    }
  }
  return { frequencies: binFrequencies(nfft, p.fs, p.onesided), psd: Tensor.fromTypedArray(psdData, [bins], { dtype: "f64" }) };
}
