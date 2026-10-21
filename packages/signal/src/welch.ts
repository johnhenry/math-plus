/**
 * welch (issues #70, #178) — power spectral density via Welch's method:
 * the average of `|X_i(f)|^2` over windowed, overlapping segments, with
 * `scipy.signal.welch`'s meanings for `fs`, `window`, `nperseg`, `noverlap`,
 * `nfft`, `detrend`, `scaling` and `onesided`.
 *
 * Defaults keep the original behaviour: fs 1, periodic Hann, two-sided
 * (`return_onesided=False`), density scaling, NO detrend (SciPy's own
 * default is `"constant"`), `nfft` = next power of two at or above `nperseg`.
 * Disclosed limits: 1-D real input only; `nfft` must be a power of two
 * (the FFT package's requirement); no `average="median"`.
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { binFrequencies, resolveSegments, segmentSpectrum, type SegmentOptions } from "./spectral.ts";

export interface WelchOptions extends SegmentOptions {
  /** `"density"` (default): power per unit frequency, `|X|² / (fs · Σw²)`. `"spectrum"`: power per bin, `|X|² / (Σw)²`. */
  readonly scaling?: "density" | "spectrum";
}

export function welch(signal: Tensor, options: WelchOptions = {}): { frequencies: number[]; psd: Tensor } {
  if (signal.shape.length !== 1) throw new RangeError("welch: v1 supports 1-D Tensor only");
  const scaling = options.scaling ?? "density";
  if (scaling !== "density" && scaling !== "spectrum") {
    throw new RangeError(`welch: scaling must be "density" or "spectrum", got ${String(scaling)}`);
  }
  // Read-only below, so no defensive copy is needed even when `.data`
  // aliases `signal`'s own storage.
  const data = signal.contiguous().data as Float64Array;
  const p = resolveSegments("welch", data.length, options);
  const { nfft } = p;

  const accumulator = new Float64Array(nfft);
  for (let f = 0; f < p.numFrames; f++) {
    const { real, imag } = segmentSpectrum(data, f * p.hop, p);
    for (let k = 0; k < nfft; k++) {
      accumulator[k] = (accumulator[k] as number) + ((real[k] as number) ** 2 + (imag[k] as number) ** 2);
    }
  }

  const scale = (scaling === "density" ? 1 / (p.fs * p.sumW2) : 1 / (p.sumW * p.sumW)) / p.numFrames;
  const bins = p.onesided ? nfft / 2 + 1 : nfft;
  const psdData = new Float64Array(bins);
  for (let k = 0; k < bins; k++) {
    let v = (accumulator[k] as number) * scale;
    // One-sided: fold the negative frequencies in (everything but DC and Nyquist counts twice).
    if (p.onesided && k > 0 && k < nfft / 2) v *= 2;
    psdData[k] = v;
  }
  return { frequencies: binFrequencies(nfft, p.fs, p.onesided), psd: Tensor.fromTypedArray(psdData, [bins], { dtype: "f64" }) };
}
