/**
 * stft/istft (issues #44, #178). Standard windowed short-time Fourier
 * transform: each frame is `window * signal[start:start+nperseg]` (optionally
 * detrended first), zero-padded to `nfft` and transformed via
 * @johnhenry/math-plus-fft's `fft`; `istft` reconstructs via weighted
 * overlap-add (WOLA) with window-power normalization -- the standard COLA
 * reconstruction, exact (to floating-point precision) for the default
 * Hann-window-at-50%-overlap configuration.
 *
 * `stft` returns the `[numFrames, bins]` ComplexTensor it always did, now
 * also carrying `freqs`, `times` (segment centres, `(start + nperseg/2) / fs`)
 * and `frames` (the tensor itself), so `const { freqs, times, frames } =
 * stft(x, opts)` works. Options follow `scipy.signal.stft` with
 * `boundary=None, padded=False` (no edge extension: samples past the last
 * full frame are dropped); `scaling` adds `"none"` (default, the raw FFT)
 * to SciPy's `"spectrum"` (`X / Σw`) and `"density"` (`X / sqrt(fs · Σw²)`,
 * SciPy's `"psd"`). Unlike `welch`, one-sided bins are NOT doubled
 * (SciPy doesn't either).
 *
 * Disclosed limits: `nfft` must be a power of two (the FFT package's
 * requirement). `istft` only inverts a two-sided, un-padded (`nfft ===
 * nperseg`), un-detrended, `scaling: "none"` spectrogram and throws on a
 * one-sided or zero-padded one.
 */
import { ComplexTensor, ifft } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { binFrequencies, resolveSegments, segmentSpectrum, type SegmentOptions } from "./spectral.ts";
import { getWindow } from "./window.ts";

export interface StftOptions extends SegmentOptions {
  /** `"none"` (default): raw FFT of each windowed frame. `"spectrum"`: divided by `Σw`. `"density"`: divided by `sqrt(fs · Σw²)`. */
  readonly scaling?: "none" | "density" | "spectrum";
}

/** What `stft` returns: the `[numFrames, bins]` complex spectrogram, plus its axes. */
export type Spectrogram = ComplexTensor & {
  readonly freqs: number[];
  readonly times: number[];
  readonly frames: ComplexTensor;
};

/** `signal[T]` -> a `[numFrames, bins]` Spectrogram (one `nfft`-point FFT per windowed frame; `bins` is `nfft`, or `nfft/2+1` one-sided). */
export function stft(signal: Tensor, options: StftOptions = {}): Spectrogram {
  if (signal.shape.length !== 1) throw new RangeError("stft: v1 supports 1-D Tensor only");
  const scaling = options.scaling ?? "none";
  if (scaling !== "none" && scaling !== "density" && scaling !== "spectrum") {
    throw new RangeError(`stft: scaling must be "none", "density" or "spectrum", got ${String(scaling)}`);
  }
  // Read-only below, so no defensive copy is needed even when `.data`
  // aliases `signal`'s own storage.
  const data = signal.contiguous().data as Float64Array;
  const p = resolveSegments("stft", data.length, options);
  const bins = p.onesided ? p.nfft / 2 + 1 : p.nfft;
  const scale = scaling === "none" ? 1 : scaling === "spectrum" ? 1 / p.sumW : 1 / Math.sqrt(p.fs * p.sumW2);

  const realOut = new Float64Array(p.numFrames * bins);
  const imagOut = new Float64Array(p.numFrames * bins);
  const times: number[] = [];
  for (let f = 0; f < p.numFrames; f++) {
    const start = f * p.hop;
    const { real, imag } = segmentSpectrum(data, start, p);
    for (let k = 0; k < bins; k++) {
      realOut[f * bins + k] = (real[k] as number) * scale;
      imagOut[f * bins + k] = (imag[k] as number) * scale;
    }
    times.push((start + p.nperseg / 2) / p.fs);
  }
  const out = ComplexTensor.fromParts(
    Tensor.fromTypedArray(realOut, [p.numFrames, bins], { dtype: "f64" }),
    Tensor.fromTypedArray(imagOut, [p.numFrames, bins], { dtype: "f64" }),
  );
  return Object.assign(out, { freqs: binFrequencies(p.nfft, p.fs, p.onesided), times, frames: out });
}

function resolveInverse(spectrogramNperseg: number, options: StftOptions): { window: Float64Array; nperseg: number; hop: number } {
  const nperseg = options.nperseg ?? spectrogramNperseg;
  if (!Number.isInteger(nperseg) || nperseg < 1) throw new RangeError(`istft: nperseg must be a positive integer, got ${nperseg}`);
  const noverlap = options.noverlap ?? Math.floor(nperseg / 2);
  if (!Number.isInteger(noverlap) || noverlap < 0 || noverlap >= nperseg) {
    throw new RangeError(`istft: noverlap (${noverlap}) must be in [0, nperseg) (nperseg=${nperseg})`);
  }
  let window: Float64Array;
  try {
    window = getWindow(options.window ?? "hann", nperseg);
  } catch (e) {
    throw new RangeError(`istft: ${(e as Error).message}`);
  }
  return { window, nperseg, hop: nperseg - noverlap };
}

/** Inverse of `stft` -- weighted overlap-add reconstruction. `options` should match what `stft` was called with (nperseg is inferred from `spectrogram`'s own shape if omitted; the spectrogram must be two-sided, un-padded and unscaled, see the header). */
export function istft(spectrogram: ComplexTensor, options: StftOptions = {}): Tensor {
  if (spectrogram.shape.length !== 2) throw new RangeError("istft: expected a [numFrames, nperseg] ComplexTensor");
  const [numFrames, spectrogramNperseg] = spectrogram.shape as [number, number];
  const { window, nperseg, hop } = resolveInverse(spectrogramNperseg, options);
  if (nperseg !== spectrogramNperseg) {
    throw new RangeError(`istft: nperseg option (${nperseg}) doesn't match the spectrogram's own frame length (${spectrogramNperseg})`);
  }

  const outLen = (numFrames - 1) * hop + nperseg;
  const out = new Float64Array(outLen);
  const norm = new Float64Array(outLen);

  for (let f = 0; f < numFrames; f++) {
    const frameReal = new Float64Array(nperseg);
    const frameImag = new Float64Array(nperseg);
    for (let k = 0; k < nperseg; k++) {
      frameReal[k] = spectrogram.real.at(f, k) as number;
      frameImag[k] = spectrogram.imag.at(f, k) as number;
    }
    const reconstructed = ifft(
      ComplexTensor.fromParts(
        Tensor.fromTypedArray(frameReal, [nperseg], { dtype: "f64" }),
        Tensor.fromTypedArray(frameImag, [nperseg], { dtype: "f64" }),
      ),
    ).real.contiguous().data as Float64Array;
    const start = f * hop;
    for (let i = 0; i < nperseg; i++) {
      out[start + i] = (out[start + i] as number) + (reconstructed[i] as number) * (window[i] as number);
      norm[start + i] = (norm[start + i] as number) + (window[i] as number) * (window[i] as number);
    }
  }

  for (let i = 0; i < outLen; i++) {
    out[i] = (norm[i] as number) > 1e-10 ? (out[i] as number) / (norm[i] as number) : 0;
  }
  return Tensor.fromTypedArray(out, [outLen], { dtype: "f64" });
}
