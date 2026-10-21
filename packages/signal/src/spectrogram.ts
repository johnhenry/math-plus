/**
 * spectrogram (SciPy parity): `scipy.signal.spectrogram(x, fs, window, nperseg, noverlap, nfft, detrend,
 * return_onesided, scaling, axis, mode)`. Defaults are SciPy's: fs 1, a Tukey window with alpha 0.25
 * (`["tukey", 0.25]`), `nperseg = min(256, n)`, `noverlap = floor(nperseg / 8)`, `nfft = nperseg`,
 * `detrend="constant"`, one-sided (real input), `scaling="density"`, `axis=-1`, `mode="psd"`. It shares everything
 * with `stft` and `welch` (the segmentation, windowing and DFT in spectral.ts); unlike `stft` it adds no boundary
 * and no padding, and it does not average.
 *
 * `Sxx` has the input's shape with `axis` replaced by the frequencies and the segment (time) axis appended, like
 * `stft`'s `Zxx`. `mode` picks what it holds: `"psd"` (default: power, real, the one-sided bins doubled as
 * `welch`'s are), `"complex"` (the `stft` values, a `ComplexTensor`), `"magnitude"` (`|stft|`), `"angle"` (the
 * phase in radians) or `"phase"` (the angle unwrapped, `np.unwrap`, along the FREQUENCY axis, as SciPy does).
 * `scaling` is `"density"` or `"spectrum"` (SciPy's spelling for these is `"psd"` too in `stft`; here, as in
 * `scipy.signal.spectrogram`, only these two).
 *
 * An `nperseg` longer than the signal is clamped, with SciPy's warning. An empty input gives empty `freqs`, `times`
 * and `Sxx` (of the input's shape) after the window and `nperseg` checks, which SciPy runs first.
 */
import { ComplexTensor } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { emptyTensor, normalizeAxis, rowsAlong, type SignalInput, shapeOf, sizeOf } from "./nd.ts";
import { binFrequencies, frameSpectra, frameTimes, resolveSegments, type SegmentOptions, toInt, triageSegments, unwrapAlong } from "./spectral.ts";

export type SpectrogramMode = "psd" | "complex" | "magnitude" | "angle" | "phase";

export interface SpectrogramOptions extends SegmentOptions {
  /** `"density"` (default) or `"spectrum"`: the power scale of `mode: "psd"`, and, as `X / sqrt(fs · Σw²)` or `X / Σw`, of the amplitude modes. */
  readonly scaling?: "density" | "spectrum";
  readonly axis?: number;
  /** What `Sxx` holds; default `"psd"`. */
  readonly mode?: SpectrogramMode;
}

export interface SpectrogramResult<S extends Tensor | ComplexTensor = Tensor | ComplexTensor> {
  readonly freqs: number[];
  /** Segment-centre times in `1/fs` units. */
  readonly times: number[];
  /** The input's shape with `axis` replaced by `freqs.length` and `times.length` appended. A `ComplexTensor` for `mode: "complex"`, else a real `Tensor`. */
  readonly Sxx: S;
}

const MODES = ["psd", "complex", "magnitude", "angle", "phase"];

export function spectrogram(signal: SignalInput, options: SpectrogramOptions & { readonly mode: "complex" }): SpectrogramResult<ComplexTensor>;
export function spectrogram(signal: SignalInput, options?: SpectrogramOptions & { readonly mode?: "psd" | "magnitude" | "angle" | "phase" }): SpectrogramResult<Tensor>;
export function spectrogram(signal: SignalInput, options?: SpectrogramOptions): SpectrogramResult;
export function spectrogram(signal: SignalInput, options: SpectrogramOptions = {}): SpectrogramResult {
  const mode = options.mode ?? "psd";
  if (!MODES.includes(mode)) throw new RangeError(`spectrogram: unknown value for mode ${String(mode)}, must be one of [${MODES.map((m) => `'${m}'`).join(", ")}]`);
  const scaling = options.scaling ?? "density";
  const shape = shapeOf(signal);
  const axis = toInt("spectrogram", "axis", options.axis ?? -1);
  // unlike stft, spectrogram does not int() nperseg before it builds the window: a fractional one is an error
  if (options.nperseg !== undefined && !Number.isInteger(options.nperseg)) throw new RangeError("spectrogram: Window length M must be a non-negative integer");
  const given: SegmentOptions = { ...options, window: options.window ?? ["tukey", 0.25] };
  if (sizeOf(shape) === 0) {
    // SciPy sets up the window and nperseg first (so its warning and its errors come first), then returns empties
    triageSegments("spectrogram", shape[normalizeAxis("spectrogram", axis, shape.length)] as number, given, "helper");
    if (mode === "phase") {
      // np.unwrap(Sxx, axis=axis - 1) on the empty result, which has no time axis yet: out of range for a 1-D input
      const ax = axis < 0 ? axis - 1 : axis;
      if (ax < -shape.length || ax >= shape.length) throw new RangeError(`spectrogram: axis ${ax} is out of bounds for array of dimension ${shape.length}`);
    }
    const empty = emptyTensor(shape);
    return { freqs: [], times: [], Sxx: mode === "complex" ? ComplexTensor.fromParts(empty, emptyTensor(shape)) : empty };
  }
  const rows = rowsAlong("spectrogram", signal, axis);
  const p = resolveSegments("spectrogram", rows.n, rows.im !== null, given, {
    defaultDetrend: "constant",
    warnComplexOnesided: true,
    family: "helper",
    defaultNoverlap: (n) => Math.floor(n / 8),
  });
  if (scaling !== "density" && scaling !== "spectrum") throw new RangeError(`spectrogram: Unknown scaling: ${JSON.stringify(scaling)}`);
  const amp = scaling === "density" ? 1 / Math.sqrt(p.fs * p.sumW2) : 1 / p.sumW;
  const freqs = binFrequencies(p.nfft, p.fs, p.onesided);
  const times = frameTimes(p);

  if (mode === "psd") {
    const s = frameSpectra("spectrogram", rows, p, amp, true);
    if (p.onesided) {
      // fold the negative frequencies in: every bin but DC (and Nyquist, for even nfft) counts twice
      const last = p.nfft % 2 === 0 ? p.bins - 1 : p.bins;
      const inner = rows.inner * p.numFrames;
      for (let i = 0; i < s.re.length; i++) {
        const k = Math.floor(i / inner) % p.bins;
        if (k >= 1 && k < last) s.re[i] = (s.re[i] as number) * 2;
      }
    }
    return { freqs, times, Sxx: Tensor.fromTypedArray(s.re, s.shape, { dtype: "f64" }) };
  }
  const s = frameSpectra("spectrogram", rows, p, amp, false);
  const im = s.im as Float64Array;
  if (mode === "complex") {
    return { freqs, times, Sxx: ComplexTensor.fromParts(Tensor.fromTypedArray(s.re, s.shape, { dtype: "f64" }), Tensor.fromTypedArray(im, s.shape, { dtype: "f64" })) };
  }
  const out = new Float64Array(s.re.length);
  for (let i = 0; i < out.length; i++) out[i] = mode === "magnitude" ? Math.hypot(s.re[i] as number, im[i] as number) : Math.atan2(im[i] as number, s.re[i] as number);
  if (mode === "phase") unwrapAlong(out, s.shape, rows.axis);
  return { freqs, times, Sxx: Tensor.fromTypedArray(out, s.shape, { dtype: "f64" }) };
}
