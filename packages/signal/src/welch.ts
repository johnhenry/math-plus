/**
 * welch (issues #70, #178) — power spectral density via Welch's method:
 * the average of `|X_i(f)|^2` over windowed, overlapping segments. With no
 * options `welch(x)` is `scipy.signal.welch(x)`: fs 1, periodic Hann window,
 * `nperseg = min(256, n)`, 50% overlap, `nfft = nperseg`, `detrend="constant"`,
 * `scaling="density"`, one-sided (`return_onesided=True`), `average="mean"`,
 * `axis=-1`. As in SciPy it is `csd(x, x).real`: the two share one routine (csd.ts).
 *
 * N-D input is handled along `axis` (default -1): the result has the input's
 * shape with that axis replaced by the frequency axis, as SciPy's does. Complex
 * input (a `ComplexTensor`) is always two-sided, in FFT order, and silently so
 * (SciPy's `welch` does not warn, unlike its `stft`).
 *
 * An `nperseg` longer than the signal is clamped, with SciPy's warning (through
 * `onWarning`, else `console.warn`). An empty input gives an empty `Pxx` of the
 * input's shape and `freqs: []`, as SciPy does (its `freqs` is that empty array).
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { type CsdOptions, csdCore, realResult } from "./csd.ts";
import type { SignalInput } from "./nd.ts";

export type WelchOptions = CsdOptions;

export interface WelchResult {
  /** Bin frequencies (`rfftfreq` one-sided, `fftfreq` order two-sided), SciPy's `f`. */
  readonly freqs: number[];
  /** The power spectral density (or spectrum), real, SciPy's `Pxx`: the input's shape with `axis` replaced by `freqs.length`. */
  readonly Pxx: Tensor;
}

export function welch(signal: SignalInput, options: WelchOptions = {}): WelchResult {
  const core = csdCore("welch", signal, null, options);
  return { freqs: core.freqs, Pxx: realResult(core) };
}
