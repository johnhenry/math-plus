/**
 * periodogram (SciPy parity): `scipy.signal.periodogram(x, fs, window, nfft, detrend, return_onesided, scaling, axis)`,
 * which SciPy itself builds as `welch` with one segment (`noverlap=0`, `nperseg` the signal length). Defaults are
 * SciPy's: fs 1, a rectangular (`"boxcar"`) window, `nfft` the signal length, `detrend="constant"`,
 * `scaling="density"`, one-sided (real input), `axis=-1`.
 *
 * `nfft` longer than the signal zero-pads it; shorter than the signal truncates the signal to its first `nfft`
 * samples first. An array `window` must be as long as the signal along `axis`. An empty input gives an empty `Pxx`
 * of the input's shape and `freqs: []`.
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { type CsdOptions, csdCore, realResult } from "./csd.ts";
import { emptyTensor, normalizeAxis, type SignalInput, shapeOf, sizeOf, sliceAxis } from "./nd.ts";
import { toInt } from "./spectral.ts";

export interface PeriodogramOptions {
  readonly fs?: CsdOptions["fs"];
  /** Default `"boxcar"`. A `Float64Array` must have the signal's length along `axis`. */
  readonly window?: CsdOptions["window"];
  /** DFT length; default the signal length. */
  readonly nfft?: number;
  /** Default `"constant"`. */
  readonly detrend?: CsdOptions["detrend"];
  readonly onesided?: boolean;
  readonly scaling?: "density" | "spectrum";
  readonly axis?: number;
  readonly onWarning?: CsdOptions["onWarning"];
}

export interface PeriodogramResult {
  readonly freqs: number[];
  /** The power spectral density (or spectrum), real: the input's shape with `axis` replaced by `freqs.length`. */
  readonly Pxx: Tensor;
}

export function periodogram(signal: SignalInput, options: PeriodogramOptions = {}): PeriodogramResult {
  const shape = shapeOf(signal);
  if (sizeOf(shape) === 0) return { freqs: [], Pxx: emptyTensor(shape) };
  const axis = toInt("periodogram", "axis", options.axis ?? -1);
  const ax = normalizeAxis("periodogram", axis, shape.length);
  const n = shape[ax] as number;
  const window = options.window ?? "boxcar";
  let x = signal;
  let nperseg: number;
  let nfft: number | undefined = options.nfft;
  if (nfft === undefined) nperseg = n;
  else if (nfft === n) nperseg = nfft;
  else if (nfft > n) nperseg = n;
  else {
    const cut = toInt("periodogram", "nfft", nfft);
    x = sliceAxis(signal, ax, cut);
    nperseg = cut;
    nfft = undefined;
  }
  if (window instanceof Float64Array && window.length !== nperseg) {
    throw new RangeError("periodogram: the size of the window must be the same size of the input on the specified axis");
  }
  const core = csdCore("periodogram", x, null, {
    fs: options.fs,
    window,
    nperseg,
    noverlap: 0,
    nfft,
    detrend: options.detrend,
    onesided: options.onesided,
    scaling: options.scaling,
    axis,
    onWarning: options.onWarning,
  });
  return { freqs: core.freqs, Pxx: realResult(core) };
}
