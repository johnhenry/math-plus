/**
 * coherence (SciPy parity): `scipy.signal.coherence(x, y, fs, window, nperseg, noverlap, nfft, detrend, axis)`,
 * the magnitude-squared coherence `Cxy = |Pxy|² / Pxx / Pyy` from three Welch estimates (SciPy calls `welch(x)`,
 * `welch(y)` and `csd(x, y)` with the same options, and so does this: each one warns on its own, so an `nperseg`
 * longer than the signal warns three times). Defaults are those of `welch`; like SciPy there is no `scaling`,
 * `onesided` or `average` option (one-sided for real input, mean).
 *
 * `x` and `y` follow `csd`'s shape rules; the three spectra are then combined with NumPy broadcasting, so inputs of
 * different lengths along `axis` (whose spectra differ in size) raise SciPy's broadcasting error, and empty inputs
 * give an empty `Cxy`. Values are in `[0, 1]`; a bin with no power gives NaN, as in SciPy.
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { csdCore, type CsdOptions } from "./csd.ts";
import { broadcastMap, broadcastShapes, type SignalInput, sizeOf } from "./nd.ts";

export interface CoherenceOptions {
  readonly fs?: CsdOptions["fs"];
  readonly window?: CsdOptions["window"];
  readonly nperseg?: number;
  readonly noverlap?: number;
  readonly nfft?: number;
  readonly detrend?: CsdOptions["detrend"];
  readonly axis?: number;
  readonly onWarning?: CsdOptions["onWarning"];
}

export interface CoherenceResult {
  readonly freqs: number[];
  /** The coherence, real: the (broadcast) input shape with `axis` replaced by `freqs.length`. */
  readonly Cxy: Tensor;
}

/** `a OP b` with NumPy broadcasting (`a` and `b` flat row-major with their shapes). */
function broadcastOp(a: Float64Array, aShape: readonly number[], b: Float64Array, bShape: readonly number[], op: (x: number, y: number) => number): { values: Float64Array; shape: number[] } {
  const shape = broadcastShapes(aShape, bShape);
  if (shape === null) {
    throw new RangeError(`coherence: operands could not be broadcast together with shapes (${aShape.join(",")}${aShape.length === 1 ? "," : ""}) (${bShape.join(",")}${bShape.length === 1 ? "," : ""})`);
  }
  const mapA = broadcastMap(shape, aShape);
  const mapB = broadcastMap(shape, bShape);
  const values = new Float64Array(sizeOf(shape));
  for (let i = 0; i < values.length; i++) values[i] = op(a[mapA[i] as number] as number, b[mapB[i] as number] as number);
  return { values, shape };
}

export function coherence(x: SignalInput, y: SignalInput, options: CoherenceOptions = {}): CoherenceResult {
  // only the options SciPy's coherence has (it has no scaling, onesided or average)
  const o: CsdOptions = {
    fs: options.fs,
    window: options.window,
    nperseg: options.nperseg,
    noverlap: options.noverlap,
    nfft: options.nfft,
    detrend: options.detrend,
    axis: options.axis,
    onWarning: options.onWarning,
  };
  const xx = csdCore("welch", x, null, o);
  const yy = csdCore("welch", y, null, o);
  const xy = csdCore("csd", x, y, o);
  // |Pxy|**2 (an auto-spectrum Pxy is real)
  const power = new Float64Array(xy.re.length);
  for (let i = 0; i < power.length; i++) power[i] = (xy.im === null ? Math.abs(xy.re[i] as number) : Math.hypot(xy.re[i] as number, xy.im[i] as number)) ** 2;
  const over = broadcastOp(power, xy.shape, xx.re, xx.shape, (p, q) => p / q);
  const cxy = broadcastOp(over.values, over.shape, yy.re, yy.shape, (p, q) => p / q);
  return { freqs: xx.freqs, Cxy: Tensor.fromTypedArray(cxy.values, cxy.shape, { dtype: "f64" }) };
}
