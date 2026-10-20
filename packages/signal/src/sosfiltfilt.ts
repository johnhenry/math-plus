/**
 * sosfiltfilt (Viaframe request, math-plus#173) -- zero-phase forward-backward
 * filtering with a cascade of second-order sections, following
 * `scipy.signal.sosfiltfilt` with its defaults: odd (anti-symmetric) edge
 * extension of `padlen` samples, and initial conditions from the steady-state
 * step response (`sosfilt_zi`) scaled by the first sample of each pass, so
 * the edges show no start-up transient.
 *
 * Default `padlen` is scipy's: `3 * (2 * nSections + 1 - k)` where `k` is the
 * smaller of the number of sections with `b2 == 0` and with `a2 == 0`.
 *
 * Scope, disclosed: 1-D input only; only scipy's default `padtype="odd"` is
 * implemented; filters must have a nonzero DC denominator gain (`sum(a) != 0`),
 * i.e. no pole at z = 1 (true of `butter` lowpass/bandpass/highpass output).
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { sosFilterRaw, type Sos, type SosSection } from "./sos-filter.ts";

export interface SosfiltfiltOptions {
  /** Samples of odd extension at each end. Default: scipy's. Must satisfy `0 <= padlen < signal.length`. */
  readonly padlen?: number;
}

function defaultPadlen(sos: Sos): number {
  let zeroB2 = 0;
  let zeroA2 = 0;
  for (const s of sos) {
    if (s[2] === 0) zeroB2++;
    if (s[5] === 0) zeroA2++;
  }
  return 3 * (2 * sos.length + 1 - Math.min(zeroB2, zeroA2));
}

/** Steady-state DF2T state of every section for a unit step input (scipy's `sosfilt_zi`). */
function sosZi(sos: Sos): [number, number][] {
  const out: [number, number][] = [];
  let scale = 1;
  for (const section of sos) {
    const [b0r, b1r, b2r, a0, a1r, a2r] = section as SosSection;
    const b0 = b0r / a0;
    const b1 = b1r / a0;
    const b2 = b2r / a0;
    const a1 = a1r / a0;
    const a2 = a2r / a0;
    const den = 1 + a1 + a2;
    if (den === 0) throw new RangeError("sosfiltfilt: a section has a pole at z = 1; initial conditions are undefined");
    const g = (b0 + b1 + b2) / den;
    const z2 = b2 - a2 * g;
    const z1 = b1 - a1 * g + z2;
    out.push([z1 * scale, z2 * scale]);
    scale *= g;
  }
  return out;
}

function oddExtend(x: Float64Array, pad: number): Float64Array {
  const n = x.length;
  const out = new Float64Array(n + 2 * pad);
  const first = x[0] as number;
  const last = x[n - 1] as number;
  for (let k = 0; k < pad; k++) {
    out[k] = 2 * first - (x[pad - k] as number);
    out[pad + n + k] = 2 * last - (x[n - 2 - k] as number);
  }
  out.set(x, pad);
  return out;
}

function pass(sos: Sos, zi: [number, number][], data: Float64Array): Float64Array {
  const x0 = data[0] as number;
  return sosFilterRaw(
    sos,
    data,
    zi.map(([a, b]) => [a * x0, b * x0] as const),
  );
}

export function sosfiltfilt(sos: Sos, signal: Tensor, options: SosfiltfiltOptions = {}): Tensor {
  if (signal.shape.length !== 1) throw new RangeError("sosfiltfilt: v1 supports 1-D Tensor only");
  if (sos.length === 0) throw new RangeError("sosfiltfilt: sos must have at least one section");
  const x = signal.contiguous().data as Float64Array;
  const n = x.length;
  const padlen = options.padlen ?? defaultPadlen(sos);
  if (!Number.isInteger(padlen) || padlen < 0) throw new RangeError("sosfiltfilt: padlen must be a non-negative integer");
  if (padlen >= n) throw new RangeError(`sosfiltfilt: signal length (${n}) must be greater than padlen (${padlen})`);

  const zi = sosZi(sos);
  const ext = padlen > 0 ? oddExtend(x, padlen) : x;
  const fwd = pass(sos, zi, ext);
  const rev = new Float64Array(fwd.length);
  for (let i = 0; i < fwd.length; i++) rev[i] = fwd[fwd.length - 1 - i] as number;
  const back = pass(sos, zi, rev);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = back[back.length - 1 - padlen - i] as number;
  return Tensor.fromTypedArray(out, [n], { dtype: "f64" });
}
