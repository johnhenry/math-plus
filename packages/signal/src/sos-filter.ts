/**
 * sosFilter (issue #44) — apply a cascade of second-order sections (SOS,
 * the numerically-stable filter representation `butter`'s `output: "sos"`
 * produces) to a signal, via Direct Form II Transposed per section (the
 * same structure `scipy.signal.sosfilt` uses) -- each section's output
 * feeds the next section's input.
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";

/** One second-order section: `[b0, b1, b2, a0, a1, a2]` (numerator then denominator coefficients; `a0` need not be pre-normalized to 1 -- this function normalizes internally). */
export type SosSection = readonly [number, number, number, number, number, number];
export type Sos = readonly SosSection[];

export function sosFilter(sos: Sos, signal: Tensor): Tensor {
  if (signal.shape.length !== 1) throw new RangeError("sosFilter: v1 supports 1-D Tensor only");
  if (sos.length === 0) throw new RangeError("sosFilter: sos must have at least one section");

  // Read-only below (each section reads `data[n]` into a fresh `out`
  // array, then rebinds `data = out` -- the original array is never
  // mutated in place), so no defensive copy is needed here.
  const data = sosFilterRaw(sos, signal.contiguous().data as Float64Array, undefined);
  return Tensor.fromTypedArray(data, [data.length], { dtype: "f64" });
}

/**
 * The shared cascade kernel behind `sosFilter` and `sosfiltfilt`. `zi`, when
 * given, holds one `[z1, z2]` initial state per section, already scaled to
 * the signal (scipy's `sosfilt(..., zi=)`).
 *
 * @internal
 */
export function sosFilterRaw(sos: Sos, input: Float64Array, zi: ReadonlyArray<readonly [number, number]> | undefined): Float64Array {
  let data = input;
  for (let s = 0; s < sos.length; s++) {
    const section = sos[s] as SosSection;
    const [b0raw, b1raw, b2raw, a0, a1raw, a2raw] = section;
    if (a0 === 0) throw new RangeError("sosFilter: a section's a0 coefficient must be nonzero");
    const b0 = b0raw / a0;
    const b1 = b1raw / a0;
    const b2 = b2raw / a0;
    const a1 = a1raw / a0;
    const a2 = a2raw / a0;

    const out = new Float64Array(data.length);
    let z1 = zi ? (zi[s] as readonly [number, number])[0] : 0;
    let z2 = zi ? (zi[s] as readonly [number, number])[1] : 0;
    for (let n = 0; n < data.length; n++) {
      const x = data[n] as number;
      const y = b0 * x + z1;
      z1 = b1 * x - a1 * y + z2;
      z2 = b2 * x - a2 * y;
      out[n] = y;
    }
    data = out;
  }
  return data;
}
