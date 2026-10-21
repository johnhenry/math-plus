/**
 * N-D plumbing shared by `welch`, `stft` and `istft`: the transform axis is moved to the end as a set of 1-D rows
 * (the way SciPy does with `np.moveaxis`), each row is processed on its own, and the results are written straight
 * into their final layout, so no transposed copies are made. Plain typed-array code, browser-safe.
 */
import { ComplexTensor } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";

/** A real `Tensor`, or a `ComplexTensor` (a `real` / `imag` pair of equal shape). */
export type SignalInput = Tensor | ComplexTensor;

export const isComplexTensor = (t: unknown): t is ComplexTensor =>
  t instanceof ComplexTensor || (typeof t === "object" && t !== null && "real" in t && "imag" in t && (t as ComplexTensor).real instanceof Tensor);

/** The values of `t` in row-major order as `Float64Array` (shared with `t` when it already is one: read only). */
export function valuesOf(t: Tensor): Float64Array {
  const data = t.contiguous().data;
  return data instanceof Float64Array ? data : Float64Array.from(data as ArrayLike<number>);
}

/** `np`-style axis normalisation: `-ndim <= axis < ndim`, negative counting from the end. */
export function normalizeAxis(who: string, axis: number, ndim: number, name = "axis"): number {
  if (!Number.isInteger(axis) || axis < -ndim || axis >= ndim) {
    throw new RangeError(`${who}: ${name} ${axis} is out of bounds for a ${ndim}-D input`);
  }
  return axis < 0 ? axis + ndim : axis;
}

export interface Rows {
  /** `rows * n` samples, row `r` being `[r * n, (r + 1) * n)`; the rows run over the other axes in row-major order. */
  readonly re: Float64Array;
  /** Imaginary parts in the same layout; `null` for real input. */
  readonly im: Float64Array | null;
  readonly rows: number;
  /** Length along the transform axis. */
  readonly n: number;
  /** Product of the dimensions before / after the axis: row `r` is `(r / inner | 0, r % inner)`. */
  readonly pre: number;
  readonly inner: number;
  /** The input's shape and the (non-negative) axis. */
  readonly shape: readonly number[];
  readonly axis: number;
}

function gather(src: Float64Array, pre: number, n: number, inner: number): Float64Array {
  if (inner === 1) return src; // already rows
  const out = new Float64Array(pre * inner * n);
  for (let p = 0; p < pre; p++) {
    for (let i = 0; i < inner; i++) {
      const dst = (p * inner + i) * n;
      const from = p * n * inner + i;
      for (let j = 0; j < n; j++) out[dst + j] = src[from + j * inner] as number;
    }
  }
  return out;
}

/** `np.moveaxis(input, axis, -1)` as rows. */
export function rowsAlong(who: string, input: SignalInput, axis: number): Rows {
  const complex = isComplexTensor(input);
  const shape = (complex ? input.real : input).shape as readonly number[];
  if (shape.length === 0) throw new RangeError(`${who}: input must have at least one dimension`);
  const ax = normalizeAxis(who, axis, shape.length);
  const n = shape[ax] as number;
  let pre = 1;
  for (let d = 0; d < ax; d++) pre *= shape[d] as number;
  let inner = 1;
  for (let d = ax + 1; d < shape.length; d++) inner *= shape[d] as number;
  const re = gather(valuesOf(complex ? input.real : input), pre, n, inner);
  const im = complex ? gather(valuesOf(input.imag), pre, n, inner) : null;
  return { re, im, rows: pre * inner, n, pre, inner, shape, axis: ax };
}

/** `shape` with `size` put at `axis` (replacing the dimension there when `replace`, else inserted before it). */
export function withSize(shape: readonly number[], axis: number, size: number, replace: boolean): number[] {
  const out = shape.slice();
  out.splice(axis, replace ? 1 : 0, size);
  return out;
}

export function checkAxesDistinct(who: string, a: number, b: number): void {
  if (a === b) throw new RangeError(`${who}: must specify differing time and frequency axes`);
}
