/**
 * @johnhenry/math-plus-tensor-core — typed n-dimensional arrays (M1 slice).
 *
 * Pure-JS/TypedArray execution path; WASM kernels swap in underneath via
 * @johnhenry/math-plus-tensor-wasm without changing this API. Architectural rules
 * enforced from day one (docs/PLAN.md §2, §6.1): no Proxy-based indexing,
 * views vs. contiguous are semantically distinct (permute/transpose/reshape
 * never copy; contiguous() copies iff needed), no implicit copies, no
 * implicit dtype promotion (M1: elementwise ops require matching dtypes).
 */
import {
  allocate,
  isBigIntDType,
  type AnyTypedArray,
  type DType,
} from "./dtype.ts";
import { parseNpy, serializeNpy } from "./npy.ts";
import {
  fillFrom,
  normalSample,
  randintSample,
  seed as rngSeed,
  uniformSample,
  type RandomOptions,
  type Rng,
} from "./random.ts";

export {
  allocate,
  isBigIntDType,
  BYTES_PER_ELEMENT,
  type AnyTypedArray,
  type DType,
  type TypedArrayFor,
} from "./dtype.ts";
export { Rng, type RandomOptions } from "./random.ts";

export type Shape = readonly number[];
export type Axis = number;

function shapeSize(shape: Shape): number {
  let size = 1;
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim < 0) {
      throw new RangeError(`invalid dimension ${dim} in shape [${shape}]`);
    }
    size *= dim;
  }
  return size;
}

/** Row-major (C-order) strides, in elements. */
function contiguousStrides(shape: Shape): number[] {
  const strides = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i] as number;
  }
  return strides;
}

/**
 * NumPy broadcasting: align from trailing axes; dims must match or be 1.
 */
export function broadcastShapes(a: Shape, b: Shape): number[] {
  const ndim = Math.max(a.length, b.length);
  const out = new Array<number>(ndim);
  for (let i = 0; i < ndim; i++) {
    const da = a[a.length - 1 - i] ?? 1;
    const db = b[b.length - 1 - i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) {
      throw new RangeError(
        `shapes [${a}] and [${b}] are not broadcast-compatible at axis ${ndim - 1 - i}`,
      );
    }
    // A size-1 dim takes the other side's size — including 0 (NumPy:
    // broadcasting [0, 4] with [4] is [0, 4]; `Math.max` wrongly gave [1, 4]).
    out[ndim - 1 - i] = da === 1 ? db : da;
  }
  return out;
}

/** Strides for reading `shape`-shaped data as if it were `target`-shaped (0-stride on broadcast axes). */
export interface SliceSpec {
  start?: number;
  end?: number;
  step?: number;
}

/**
 * Python `slice.indices(length)` semantics: negative start/end count from the
 * end, a negative step reverses direction, and out-of-range bounds clamp
 * rather than throw. Returns `{start, length, step}` where `start` is the
 * first element index and `length` is the number of elements selected — the
 * inputs a strided VIEW needs (`offset += start*stride; stride *= step`).
 */
function resolveSlice(spec: SliceSpec, dim: number): {
  start: number;
  length: number;
  step: number;
} {
  const step = spec.step ?? 1;
  if (step === 0) throw new RangeError("slice step must be non-zero");

  const clamp = (i: number, lo: number, hi: number) =>
    Math.max(lo, Math.min(hi, i));
  const resolve = (raw: number | undefined, defaultValue: number): number => {
    if (raw === undefined) return defaultValue;
    let i = raw;
    if (i < 0) i += dim;
    return step > 0 ? clamp(i, 0, dim) : clamp(i, -1, dim - 1);
  };

  const start = resolve(spec.start, step > 0 ? 0 : dim - 1);
  const end = resolve(spec.end, step > 0 ? dim : -1);
  const length =
    step > 0
      ? Math.max(0, Math.ceil((end - start) / step))
      : Math.max(0, Math.ceil((end - start) / step));
  return { start, length, step };
}

function broadcastStrides(
  shape: Shape,
  strides: readonly number[],
  target: Shape,
): number[] {
  const out = new Array<number>(target.length).fill(0);
  for (let i = 0; i < shape.length; i++) {
    const targetAxis = target.length - shape.length + i;
    if (shape[i] === target[targetAxis]) {
      out[targetAxis] = strides[i] as number;
    } else {
      // shape[i] === 1 (validated by broadcastShapes): stride 0 repeats it.
      out[targetAxis] = 0;
    }
  }
  return out;
}

type BinaryOp = "add" | "sub" | "mul" | "div";

const NUMBER_OPS: Record<BinaryOp, (a: number, b: number) => number> = {
  add: (a, b) => a + b,
  sub: (a, b) => a - b,
  mul: (a, b) => a * b,
  div: (a, b) => a / b,
};

const BIGINT_OPS: Partial<Record<BinaryOp, (a: bigint, b: bigint) => bigint>> =
  {
    add: (a, b) => a + b,
    sub: (a, b) => a - b,
    mul: (a, b) => a * b,
    // div omitted: integer division semantics differ from NumPy's true
    // division (i64/i64 -> f64); cast to a float dtype first.
  };

type CompareOp = "eq" | "ne" | "lt" | "lte" | "gt" | "gte";

const NUMBER_CMP: Record<CompareOp, (a: number, b: number) => boolean> = {
  eq: (a, b) => a === b,
  ne: (a, b) => a !== b,
  lt: (a, b) => a < b,
  lte: (a, b) => a <= b,
  gt: (a, b) => a > b,
  gte: (a, b) => a >= b,
};

const BIGINT_CMP: Record<CompareOp, (a: bigint, b: bigint) => boolean> = {
  eq: (a, b) => a === b,
  ne: (a, b) => a !== b,
  lt: (a, b) => a < b,
  lte: (a, b) => a <= b,
  gt: (a, b) => a > b,
  gte: (a, b) => a >= b,
};

export interface TensorOptions {
  dtype?: DType;
}

export class Tensor {
  readonly shape: Shape;
  readonly strides: readonly number[];
  readonly dtype: DType;
  readonly size: number;
  /** Backing storage; shared between views (compare identity to detect views). */
  readonly data: AnyTypedArray;
  /** Element offset of this tensor's [0, 0, ...] within `data`. */
  readonly offset: number;

  private constructor(
    data: AnyTypedArray,
    shape: Shape,
    strides: readonly number[],
    dtype: DType,
    offset: number,
  ) {
    this.data = data;
    this.shape = Object.freeze([...shape]);
    this.strides = Object.freeze([...strides]);
    this.dtype = dtype;
    this.size = shapeSize(shape);
    this.offset = offset;
  }

  get ndim(): number {
    return this.shape.length;
  }

  get isContiguous(): boolean {
    const expected = contiguousStrides(this.shape);
    for (let i = 0; i < expected.length; i++) {
      // A dim of size 1 makes its stride irrelevant.
      if (this.shape[i] !== 1 && this.strides[i] !== expected[i]) return false;
    }
    return true;
  }

  // ---- constructors -------------------------------------------------------

  static zeros(shape: Shape, options: TensorOptions = {}): Tensor {
    const dtype = options.dtype ?? "f32";
    return new Tensor(
      allocate(dtype, shapeSize(shape)),
      shape,
      contiguousStrides(shape),
      dtype,
      0,
    );
  }

  static ones(shape: Shape, options: TensorOptions = {}): Tensor {
    return Tensor.full(shape, 1, options);
  }

  static full(shape: Shape, value: number, options: TensorOptions = {}): Tensor {
    const dtype = options.dtype ?? "f32";
    const data = allocate(dtype, shapeSize(shape));
    if (isBigIntDType(dtype)) {
      (data as BigInt64Array | BigUint64Array).fill(BigInt(value));
    } else {
      (data as Float64Array).fill(value);
    }
    return new Tensor(data, shape, contiguousStrides(shape), dtype, 0);
  }

  static arange(
    start: number,
    stop?: number,
    step = 1,
    options: TensorOptions = {},
  ): Tensor {
    if (stop === undefined) {
      stop = start;
      start = 0;
    }
    if (step === 0) throw new RangeError("arange step must be non-zero");
    const dtype = options.dtype ?? "f32";
    const length = Math.max(0, Math.ceil((stop - start) / step));
    const data = allocate(dtype, length);
    if (isBigIntDType(dtype)) {
      const big = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < length; i++) big[i] = BigInt(start + i * step);
    } else {
      const num = data as Float64Array;
      for (let i = 0; i < length; i++) num[i] = start + i * step;
    }
    return new Tensor(data, [length], [1], dtype, 0);
  }

  static from(values: readonly number[], options: TensorOptions = {}): Tensor {
    const dtype = options.dtype ?? "f32";
    const data = allocate(dtype, values.length);
    if (isBigIntDType(dtype)) {
      const big = data as BigInt64Array | BigUint64Array;
      for (let i = 0; i < values.length; i++) {
        big[i] = BigInt(values[i] as number);
      }
    } else {
      (data as Float64Array).set(values);
    }
    return new Tensor(data, [values.length], [1], dtype, 0);
  }

  /** Wrap an existing TypedArray WITHOUT copying; caller owns aliasing. */
  static fromTypedArray(
    data: AnyTypedArray,
    shape: Shape,
    options: { dtype: DType },
  ): Tensor {
    if (shapeSize(shape) !== data.length) {
      throw new RangeError(
        `shape [${shape}] (${shapeSize(shape)} elements) does not match data length ${data.length}`,
      );
    }
    return new Tensor(data, shape, contiguousStrides(shape), options.dtype, 0);
  }

  // ---- combining tensors (issue #4) ------------------------------------------

  /** Join tensors along an existing axis (all dims but `axis` must match). Copies. */
  static concat(tensors: readonly Tensor[], options: { axis?: number } = {}): Tensor {
    if (tensors.length === 0) {
      throw new RangeError("concat requires at least one tensor");
    }
    const first = tensors[0] as Tensor;
    const axis = options.axis ?? 0;
    let ax = axis;
    if (ax < 0) ax += first.ndim;
    if (ax < 0 || ax >= first.ndim) {
      throw new RangeError(`axis ${axis} out of range for ndim ${first.ndim}`);
    }
    for (const t of tensors) {
      if (t.dtype !== first.dtype) {
        throw new TypeError(`concat: dtype mismatch ${first.dtype} vs ${t.dtype}`);
      }
      if (t.ndim !== first.ndim) {
        throw new RangeError(`concat: ndim mismatch ${first.ndim} vs ${t.ndim}`);
      }
      for (let i = 0; i < first.ndim; i++) {
        if (i !== ax && t.shape[i] !== first.shape[i]) {
          throw new RangeError(
            `concat: shape mismatch on axis ${i} (${first.shape[i]} vs ${t.shape[i]})`,
          );
        }
      }
    }
    const axisSize = tensors.reduce((sum, t) => sum + (t.shape[ax] as number), 0);
    const outShape = first.shape.map((d, i) => (i === ax ? axisSize : d));
    const out = Tensor.zeros(outShape, { dtype: first.dtype });
    let cursor = 0;
    for (const t of tensors) {
      const dim = t.shape[ax] as number;
      const specs: Array<SliceSpec | null> = new Array(out.ndim).fill(null);
      specs[ax] = { start: cursor, end: cursor + dim };
      out.slice(...specs).#copyFrom(t);
      cursor += dim;
    }
    return out;
  }

  /** Join tensors along a NEW axis (all inputs must share shape/dtype). Copies. */
  static stack(tensors: readonly Tensor[], options: { axis?: number } = {}): Tensor {
    if (tensors.length === 0) {
      throw new RangeError("stack requires at least one tensor");
    }
    const axis = options.axis ?? 0;
    return Tensor.concat(
      tensors.map((t) => t.unsqueeze(axis)),
      { axis },
    );
  }

  /** Elementwise select: `condition[i] ? a[i] : b[i]`, broadcasting all three. */
  static where(condition: Tensor, a: Tensor, b: Tensor): Tensor {
    if (condition.dtype !== "bool") {
      throw new TypeError(`where: condition must be a bool tensor, got ${condition.dtype}`);
    }
    if (a.dtype !== b.dtype) {
      throw new TypeError(`where: dtype mismatch ${a.dtype} vs ${b.dtype}`);
    }
    const outShape = broadcastShapes(broadcastShapes(condition.shape, a.shape), b.shape);
    const out = Tensor.zeros(outShape, { dtype: a.dtype });
    const condStrides = broadcastStrides(condition.shape, condition.strides, outShape);
    const aStrides = broadcastStrides(a.shape, a.strides, outShape);
    const bStrides = broadcastStrides(b.shape, b.strides, outShape);
    const index = new Array<number>(outShape.length).fill(0);
    let condOff = condition.offset;
    let aOff = a.offset;
    let bOff = b.offset;
    for (let i = 0; i < out.size; i++) {
      const pick = (condition.data[condOff] as number) !== 0;
      out.data[i] = (pick ? a.data[aOff] : b.data[bOff]) as never;
      for (let axis = outShape.length - 1; axis >= 0; axis--) {
        index[axis] = (index[axis] as number) + 1;
        condOff += condStrides[axis] as number;
        aOff += aStrides[axis] as number;
        bOff += bStrides[axis] as number;
        if ((index[axis] as number) < (outShape[axis] as number)) break;
        index[axis] = 0;
        condOff -= (outShape[axis] as number) * (condStrides[axis] as number);
        aOff -= (outShape[axis] as number) * (aStrides[axis] as number);
        bOff -= (outShape[axis] as number) * (bStrides[axis] as number);
      }
    }
    return out;
  }

  // ---- views (never copy) -------------------------------------------------

  /** View with a new shape. Supports one -1 (inferred). Requires contiguity. */
  reshape(shape: Shape): Tensor {
    if (!this.isContiguous) {
      throw new TypeError(
        "reshape of a non-contiguous tensor is not a view; call .contiguous() first (no implicit copies)",
      );
    }
    const inferred = [...shape];
    const negatives = inferred.filter((d) => d === -1).length;
    if (negatives > 1) {
      throw new RangeError("reshape allows at most one -1 dimension");
    }
    if (negatives === 1) {
      const known = inferred.reduce((a, d) => (d === -1 ? a : a * d), 1);
      if (known === 0 || this.size % known !== 0) {
        throw new RangeError(
          `cannot infer -1: ${this.size} elements do not divide by ${known}`,
        );
      }
      inferred[inferred.indexOf(-1)] = this.size / known;
    }
    if (shapeSize(inferred) !== this.size) {
      throw new RangeError(
        `cannot reshape ${this.size} elements into [${shape}]`,
      );
    }
    return new Tensor(
      this.data,
      inferred,
      contiguousStrides(inferred),
      this.dtype,
      this.offset,
    );
  }

  /** Axis-permutation view (shares storage; never copies). */
  permute(axes: readonly number[]): Tensor {
    if (axes.length !== this.ndim) {
      throw new RangeError(`permute needs ${this.ndim} axes, got ${axes.length}`);
    }
    const seen = new Set(axes);
    if (seen.size !== this.ndim || axes.some((a) => a < 0 || a >= this.ndim)) {
      throw new RangeError(`invalid permutation [${axes}] for ndim ${this.ndim}`);
    }
    const shape = axes.map((a) => this.shape[a] as number);
    const strides = axes.map((a) => this.strides[a] as number);
    return new Tensor(this.data, shape, strides, this.dtype, this.offset);
  }

  /** Reverse (or permute) axes — a view. */
  transpose(axes?: readonly number[]): Tensor {
    if (axes) return this.permute(axes);
    const reversed = Array.from({ length: this.ndim }, (_, i) => this.ndim - 1 - i);
    return this.permute(reversed);
  }

  /** Packed C-order copy — or `this` (identical object) when already packed. */
  contiguous(): Tensor {
    if (this.isContiguous && this.offset === 0 && this.size === this.data.length) {
      return this;
    }
    const data = allocate(this.dtype, this.size);
    const source = this.data;
    let i = 0;
    for (const elementOffset of this.elementOffsets()) {
      data[i++] = source[elementOffset] as never;
    }
    return new Tensor(
      data,
      this.shape,
      contiguousStrides(this.shape),
      this.dtype,
      0,
    );
  }

  /**
   * C-order-strided tensor sharing the SAME storage when the source is
   * already contiguous — regardless of `offset` or whether it spans the
   * whole underlying buffer. Cheaper than {@link contiguous}, whose fast
   * path additionally requires `offset === 0` and a fully-occupied buffer
   * (by design: `contiguous()` hands back a tensor that owns its buffer
   * outright, e.g. for NPY serialization). Callers that only need
   * C-order strides to `reshape()` against — `flatten()`, `roll()`,
   * `#cumulative()` — don't need that stronger guarantee, so they route
   * through this instead of unconditionally copying.
   */
  #contiguousView(): Tensor {
    return this.isContiguous ? this : this.contiguous();
  }

  /** Drop size-1 axes — all of them, or just `axis` if given. A VIEW. */
  squeeze(axis?: Axis): Tensor {
    let axesToRemove: Set<number>;
    if (axis === undefined) {
      axesToRemove = new Set(
        this.shape.flatMap((d, i) => (d === 1 ? [i] : [])),
      );
    } else {
      const ax = this.#normalizeAxis(axis);
      if (this.shape[ax] !== 1) {
        throw new RangeError(
          `squeeze: axis ${axis} has size ${this.shape[ax]}, not 1`,
        );
      }
      axesToRemove = new Set([ax]);
    }
    return new Tensor(
      this.data,
      this.shape.filter((_, i) => !axesToRemove.has(i)),
      this.strides.filter((_, i) => !axesToRemove.has(i)),
      this.dtype,
      this.offset,
    );
  }

  /** Insert a size-1 axis at `axis` (supports `axis === ndim`, appending). A VIEW. */
  unsqueeze(axis: number): Tensor {
    let ax = axis;
    if (ax < 0) ax += this.ndim + 1;
    if (ax < 0 || ax > this.ndim) {
      throw new RangeError(`unsqueeze: axis ${axis} out of range for ndim ${this.ndim}`);
    }
    return new Tensor(
      this.data,
      [...this.shape.slice(0, ax), 1, ...this.shape.slice(ax)],
      // Stride is irrelevant on a size-1 axis (never advances); 0 matches the
      // broadcast convention used elsewhere (broadcastStrides).
      [...this.strides.slice(0, ax), 0, ...this.strides.slice(ax)],
      this.dtype,
      this.offset,
    );
  }

  /**
   * Collapse axes `[start, end]` (inclusive, NumPy-style negative indices)
   * into one. Packs into contiguous storage first if needed, so this is a
   * VIEW only when the source already is contiguous over that range.
   */
  flatten(start = 0, end = -1): Tensor {
    const s = start < 0 ? start + this.ndim : start;
    const e = end < 0 ? end + this.ndim : end;
    if (s < 0 || e >= this.ndim || s > e) {
      throw new RangeError(
        `flatten: invalid range [${start}, ${end}] for ndim ${this.ndim}`,
      );
    }
    const packed = this.#contiguousView();
    const flatSize = packed.shape.slice(s, e + 1).reduce((a, b) => a * b, 1);
    return packed.reshape([
      ...packed.shape.slice(0, s),
      flatSize,
      ...packed.shape.slice(e + 1),
    ]);
  }

  /** Expand size-1 (or missing leading) axes to `shape` via stride-0 broadcasting. A VIEW. */
  broadcastTo(shape: Shape): Tensor {
    const target = [...shape];
    const validated = broadcastShapes(this.shape, target);
    if (
      validated.length !== target.length ||
      validated.some((d, i) => d !== target[i])
    ) {
      throw new RangeError(
        `cannot broadcastTo [${shape}] from [${this.shape}] (broadcasting the two gives [${validated}])`,
      );
    }
    return new Tensor(
      this.data,
      target,
      broadcastStrides(this.shape, this.strides, target),
      this.dtype,
      this.offset,
    );
  }

  /**
   * Sliding-window ("patch") view — NumPy `sliding_window_view` semantics
   * (issue #84, upstream for the generalized Wang tile laboratory's patch
   * census / subshift-language machinery). `windowShape` gives the window
   * size along each of `axes` (default: every axis, so `windowShape.length`
   * must equal `ndim`). Output shape: the source axes with each windowed
   * one shrunk in place to `dim - window + 1` (the number of window
   * positions), followed by the window axes themselves in `axes` order —
   * e.g. a 2D tensor with `windowShape=[h,w]` and no `axes` given produces
   * `[outH, outW, h, w]`.
   *
   * A pure VIEW, never copies: each window axis reuses the SAME stride as
   * its source axis (deliberately "double-counting" strides) — stepping
   * one position along the outer (window-position) axis and one position
   * along its paired window axis both advance the same distance through
   * `data`, which is exactly what overlapping windows need. This is the
   * textbook stride trick `sliding_window_view` itself is built on.
   */
  unfold(windowShape: readonly number[], axes?: readonly number[]): Tensor {
    const rawAxes = axes ?? Array.from({ length: this.ndim }, (_, i) => i);
    if (rawAxes.length !== windowShape.length) {
      throw new RangeError(`unfold: windowShape (${windowShape.length}) and axes (${rawAxes.length}) must have the same length`);
    }
    const normalizedAxes = rawAxes.map((a) => this.#normalizeAxis(a));
    const seen = new Set(normalizedAxes);
    if (seen.size !== normalizedAxes.length) {
      throw new RangeError(`unfold: duplicate axis in [${rawAxes}]`);
    }
    normalizedAxes.forEach((axis, i) => {
      const w = windowShape[i] as number;
      const dim = this.shape[axis] as number;
      if (!Number.isInteger(w) || w <= 0 || w > dim) {
        throw new RangeError(`unfold: window size ${w} invalid for axis ${axis} of size ${dim}`);
      }
    });
    const outerShape = this.shape.map((d, i) => {
      const windowIdx = normalizedAxes.indexOf(i);
      return windowIdx === -1 ? d : d - (windowShape[windowIdx] as number) + 1;
    });
    const windowStrides = normalizedAxes.map((a) => this.strides[a] as number);
    return new Tensor(this.data, [...outerShape, ...windowShape], [...this.strides, ...windowStrides], this.dtype, this.offset);
  }

  // ---- structural/manipulation ops (issue #65) -----------------------------
  //
  // A cluster of missing NumPy-standard ops, several implemented as thin
  // compositions of existing primitives (`take`/`slice`/`where`/`full`)
  // rather than new elementwise loops — lower risk, and it's what those
  // primitives are for.

  /** Elementwise bound: values below `min` become `min`, above `max` become `max`. Either bound may be omitted. Distinct from the REDUCTION `min()`/`max()`. Any dtype. */
  clip(min?: number, max?: number): Tensor {
    let out: Tensor = this;
    if (min !== undefined) {
      out = Tensor.where(out.lt(min), Tensor.full(out.shape, min, { dtype: out.dtype }), out);
    }
    if (max !== undefined) {
      out = Tensor.where(out.gt(max), Tensor.full(out.shape, max, { dtype: out.dtype }), out);
    }
    return out;
  }

  /** Product over all elements (axis omitted) or along one axis — the full-reduction dual of {@link sum} (which `cumsum`/`cumprod` already have a cumulative pair for, but `sum` didn't have a `prod` counterpart). Built on the existing `cumprod`: the running product's LAST entry along the scan axis IS the full product. */
  prod(axis?: Axis): Tensor {
    if (axis === undefined) return this.cumprod().select(0, -1);
    const ax = this.#normalizeAxis(axis);
    return this.cumprod(ax).select(ax, -1);
  }

  /**
   * Pad the LAST `padding.length` axes with `[before, after]` element counts
   * each (leading axes untouched) — NumPy `pad`-style spec, constant-value
   * mode only (`options.value`, default 0). Concretely unblocks
   * `@johnhenry/math-plus-data`'s `collate.vectors()`, which currently throws on ragged
   * batches for lack of exactly this primitive.
   */
  pad(padding: ReadonlyArray<readonly [number, number]>, options: { value?: number } = {}): Tensor {
    const ndim = this.ndim;
    if (padding.length > ndim) {
      throw new RangeError(`pad: ${padding.length} axis pairs given but tensor has only ${ndim} axes`);
    }
    const fullPadding: Array<[number, number]> = this.shape.map((_, i) => {
      const fromEnd = ndim - i;
      const idx = padding.length - fromEnd;
      const pair = idx >= 0 ? padding[idx] : undefined;
      if (pair && (pair[0] < 0 || pair[1] < 0)) {
        throw new RangeError("pad: negative padding is not supported (use slice() to shrink instead)");
      }
      return pair ? [pair[0], pair[1]] : [0, 0];
    });
    const outShape = this.shape.map((d, i) => d + (fullPadding[i] as [number, number])[0] + (fullPadding[i] as [number, number])[1]);
    const out = Tensor.full(outShape, options.value ?? 0, { dtype: this.dtype });
    const specs: SliceSpec[] = fullPadding.map(([before], i) => ({
      start: before,
      end: before + (this.shape[i] as number),
    }));
    out.slice(...specs).#copyFrom(this);
    return out;
  }

  /**
   * Split along `axis` (default 0) into VIEWS (no copy — built on {@link slice}).
   * `sections` as a number: that many EQUAL parts (throws if it doesn't evenly
   * divide the axis, matching NumPy `split`'s strict form — see
   * {@link DType} module docs for the "no silent surprises" convention).
   * `sections` as an array: explicit CUT-POINT indices along the axis
   * (NumPy `split`'s array form — `[2, 5]` on a length-8 axis yields three
   * parts of length 2, 3, 3), not part *sizes*.
   */
  split(sections: number | readonly number[], options: { axis?: number } = {}): Tensor[] {
    const ax = this.#normalizeAxis(options.axis ?? 0);
    const dim = this.shape[ax] as number;
    let points: number[];
    if (typeof sections === "number") {
      if (sections <= 0 || dim % sections !== 0) {
        throw new RangeError(`split: axis size ${dim} is not evenly divisible into ${sections} equal sections`);
      }
      const size = dim / sections;
      points = Array.from({ length: sections - 1 }, (_, i) => (i + 1) * size);
    } else {
      points = [...sections];
    }
    const bounds = [0, ...points, dim];
    const out: Tensor[] = [];
    for (let i = 0; i < bounds.length - 1; i++) {
      const specs: Array<SliceSpec | null> = this.shape.map((_, a) =>
        a === ax ? { start: bounds[i], end: bounds[i + 1] } : null,
      );
      out.push(this.slice(...specs));
    }
    return out;
  }

  /**
   * Repeat each element along `axis` (default 0) `counts` times — a uniform
   * count, or an array giving a per-index count (NumPy `repeat`'s full
   * form). Distinct from {@link broadcastTo}, which requires shape-compatible
   * broadcasting (size-1 axes only), not arbitrary per-element repetition.
   * Built on {@link take} — no new copy logic.
   */
  repeat(counts: number | readonly number[], options: { axis?: number } = {}): Tensor {
    const ax = this.#normalizeAxis(options.axis ?? 0);
    const dim = this.shape[ax] as number;
    const countArr = typeof counts === "number" ? new Array<number>(dim).fill(counts) : counts;
    if (countArr.length !== dim) {
      throw new RangeError(`repeat: counts length ${countArr.length} does not match axis size ${dim}`);
    }
    const indices: number[] = [];
    for (let i = 0; i < dim; i++) {
      const c = countArr[i] as number;
      for (let j = 0; j < c; j++) indices.push(i);
    }
    return this.take(indices, { axis: ax });
  }

  /** Reverse along one axis, several axes, or (default) every axis. Built on {@link take}. */
  flip(axis?: number | readonly number[]): Tensor {
    const axes = axis === undefined ? this.shape.map((_, i) => i) : Array.isArray(axis) ? axis : [axis];
    let out: Tensor = this;
    for (const a of axes) {
      const ax = out.#normalizeAxis(a);
      const dim = out.shape[ax] as number;
      const indices = Array.from({ length: dim }, (_, i) => dim - 1 - i);
      out = out.take(indices, { axis: ax });
    }
    return out;
  }

  /** Circular shift by `shift` along `axis` (default: flatten, roll, reshape back — NumPy's no-axis convention). Built on {@link take}. */
  roll(shift: number, options: { axis?: number } = {}): Tensor {
    if (options.axis === undefined) {
      return this.#contiguousView().reshape([this.size]).roll(shift, { axis: 0 }).reshape(this.shape);
    }
    const ax = this.#normalizeAxis(options.axis);
    const dim = this.shape[ax] as number;
    if (dim === 0) return this;
    const s = ((shift % dim) + dim) % dim;
    const indices = Array.from({ length: dim }, (_, i) => (i - s + dim) % dim);
    return this.take(indices, { axis: ax });
  }

  /**
   * Coordinates of every truthy (non-zero) element, as an `[count, ndim]`
   * `i64` Tensor — one row per element, matching NumPy `argwhere`'s shape
   * (rather than `nonzero`'s classic tuple-of-1-D-arrays-per-axis form,
   * which doesn't fit this repo's one-Tensor-return convention as cleanly;
   * documented deviation, not an oversight).
   */
  nonzero(): Tensor {
    const ndim = this.ndim;
    const shape = this.shape;
    const big = isBigIntDType(this.dtype);
    const rows: number[] = [];
    let count = 0;
    const idx = new Array<number>(ndim).fill(0);
    for (const off of this.elementOffsets()) {
      const v = this.data[off];
      const truthy = big ? (v as bigint) !== 0n : (v as number) !== 0;
      if (truthy) {
        rows.push(...idx);
        count++;
      }
      for (let a = ndim - 1; a >= 0; a--) {
        idx[a] = (idx[a] as number) + 1;
        if ((idx[a] as number) < (shape[a] as number)) break;
        idx[a] = 0;
      }
    }
    return Tensor.from(rows, { dtype: "i64" }).reshape([count, ndim]);
  }

  // ---- indexing & slicing --------------------------------------------------

  /**
   * Strided range selection — a VIEW, never a copy (issue #1).
   *
   * Specs align to leading axes; omitted trailing axes are taken whole, as is
   * `null`. Follows Python's `slice.indices()` semantics exactly, including
   * negative indices and negative `step` (which produces a negative stride
   * rather than reordering data).
   *
   * `x.slice({ start: 1, end: 10, step: 2 })` — deliberately not `x[1:10:2]`;
   * see non-goal 12 (no Proxy-based pseudo-Python indexing).
   */
  slice(...specs: Array<SliceSpec | null>): Tensor {
    if (specs.length > this.ndim) {
      throw new RangeError(
        `got ${specs.length} slice specs for a ${this.ndim}-d tensor`,
      );
    }
    const shape: number[] = [];
    const strides: number[] = [];
    let offset = this.offset;

    for (let axis = 0; axis < this.ndim; axis++) {
      const dim = this.shape[axis] as number;
      const stride = this.strides[axis] as number;
      const spec = specs[axis];
      if (spec === undefined || spec === null) {
        shape.push(dim);
        strides.push(stride);
        continue;
      }
      const { start, length, step } = resolveSlice(spec, dim);
      offset += start * stride;
      shape.push(length);
      strides.push(stride * step);
    }
    return new Tensor(this.data, shape, strides, this.dtype, offset);
  }

  /** Pick one index along `axis`, dropping that axis. A VIEW. */
  select(axis: number, index: number): Tensor {
    const ax = this.#normalizeAxis(axis);
    const dim = this.shape[ax] as number;
    let i = index;
    if (i < 0) i += dim;
    if (i < 0 || i >= dim) {
      throw new RangeError(
        `index ${index} out of bounds for axis ${ax} (size ${dim})`,
      );
    }
    return new Tensor(
      this.data,
      this.shape.filter((_, a) => a !== ax),
      this.strides.filter((_, a) => a !== ax),
      this.dtype,
      this.offset + i * (this.strides[ax] as number),
    );
  }

  /**
   * Gather arbitrary indices along an axis. COPIES — arbitrary index lists
   * cannot be expressed as a stride.
   */
  take(indices: readonly number[], options: { axis?: number } = {}): Tensor {
    const ax = this.#normalizeAxis(options.axis ?? 0);
    const dim = this.shape[ax] as number;
    const resolved = indices.map((raw) => {
      const i = raw < 0 ? raw + dim : raw;
      if (i < 0 || i >= dim) {
        throw new RangeError(
          `index ${raw} out of bounds for axis ${ax} (size ${dim})`,
        );
      }
      return i;
    });
    const outShape = this.shape.map((d, a) => (a === ax ? resolved.length : d));
    const out = Tensor.zeros(outShape, { dtype: this.dtype });
    for (let k = 0; k < resolved.length; k++) {
      const source = this.select(ax, resolved[k] as number);
      const target = out.select(ax, k);
      target.#copyFrom(source);
    }
    return out;
  }

  /** Alias of {@link take} matching the source design's `gather(axis, indices)` order. */
  gather(axis: number, indices: readonly number[]): Tensor {
    return this.take(indices, { axis });
  }

  /**
   * Boolean selection. COPIES, and always yields a 1-D result — the number of
   * selected elements isn't known until runtime, so no shape survives.
   */
  mask(condition: Tensor): Tensor {
    if (condition.dtype !== "bool") {
      throw new TypeError(`mask expects a bool tensor, got ${condition.dtype}`);
    }
    if (
      condition.ndim !== this.ndim ||
      condition.shape.some((d, i) => d !== this.shape[i])
    ) {
      throw new RangeError(
        `mask shape [${condition.shape}] does not match [${this.shape}]`,
      );
    }
    const selected: number[] = [];
    const condOffsets = [...condition.elementOffsets()];
    const selfOffsets = [...this.elementOffsets()];
    for (let i = 0; i < condOffsets.length; i++) {
      if (condition.data[condOffsets[i] as number]) {
        selected.push(selfOffsets[i] as number);
      }
    }
    const out = Tensor.zeros([selected.length], { dtype: this.dtype });
    for (let i = 0; i < selected.length; i++) {
      out.data[i] = this.data[selected[i] as number] as never;
    }
    return out;
  }

  #normalizeAxis(axis: number): number {
    let ax = axis;
    if (ax < 0) ax += this.ndim;
    if (ax < 0 || ax >= this.ndim) {
      throw new RangeError(`axis ${axis} out of range for ndim ${this.ndim}`);
    }
    return ax;
  }

  // ---- element access ------------------------------------------------------

  /** Scalar element read by multi-index. Returns bigint for i64/u64. */
  at(...indices: number[]): number | bigint {
    if (indices.length !== this.ndim) {
      throw new RangeError(`expected ${this.ndim} indices, got ${indices.length}`);
    }
    let elementOffset = this.offset;
    for (let axis = 0; axis < indices.length; axis++) {
      const dim = this.shape[axis] as number;
      let index = indices[axis] as number;
      if (index < 0) index += dim;
      if (index < 0 || index >= dim) {
        throw new RangeError(
          `index ${indices[axis]} out of bounds for axis ${axis} (size ${dim})`,
        );
      }
      elementOffset += index * (this.strides[axis] as number);
    }
    return this.data[elementOffset] as number | bigint;
  }

  /** The single element of a size-1 tensor. */
  item(): number | bigint {
    if (this.size !== 1) {
      throw new RangeError(`item() requires size 1, got ${this.size}`);
    }
    return this.data[this.offset] as number | bigint;
  }

  /** Nested plain-array copy (edge/API use only — never a kernel format). */
  toArray(): unknown[] {
    const build = (axis: number, offset: number): unknown[] => {
      const dim = this.shape[axis] as number;
      const stride = this.strides[axis] as number;
      const out = new Array(dim);
      for (let i = 0; i < dim; i++) {
        out[i] =
          axis === this.ndim - 1
            ? this.data[offset + i * stride]
            : build(axis + 1, offset + i * stride);
      }
      return out;
    };
    if (this.ndim === 0) return [];
    return build(0, this.offset);
  }

  /** Iterate storage offsets of every element in C-order of this view. */
  private *elementOffsets(): Generator<number> {
    if (this.size === 0) return;
    if (this.ndim === 0) {
      yield this.offset;
      return;
    }
    const index = new Array<number>(this.ndim).fill(0);
    let offset = this.offset;
    outer: for (;;) {
      yield offset;
      for (let axis = this.ndim - 1; axis >= 0; axis--) {
        index[axis] = (index[axis] as number) + 1;
        offset += this.strides[axis] as number;
        if ((index[axis] as number) < (this.shape[axis] as number)) {
          continue outer;
        }
        index[axis] = 0;
        offset -=
          (this.shape[axis] as number) * (this.strides[axis] as number);
      }
      return;
    }
  }

  /** Strided element-by-element copy from `source` into `this` (shapes must match). */
  #copyFrom(source: Tensor): void {
    if (
      this.shape.length !== source.shape.length ||
      this.shape.some((d, i) => d !== source.shape[i])
    ) {
      throw new RangeError(
        `copy shape mismatch: [${this.shape}] vs [${source.shape}]`,
      );
    }
    const targetOffsets = this.elementOffsets();
    const sourceOffsets = source.elementOffsets();
    let target = targetOffsets.next();
    let src = sourceOffsets.next();
    while (!target.done && !src.done) {
      this.data[target.value] = source.data[src.value] as never;
      target = targetOffsets.next();
      src = sourceOffsets.next();
    }
  }

  // ---- elementwise (broadcasting) -----------------------------------------

  #binary(op: BinaryOp, other: Tensor | number): Tensor {
    const rhs =
      typeof other === "number"
        ? Tensor.full([], other, { dtype: this.dtype })
        : other;
    if (rhs.dtype !== this.dtype) {
      throw new TypeError(
        `dtype mismatch: ${this.dtype} vs ${rhs.dtype} (M1 has no implicit promotion; cast() first)`,
      );
    }
    const outShape = broadcastShapes(this.shape, rhs.shape);
    const out = Tensor.zeros(outShape, { dtype: this.dtype });
    const aStrides = broadcastStrides(this.shape, this.strides, outShape);
    const bStrides = broadcastStrides(rhs.shape, rhs.strides, outShape);

    const size = out.size;
    const ndim = outShape.length;
    const index = new Array<number>(ndim).fill(0);
    let aOff = this.offset;
    let bOff = rhs.offset;

    if (isBigIntDType(this.dtype)) {
      const fn = BIGINT_OPS[op];
      if (!fn) {
        throw new TypeError(
          `${op} is not defined for ${this.dtype} (NumPy true-division returns f64; cast() first)`,
        );
      }
      const a = this.data as BigInt64Array;
      const b = rhs.data as BigInt64Array;
      const o = out.data as BigInt64Array;
      for (let i = 0; i < size; i++) {
        o[i] = fn(a[aOff] as bigint, b[bOff] as bigint);
        for (let axis = ndim - 1; axis >= 0; axis--) {
          index[axis] = (index[axis] as number) + 1;
          aOff += aStrides[axis] as number;
          bOff += bStrides[axis] as number;
          if ((index[axis] as number) < (outShape[axis] as number)) break;
          index[axis] = 0;
          aOff -= (outShape[axis] as number) * (aStrides[axis] as number);
          bOff -= (outShape[axis] as number) * (bStrides[axis] as number);
        }
      }
    } else {
      const fn = NUMBER_OPS[op];
      const a = this.data as Float64Array;
      const b = rhs.data as Float64Array;
      const o = out.data as Float64Array;
      for (let i = 0; i < size; i++) {
        o[i] = fn(a[aOff] as number, b[bOff] as number);
        for (let axis = ndim - 1; axis >= 0; axis--) {
          index[axis] = (index[axis] as number) + 1;
          aOff += aStrides[axis] as number;
          bOff += bStrides[axis] as number;
          if ((index[axis] as number) < (outShape[axis] as number)) break;
          index[axis] = 0;
          aOff -= (outShape[axis] as number) * (aStrides[axis] as number);
          bOff -= (outShape[axis] as number) * (bStrides[axis] as number);
        }
      }
    }
    return out;
  }

  add(other: Tensor | number): Tensor {
    return this.#binary("add", other);
  }
  sub(other: Tensor | number): Tensor {
    return this.#binary("sub", other);
  }
  mul(other: Tensor | number): Tensor {
    return this.#binary("mul", other);
  }
  div(other: Tensor | number): Tensor {
    return this.#binary("div", other);
  }

  /** Elementwise square root. Float dtypes only (cast() first). */
  sqrt(): Tensor {
    if (isBigIntDType(this.dtype)) {
      throw new TypeError(`sqrt requires a float dtype, got ${this.dtype}`);
    }
    const out = Tensor.zeros(this.shape, { dtype: this.dtype });
    const outData = out.data as Float64Array;
    let i = 0;
    for (const off of this.elementOffsets()) {
      outData[i++] = Math.sqrt(this.data[off] as number);
    }
    return out;
  }

  /** Elementwise natural log. Float dtypes only; domain is (0, +Inf) same as Math.log. */
  log(): Tensor {
    if (isBigIntDType(this.dtype)) {
      throw new TypeError(`log requires a float dtype, got ${this.dtype}`);
    }
    const out = Tensor.zeros(this.shape, { dtype: this.dtype });
    const outData = out.data as Float64Array;
    let i = 0;
    for (const off of this.elementOffsets()) {
      outData[i++] = Math.log(this.data[off] as number);
    }
    return out;
  }

  #unaryFloat(fn: (v: number) => number): Tensor {
    if (isBigIntDType(this.dtype)) {
      throw new TypeError(`this op requires a float dtype, got ${this.dtype}`);
    }
    const out = Tensor.zeros(this.shape, { dtype: this.dtype });
    const outData = out.data as Float64Array;
    let i = 0;
    for (const off of this.elementOffsets()) {
      outData[i++] = fn(this.data[off] as number);
    }
    return out;
  }

  /** Elementwise ReLU: `max(x, 0)`. Documented tensor-core v1 scope (PLAN.md §6.1); float dtypes only. */
  relu(): Tensor {
    return this.#unaryFloat((v) => (v > 0 ? v : 0));
  }

  /** Elementwise sigmoid: `1 / (1 + exp(-x))`. Float dtypes only. */
  sigmoid(): Tensor {
    return this.#unaryFloat((v) => 1 / (1 + Math.exp(-v)));
  }

  /** Elementwise GELU (tanh approximation, matching common ML-library defaults). Float dtypes only. */
  gelu(): Tensor {
    const c = Math.sqrt(2 / Math.PI);
    return this.#unaryFloat((v) => 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v ** 3))));
  }

  // ---- unary op-table parity with the compiled IR (issue #64) --------------
  //
  // `sqrt`/`log` above already existed; these fill the rest of the set
  // `tensor-compile`'s `ir.ts` (`unaryValueAndDeriv`) supports for its
  // compiled/traced path but eager `Tensor` never got. Every formula here
  // is the VALUE half of that same function, copied (not re-derived) so the
  // two backends compute identically — tensor-core has no gradient concept,
  // so only the value side applies here; the derivative halves stay in
  // ir.ts, which is the autograd-facing surface.

  /** Elementwise `e^x`. Float dtypes only. */
  exp(): Tensor {
    return this.#unaryFloat((v) => Math.exp(v));
  }

  /** Elementwise `x^exponent` for a SCALAR exponent (the common case `tensor-compile`'s `pow` node covers) — binary tensor-tensor `pow` is out of scope here. Float dtypes only. */
  pow(exponent: number): Tensor {
    return this.#unaryFloat((v) => Math.pow(v, exponent));
  }

  /** Elementwise absolute value. Works on any numeric dtype (unlike the float-only ops above — `Math.abs` is exact on integers too), so this bypasses `#unaryFloat`'s float-dtype guard. */
  abs(): Tensor {
    const out = Tensor.zeros(this.shape, { dtype: this.dtype });
    if (isBigIntDType(this.dtype)) {
      const outData = out.data as BigInt64Array | BigUint64Array;
      let i = 0;
      for (const off of this.elementOffsets()) {
        const v = this.data[off] as bigint;
        outData[i++] = v < 0n ? -v : v;
      }
      return out;
    }
    const outData = out.data as Float32Array | Float64Array;
    let i = 0;
    for (const off of this.elementOffsets()) outData[i++] = Math.abs(this.data[off] as number);
    return out;
  }

  /** Elementwise negation: `-x`. Same any-dtype scope as {@link abs}. */
  neg(): Tensor {
    return this.mul(-1);
  }

  /** Elementwise sign: `-1`/`0`/`1`. Float dtypes only (matches `ir.ts`'s convention: gradient-free, defined identically to `Math.sign`). */
  sign(): Tensor {
    return this.#unaryFloat((v) => Math.sign(v));
  }

  /** Elementwise sine. Float dtypes only. */
  sin(): Tensor {
    return this.#unaryFloat((v) => Math.sin(v));
  }
  /** Elementwise cosine. Float dtypes only. */
  cos(): Tensor {
    return this.#unaryFloat((v) => Math.cos(v));
  }
  /** Elementwise tangent. Float dtypes only. */
  tan(): Tensor {
    return this.#unaryFloat((v) => Math.tan(v));
  }
  /** Elementwise arcsine. Domain `[-1, 1]`; float dtypes only. */
  asin(): Tensor {
    return this.#unaryFloat((v) => Math.asin(v));
  }
  /** Elementwise arccosine. Domain `[-1, 1]`; float dtypes only. */
  acos(): Tensor {
    return this.#unaryFloat((v) => Math.acos(v));
  }
  /** Elementwise arctangent. Float dtypes only. */
  atan(): Tensor {
    return this.#unaryFloat((v) => Math.atan(v));
  }

  /** Elementwise hyperbolic sine. Float dtypes only. */
  sinh(): Tensor {
    return this.#unaryFloat((v) => Math.sinh(v));
  }
  /** Elementwise hyperbolic cosine. Float dtypes only. */
  cosh(): Tensor {
    return this.#unaryFloat((v) => Math.cosh(v));
  }
  /** Elementwise hyperbolic tangent. Float dtypes only. */
  tanh(): Tensor {
    return this.#unaryFloat((v) => Math.tanh(v));
  }
  /** Elementwise inverse hyperbolic sine. Float dtypes only. */
  asinh(): Tensor {
    return this.#unaryFloat((v) => Math.asinh(v));
  }
  /** Elementwise inverse hyperbolic cosine. Domain `x >= 1`; float dtypes only. */
  acosh(): Tensor {
    return this.#unaryFloat((v) => Math.acosh(v));
  }
  /** Elementwise inverse hyperbolic tangent. Domain `(-1, 1)`; float dtypes only. */
  atanh(): Tensor {
    return this.#unaryFloat((v) => Math.atanh(v));
  }

  /** Elementwise real cube root (sign-preserving — unlike `x^(1/3)`, `cbrt(-8) === -2`). Float dtypes only. */
  cbrt(): Tensor {
    return this.#unaryFloat((v) => Math.cbrt(v));
  }
  /** Elementwise base-10 logarithm. Domain `x > 0`; float dtypes only. */
  log10(): Tensor {
    return this.#unaryFloat((v) => Math.log10(v));
  }
  /** Elementwise base-2 logarithm. Domain `x > 0`; float dtypes only. */
  log2(): Tensor {
    return this.#unaryFloat((v) => Math.log2(v));
  }
  /** Elementwise `e^x - 1`, precision-preserving for small `x` (vs. `exp(x).sub(1)`). Float dtypes only. */
  expm1(): Tensor {
    return this.#unaryFloat((v) => Math.expm1(v));
  }
  /** Elementwise `ln(1 + x)`, precision-preserving for small `x`. Domain `x > -1`; float dtypes only. */
  log1p(): Tensor {
    return this.#unaryFloat((v) => Math.log1p(v));
  }

  /** Elementwise floor. Gradient-free (matches `ir.ts`'s "locally constant" convention); float dtypes only. */
  floor(): Tensor {
    return this.#unaryFloat((v) => Math.floor(v));
  }
  /** Elementwise ceiling. Gradient-free; float dtypes only. */
  ceil(): Tensor {
    return this.#unaryFloat((v) => Math.ceil(v));
  }
  /** Elementwise round, half-up (matches `Math.round`'s convention, not round-half-to-even). Gradient-free; float dtypes only. */
  round(): Tensor {
    return this.#unaryFloat((v) => Math.round(v));
  }
  /** Elementwise truncation toward zero. Gradient-free; float dtypes only. */
  trunc(): Tensor {
    return this.#unaryFloat((v) => Math.trunc(v));
  }

  /** Softmax along `axis` (default: last axis). Numerically stable (subtracts the per-row max first). */
  softmax(axis: Axis = -1): Tensor {
    if (isBigIntDType(this.dtype)) {
      throw new TypeError(`softmax requires a float dtype, got ${this.dtype}`);
    }
    const ax = this.#normalizeAxis(axis);
    const shifted = this.sub(this.max(ax).unsqueeze(ax).broadcastTo(this.shape).contiguous());
    const expd = shifted.#unaryFloat(Math.exp);
    return expd.div(expd.sum(ax).unsqueeze(ax).broadcastTo(this.shape).contiguous());
  }

  // ---- matmul (issue #2) ---------------------------------------------------

  /**
   * NumPy `matmul` semantics: the last two axes of each operand are treated
   * as matrices, leading axes broadcast (batched matmul). A 1-D operand gets
   * a size-1 axis prepended (lhs) or appended (rhs) for the multiply, then
   * that axis is squeezed back out of the result — so `(n,) @ (n,)` yields a
   * 0-d scalar, `(n,) @ (n,k)` yields `(k,)`, `(m,n) @ (n,)` yields `(m,)`.
   *
   * Pure-JS reference (naive triple loop); operates directly on strided
   * views — a transposed operand is never implicitly copied. A WASM GEMM
   * kernel swaps in underneath this same signature later (issue #3).
   */
  matmul(other: Tensor): Tensor {
    if (other.dtype !== this.dtype) {
      throw new TypeError(
        `dtype mismatch: ${this.dtype} vs ${other.dtype} (no implicit promotion; cast() first)`,
      );
    }
    if (this.ndim === 0 || other.ndim === 0) {
      throw new RangeError("matmul operands must have ndim >= 1 (scalars aren't matrices)");
    }

    const lhsIs1D = this.ndim === 1;
    const rhsIs1D = other.ndim === 1;
    const lhsShape = lhsIs1D ? [1, this.shape[0] as number] : [...this.shape];
    const rhsShape = rhsIs1D
      ? [other.shape[0] as number, 1]
      : [...other.shape];

    const m = lhsShape[lhsShape.length - 2] as number;
    const k = lhsShape[lhsShape.length - 1] as number;
    const k2 = rhsShape[rhsShape.length - 2] as number;
    const n = rhsShape[rhsShape.length - 1] as number;
    if (k !== k2) {
      throw new RangeError(
        `matmul: inner dimensions ${k} and ${k2} do not match (shapes [${this.shape}] @ [${other.shape}])`,
      );
    }

    const lhsBatch = lhsShape.slice(0, -2);
    const rhsBatch = rhsShape.slice(0, -2);
    const batchShape = broadcastShapes(lhsBatch, rhsBatch);

    const fullOutShape = [...batchShape, m, n];
    const out = Tensor.zeros(fullOutShape, { dtype: this.dtype });

    const lhsBatchStrides = broadcastStrides(
      lhsBatch,
      this.strides.slice(0, lhsBatch.length),
      batchShape,
    );
    const rhsBatchStrides = broadcastStrides(
      rhsBatch,
      other.strides.slice(0, rhsBatch.length),
      batchShape,
    );
    const lhsRowStride = this.strides[lhsIs1D ? 0 : this.ndim - 2] as number;
    const lhsColStride = this.strides[this.ndim - 1] as number;
    const rhsRowStride = other.strides[other.ndim - (rhsIs1D ? 1 : 2)] as number;
    const rhsColStride = rhsIs1D ? 0 : (other.strides[other.ndim - 1] as number);

    const batchSize = batchShape.reduce((a, b) => a * b, 1);
    const batchIndex = new Array<number>(batchShape.length).fill(0);
    let lhsBatchOffset = this.offset;
    let rhsBatchOffset = other.offset;
    const big = isBigIntDType(this.dtype);

    for (let b = 0; b < batchSize; b++) {
      const outBase = b * m * n;
      for (let i = 0; i < m; i++) {
        const lhsRowBase = lhsBatchOffset + i * lhsRowStride;
        for (let j = 0; j < n; j++) {
          const rhsColBase = rhsBatchOffset + j * rhsColStride;
          if (big) {
            let acc = 0n;
            for (let p = 0; p < k; p++) {
              acc +=
                (this.data[lhsRowBase + p * lhsColStride] as bigint) *
                (other.data[rhsColBase + p * rhsRowStride] as bigint);
            }
            (out.data as BigInt64Array)[outBase + i * n + j] = acc;
          } else {
            let acc = 0;
            for (let p = 0; p < k; p++) {
              acc +=
                (this.data[lhsRowBase + p * lhsColStride] as number) *
                (other.data[rhsColBase + p * rhsRowStride] as number);
            }
            (out.data as Float64Array)[outBase + i * n + j] = acc;
          }
        }
      }
      for (let axis = batchShape.length - 1; axis >= 0; axis--) {
        batchIndex[axis] = (batchIndex[axis] as number) + 1;
        lhsBatchOffset += lhsBatchStrides[axis] as number;
        rhsBatchOffset += rhsBatchStrides[axis] as number;
        if ((batchIndex[axis] as number) < (batchShape[axis] as number)) break;
        batchIndex[axis] = 0;
        lhsBatchOffset -=
          (batchShape[axis] as number) * (lhsBatchStrides[axis] as number);
        rhsBatchOffset -=
          (batchShape[axis] as number) * (rhsBatchStrides[axis] as number);
      }
    }

    // Squeeze back the axes synthesized for 1-D operands: drop `m` if the lhs
    // was 1-D, drop `n` if the rhs was 1-D — independently, so 1-D @ 1-D
    // collapses batchShape (m, n) all the way down to a 0-d scalar.
    const tail: number[] = [];
    if (!lhsIs1D) tail.push(m);
    if (!rhsIs1D) tail.push(n);
    const resultShape = [...batchShape, ...tail];
    return resultShape.length === out.ndim ? out : out.reshape(resultShape);
  }

  /** Inner product of two 1-D tensors, returned as a 0-d tensor. */
  dot(other: Tensor): Tensor {
    if (this.ndim !== 1 || other.ndim !== 1) {
      throw new RangeError(
        `dot expects two 1-D tensors, got ndim ${this.ndim} and ${other.ndim} (use matmul for higher rank)`,
      );
    }
    return this.matmul(other);
  }

  // ---- dtype conversion (issue #4) ------------------------------------------

  /**
   * Explicit dtype conversion — the only way values cross dtypes (no
   * implicit promotion anywhere else, per non-goal). Truncates toward zero
   * when converting to an integer type (matches `Math.trunc`, not rounding);
   * `bool` maps any non-zero value to 1. Always a copy, even when `dtype`
   * equals `this.dtype`, so callers can rely on `cast()` never aliasing.
   */
  cast(dtype: DType): Tensor {
    const out = Tensor.zeros(this.shape, { dtype });
    const outBig = isBigIntDType(dtype);
    const srcBig = isBigIntDType(this.dtype);
    const targetOffsets = out.elementOffsets();
    const sourceOffsets = this.elementOffsets();
    let t = targetOffsets.next();
    let s = sourceOffsets.next();
    while (!t.done && !s.done) {
      const raw = this.data[s.value];
      let converted: number | bigint;
      if (outBig) {
        converted = srcBig ? (raw as bigint) : BigInt(Math.trunc(raw as number));
      } else {
        const num = srcBig ? Number(raw as bigint) : (raw as number);
        converted =
          dtype === "bool"
            ? (num !== 0 ? 1 : 0)
            : dtype === "f32" || dtype === "f64"
              ? num
              : Math.trunc(num);
      }
      out.data[t.value] = converted as never;
      t = targetOffsets.next();
      s = sourceOffsets.next();
    }
    return out;
  }

  // ---- comparisons & logicals (issue #4) -------------------------------------

  #compare(op: CompareOp, other: Tensor | number): Tensor {
    const rhs =
      typeof other === "number"
        ? Tensor.full([], other, { dtype: this.dtype })
        : other;
    if (rhs.dtype !== this.dtype) {
      throw new TypeError(
        `dtype mismatch: ${this.dtype} vs ${rhs.dtype} (no implicit promotion; cast() first)`,
      );
    }
    const outShape = broadcastShapes(this.shape, rhs.shape);
    const out = Tensor.zeros(outShape, { dtype: "bool" });
    const aStrides = broadcastStrides(this.shape, this.strides, outShape);
    const bStrides = broadcastStrides(rhs.shape, rhs.strides, outShape);
    const ndim = outShape.length;
    const index = new Array<number>(ndim).fill(0);
    let aOff = this.offset;
    let bOff = rhs.offset;
    const big = isBigIntDType(this.dtype);
    const numFn = NUMBER_CMP[op];
    const bigFn = BIGINT_CMP[op];
    const outData = out.data as Uint8Array;

    for (let i = 0; i < out.size; i++) {
      outData[i] = big
        ? bigFn(this.data[aOff] as bigint, rhs.data[bOff] as bigint)
          ? 1
          : 0
        : numFn(this.data[aOff] as number, rhs.data[bOff] as number)
          ? 1
          : 0;
      for (let axis = ndim - 1; axis >= 0; axis--) {
        index[axis] = (index[axis] as number) + 1;
        aOff += aStrides[axis] as number;
        bOff += bStrides[axis] as number;
        if ((index[axis] as number) < (outShape[axis] as number)) break;
        index[axis] = 0;
        aOff -= (outShape[axis] as number) * (aStrides[axis] as number);
        bOff -= (outShape[axis] as number) * (bStrides[axis] as number);
      }
    }
    return out;
  }

  eq(other: Tensor | number): Tensor {
    return this.#compare("eq", other);
  }
  ne(other: Tensor | number): Tensor {
    return this.#compare("ne", other);
  }
  lt(other: Tensor | number): Tensor {
    return this.#compare("lt", other);
  }
  lte(other: Tensor | number): Tensor {
    return this.#compare("lte", other);
  }
  gt(other: Tensor | number): Tensor {
    return this.#compare("gt", other);
  }
  gte(other: Tensor | number): Tensor {
    return this.#compare("gte", other);
  }

  #assertBool(name: string): void {
    if (this.dtype !== "bool") {
      throw new TypeError(`${name} expects a bool tensor, got ${this.dtype}`);
    }
  }

  /** Elementwise logical AND. Both operands must be `bool` tensors (cast() first). */
  logicalAnd(other: Tensor): Tensor {
    this.#assertBool("logicalAnd");
    other.#assertBool("logicalAnd");
    const outShape = broadcastShapes(this.shape, other.shape);
    const out = Tensor.zeros(outShape, { dtype: "bool" });
    const aStrides = broadcastStrides(this.shape, this.strides, outShape);
    const bStrides = broadcastStrides(other.shape, other.strides, outShape);
    const index = new Array<number>(outShape.length).fill(0);
    let aOff = this.offset;
    let bOff = other.offset;
    const outData = out.data as Uint8Array;
    for (let i = 0; i < out.size; i++) {
      outData[i] =
        (this.data[aOff] as number) !== 0 && (other.data[bOff] as number) !== 0
          ? 1
          : 0;
      for (let axis = outShape.length - 1; axis >= 0; axis--) {
        index[axis] = (index[axis] as number) + 1;
        aOff += aStrides[axis] as number;
        bOff += bStrides[axis] as number;
        if ((index[axis] as number) < (outShape[axis] as number)) break;
        index[axis] = 0;
        aOff -= (outShape[axis] as number) * (aStrides[axis] as number);
        bOff -= (outShape[axis] as number) * (bStrides[axis] as number);
      }
    }
    return out;
  }

  /** Elementwise logical OR. Both operands must be `bool` tensors (cast() first). */
  logicalOr(other: Tensor): Tensor {
    this.#assertBool("logicalOr");
    other.#assertBool("logicalOr");
    const outShape = broadcastShapes(this.shape, other.shape);
    const out = Tensor.zeros(outShape, { dtype: "bool" });
    const aStrides = broadcastStrides(this.shape, this.strides, outShape);
    const bStrides = broadcastStrides(other.shape, other.strides, outShape);
    const index = new Array<number>(outShape.length).fill(0);
    let aOff = this.offset;
    let bOff = other.offset;
    const outData = out.data as Uint8Array;
    for (let i = 0; i < out.size; i++) {
      outData[i] =
        (this.data[aOff] as number) !== 0 || (other.data[bOff] as number) !== 0
          ? 1
          : 0;
      for (let axis = outShape.length - 1; axis >= 0; axis--) {
        index[axis] = (index[axis] as number) + 1;
        aOff += aStrides[axis] as number;
        bOff += bStrides[axis] as number;
        if ((index[axis] as number) < (outShape[axis] as number)) break;
        index[axis] = 0;
        aOff -= (outShape[axis] as number) * (aStrides[axis] as number);
        bOff -= (outShape[axis] as number) * (bStrides[axis] as number);
      }
    }
    return out;
  }

  /** Elementwise logical NOT. Must be a `bool` tensor (cast() first). */
  logicalNot(): Tensor {
    this.#assertBool("logicalNot");
    const out = Tensor.zeros(this.shape, { dtype: "bool" });
    const outData = out.data as Uint8Array;
    const sourceOffsets = this.elementOffsets();
    let i = 0;
    for (const off of sourceOffsets) {
      outData[i++] = (this.data[off] as number) === 0 ? 1 : 0;
    }
    return out;
  }

  /** True if any element is non-zero/true (axis omitted) or along one axis. */
  any(axis?: Axis): Tensor {
    return this.#boolReduce(axis, false);
  }

  /** True if every element is non-zero/true (axis omitted) or along one axis. */
  all(axis?: Axis): Tensor {
    return this.#boolReduce(axis, true);
  }

  #boolReduce(axis: Axis | undefined, requireAll: boolean): Tensor {
    if (axis === undefined) {
      let result = requireAll;
      for (const off of this.elementOffsets()) {
        const truthy = (this.data[off] as number | bigint) != 0;
        result = requireAll ? result && truthy : result || truthy;
      }
      const out = Tensor.zeros([], { dtype: "bool" });
      (out.data as Uint8Array)[0] = result ? 1 : 0;
      return out;
    }
    const ax = this.#normalizeAxis(axis);
    const outShape = this.shape.filter((_, i) => i !== ax);
    const reduceDim = this.shape[ax] as number;
    const reduceStride = this.strides[ax] as number;
    const out = Tensor.zeros(outShape, { dtype: "bool" });
    const leadStrides = this.strides.filter((_, i) => i !== ax);
    const leadSize = shapeSize(outShape);
    const index = new Array<number>(outShape.length).fill(0);
    let base = this.offset;
    const outData = out.data as Uint8Array;
    for (let i = 0; i < leadSize; i++) {
      let result = requireAll;
      for (let j = 0; j < reduceDim; j++) {
        const truthy = (this.data[base + j * reduceStride] as number | bigint) != 0;
        result = requireAll ? result && truthy : result || truthy;
      }
      outData[i] = result ? 1 : 0;
      for (let a = outShape.length - 1; a >= 0; a--) {
        index[a] = (index[a] as number) + 1;
        base += leadStrides[a] as number;
        if ((index[a] as number) < (outShape[a] as number)) break;
        index[a] = 0;
        base -= (outShape[a] as number) * (leadStrides[a] as number);
      }
    }
    return out;
  }

  // ---- reductions -----------------------------------------------------------

  /** Sum over all elements (axis omitted) or along one axis. */
  sum(axis?: Axis): Tensor {
    return this.#reduce(axis, false);
  }

  /** Mean over all elements (axis omitted) or along one axis. Always float output. */
  mean(axis?: Axis): Tensor {
    return this.#reduce(axis, true);
  }

  #reduce(axis: Axis | undefined, mean: boolean): Tensor {
    const big = isBigIntDType(this.dtype);
    // NumPy: mean of any integer dtype yields f64; sum keeps the dtype.
    const isFloat = this.dtype === "f32" || this.dtype === "f64";
    const outDtype: DType = mean && !isFloat ? "f64" : this.dtype;

    if (axis === undefined) {
      const count = this.size;
      const out = Tensor.zeros([], { dtype: outDtype });
      if (big) {
        let acc = 0n;
        for (const off of this.elementOffsets()) {
          acc += this.data[off] as bigint;
        }
        if (mean) (out.data as Float64Array)[0] = Number(acc) / count;
        else (out.data as BigInt64Array)[0] = acc;
      } else {
        let acc = 0;
        for (const off of this.elementOffsets()) {
          acc += this.data[off] as number;
        }
        (out.data as Float64Array)[0] = mean ? acc / count : acc;
      }
      return out;
    }

    let ax = axis;
    if (ax < 0) ax += this.ndim;
    if (ax < 0 || ax >= this.ndim) {
      throw new RangeError(`axis ${axis} out of range for ndim ${this.ndim}`);
    }
    const outShape = this.shape.filter((_, i) => i !== ax);
    const reduceDim = this.shape[ax] as number;
    const reduceStride = this.strides[ax] as number;
    const out = Tensor.zeros(outShape, { dtype: outDtype });

    // Walk the non-reduced axes in C-order; for each position, accumulate
    // along the pinned axis via its stride.
    const leadShape = outShape;
    const leadStrides = this.strides.filter((_, i) => i !== ax);
    const leadSize = shapeSize(outShape);
    const index = new Array<number>(leadShape.length).fill(0);
    let base = this.offset;
    for (let i = 0; i < leadSize; i++) {
      if (big) {
        let acc = 0n;
        for (let j = 0; j < reduceDim; j++) {
          acc += this.data[base + j * reduceStride] as bigint;
        }
        if (mean) (out.data as Float64Array)[i] = Number(acc) / reduceDim;
        else (out.data as BigInt64Array)[i] = acc;
      } else {
        let acc = 0;
        for (let j = 0; j < reduceDim; j++) {
          acc += this.data[base + j * reduceStride] as number;
        }
        (out.data as Float64Array)[i] = mean ? acc / reduceDim : acc;
      }
      for (let a = leadShape.length - 1; a >= 0; a--) {
        index[a] = (index[a] as number) + 1;
        base += leadStrides[a] as number;
        if ((index[a] as number) < (leadShape[a] as number)) break;
        index[a] = 0;
        base -= (leadShape[a] as number) * (leadStrides[a] as number);
      }
    }
    return out;
  }

  // ---- min/max & argmin/argmax (issue #4) ------------------------------------

  min(axis?: Axis): Tensor {
    return this.#extremum(axis, false);
  }

  max(axis?: Axis): Tensor {
    return this.#extremum(axis, true);
  }

  #extremum(axis: Axis | undefined, wantMax: boolean): Tensor {
    const label = wantMax ? "max" : "min";
    const big = isBigIntDType(this.dtype);
    const better = (v: number | bigint, best: number | bigint): boolean =>
      big
        ? wantMax
          ? (v as bigint) > (best as bigint)
          : (v as bigint) < (best as bigint)
        : wantMax
          ? (v as number) > (best as number)
          : (v as number) < (best as number);

    if (axis === undefined) {
      if (this.size === 0) throw new RangeError(`${label} of an empty tensor`);
      let best: number | bigint | undefined;
      for (const off of this.elementOffsets()) {
        const v = this.data[off] as number | bigint;
        if (best === undefined || better(v, best)) best = v;
      }
      const out = Tensor.zeros([], { dtype: this.dtype });
      out.data[0] = best as never;
      return out;
    }

    const ax = this.#normalizeAxis(axis);
    const reduceDim = this.shape[ax] as number;
    if (reduceDim === 0) throw new RangeError(`${label} of an empty axis`);
    const outShape = this.shape.filter((_, i) => i !== ax);
    const reduceStride = this.strides[ax] as number;
    const out = Tensor.zeros(outShape, { dtype: this.dtype });
    const leadStrides = this.strides.filter((_, i) => i !== ax);
    const leadSize = shapeSize(outShape);
    const index = new Array<number>(outShape.length).fill(0);
    let base = this.offset;
    for (let i = 0; i < leadSize; i++) {
      let best = this.data[base] as number | bigint;
      for (let j = 1; j < reduceDim; j++) {
        const v = this.data[base + j * reduceStride] as number | bigint;
        if (better(v, best)) best = v;
      }
      out.data[i] = best as never;
      for (let a = outShape.length - 1; a >= 0; a--) {
        index[a] = (index[a] as number) + 1;
        base += leadStrides[a] as number;
        if ((index[a] as number) < (outShape[a] as number)) break;
        index[a] = 0;
        base -= (outShape[a] as number) * (leadStrides[a] as number);
      }
    }
    return out;
  }

  /** Index of the minimum: flattened C-order index if `axis` is omitted. */
  argmin(axis?: Axis): Tensor {
    return this.#argExtremum(axis, false);
  }

  /** Index of the maximum: flattened C-order index if `axis` is omitted. */
  argmax(axis?: Axis): Tensor {
    return this.#argExtremum(axis, true);
  }

  #argExtremum(axis: Axis | undefined, wantMax: boolean): Tensor {
    const label = wantMax ? "argmax" : "argmin";
    const big = isBigIntDType(this.dtype);
    const better = (v: number | bigint, best: number | bigint): boolean =>
      big
        ? wantMax
          ? (v as bigint) > (best as bigint)
          : (v as bigint) < (best as bigint)
        : wantMax
          ? (v as number) > (best as number)
          : (v as number) < (best as number);

    if (axis === undefined) {
      if (this.size === 0) throw new RangeError(`${label} of an empty tensor`);
      let best: number | bigint | undefined;
      let bestIndex = 0;
      let i = 0;
      for (const off of this.elementOffsets()) {
        const v = this.data[off] as number | bigint;
        if (best === undefined || better(v, best)) {
          best = v;
          bestIndex = i;
        }
        i++;
      }
      const out = Tensor.zeros([], { dtype: "i32" });
      (out.data as Int32Array)[0] = bestIndex;
      return out;
    }

    const ax = this.#normalizeAxis(axis);
    const reduceDim = this.shape[ax] as number;
    if (reduceDim === 0) throw new RangeError(`${label} of an empty axis`);
    const outShape = this.shape.filter((_, i) => i !== ax);
    const reduceStride = this.strides[ax] as number;
    const out = Tensor.zeros(outShape, { dtype: "i32" });
    const leadStrides = this.strides.filter((_, i) => i !== ax);
    const leadSize = shapeSize(outShape);
    const index = new Array<number>(outShape.length).fill(0);
    let base = this.offset;
    const outData = out.data as Int32Array;
    for (let i = 0; i < leadSize; i++) {
      let best = this.data[base] as number | bigint;
      let bestIndex = 0;
      for (let j = 1; j < reduceDim; j++) {
        const v = this.data[base + j * reduceStride] as number | bigint;
        if (better(v, best)) {
          best = v;
          bestIndex = j;
        }
      }
      outData[i] = bestIndex;
      for (let a = outShape.length - 1; a >= 0; a--) {
        index[a] = (index[a] as number) + 1;
        base += leadStrides[a] as number;
        if ((index[a] as number) < (outShape[a] as number)) break;
        index[a] = 0;
        base -= (outShape[a] as number) * (leadStrides[a] as number);
      }
    }
    return out;
  }

  // ---- variance & standard deviation (issue #4) -------------------------------

  /**
   * Sample/population variance (`ddof=0` by default — population variance,
   * matching NumPy's default). Built on existing ops rather than a new
   * strided loop: `cast` (int dtypes upcast to f64, matching #reduce's mean
   * semantics) → `mean` → broadcast-subtract → square → `sum` → `div`.
   */
  variance(axis?: Axis, options: { ddof?: number } = {}): Tensor {
    const ddof = options.ddof ?? 0;
    const isFloat = this.dtype === "f32" || this.dtype === "f64";
    const working = isFloat ? this : this.cast("f64");
    const meanTensor = working.mean(axis);

    let meanBroadcastable: Tensor;
    if (axis === undefined) {
      meanBroadcastable = meanTensor; // 0-d broadcasts against anything
    } else {
      const ax = this.#normalizeAxis(axis);
      const expanded = [...meanTensor.shape];
      expanded.splice(ax, 0, 1);
      meanBroadcastable = meanTensor.reshape(expanded); // #reduce's output is always contiguous
    }

    const centered = working.sub(meanBroadcastable);
    const count =
      axis === undefined ? this.size : (this.shape[this.#normalizeAxis(axis)] as number);
    const denom = count - ddof;
    if (denom <= 0) {
      throw new RangeError(`variance: ddof=${ddof} >= count=${count}`);
    }
    return centered.mul(centered).sum(axis).div(denom);
  }

  /** Standard deviation — `sqrt(variance(axis, options))`. */
  std(axis?: Axis, options: { ddof?: number } = {}): Tensor {
    return this.variance(axis, options).sqrt();
  }

  // ---- cumulative scans (issue #4) --------------------------------------------

  /** Running sum along `axis` (flattens first if `axis` is omitted, per NumPy). */
  cumsum(axis?: Axis): Tensor {
    return this.#cumulative(axis, "add");
  }

  /** Running product along `axis` (flattens first if `axis` is omitted, per NumPy). */
  cumprod(axis?: Axis): Tensor {
    return this.#cumulative(axis, "mul");
  }

  #cumulative(axis: Axis | undefined, op: "add" | "mul"): Tensor {
    const source = axis === undefined ? this.#contiguousView().reshape([this.size]) : this;
    const ax = axis === undefined ? 0 : source.#normalizeAxis(axis);
    const big = isBigIntDType(source.dtype);
    const out = Tensor.zeros(source.shape, { dtype: source.dtype });

    const scanDim = source.shape[ax] as number;
    const scanStrideIn = source.strides[ax] as number;
    const scanStrideOut = out.strides[ax] as number; // out is fresh & contiguous
    const leadShape = source.shape.filter((_, i) => i !== ax);
    const leadStridesIn = source.strides.filter((_, i) => i !== ax);
    const leadStridesOut = out.strides.filter((_, i) => i !== ax);
    const leadSize = shapeSize(leadShape);
    const index = new Array<number>(leadShape.length).fill(0);
    let baseIn = source.offset;
    let baseOut = 0;

    for (let i = 0; i < leadSize; i++) {
      if (big) {
        let acc = op === "add" ? 0n : 1n;
        for (let j = 0; j < scanDim; j++) {
          const v = source.data[baseIn + j * scanStrideIn] as bigint;
          acc = op === "add" ? acc + v : acc * v;
          (out.data as BigInt64Array)[baseOut + j * scanStrideOut] = acc;
        }
      } else {
        let acc = op === "add" ? 0 : 1;
        for (let j = 0; j < scanDim; j++) {
          const v = source.data[baseIn + j * scanStrideIn] as number;
          acc = op === "add" ? acc + v : acc * v;
          (out.data as Float64Array)[baseOut + j * scanStrideOut] = acc;
        }
      }
      for (let a = leadShape.length - 1; a >= 0; a--) {
        index[a] = (index[a] as number) + 1;
        baseIn += leadStridesIn[a] as number;
        baseOut += leadStridesOut[a] as number;
        if ((index[a] as number) < (leadShape[a] as number)) break;
        index[a] = 0;
        baseIn -= (leadShape[a] as number) * (leadStridesIn[a] as number);
        baseOut -= (leadShape[a] as number) * (leadStridesOut[a] as number);
      }
    }
    return out;
  }

  // ---- sorting & top-k (issue #4) ---------------------------------------------

  /** Sort along `axis` (default: last axis, matching NumPy). */
  sort(axis: Axis = -1): Tensor {
    return this.#sortAlong(axis).values;
  }

  /** Indices that would sort along `axis` (default: last axis). */
  argsort(axis: Axis = -1): Tensor {
    return this.#sortAlong(axis).indices;
  }

  #sortAlong(axis: Axis): { values: Tensor; indices: Tensor } {
    const ax = this.#normalizeAxis(axis);
    const reduceDim = this.shape[ax] as number;
    const reduceStride = this.strides[ax] as number;
    const values = Tensor.zeros(this.shape, { dtype: this.dtype });
    const indices = Tensor.zeros(this.shape, { dtype: "i32" });
    const outAxisStride = values.strides[ax] as number; // values/indices share layout
    const leadShape = this.shape.filter((_, i) => i !== ax);
    const leadStridesIn = this.strides.filter((_, i) => i !== ax);
    const leadStridesOut = values.strides.filter((_, i) => i !== ax);
    const leadSize = shapeSize(leadShape);
    const index = new Array<number>(leadShape.length).fill(0);
    let baseIn = this.offset;
    let baseOut = 0;
    const big = isBigIntDType(this.dtype);
    const indicesData = indices.data as Int32Array;

    for (let i = 0; i < leadSize; i++) {
      const positions = Array.from({ length: reduceDim }, (_, j) => j);
      if (big) {
        positions.sort((p, q) => {
          const a = this.data[baseIn + p * reduceStride] as bigint;
          const b = this.data[baseIn + q * reduceStride] as bigint;
          return a < b ? -1 : a > b ? 1 : 0;
        });
      } else {
        positions.sort(
          (p, q) =>
            (this.data[baseIn + p * reduceStride] as number) -
            (this.data[baseIn + q * reduceStride] as number),
        );
      }
      for (let k = 0; k < reduceDim; k++) {
        const srcPos = positions[k] as number;
        values.data[baseOut + k * outAxisStride] = this.data[
          baseIn + srcPos * reduceStride
        ] as never;
        indicesData[baseOut + k * outAxisStride] = srcPos;
      }
      for (let a = leadShape.length - 1; a >= 0; a--) {
        index[a] = (index[a] as number) + 1;
        baseIn += leadStridesIn[a] as number;
        baseOut += leadStridesOut[a] as number;
        if ((index[a] as number) < (leadShape[a] as number)) break;
        index[a] = 0;
        baseIn -= (leadShape[a] as number) * (leadStridesIn[a] as number);
        baseOut -= (leadShape[a] as number) * (leadStridesOut[a] as number);
      }
    }
    return { values, indices };
  }

  /**
   * The `k` largest (default) or smallest values along `axis` (default: last
   * axis), plus their original indices. `largest: true` returns descending
   * order; `largest: false` returns ascending order. A VIEW into a freshly
   * sorted buffer (built on {@link sort}/{@link slice}, not a new algorithm).
   */
  topK(
    k: number,
    options: { axis?: Axis; largest?: boolean } = {},
  ): { values: Tensor; indices: Tensor } {
    const axis = options.axis ?? -1;
    const largest = options.largest ?? true;
    const ax = this.#normalizeAxis(axis);
    const dim = this.shape[ax] as number;
    if (!Number.isInteger(k) || k <= 0 || k > dim) {
      throw new RangeError(`topK: k=${k} out of range for axis size ${dim}`);
    }
    const { values, indices } = this.#sortAlong(axis); // ascending
    const specs: Array<SliceSpec | null> = new Array(this.ndim).fill(null);
    specs[ax] = largest
      ? { start: -1, end: -(k + 1), step: -1 } // last k, reversed -> descending
      : { start: 0, end: k }; // first k, already ascending
    return { values: values.slice(...specs), indices: indices.slice(...specs) };
  }

  // ---- .npy I/O -------------------------------------------------------------

  /** Serialize to NPY v1.0 bytes (packs a non-contiguous view first). */
  toNpy(): Uint8Array {
    const packed = this.contiguous();
    return serializeNpy({
      data:
        packed.data.length === packed.size
          ? packed.data
          : (packed.data.slice(
              packed.offset,
              packed.offset + packed.size,
            ) as AnyTypedArray),
      shape: [...packed.shape],
      dtype: packed.dtype,
    });
  }

  static fromNpy(bytes: Uint8Array): Tensor {
    const { data, shape, dtype } = parseNpy(bytes);
    return Tensor.fromTypedArray(data, shape, { dtype });
  }
}

/**
 * `random` namespace (issue #5) — exposed as a named object rather than
 * static Tensor methods, matching the source design's "expose as named
 * namespaces" convention. `seed(n)` returns a reproducible generator; pass
 * it as `{ rng }` for deterministic output, or omit it for a fresh
 * non-deterministic one each call (no shared global state).
 */
export const random = {
  seed: rngSeed,

  uniform(
    shape: Shape,
    options: RandomOptions & { min?: number; max?: number } = {},
  ): Tensor {
    const dtype = options.dtype ?? "f32";
    const min = options.min ?? 0;
    const max = options.max ?? 1;
    const data = fillFrom(shape, dtype, options.rng, (rng) =>
      uniformSample(rng, min, max),
    );
    return Tensor.fromTypedArray(data, shape, { dtype });
  },

  normal(
    shape: Shape,
    options: RandomOptions & { mean?: number; std?: number } = {},
  ): Tensor {
    const dtype = options.dtype ?? "f32";
    const mean = options.mean ?? 0;
    const std = options.std ?? 1;
    const data = fillFrom(shape, dtype, options.rng, (rng) =>
      normalSample(rng, mean, std),
    );
    return Tensor.fromTypedArray(data, shape, { dtype });
  },

  randint(
    low: number,
    high: number,
    shape: Shape,
    options: RandomOptions = {},
  ): Tensor {
    const dtype = options.dtype ?? "i32";
    const data = fillFrom(shape, dtype, options.rng, (rng) =>
      randintSample(rng, low, high),
    );
    return Tensor.fromTypedArray(data, shape, { dtype });
  },
};
