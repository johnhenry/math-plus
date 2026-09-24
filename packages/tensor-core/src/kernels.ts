/**
 * Flat typed-array kernels behind tensor-core's contiguous fast paths
 * (issue #120). Not re-exported from the package entry; device packages
 * that store flat typed arrays (the CPU `Backend`, issue #144) import them
 * from the `@johnhenry/math-plus-tensor-core/kernels` subpath
 * (src/public-kernels.ts), together with the fused NN kernels in
 * src/nn-kernels.ts.
 *
 * Contract every kernel here keeps: it produces results BIT-IDENTICAL to the
 * general strided path in `index.ts` it replaces. Same operation per element,
 * same accumulation order (ascending along the reduced axis, accumulated in a
 * JS number, i.e. f64), and the same points at which a value is rounded to
 * the storage dtype (a typed-array store, or `Math.fround` where the general
 * path would have materialized an f32 temporary). That is what lets the
 * fast-path/strided equivalence tests assert exact equality rather than a
 * tolerance, and what keeps the NumPy differential suites unchanged.
 *
 * Scope boundary: these handle every NON-bigint dtype (the Number-valued
 * typed arrays). i64/u64 always take the general path, as do strided views
 * (transposes, negative-step slices, broadcast stride-0 views) — callers
 * check `isContiguous` before dispatching here.
 */

/** Every Number-valued typed array tensor-core allocates (all dtypes except i64/u64). */
export type NumArray =
  | Float64Array
  | Float32Array
  | Int32Array
  | Uint32Array
  | Int16Array
  | Uint16Array
  | Int8Array
  | Uint8Array;

// Binary op codes (plain numbers: node's type stripping doesn't allow `enum`).
export const OP_ADD = 0;
export const OP_SUB = 1;
export const OP_MUL = 2;
export const OP_DIV = 3;

/** `o[oo+i] = a[ao+i] (op) b[bo+i]` for `i < n`. One loop per op so the op is never a per-element call. */
export function binaryFlat(
  op: number,
  a: NumArray, ao: number,
  b: NumArray, bo: number,
  o: NumArray, oo: number,
  n: number,
): void {
  switch (op) {
    case OP_ADD: for (let i = 0; i < n; i++) o[oo + i] = a[ao + i]! + b[bo + i]!; break;
    case OP_SUB: for (let i = 0; i < n; i++) o[oo + i] = a[ao + i]! - b[bo + i]!; break;
    case OP_MUL: for (let i = 0; i < n; i++) o[oo + i] = a[ao + i]! * b[bo + i]!; break;
    default: for (let i = 0; i < n; i++) o[oo + i] = a[ao + i]! / b[bo + i]!;
  }
}

/** `o[oo+i] = a[ao+i] (op) s` — the scalar on the RIGHT. */
export function binaryScalarRight(
  op: number,
  a: NumArray, ao: number,
  s: number,
  o: NumArray, oo: number,
  n: number,
): void {
  switch (op) {
    case OP_ADD: for (let i = 0; i < n; i++) o[oo + i] = a[ao + i]! + s; break;
    case OP_SUB: for (let i = 0; i < n; i++) o[oo + i] = a[ao + i]! - s; break;
    case OP_MUL: for (let i = 0; i < n; i++) o[oo + i] = a[ao + i]! * s; break;
    default: for (let i = 0; i < n; i++) o[oo + i] = a[ao + i]! / s;
  }
}

/** `o[oo+i] = s (op) b[bo+i]` — the scalar on the LEFT (order matters for sub/div). */
export function binaryScalarLeft(
  op: number,
  s: number,
  b: NumArray, bo: number,
  o: NumArray, oo: number,
  n: number,
): void {
  switch (op) {
    case OP_ADD: for (let i = 0; i < n; i++) o[oo + i] = s + b[bo + i]!; break;
    case OP_SUB: for (let i = 0; i < n; i++) o[oo + i] = s - b[bo + i]!; break;
    case OP_MUL: for (let i = 0; i < n; i++) o[oo + i] = s * b[bo + i]!; break;
    default: for (let i = 0; i < n; i++) o[oo + i] = s / b[bo + i]!;
  }
}

// Comparison op codes.
export const CMP_EQ = 0;
export const CMP_NE = 1;
export const CMP_LT = 2;
export const CMP_LTE = 3;
export const CMP_GT = 4;
export const CMP_GTE = 5;

/**
 * `o[i] = a[ao+i] (cmp) b[bo + i*bStep] ? 1 : 0` — `bStep` 1 for same-shape,
 * 0 for a scalar right operand.
 */
export function compareFlat(
  op: number,
  a: NumArray, ao: number,
  b: NumArray, bo: number, bStep: number,
  o: Uint8Array,
  n: number,
): void {
  switch (op) {
    case CMP_EQ: for (let i = 0; i < n; i++) o[i] = a[ao + i]! === b[bo + i * bStep]! ? 1 : 0; break;
    case CMP_NE: for (let i = 0; i < n; i++) o[i] = a[ao + i]! !== b[bo + i * bStep]! ? 1 : 0; break;
    case CMP_LT: for (let i = 0; i < n; i++) o[i] = a[ao + i]! < b[bo + i * bStep]! ? 1 : 0; break;
    case CMP_LTE: for (let i = 0; i < n; i++) o[i] = a[ao + i]! <= b[bo + i * bStep]! ? 1 : 0; break;
    case CMP_GT: for (let i = 0; i < n; i++) o[i] = a[ao + i]! > b[bo + i * bStep]! ? 1 : 0; break;
    default: for (let i = 0; i < n; i++) o[i] = a[ao + i]! >= b[bo + i * bStep]! ? 1 : 0;
  }
}

/**
 * Sum (or mean) along the middle axis of a contiguous `[outer, dim, inner]`
 * block starting at `xo`, into `o[0 .. outer*inner)`. Per output the
 * accumulation runs over `j = 0..dim-1` in order — the same order as the
 * strided path — but the `inner > 1` case walks memory row-by-row (into an
 * f64 accumulator row) instead of striding down columns.
 */
export function sumAxis(
  x: NumArray, xo: number,
  outer: number, dim: number, inner: number,
  o: NumArray,
  mean: boolean,
): void {
  if (inner === 1) {
    for (let r = 0; r < outer; r++) {
      const base = xo + r * dim;
      let acc = 0;
      for (let j = 0; j < dim; j++) acc += x[base + j]!;
      o[r] = mean ? acc / dim : acc;
    }
    return;
  }
  const acc = new Float64Array(inner);
  for (let r = 0; r < outer; r++) {
    acc.fill(0);
    const base = xo + r * dim * inner;
    for (let j = 0; j < dim; j++) {
      const row = base + j * inner;
      for (let t = 0; t < inner; t++) acc[t] = acc[t]! + x[row + t]!;
    }
    const oo = r * inner;
    for (let t = 0; t < inner; t++) o[oo + t] = mean ? acc[t]! / dim : acc[t]!;
  }
}

/**
 * Max/min along the middle axis of a contiguous `[outer, dim, inner]` block
 * (`dim >= 1`). Keeps the strided path's exact comparison semantics: the
 * running best starts at element 0 and is replaced only when `v > best`
 * (`v < best` for min), so NaN is sticky only in position 0.
 */
export function extremumAxis(
  x: NumArray, xo: number,
  outer: number, dim: number, inner: number,
  o: NumArray,
  wantMax: boolean,
): void {
  if (inner === 1) {
    for (let r = 0; r < outer; r++) {
      const base = xo + r * dim;
      let best = x[base]!;
      if (wantMax) {
        for (let j = 1; j < dim; j++) {
          const v = x[base + j]!;
          if (v > best) best = v;
        }
      } else {
        for (let j = 1; j < dim; j++) {
          const v = x[base + j]!;
          if (v < best) best = v;
        }
      }
      o[r] = best;
    }
    return;
  }
  const best = new Float64Array(inner);
  for (let r = 0; r < outer; r++) {
    const base = xo + r * dim * inner;
    for (let t = 0; t < inner; t++) best[t] = x[base + t]!;
    for (let j = 1; j < dim; j++) {
      const row = base + j * inner;
      if (wantMax) {
        for (let t = 0; t < inner; t++) {
          const v = x[row + t]!;
          if (v > best[t]!) best[t] = v;
        }
      } else {
        for (let t = 0; t < inner; t++) {
          const v = x[row + t]!;
          if (v < best[t]!) best[t] = v;
        }
      }
    }
    const oo = r * inner;
    for (let t = 0; t < inner; t++) o[oo + t] = best[t]!;
  }
}

/**
 * Fused, numerically-stable softmax along the middle axis of a contiguous
 * f32/f64 `[outer, dim, inner]` block (`dim >= 1`), written into the fresh
 * contiguous `o`. Three passes over each lane (max, exp+sum, scale) and no
 * temporaries beyond the output — replacing the general path's
 * max/sub/broadcast-copy/exp/sum/broadcast-copy/div chain (~6 temporaries).
 *
 * Bit-identical to that chain: with `f32`, every intermediate the chain
 * would have stored in a Float32Array (`x - max`, `exp(.)`, the row sum) is
 * rounded with `Math.fround` at the same point; the exp values themselves
 * are stored into `o` (rounding them) and re-read for the sum and divide.
 */
export function softmaxAxis(
  x: Float32Array | Float64Array, xo: number,
  outer: number, dim: number, inner: number,
  o: Float32Array | Float64Array,
  f32: boolean,
): void {
  for (let r = 0; r < outer; r++) {
    const base = r * dim * inner;
    for (let t = 0; t < inner; t++) {
      const lane = base + t;
      let max = x[xo + lane]!;
      for (let j = 1; j < dim; j++) {
        const v = x[xo + lane + j * inner]!;
        if (v > max) max = v;
      }
      let sum = 0;
      for (let j = 0; j < dim; j++) {
        const k = lane + j * inner;
        const d = x[xo + k]! - max;
        o[k] = Math.exp(f32 ? Math.fround(d) : d);
        sum += o[k]!;
      }
      const s = f32 ? Math.fround(sum) : sum;
      for (let j = 0; j < dim; j++) {
        const k = lane + j * inner;
        o[k] = o[k]! / s;
      }
    }
  }
}

/**
 * Fused variance along the middle axis of a contiguous f32/f64
 * `[outer, dim, inner]` block: two passes per lane (mean, then centered sum
 * of squares), no temporaries. `denom` is `dim - ddof`.
 *
 * Bit-identical to the general path's `mean → sub → mul → sum → div` chain:
 * with `f32`, the mean, each centered value, each square, the sum and the
 * denominator are rounded with `Math.fround` exactly where that chain
 * materialized an f32 tensor (or an f32 0-d scalar, for `denom`).
 */
export function varianceAxis(
  x: Float32Array | Float64Array, xo: number,
  outer: number, dim: number, inner: number,
  o: Float32Array | Float64Array,
  denom: number,
  f32: boolean,
): void {
  const d = f32 ? Math.fround(denom) : denom;
  for (let r = 0; r < outer; r++) {
    const base = xo + r * dim * inner;
    for (let t = 0; t < inner; t++) {
      const lane = base + t;
      let acc = 0;
      for (let j = 0; j < dim; j++) acc += x[lane + j * inner]!;
      const mean = f32 ? Math.fround(acc / dim) : acc / dim;
      let ss = 0;
      for (let j = 0; j < dim; j++) {
        const c = f32 ? Math.fround(x[lane + j * inner]! - mean) : x[lane + j * inner]! - mean;
        ss += f32 ? Math.fround(c * c) : c * c;
      }
      o[r * inner + t] = (f32 ? Math.fround(ss) : ss) / d;
    }
  }
}

/**
 * C[cOff + i*ldc + j] = Σ_p A[i*k + p] · B[j*k + p]   (A · Bᵀ, "NT" layout)
 *
 * Both operands are PACKED f64 panels (row-major, leading dimension `k`) so
 * every inner-loop load is unit-stride and the kernel is monomorphic. A 4×4
 * register block gives 16 independent accumulators per `p` step; `j` is
 * blocked so a panel of B stays cache-resident while all rows of A pass over
 * it. Each output accumulates `p = 0..k-1` in order starting from 0 — the
 * exact sequence of f64 operations the naive triple loop performs, so the
 * result is bit-identical to it.
 *
 * Adapted from laya-js `packages/backend-cpu/src/gemm.ts` `gemmNT`
 * (Apache-2.0, same author), generalized from f32 to packed f64 panels.
 */
export function gemmNT(
  A: Float64Array,
  B: Float64Array,
  C: Float64Array, cOff: number, ldc: number,
  M: number, N: number, K: number,
): void {
  const JB = 64;
  for (let j0 = 0; j0 < N; j0 += JB) {
    const j1 = Math.min(N, j0 + JB);
    let i = 0;
    for (; i + 4 <= M; i += 4) {
      const a0 = i * K, a1 = a0 + K, a2 = a1 + K, a3 = a2 + K;
      const c0 = cOff + i * ldc, c1 = c0 + ldc, c2 = c1 + ldc, c3 = c2 + ldc;
      let j = j0;
      for (; j + 4 <= j1; j += 4) {
        const b0 = j * K, b1 = b0 + K, b2 = b1 + K, b3 = b2 + K;
        let s00 = 0, s01 = 0, s02 = 0, s03 = 0;
        let s10 = 0, s11 = 0, s12 = 0, s13 = 0;
        let s20 = 0, s21 = 0, s22 = 0, s23 = 0;
        let s30 = 0, s31 = 0, s32 = 0, s33 = 0;
        for (let p = 0; p < K; p++) {
          const x0 = A[a0 + p]!, x1 = A[a1 + p]!, x2 = A[a2 + p]!, x3 = A[a3 + p]!;
          const y0 = B[b0 + p]!, y1 = B[b1 + p]!, y2 = B[b2 + p]!, y3 = B[b3 + p]!;
          s00 += x0 * y0; s01 += x0 * y1; s02 += x0 * y2; s03 += x0 * y3;
          s10 += x1 * y0; s11 += x1 * y1; s12 += x1 * y2; s13 += x1 * y3;
          s20 += x2 * y0; s21 += x2 * y1; s22 += x2 * y2; s23 += x2 * y3;
          s30 += x3 * y0; s31 += x3 * y1; s32 += x3 * y2; s33 += x3 * y3;
        }
        C[c0 + j] = s00; C[c0 + j + 1] = s01; C[c0 + j + 2] = s02; C[c0 + j + 3] = s03;
        C[c1 + j] = s10; C[c1 + j + 1] = s11; C[c1 + j + 2] = s12; C[c1 + j + 3] = s13;
        C[c2 + j] = s20; C[c2 + j + 1] = s21; C[c2 + j + 2] = s22; C[c2 + j + 3] = s23;
        C[c3 + j] = s30; C[c3 + j + 1] = s31; C[c3 + j + 2] = s32; C[c3 + j + 3] = s33;
      }
      for (; j < j1; j++) {
        const b0 = j * K;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let p = 0; p < K; p++) {
          const y = B[b0 + p]!;
          s0 += A[a0 + p]! * y; s1 += A[a1 + p]! * y; s2 += A[a2 + p]! * y; s3 += A[a3 + p]! * y;
        }
        C[c0 + j] = s0; C[c1 + j] = s1; C[c2 + j] = s2; C[c3 + j] = s3;
      }
    }
    for (; i < M; i++) {
      const a0 = i * K, c0 = cOff + i * ldc;
      let j = j0;
      for (; j + 4 <= j1; j += 4) {
        const b0 = j * K, b1 = b0 + K, b2 = b1 + K, b3 = b2 + K;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let p = 0; p < K; p++) {
          const x = A[a0 + p]!;
          s0 += x * B[b0 + p]!; s1 += x * B[b1 + p]!; s2 += x * B[b2 + p]!; s3 += x * B[b3 + p]!;
        }
        C[c0 + j] = s0; C[c0 + j + 1] = s1; C[c0 + j + 2] = s2; C[c0 + j + 3] = s3;
      }
      for (; j < j1; j++) {
        const b0 = j * K;
        let s = 0;
        for (let p = 0; p < K; p++) s += A[a0 + p]! * B[b0 + p]!;
        C[c0 + j] = s;
      }
    }
  }
}

/**
 * Pack a strided `[rows, cols]` matrix view into a dense row-major f64 panel
 * (`dst[r*cols + c]`). Used for A as-is and for B transposed (pass B's column
 * stride as `rowStride` and its row stride as `colStride`). Reads raw storage
 * values, so it works for any Number-valued dtype; f32/int → f64 is exact.
 */
export function packPanel(
  src: NumArray, off: number, rowStride: number, colStride: number,
  rows: number, cols: number,
  dst: Float64Array,
): void {
  if (colStride === 1) {
    for (let r = 0; r < rows; r++) {
      const s = off + r * rowStride, d = r * cols;
      for (let c = 0; c < cols; c++) dst[d + c] = src[s + c]!;
    }
    return;
  }
  for (let r = 0; r < rows; r++) {
    const s = off + r * rowStride, d = r * cols;
    for (let c = 0; c < cols; c++) dst[d + c] = src[s + c * colStride]!;
  }
}

// ---------------------------------------------------------------------------
// Strided / broadcasting building blocks (issue #144). Shared by tensor-core's
// own fast paths (`contiguous()`, `argmax`/`argmin`, `cumsum`) and by device
// packages that store flat typed arrays (the CPU `Backend` in
// @johnhenry/math-plus-tensor-cpu), so there is one copy of each loop.
// ---------------------------------------------------------------------------

/**
 * Row-major strides of `shape` aligned to `outShape` (right-aligned numpy
 * broadcasting), with 0 on every axis `shape` broadcasts along (size 1, or
 * missing on the left).
 */
export function alignedStrides(shape: readonly number[], outShape: readonly number[]): number[] {
  const out = new Array<number>(outShape.length).fill(0);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    const d = shape[i]!;
    out[outShape.length - shape.length + i] = d === 1 ? 0 : acc;
    acc *= d;
  }
  return out;
}

/**
 * Offset of the first element of every "row" (all axes but the last, in C
 * order) of a `shape`-shaped walk over storage laid out with `strides`,
 * starting at `base`. Rank 0 and rank 1 give a single row at `base`.
 */
export function rowOffsets(shape: readonly number[], strides: readonly number[], base = 0): Float64Array {
  const rank = shape.length;
  let rows = 1;
  for (let i = 0; i < rank - 1; i++) rows *= shape[i]!;
  const offs = new Float64Array(rows);
  if (rows === 0) return offs;
  const idx = new Array<number>(Math.max(0, rank - 1)).fill(0);
  let off = base;
  for (let r = 0; r < rows; r++) {
    offs[r] = off;
    for (let d = rank - 2; d >= 0; d--) {
      idx[d]!++;
      off += strides[d]!;
      if (idx[d]! < shape[d]!) break;
      off -= strides[d]! * shape[d]!;
      idx[d] = 0;
    }
  }
  return offs;
}

/**
 * Copies the strided view (`src`, `off`, `shape`, `strides`) into `dst` in C
 * order starting at `dstOff`. A plain copy, so exact for every Number-valued
 * dtype (including f16/bf16 bit patterns stored as Uint16Array).
 */
export function stridedCopy(
  src: NumArray, off: number,
  shape: readonly number[], strides: readonly number[],
  dst: NumArray, dstOff = 0,
): void {
  const rank = shape.length;
  if (rank === 0) {
    dst[dstOff] = src[off]!;
    return;
  }
  const n = shape[rank - 1]!;
  if (n === 0) return;
  const s = strides[rank - 1]!;
  const offs = rowOffsets(shape, strides, off);
  const sameKind = src.constructor === dst.constructor;
  for (let r = 0; r < offs.length; r++) {
    const o = offs[r]!, d = dstOff + r * n;
    if (s === 1 && sameKind) dst.set(src.subarray(o, o + n), d);
    else for (let j = 0; j < n; j++) dst[d + j] = src[o + j * s]!;
  }
}

/** Binary op code for {@link binaryStrided}: NaN-propagating `max(a, b)`. */
export const OP_MAXIMUM = 4;
/** Binary op code for {@link binaryStrided}: `Math.pow(a, b)`. */
export const OP_POW = 5;

/**
 * `o[oo+j] = A[ao + j*as] (op) B[bo + j*bs]` for `j < n` — one row of a
 * broadcast binary op (a stride of 0 repeats an operand). Ops: OP_ADD,
 * OP_SUB, OP_MUL, OP_DIV, OP_MAXIMUM (NaN if either side is NaN), OP_POW.
 */
export function binaryStrided(
  op: number,
  A: NumArray, ao: number, as: number,
  B: NumArray, bo: number, bs: number,
  O: NumArray, oo: number,
  n: number,
): void {
  switch (op) {
    case OP_ADD: for (let j = 0; j < n; j++) O[oo + j] = A[ao + j * as]! + B[bo + j * bs]!; break;
    case OP_SUB: for (let j = 0; j < n; j++) O[oo + j] = A[ao + j * as]! - B[bo + j * bs]!; break;
    case OP_MUL: for (let j = 0; j < n; j++) O[oo + j] = A[ao + j * as]! * B[bo + j * bs]!; break;
    case OP_DIV: for (let j = 0; j < n; j++) O[oo + j] = A[ao + j * as]! / B[bo + j * bs]!; break;
    case OP_MAXIMUM:
      for (let j = 0; j < n; j++) {
        const x = A[ao + j * as]!, y = B[bo + j * bs]!;
        O[oo + j] = x !== x || y !== y ? NaN : x > y ? x : y;
      }
      break;
    default: for (let j = 0; j < n; j++) O[oo + j] = Math.pow(A[ao + j * as]!, B[bo + j * bs]!);
  }
}

/** Comparison code for {@link compareStrided}: both nonzero. */
export const CMP_AND = 6;
/** Comparison code for {@link compareStrided}: either nonzero. */
export const CMP_OR = 7;

/**
 * `o[oo+j] = A[ao + j*as] (cmp) B[bo + j*bs] ? 1 : 0` — one row of a
 * broadcast comparison. Codes: CMP_EQ … CMP_GTE, CMP_AND, CMP_OR
 * (nonzero-is-true).
 */
export function compareStrided(
  op: number,
  A: NumArray, ao: number, as: number,
  B: NumArray, bo: number, bs: number,
  O: Uint8Array, oo: number,
  n: number,
): void {
  switch (op) {
    case CMP_EQ: for (let j = 0; j < n; j++) O[oo + j] = A[ao + j * as]! === B[bo + j * bs]! ? 1 : 0; break;
    case CMP_NE: for (let j = 0; j < n; j++) O[oo + j] = A[ao + j * as]! !== B[bo + j * bs]! ? 1 : 0; break;
    case CMP_LT: for (let j = 0; j < n; j++) O[oo + j] = A[ao + j * as]! < B[bo + j * bs]! ? 1 : 0; break;
    case CMP_LTE: for (let j = 0; j < n; j++) O[oo + j] = A[ao + j * as]! <= B[bo + j * bs]! ? 1 : 0; break;
    case CMP_GT: for (let j = 0; j < n; j++) O[oo + j] = A[ao + j * as]! > B[bo + j * bs]! ? 1 : 0; break;
    case CMP_GTE: for (let j = 0; j < n; j++) O[oo + j] = A[ao + j * as]! >= B[bo + j * bs]! ? 1 : 0; break;
    case CMP_AND: for (let j = 0; j < n; j++) O[oo + j] = A[ao + j * as]! !== 0 && B[bo + j * bs]! !== 0 ? 1 : 0; break;
    default: for (let j = 0; j < n; j++) O[oo + j] = A[ao + j * as]! !== 0 || B[bo + j * bs]! !== 0 ? 1 : 0;
  }
}

/** `o[oo+j] = C[co + j*cs] ? A[ao + j*as] : B[bo + j*bs]` — one row of a broadcast select. */
export function whereStrided(
  C: NumArray, co: number, cs: number,
  A: NumArray, ao: number, as: number,
  B: NumArray, bo: number, bs: number,
  O: NumArray, oo: number,
  n: number,
): void {
  for (let j = 0; j < n; j++) O[oo + j] = C[co + j * cs] ? A[ao + j * as]! : B[bo + j * bs]!;
}

/**
 * Index of the first maximum (`wantMax`) or minimum along the middle axis of
 * a contiguous `[outer, dim, inner]` block (`dim >= 1`), into `o`. Same
 * comparison semantics as {@link extremumAxis}: the running best starts at
 * element 0 and moves only on a strict `>` (`<`), so a NaN wins only in
 * position 0.
 */
export function argExtremumAxis(
  x: NumArray, xo: number,
  outer: number, dim: number, inner: number,
  o: NumArray,
  wantMax: boolean,
): void {
  for (let r = 0; r < outer; r++) {
    const base = xo + r * dim * inner;
    for (let t = 0; t < inner; t++) {
      const lane = base + t;
      let best = x[lane]!, bi = 0;
      if (wantMax) {
        for (let j = 1; j < dim; j++) {
          const v = x[lane + j * inner]!;
          if (v > best) { best = v; bi = j; }
        }
      } else {
        for (let j = 1; j < dim; j++) {
          const v = x[lane + j * inner]!;
          if (v < best) { best = v; bi = j; }
        }
      }
      o[r * inner + t] = bi;
    }
  }
}

/**
 * Inclusive prefix sum along the middle axis of a contiguous
 * `[outer, dim, inner]` block into the contiguous `o` (same layout). The
 * running sum is a JS number (f64) and each partial is rounded only by the
 * store into `o`, exactly as tensor-core's general `cumsum` path does; an
 * Int32Array `o` therefore wraps like two's complement.
 */
export function cumsumAxis(
  x: NumArray, xo: number,
  outer: number, dim: number, inner: number,
  o: NumArray,
): void {
  for (let r = 0; r < outer; r++) {
    const base = r * dim * inner;
    for (let t = 0; t < inner; t++) {
      const lane = base + t;
      let acc = 0;
      for (let j = 0; j < dim; j++) {
        acc += x[xo + lane + j * inner]!;
        o[lane + j * inner] = acc;
      }
    }
  }
}

/**
 * Ascending sort of every lane along the middle axis of a contiguous
 * `[outer, dim, inner]` block into `o` (same layout). Uses the typed-array
 * numeric sort, so NaNs sort last. `lane` is scratch of length >= `dim` with
 * the same element type as `x`.
 */
export function sortAxis(
  x: NumArray, xo: number,
  outer: number, dim: number, inner: number,
  o: NumArray,
  lane: NumArray,
): void {
  const buf = lane.subarray(0, dim);
  for (let r = 0; r < outer; r++) {
    for (let t = 0; t < inner; t++) {
      const base = r * dim * inner + t;
      for (let j = 0; j < dim; j++) buf[j] = x[xo + base + j * inner]!;
      buf.sort();
      for (let j = 0; j < dim; j++) o[base + j * inner] = buf[j]!;
    }
  }
}

