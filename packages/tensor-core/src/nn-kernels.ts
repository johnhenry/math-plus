/**
 * Fused transformer / NN kernels over flat, contiguous typed arrays
 * (issue #144). They exist for the CPU `Backend` in
 * @johnhenry/math-plus-tensor-cpu, which stores tensors as flat f32 / i32 /
 * u8 arrays and implements the @johnhenry/tensor-backend contract on top of
 * these — so GEMM, softmax, LayerNorm, RoPE, attention and GELU each have
 * exactly one implementation in math-plus (AGENTS.md, canonical-
 * implementation rule). Exported from the `@johnhenry/math-plus-tensor-core/kernels`
 * subpath, not from the package entry.
 *
 * Conventions:
 * - Inputs are read as numbers from any Number-valued typed array (`NumArray`);
 *   outputs are written by typed-array store, so an f32 output rounds once.
 * - Accumulation (dot products, row statistics, softmax sums) is in f64 (JS
 *   numbers). Matrix products all go through `gemmNT` over packed f64 panels
 *   (kernels.ts), the same GEMM `Tensor.matmul` uses.
 * - No allocation beyond per-call scratch; shapes are validated by callers.
 *
 * Adapted from laya-js `packages/backend-cpu/src/index.ts` (MIT, same author).
 */
import { erf, geluErf } from "@johnhenry/math-plus-special";
import { gemmNT, packPanel, type NumArray } from "./kernels.ts";

// Unary op codes for unaryFlat (plain numbers: node's type stripping has no `enum`).
export const UN_EXP = 0;
export const UN_LOG = 1;
/** `x > 0 ? x : 0` (NaN → 0). */
export const UN_RELU = 2;
/** Exact erf GELU (`geluErf` from @johnhenry/math-plus-special). */
export const UN_GELU = 3;
export const UN_SQRT = 4;
export const UN_RSQRT = 5;
export const UN_TANH = 6;
/** Overflow-free logistic sigmoid. */
export const UN_SIGMOID = 7;
/** The canonical double-precision erf from @johnhenry/math-plus-special. */
export const UN_ERF = 8;
export const UN_NEG = 9;
export const UN_ABS = 10;

/** `o[oo+i] = f(x[xo+i])` for `i < n`; one loop per op so the op is never a per-element call. Evaluated in f64. */
export function unaryFlat(op: number, x: NumArray, xo: number, o: NumArray, oo: number, n: number): void {
  switch (op) {
    case UN_EXP: for (let i = 0; i < n; i++) o[oo + i] = Math.exp(x[xo + i]!); break;
    case UN_LOG: for (let i = 0; i < n; i++) o[oo + i] = Math.log(x[xo + i]!); break;
    case UN_RELU: for (let i = 0; i < n; i++) { const v = x[xo + i]!; o[oo + i] = v > 0 ? v : 0; } break;
    case UN_GELU: for (let i = 0; i < n; i++) o[oo + i] = geluErf(x[xo + i]!); break;
    case UN_SQRT: for (let i = 0; i < n; i++) o[oo + i] = Math.sqrt(x[xo + i]!); break;
    case UN_RSQRT: for (let i = 0; i < n; i++) o[oo + i] = 1 / Math.sqrt(x[xo + i]!); break;
    case UN_TANH: for (let i = 0; i < n; i++) o[oo + i] = Math.tanh(x[xo + i]!); break;
    case UN_SIGMOID:
      for (let i = 0; i < n; i++) {
        const v = x[xo + i]!;
        if (v >= 0) o[oo + i] = 1 / (1 + Math.exp(-v));
        else { const e = Math.exp(v); o[oo + i] = e / (1 + e); }
      }
      break;
    case UN_ERF: for (let i = 0; i < n; i++) o[oo + i] = erf(x[xo + i]!); break;
    case UN_NEG: for (let i = 0; i < n; i++) o[oo + i] = -x[xo + i]!; break;
    case UN_ABS: for (let i = 0; i < n; i++) o[oo + i] = Math.abs(x[xo + i]!); break;
    default: throw new RangeError(`unaryFlat: unknown op ${op}`);
  }
}

/** Row block for {@link linearNT}: bounds the packed-A and accumulator scratch to ROW_BLOCK·K and ROW_BLOCK·N f64s. */
const ROW_BLOCK = 256;

/**
 * `out[r, j] = Σ_p X[xo + r·K + p] · W[j·K + p] (+ bias[j])` — a PyTorch
 * `linear` (`W` is `[N, K]`, row-major) over `rows` contiguous rows of `X`.
 * `W` is packed to f64 once per call; `X` is packed and multiplied in
 * blocks of 256 rows so scratch stays bounded for long sequences.
 */
export function linearNT(
  X: NumArray, xo: number, rows: number, K: number,
  W: NumArray, N: number,
  bias: NumArray | null,
  out: NumArray, oo = 0,
): void {
  const pw = new Float64Array(N * K);
  packPanel(W, 0, K, 1, N, K, pw);
  const rb = Math.min(ROW_BLOCK, rows);
  const pa = new Float64Array(rb * K);
  const acc = new Float64Array(rb * N);
  for (let r0 = 0; r0 < rows; r0 += rb) {
    const m = Math.min(rb, rows - r0);
    packPanel(X, xo + r0 * K, K, 1, m, K, pa);
    gemmNT(pa, pw, acc, 0, N, m, N, K);
    const base = oo + r0 * N;
    if (bias) {
      for (let i = 0; i < m; i++) {
        const a = i * N, o = base + a;
        for (let j = 0; j < N; j++) out[o + j] = acc[a + j]! + bias[j]!;
      }
    } else if (m === rb) {
      out.set(acc, base);
    } else {
      out.set(acc.subarray(0, m * N), base);
    }
  }
}

/**
 * LayerNorm over the last axis of `rows` contiguous rows of length `D`:
 * `(x - mean) / sqrt(var + eps) · weight + bias` with population variance
 * and f64 statistics (two passes). `weight`/`bias` may be null.
 */
export function layerNormRows(
  x: NumArray, xo: number, rows: number, D: number,
  weight: NumArray | null, bias: NumArray | null, eps: number,
  o: NumArray, oo = 0,
): void {
  for (let r = 0; r < rows; r++) {
    const s = xo + r * D, d = oo + r * D;
    let mean = 0;
    for (let j = 0; j < D; j++) mean += x[s + j]!;
    mean /= D;
    let v = 0;
    for (let j = 0; j < D; j++) {
      const c = x[s + j]! - mean;
      v += c * c;
    }
    const inv = 1 / Math.sqrt(v / D + eps);
    if (weight && bias) for (let j = 0; j < D; j++) o[d + j] = (x[s + j]! - mean) * inv * weight[j]! + bias[j]!;
    else if (weight) for (let j = 0; j < D; j++) o[d + j] = (x[s + j]! - mean) * inv * weight[j]!;
    else if (bias) for (let j = 0; j < D; j++) o[d + j] = (x[s + j]! - mean) * inv + bias[j]!;
    else for (let j = 0; j < D; j++) o[d + j] = (x[s + j]! - mean) * inv;
  }
}

/**
 * Split-half ("NeoX" / Hugging Face, MLX `traditional=False`) rotary
 * embedding over `BH` contiguous `[L, Dh]` blocks, positions `0..L-1`, full
 * head dim (`Dh` even), scale 1. The angle is computed the way MLX and
 * PyTorch do in f32: `inv_freq = base^(-i/half)` and `pos · inv_freq` are
 * each rounded to f32 before cos/sin.
 */
export function ropeHalf(x: NumArray, xo: number, BH: number, L: number, Dh: number, base: number, o: NumArray, oo = 0): void {
  const half = Dh >> 1;
  const lb = Math.fround(Math.log2(base));
  const cos = new Float64Array(L * half), sin = new Float64Array(L * half);
  for (let i = 0; i < half; i++) {
    const invf = Math.fround(2 ** Math.fround(-Math.fround(i / half) * lb));
    for (let p = 0; p < L; p++) {
      const th = Math.fround(p * invf);
      cos[p * half + i] = Math.cos(th);
      sin[p * half + i] = Math.sin(th);
    }
  }
  for (let bh = 0; bh < BH; bh++) {
    for (let p = 0; p < L; p++) {
      const s = xo + (bh * L + p) * Dh, d = oo + (bh * L + p) * Dh, t = p * half;
      for (let i = 0; i < half; i++) {
        const x1 = x[s + i]!, x2 = x[s + half + i]!, c = cos[t + i]!, sn = sin[t + i]!;
        o[d + i] = x1 * c - x2 * sn;
        o[d + half + i] = x2 * c + x1 * sn;
      }
    }
  }
}

/** Mask description for {@link attention}. */
export interface AttentionMask {
  data: NumArray;
  /** Element strides aligned to `[B, H, Lq, Lk]` (0 on broadcast axes; see `alignedStrides`). */
  strides: readonly number[];
  /** `true`: nonzero = attend, zero = masked out. `false`: values are added to the scores. */
  boolean: boolean;
}

/**
 * Scaled dot-product attention, `softmax(scale · Q·Kᵀ + mask) · V`, over
 * contiguous `Q [B, H, Lq, Dh]`, `K [B, Hk, Lk, Dh]`, `V [B, Hk, Lk, Dv]`
 * into `out [B, H, Lq, Dv]`. `H` must be a multiple of `Hk` (grouped-query
 * heads share K/V). Both products run through `gemmNT`; scores, softmax and
 * the output accumulate in f64 and are rounded once on store. A row whose
 * every key is masked out yields zeros (the contract leaves it undefined).
 */
export function attention(
  Q: NumArray, K: NumArray, V: NumArray,
  B: number, H: number, Hk: number, Lq: number, Lk: number, Dh: number, Dv: number,
  scale: number,
  mask: AttentionMask | null,
  out: NumArray,
): void {
  const rep = H / Hk;
  const pq = new Float64Array(Lq * Dh);
  const pk = new Float64Array(Lk * Dh);
  const pvt = new Float64Array(Dv * Lk);
  const S = new Float64Array(Lq * Lk);
  const acc = new Float64Array(Lq * Dv);
  const M = mask?.data ?? null;
  const [msB, msH, msI, msJ] = (mask?.strides ?? [0, 0, 0, 0]) as [number, number, number, number];
  const isBool = mask?.boolean ?? true;
  let packedKV = -1;
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const kvh = b * Hk + Math.floor(h / rep);
      packPanel(Q, (b * H + h) * Lq * Dh, Dh, 1, Lq, Dh, pq);
      if (kvh !== packedKV) {
        packPanel(K, kvh * Lk * Dh, Dh, 1, Lk, Dh, pk);
        // Vᵀ: row d of the panel is column d of this head's V.
        packPanel(V, kvh * Lk * Dv, 1, Dv, Dv, Lk, pvt);
        packedKV = kvh;
      }
      gemmNT(pq, pk, S, 0, Lk, Lq, Lk, Dh);
      for (let i = 0; i < Lq; i++) {
        const row = i * Lk;
        const mo = b * msB + h * msH + i * msI;
        let m = -Infinity;
        for (let j = 0; j < Lk; j++) {
          let s = S[row + j]! * scale;
          if (M) {
            const mv = M[mo + j * msJ]!;
            if (isBool) {
              if (!mv) s = -Infinity;
            } else s += mv;
          }
          S[row + j] = s;
          if (s > m) m = s;
        }
        if (m === -Infinity) {
          S.fill(0, row, row + Lk);
          continue;
        }
        let sum = 0;
        for (let j = 0; j < Lk; j++) {
          const e = Math.exp(S[row + j]! - m);
          S[row + j] = e;
          sum += e;
        }
        const inv = 1 / sum;
        for (let j = 0; j < Lk; j++) S[row + j]! *= inv;
      }
      gemmNT(S, pvt, acc, 0, Dv, Lq, Dv, Lk);
      out.set(acc, (b * H + h) * Lq * Dv);
    }
  }
}

/**
 * GEGLU over `rows` contiguous rows of length `2·D`: the first half is the
 * value, the second the gate; `o[r, j] = geluErf(x[r, j]) · x[r, D + j]`.
 */
export function gegluRows(x: NumArray, xo: number, rows: number, D: number, o: NumArray, oo = 0): void {
  for (let r = 0; r < rows; r++) {
    const s = xo + r * 2 * D, d = oo + r * D;
    for (let j = 0; j < D; j++) o[d + j] = geluErf(x[s + j]!) * x[s + D + j]!;
  }
}

/**
 * Masked mean over axis 1: `x [B, L, D]`, `mask [B, L]` (nonzero = keep) →
 * `o [B, D]`, accumulated in f64. A batch row with no kept positions gives 0.
 */
export function maskedMeanPool(x: NumArray, B: number, L: number, D: number, mask: NumArray, o: NumArray): void {
  const acc = new Float64Array(D);
  for (let b = 0; b < B; b++) {
    acc.fill(0);
    let cnt = 0;
    for (let l = 0; l < L; l++) {
      if (!mask[b * L + l]) continue;
      cnt++;
      const s = (b * L + l) * D;
      for (let j = 0; j < D; j++) acc[j]! += x[s + j]!;
    }
    const inv = 1 / Math.max(cnt, 1);
    for (let j = 0; j < D; j++) o[b * D + j] = acc[j]! * inv;
  }
}

/**
 * Row gather: `o[i] = table[idx[i]]` for rows of length `rowLen`, with
 * `idx[i]` in `[0, nRows)` (a RangeError names the offending index
 * otherwise). Used for embedding lookups and per-batch row gathers.
 */
export function takeRows(table: NumArray, rowLen: number, nRows: number, idx: ArrayLike<number>, o: NumArray): void {
  const sameKind = table.constructor === o.constructor;
  for (let i = 0; i < idx.length; i++) {
    const r = idx[i]!;
    if (!(r >= 0 && r < nRows)) throw new RangeError(`takeRows: row index ${r} out of range [0, ${nRows})`);
    const s = r * rowLen, d = i * rowLen;
    if (sameKind) o.set(table.subarray(s, s + rowLen), d);
    else for (let j = 0; j < rowLen; j++) o[d + j] = table[s + j]!;
  }
}
