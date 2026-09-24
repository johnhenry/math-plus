/**
 * WebGPU attention entry points (issue #12 primitives, issue #126 fused
 * attention), since issue #146 running on `@johnhenry/backend-webgpu`:
 *
 * - `runQKT` (batched Q·Kᵀ, unscaled), `runSoftmax` (last axis) and
 *   `runWeightedSum` (batched weights·V) are the backend's `matmul`,
 *   `softmax` and strided `transpose`.
 * - `runAttention` is the backend's fused flash `sdpa` (online softmax,
 *   a head-dim 32/64 fast path that skips masked key tiles, and a generic
 *   kernel up to head dim 256), with this package's f32 mask convention
 *   and its "a query row with no visible key produces 0" guarantee kept on
 *   top (the backend leaves that case undefined). backend-webgpu 0.3.1
 *   sizes the kernels to the device's `maxComputeWorkgroupStorageSize`, so
 *   this is fused on every device, including one with the 16 KiB default.
 *
 * Shape convention (unchanged): `Q`/`K`/`V` are `(batch, seq, dim)`
 * row-major, f32; fold heads into `batch`. Every result is a GPU-resident
 * `GPUTensor` the caller owns and must `.free()`; inputs are never freed.
 *
 * @deprecated These `GPUDevice` + `GPUTensor` entry points are kept through
 * the deprecation window (see the 0.2.0 changelog). New code:
 * `createWebGpuDevice()` and `gpu.backend.sdpa(q, k, v, mask, scale)` on
 * `[B, H, L, D]` tensors (bool mask), or `matmul`/`softmax` directly.
 */
import type { WebGpuBackend, WebGpuTensor } from "@johnhenry/backend-webgpu";
import { backendFor } from "./bridge.ts";
import { GPUTensor } from "./device.ts";

/** Attention is f32-only through these shims: reject f16 `GPUTensor`s up front. */
function f32Handle(op: string, t: GPUTensor): WebGpuTensor {
  if (t.dtype !== "f32") throw new TypeError(`${op}: f32 GPUTensors only (got ${t.dtype}); only GEMM supports f16`);
  return t._live(op);
}

function numel(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * `scores[b, i, j] = sum_d Q[b, i, d] * K[b, j, d]` — `Q @ K^T` per batch,
 * unscaled (callers apply `1/sqrt(dim)` themselves). `q`/`k` must already
 * be `(batch, seqQ|seqK, dim)`-shaped; the result is a GPU-resident
 * `(batch, seqQ, seqK)` `GPUTensor`.
 *
 * @deprecated See the module doc.
 */
export async function runQKT(
  device: GPUDevice,
  q: GPUTensor,
  k: GPUTensor,
  batch: number,
  seqQ: number,
  seqK: number,
  dim: number,
): Promise<GPUTensor> {
  if (q.shape.length !== 3 || q.shape[0] !== batch || q.shape[1] !== seqQ || q.shape[2] !== dim) {
    throw new RangeError(`runQKT: Q shape [${q.shape}] does not match (batch=${batch}, seqQ=${seqQ}, dim=${dim})`);
  }
  if (k.shape.length !== 3 || k.shape[0] !== batch || k.shape[1] !== seqK || k.shape[2] !== dim) {
    throw new RangeError(`runQKT: K shape [${k.shape}] does not match (batch=${batch}, seqK=${seqK}, dim=${dim})`);
  }
  const b = backendFor(device);
  const qh = f32Handle("runQKT", q);
  const kh = f32Handle("runQKT", k);
  return GPUTensor._fromHandle(device, b.scope(() => b.matmul(qh, b.transpose(kh, [0, 2, 1]))));
}

/**
 * Row-wise numerically stable softmax over a `rows x cols` row-major
 * `GPUTensor` (`x` just needs `rows * cols` elements — e.g. a
 * `(batch, seqQ, seqK)` QKᵀ result viewed as `(batch*seqQ, seqK)`).
 * Returns a GPU-resident `(rows, cols)` `GPUTensor`.
 *
 * @deprecated See the module doc.
 */
export async function runSoftmax(device: GPUDevice, x: GPUTensor, rows: number, cols: number): Promise<GPUTensor> {
  const size = numel(x.shape);
  if (size !== rows * cols) throw new RangeError(`runSoftmax: x has ${size} elements, expected rows*cols ${rows * cols}`);
  const b = backendFor(device);
  const xh = f32Handle("runSoftmax", x);
  return GPUTensor._fromHandle(device, b.scope(() => b.softmax(b.reshape(xh, [rows, cols]), -1)));
}

/**
 * `out[b, i, d] = sum_j weights[b, i, j] * V[b, j, d]` — `weights @ V` per
 * batch (`weights` needs `batch*seqQ*seqK` elements). Returns a
 * GPU-resident `(batch, seqQ, dim)` `GPUTensor`.
 *
 * @deprecated See the module doc.
 */
export async function runWeightedSum(
  device: GPUDevice,
  weights: GPUTensor,
  v: GPUTensor,
  batch: number,
  seqQ: number,
  seqK: number,
  dim: number,
): Promise<GPUTensor> {
  const weightsSize = numel(weights.shape);
  if (weightsSize !== batch * seqQ * seqK) {
    throw new RangeError(`runWeightedSum: weights has ${weightsSize} elements, expected batch*seqQ*seqK ${batch * seqQ * seqK}`);
  }
  if (v.shape.length !== 3 || v.shape[0] !== batch || v.shape[1] !== seqK || v.shape[2] !== dim) {
    throw new RangeError(`runWeightedSum: V shape [${v.shape}] does not match (batch=${batch}, seqK=${seqK}, dim=${dim})`);
  }
  const b = backendFor(device);
  const wh = f32Handle("runWeightedSum", weights);
  const vh = f32Handle("runWeightedSum", v);
  return GPUTensor._fromHandle(device, b.scope(() => b.matmul(b.reshape(wh, [batch, seqQ, seqK]), vh)));
}

// ---- fused (flash) attention ---------------------------------------------------

export interface AttentionOptions {
  /** Multiplies `Q·Kᵀ` before the softmax. Default `1 / sqrt(dim)` (unlike the unscaled {@link runQKT}). */
  scale?: number;
  /**
   * f32 `GPUTensor`, nonzero = may attend, 0 = masked out. Its shape is
   * broadcast (right-aligned, size-1 axes repeat) against
   * `(batch, seqQ, seqK)`: e.g. `[seqQ, seqK]` for one causal or
   * sliding-window mask shared by the batch, `[batch, 1, seqK]` for key
   * padding. Query rows with no visible key produce 0.
   */
  mask?: GPUTensor;
  /**
   * @deprecated Ignored. backend-webgpu's fast kernel (head dim 32/64)
   * always skips key tiles no query of a block may attend to; its generic
   * kernel never does. (Turning skipping off existed for benchmarks.)
   */
  skipMaskedTiles?: boolean;
  /**
   * @deprecated backend-webgpu picks the kernel: `"fast"` for head dim 32/64,
   * `"generic"` otherwise. Forcing `"fast"` on another head dim still
   * throws; forcing `"generic"` is ignored.
   */
  kernel?: "fast" | "generic" | "auto";
}

/** Element strides of `shape` broadcast against `target` (right-aligned; size-1 axes get stride 0) — validates the mask shape. */
function checkMaskShape(op: string, shape: readonly number[], target: readonly number[]): void {
  if (shape.length < 1 || shape.length > target.length) {
    throw new RangeError(`${op}: mask shape [${shape}] must have 1..${target.length} axes, broadcastable to [${target}]`);
  }
  const full = [...Array<number>(target.length - shape.length).fill(1), ...shape];
  full.forEach((n, i) => {
    if (n !== target[i] && n !== 1) throw new RangeError(`${op}: mask shape [${shape}] is not broadcastable to [${target}]`);
  });
}

/**
 * Attention on the backend's fused `sdpa`: `q` `[B, Lq, D]`, `k`/`v`
 * `[B, Lk, D]` (f32), `mask` f32 broadcastable to `[B, Lq, Lk]` (nonzero =
 * attend) or null. Returns a new `[B, Lq, D]` tensor (tracked by an
 * enclosing scope, if any; otherwise the caller disposes it). Query rows
 * with no visible key are 0.
 */
export function encodeAttention(
  b: WebGpuBackend,
  q: WebGpuTensor,
  k: WebGpuTensor,
  v: WebGpuTensor,
  mask: WebGpuTensor | null,
  scale: number,
): WebGpuTensor {
  const [batch, seqQ, dim] = q.shape as [number, number, number];
  const seqK = k.shape[1] as number;
  return b.scope(() => {
    // The mask right-aligned to (batch, seqQ, seqK), as bool.
    const m3 = mask ? ([...Array<number>(3 - mask.shape.length).fill(1), ...mask.shape] as [number, number, number]) : null;
    const mask3 = mask && m3 ? b.reshape(mask, m3) : null;
    const visible = mask3 ? b.cast(mask3, "bool") : null;
    const q4 = b.reshape(q, [batch, 1, seqQ, dim]);
    const k4 = b.reshape(k, [batch, 1, seqK, dim]);
    const v4 = b.reshape(v, [batch, 1, seqK, dim]);
    const m4 = visible && m3 ? b.reshape(visible, [m3[0], 1, m3[1], m3[2]]) : null;
    const out = b.reshape(b.sdpa(q4, k4, v4, m4, scale), [batch, seqQ, dim]);
    if (!mask3) return out;
    // Rows with no visible key: the kernel leaves them undefined; this API
    // promises 0. anyVisible = max|mask|
    // per (batch, query) row is nonzero iff some key is visible, and its ×0
    // is a finite zero even where `out` is NaN.
    const anyVisible = b.max(b.abs(mask3), 2, true);
    return b.where(b.cast(anyVisible, "bool"), out, b.scale(anyVisible, 0));
  });
}

/**
 * Fused scaled-dot-product attention, `softmax(scale · Q·Kᵀ + mask) · V`,
 * flash-style (no `(seqQ, seqK)` scores tensor in global memory). `q` is
 * `(batch, seqQ, dim)`, `k`/`v` are `(batch, seqK, dim)`, all f32; fold
 * heads into `batch`. Returns a GPU-resident `(batch, seqQ, dim)` f32
 * `GPUTensor`. Head dims up to 256.
 *
 * @deprecated See the module doc.
 */
export async function runAttention(
  device: GPUDevice,
  q: GPUTensor,
  k: GPUTensor,
  v: GPUTensor,
  opts: AttentionOptions = {},
): Promise<GPUTensor> {
  if (q.shape.length !== 3 || k.shape.length !== 3 || v.shape.length !== 3) {
    throw new RangeError(`runAttention: q, k, v must be 3-D (batch, seq, dim), got [${q.shape}], [${k.shape}], [${v.shape}]`);
  }
  const [batch, seqQ, dim] = q.shape as [number, number, number];
  const seqK = k.shape[1] as number;
  if (k.shape[0] !== batch || k.shape[2] !== dim || v.shape[0] !== batch || v.shape[1] !== seqK || v.shape[2] !== dim) {
    throw new RangeError(`runAttention: shapes do not agree: q [${q.shape}], k [${k.shape}], v [${v.shape}]`);
  }
  if (opts.kernel === "fast" && dim !== 32 && dim !== 64) {
    throw new RangeError(`runAttention: the fast kernel needs head dim 32 or 64 (got ${dim})`);
  }
  if (dim > 256) throw new RangeError(`runAttention: head dim ${dim} > 256 is not supported`);
  const qh = f32Handle("runAttention", q);
  const kh = f32Handle("runAttention", k);
  const vh = f32Handle("runAttention", v);
  let mh: WebGpuTensor | null = null;
  if (opts.mask) {
    mh = f32Handle("runAttention", opts.mask);
    checkMaskShape("runAttention", opts.mask.shape, [batch, seqQ, seqK]);
  }
  const b = backendFor(device);
  return GPUTensor._fromHandle(device, encodeAttention(b, qh, kh, vh, mh, opts.scale ?? 1 / Math.sqrt(dim)));
}
