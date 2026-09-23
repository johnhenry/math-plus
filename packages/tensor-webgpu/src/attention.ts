/**
 * WebGPU attention-adjacent primitives (issue #12, v1 scope item 2): the
 * three ops that make up scaled-dot-product-attention's core —
 * `runQKT` (batched Q @ K^T), `runSoftmax` (numerically stable, last axis),
 * `runWeightedSum` (batched weights @ V) — kept as separate dispatches
 * rather than one fused flash-attention-style kernel, matching the issue's
 * explicit v1 scope ("these three primitives, not a fused kernel").
 *
 * Since issue #126 there is ALSO a fused path, {@link runAttention}: one
 * flash-style dispatch (kernels in attention-kernels.ts, ported from
 * laya-js) with an optional mask and masked-key-tile skipping. The three
 * primitives stay as they were (unmasked, unscaled) for callers that want
 * the intermediates.
 *
 * Shape convention throughout: `Q`/`K`/`V` are `(batch, seq, dim)` row-major
 * contiguous, f32.
 *
 * GPU residency (issue #100): all three primitives take and return
 * {@link GPUTensor}, not `Float32Array` — chaining `runQKT` -> `runSoftmax` ->
 * `runWeightedSum` (as scaled-dot-product-attention does) used to round-trip
 * every intermediate through the CPU (`Float32Array` out of one call,
 * re-uploaded as a fresh storage buffer by the next), even though nothing
 * outside the GPU ever needed to see those intermediates. Each function now
 * dispatches directly against its inputs' existing `GPUBuffer`s and wraps its
 * output buffer as a `GPUTensor` via `GPUTensor.fromBuffer` (device.ts) — no
 * host copy happens until/unless a caller explicitly calls `.toTensor()`/
 * `.toFloat32Array()` on a result. No wait/fence is needed between chained
 * calls either: every dispatch here goes through `device.queue`, and WebGPU
 * serializes queue submissions in order, so a later dispatch reading a
 * buffer an earlier dispatch wrote is automatically ordered correctly.
 * Callers own every `GPUTensor` they get back and must `.free()` it
 * (including intermediates they don't read back) — this module never frees
 * a caller-supplied input.
 */
import { GPUTensor } from "./device.ts";
import {
  fastAttentionBytes,
  fastAttentionWGSL,
  FAST_BQ,
  genericAttentionConfig,
  genericAttentionWGSL,
  type AttentionKernel,
} from "./attention-kernels.ts";
import { allocateGPUResidentBuffer, bindingOf, dispatchKernel, getKernelChecked, type SizedBuffer } from "./gpu-runtime.ts";

const TILE = 8;

/** Attention kernels are f32-only: reject f16 `GPUTensor`s up front instead of reinterpreting their bits as f32. */
function f32Binding(op: string, t: GPUTensor): SizedBuffer {
  if (t.dtype !== "f32") throw new TypeError(`${op}: f32 GPUTensors only (got ${t.dtype}); only GEMM supports f16`);
  return bindingOf(t);
}

/**
 * Dispatch a compute shader (3-D workgroup grid) and wrap its output buffer
 * as a `GPUTensor` of `outShape` WITHOUT reading it back. `dims` travels in
 * the runtime's uniform ring (gpu-runtime.ts `dispatchKernel`), and the
 * bind group is cached across calls on the same buffers.
 */
function dispatch3DResident(
  device: GPUDevice,
  label: string,
  code: string,
  bindings: readonly SizedBuffer[],
  dims: readonly [number, number, number, number],
  outputIndex: number,
  outShape: readonly number[],
  x: number,
  y: number,
  z: number,
): GPUTensor {
  dispatchKernel(device, code, bindings, [x, y, z], { uniform: new Uint32Array(dims), label });
  return GPUTensor.fromBuffer(device, (bindings[outputIndex] as SizedBuffer).buffer, outShape);
}

const QKT_WGSL = `
struct Dims { seqQ: u32, seqK: u32, dim: u32, batch: u32 };
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(${TILE}, ${TILE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x; // index into seqK
  let row = gid.y; // index into seqQ
  let b = gid.z;   // batch
  if (col >= dims.seqK || row >= dims.seqQ || b >= dims.batch) {
    return;
  }
  let qBase = (b * dims.seqQ + row) * dims.dim;
  let kBase = (b * dims.seqK + col) * dims.dim;
  var acc: f32 = 0.0;
  for (var d: u32 = 0u; d < dims.dim; d = d + 1u) {
    acc = acc + q[qBase + d] * k[kBase + d];
  }
  out[(b * dims.seqQ + row) * dims.seqK + col] = acc;
}
`;

/**
 * `scores[b, i, j] = sum_d Q[b, i, d] * K[b, j, d]` — `Q @ K^T` per batch,
 * unscaled (callers apply `1/sqrt(dim)` themselves, e.g. by pre-scaling `Q`,
 * matching how most reference attention implementations separate the scale
 * from the matmul rather than baking it into the kernel). `q`/`k` must
 * already be `(batch, seqQ|seqK, dim)`-shaped `GPUTensor`s (e.g. via
 * `toWebGPU`); the result is a GPU-resident `(batch, seqQ, seqK)` `GPUTensor`
 * — call `.toTensor()`/`.toFloat32Array()` on it if you need it on the CPU,
 * or pass it straight into `runSoftmax` to stay on-device.
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
  const bufOut = allocateGPUResidentBuffer(device, batch * seqQ * seqK);
  return dispatch3DResident(
    device,
    "attention:qkt",
    QKT_WGSL,
    [f32Binding("runQKT", q), f32Binding("runQKT", k), bufOut],
    [seqQ, seqK, dim, batch],
    2,
    [batch, seqQ, seqK],
    Math.ceil(seqK / TILE),
    Math.ceil(seqQ / TILE),
    batch,
  );
}

const SOFTMAX_WGSL = `
struct Dims { rows: u32, cols: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= dims.rows) {
    return;
  }
  let base = row * dims.cols;
  var m: f32 = x[base];
  for (var j: u32 = 1u; j < dims.cols; j = j + 1u) {
    m = max(m, x[base + j]);
  }
  var sum: f32 = 0.0;
  for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
    sum = sum + exp(x[base + j] - m);
  }
  for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
    out[base + j] = exp(x[base + j] - m) / sum;
  }
}
`;

/**
 * Numerically stable softmax along the LAST axis of a `(rows, cols)`
 * row-major `GPUTensor` (one GPU invocation per row; `cols` is walked
 * serially within the invocation, matching `Tensor.softmax`'s per-row
 * reduction shape — a parallel-reduction version is future work once
 * profiling shows this naive one is the bottleneck, not before). `x`'s shape
 * just needs `rows * cols` elements (e.g. a `(batch, seqQ, seqK)` QKT result
 * viewed as `(batch*seqQ, seqK)` — `rows`/`cols` are taken as given, not
 * inferred from `x.shape`, matching the pre-#100 signature). Returns a
 * GPU-resident `(rows, cols)` `GPUTensor`.
 */
export async function runSoftmax(
  device: GPUDevice,
  x: GPUTensor,
  rows: number,
  cols: number,
): Promise<GPUTensor> {
  const size = x.shape.reduce((a, b) => a * b, 1);
  if (size !== rows * cols) throw new RangeError(`runSoftmax: x has ${size} elements, expected rows*cols ${rows * cols}`);
  const bufOut = allocateGPUResidentBuffer(device, rows * cols);
  return dispatch3DResident(
    device,
    "attention:softmax",
    SOFTMAX_WGSL,
    [f32Binding("runSoftmax", x), bufOut],
    [rows, cols, 0, 0],
    1,
    [rows, cols],
    Math.ceil(rows / 64),
    1,
    1,
  );
}

const WEIGHTED_SUM_WGSL = `
struct Dims { seqQ: u32, seqK: u32, dim: u32, batch: u32 };
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(${TILE}, ${TILE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;   // index into dim
  let row = gid.y; // index into seqQ
  let b = gid.z;   // batch
  if (d >= dims.dim || row >= dims.seqQ || b >= dims.batch) {
    return;
  }
  let wBase = (b * dims.seqQ + row) * dims.seqK;
  var acc: f32 = 0.0;
  for (var j: u32 = 0u; j < dims.seqK; j = j + 1u) {
    acc = acc + weights[wBase + j] * v[(b * dims.seqK + j) * dims.dim + d];
  }
  out[(b * dims.seqQ + row) * dims.dim + d] = acc;
}
`;

/**
 * `out[b, i, d] = sum_j weights[b, i, j] * V[b, j, d]` — `weights @ V` per
 * batch (the softmax'd attention weights times the value tensor; `weights`
 * is typically `runSoftmax`'s output, but this function takes any
 * `(batch, seqQ, seqK)`-shaped `GPUTensor`). Returns a GPU-resident
 * `(batch, seqQ, dim)` `GPUTensor`.
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
  const weightsSize = weights.shape.reduce((a, b) => a * b, 1);
  if (weightsSize !== batch * seqQ * seqK) {
    throw new RangeError(`runWeightedSum: weights has ${weightsSize} elements, expected batch*seqQ*seqK ${batch * seqQ * seqK}`);
  }
  if (v.shape.length !== 3 || v.shape[0] !== batch || v.shape[1] !== seqK || v.shape[2] !== dim) {
    throw new RangeError(`runWeightedSum: V shape [${v.shape}] does not match (batch=${batch}, seqK=${seqK}, dim=${dim})`);
  }
  const bufOut = allocateGPUResidentBuffer(device, batch * seqQ * dim);
  return dispatch3DResident(
    device,
    "attention:weighted-sum",
    WEIGHTED_SUM_WGSL,
    [f32Binding("runWeightedSum", weights), f32Binding("runWeightedSum", v), bufOut],
    [seqQ, seqK, dim, batch],
    2,
    [batch, seqQ, dim],
    Math.ceil(dim / TILE),
    Math.ceil(seqQ / TILE),
    batch,
  );
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
   * With a mask, skip key tiles that no query of a workgroup's block may
   * attend to (see attention-kernels.ts). Default true; `false` exists for
   * benchmarks and tests.
   */
  skipMaskedTiles?: boolean;
  /** Force a kernel (throws if it doesn't fit this head dim / device). Default `"auto"`: `fast` when it fits, else `generic`. */
  kernel?: AttentionKernel | "auto";
}

export interface AttentionPlan {
  kernel: AttentionKernel;
  /** WGSL source (also the pipeline-cache key). */
  code: string;
  /** Workgroup grid `[x, y]` (query blocks, batch). */
  groups: [number, number];
}

/**
 * Pick the attention kernel and build its shader (pure; exported for tests).
 * `workgroupStorageLimit` is the device's `maxComputeWorkgroupStorageSize`
 * (16 KiB unless raised): the fast kernel for head dim 64 needs ~20 KiB, and
 * the generic kernel's tile size shrinks until it fits.
 */
export function planAttention(
  batch: number,
  seqQ: number,
  dim: number,
  masked: boolean,
  workgroupStorageLimit: number,
  opts: Pick<AttentionOptions, "kernel" | "skipMaskedTiles"> = {},
): AttentionPlan {
  const variant = { D: dim, masked, skipMaskedTiles: opts.skipMaskedTiles ?? true };
  const fastFits = (dim === 32 || dim === 64) && fastAttentionBytes(dim, masked) <= workgroupStorageLimit;
  const want = opts.kernel ?? "auto";
  if (want === "fast" || (want === "auto" && fastFits)) {
    if (!fastFits) {
      throw new RangeError(
        `planAttention: the fast kernel needs head dim 32 or 64 and ${fastAttentionBytes(dim === 64 ? 64 : 32, masked)} bytes of workgroup memory (dim ${dim}, limit ${workgroupStorageLimit})`,
      );
    }
    return { kernel: "fast", code: fastAttentionWGSL(variant), groups: [Math.ceil(seqQ / FAST_BQ), batch] };
  }
  const cfg = genericAttentionConfig(dim, workgroupStorageLimit, masked);
  if (!cfg) throw new RangeError(`planAttention: head dim ${dim} does not fit ${workgroupStorageLimit} bytes of workgroup memory`);
  return { kernel: "generic", code: genericAttentionWGSL(variant, cfg), groups: [Math.ceil(seqQ / cfg.BQ), batch] };
}

/** Element strides of `shape` broadcast against `target` (right-aligned; size-1 axes get stride 0). */
function broadcastStrides(op: string, shape: readonly number[], target: readonly number[]): number[] {
  if (shape.length < 1 || shape.length > target.length) {
    throw new RangeError(`${op}: mask shape [${shape}] must have 1..${target.length} axes, broadcastable to [${target}]`);
  }
  const full = [...Array<number>(target.length - shape.length).fill(1), ...shape];
  const strides = new Array<number>(full.length).fill(0);
  let s = 1;
  for (let i = full.length - 1; i >= 0; i--) {
    const n = full[i] as number;
    if (n !== target[i] && n !== 1) {
      throw new RangeError(`${op}: mask shape [${shape}] is not broadcastable to [${target}]`);
    }
    strides[i] = n === 1 ? 0 : s;
    s *= n;
  }
  return strides;
}

/**
 * Fused scaled-dot-product attention, `softmax(scale · Q·Kᵀ + mask) · V`,
 * in ONE dispatch (flash-style: online softmax over key tiles, no
 * `(seqQ, seqK)` scores tensor in global memory) — the fused counterpart
 * of chaining {@link runQKT} → {@link runSoftmax} → {@link runWeightedSum},
 * plus mask support. `q` is `(batch, seqQ, dim)`, `k`/`v` are
 * `(batch, seqK, dim)`, all f32; fold heads into `batch`. Returns a
 * GPU-resident `(batch, seqQ, dim)` f32 `GPUTensor`.
 *
 * With a mask, key tiles no query in a block may see are skipped entirely
 * (sliding-window and padding masks stop paying for the masked part);
 * docs/spikes/webgpu-runtime.md has the measurements.
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
  const bindings = [f32Binding("runAttention", q), f32Binding("runAttention", k), f32Binding("runAttention", v)];
  const mask = opts.mask;
  let uniform: ArrayBuffer;
  const scale = opts.scale ?? 1 / Math.sqrt(dim);
  if (mask) {
    bindings.push(f32Binding("runAttention", mask));
    const [msb, msq, msk] = broadcastStrides("runAttention", mask.shape, [batch, seqQ, seqK]) as [number, number, number];
    uniform = new ArrayBuffer(24);
    new Uint32Array(uniform, 0, 5).set([seqQ, seqK, msb, msq, msk]);
    new Float32Array(uniform, 20, 1)[0] = scale;
  } else {
    uniform = new ArrayBuffer(12);
    new Uint32Array(uniform, 0, 2).set([seqQ, seqK]);
    new Float32Array(uniform, 8, 1)[0] = scale;
  }
  const out = allocateGPUResidentBuffer(device, batch * seqQ * dim);
  if (batch * seqQ * dim > 0) {
    const limit = device.limits.maxComputeWorkgroupStorageSize ?? 16384;
    const plan = planAttention(batch, seqQ, dim, mask !== undefined, limit, opts);
    await getKernelChecked(device, plan.code, `attention:${plan.kernel}`);
    dispatchKernel(device, plan.code, [...bindings, out], plan.groups, {
      uniform: new Uint8Array(uniform),
      label: `attention:${plan.kernel}${mask ? ((opts.skipMaskedTiles ?? true) ? ":masked" : ":masked-noskip") : ""}`,
    });
  }
  return GPUTensor.fromBuffer(device, out.buffer, [batch, seqQ, dim]);
}
