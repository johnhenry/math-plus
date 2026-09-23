/**
 * WebGPU attention-adjacent primitives (issue #12, v1 scope item 2): the
 * three ops that make up scaled-dot-product-attention's core —
 * `runQKT` (batched Q @ K^T), `runSoftmax` (numerically stable, last axis),
 * `runWeightedSum` (batched weights @ V) — kept as separate dispatches
 * rather than one fused flash-attention-style kernel, matching the issue's
 * explicit v1 scope ("these three primitives, not a fused kernel").
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
import { allocateGPUResidentBuffer, bindingOf, dispatchKernel, type SizedBuffer } from "./gpu-runtime.ts";

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

