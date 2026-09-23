/**
 * Small shared helpers for dispatching a WGSL compute shader over
 * storage buffers (f32, or f16 bits for GEMM) and reading the result back. Used by
 * fusion-wgsl's elementwise kernels, gemm.ts, and attention.ts — factored out
 * once instead of duplicated per kernel (buffer creation, staging-buffer
 * readback, and the map/unmap dance are identical across all of them).
 *
 * These functions call real `GPUDevice` methods — they only *type*-check in
 * Node (via `@webgpu/types`, no runtime browser globals needed to import this
 * module), but actually RUNNING them requires a real `GPUDevice`, which only
 * exists behind `navigator.gpu` in a browser, or Dawn in Node/Bun (dawn.ts —
 * see README "Node and Bun"). test/helpers.ts drives a real one either way.
 *
 * Two perf fixes live here (issue #100):
 *
 *  1. Shader module + compute pipeline caching (`getOrCreateComputePipeline`):
 *     every dispatch used to call `createShaderModule`/`createComputePipeline`
 *     from scratch, even when the exact same WGSL source had already been
 *     compiled on a previous call (e.g. every `runGemmWGSL`/`runQKT`/
 *     `runSoftmax`/`runWeightedSum` call with the same op re-triggers a full
 *     shader compile). Cached per-`GPUDevice` (a `WeakMap` so it doesn't keep
 *     a device alive past its own lifetime), keyed by the shader source
 *     string itself — which also transparently covers the IR-compiled
 *     elementwise shaders in `elementwise.ts` (their source already varies
 *     with `(node, numInputs)`, so the same source string implies the same
 *     compiled program).
 *  2. Buffer pooling (`acquireBuffer`/`releaseBuffer`): buffers used to be
 *     allocated fresh and `destroy()`ed every call. They're now checked out
 *     of a size+usage-keyed per-device pool and returned to it instead of
 *     being destroyed, so back-to-back calls of the same shape/dtype reuse
 *     the same underlying `GPUBuffer`s rather than round-tripping through the
 *     GPU driver's allocator every time. Safe to reuse immediately (no manual
 *     fence/wait needed): all reads/writes to a given buffer go through the
 *     same `device.queue`, and WebGPU serializes queue operations in
 *     submission order — by the time a later `acquireBuffer` call's caller
 *     writes/dispatches into a reused buffer, every previously-queued command
 *     that touched it has already been ordered ahead of the new one.
 */

/** A `GPUBuffer` plus the byte length and usage flags it was created with — every helper here needs all three (`GPUBuffer` alone doesn't expose either), and `usage` is what `releaseBuffer` needs to put it back in the right pool bucket. */
export interface SizedBuffer {
  buffer: GPUBuffer;
  byteLength: number;
  usage: GPUBufferUsageFlags;
}

// ---- buffer pool -------------------------------------------------------

/** Free list keyed by `"${byteLength}:${usage}"` — buffers are only ever reused for an identically-sized, identically-used request, so no cross-shape aliasing risk. */
class BufferPool {
  #device: GPUDevice;
  #free = new Map<string, GPUBuffer[]>();

  constructor(device: GPUDevice) {
    this.#device = device;
  }

  #key(byteLength: number, usage: GPUBufferUsageFlags): string {
    return `${byteLength}:${usage}`;
  }

  acquire(byteLength: number, usage: GPUBufferUsageFlags): SizedBuffer {
    const key = this.#key(byteLength, usage);
    const list = this.#free.get(key);
    const buffer = list?.pop() ?? this.#device.createBuffer({ size: byteLength, usage });
    return { buffer, byteLength, usage };
  }

  release(sized: SizedBuffer): void {
    const key = this.#key(sized.byteLength, sized.usage);
    const list = this.#free.get(key);
    if (list) list.push(sized.buffer);
    else this.#free.set(key, [sized.buffer]);
  }

  /** Destroys every pooled (currently-released) buffer and forgets them — for test teardown/explicit cleanup, never required for correctness. */
  destroyAll(): void {
    for (const list of this.#free.values()) for (const b of list) b.destroy();
    this.#free.clear();
  }
}

const pools = new WeakMap<GPUDevice, BufferPool>();

function poolFor(device: GPUDevice): BufferPool {
  let pool = pools.get(device);
  if (!pool) {
    pool = new BufferPool(device);
    pools.set(device, pool);
  }
  return pool;
}

/** Check a buffer of `byteLength`/`usage` out of `device`'s pool — a reused buffer if one of that exact shape+usage was previously released, otherwise a freshly created one. */
export function acquireBuffer(device: GPUDevice, byteLength: number, usage: GPUBufferUsageFlags): SizedBuffer {
  return poolFor(device).acquire(byteLength, usage);
}

/** Return a buffer to its device's pool for reuse instead of destroying it. Only call this for buffers whose lifetime this module owns (i.e. NOT a `GPUTensor`'s buffer, whose lifetime the caller controls via `.free()`). */
export function releaseBuffer(device: GPUDevice, sized: SizedBuffer): void {
  poolFor(device).release(sized);
}

/** Destroy and forget every buffer currently sitting free in `device`'s pool. Exposed for tests/teardown; never required for correctness (buffers left in the pool are just reused or eventually GC'd with the device). */
export function destroyBufferPool(device: GPUDevice): void {
  pools.get(device)?.destroyAll();
}

/** `byteLength` rounded up to WebGPU's 4-byte copy granularity (and at least 4 — a zero-size binding is invalid). f16 data (2 bytes/element) with an odd element count is the case that actually needs this. */
export function paddedByteLength(byteLength: number): number {
  return Math.max(4, (byteLength + 3) & ~3);
}

/**
 * Upload `data` into a STORAGE buffer usable as a shader input, reusing a
 * pooled buffer of the same size when available. Accepts f32 data or f16
 * *bits* (`Uint16Array`, tensor-core's f16 storage representation); the
 * buffer is padded to a multiple of 4 bytes, since `writeBuffer` (and
 * `copyBufferToBuffer`) only move 4-byte multiples.
 */
export function uploadStorageBuffer(device: GPUDevice, data: Float32Array | Uint16Array): SizedBuffer {
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const byteLength = paddedByteLength(data.byteLength);
  const sized = acquireBuffer(device, byteLength, usage);
  writePadded(device, sized.buffer, data);
  return sized;
}

/** `queue.writeBuffer` of `data` at offset 0, zero-padding a trailing partial word (only possible for 2-byte f16 data). */
export function writePadded(device: GPUDevice, buffer: GPUBuffer, data: Float32Array | Uint16Array): void {
  let bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  if (bytes.byteLength % 4 !== 0) {
    const padded = new Uint8Array(paddedByteLength(bytes.byteLength));
    padded.set(bytes);
    bytes = padded;
  }
  if (bytes.byteLength === 0) return;
  // `bytes.buffer as ArrayBuffer`: @webgpu/types' `writeBuffer` wants a
  // `BufferSource` typed over a plain `ArrayBuffer`, but TS 5.7's typed-array
  // generics widen `#buffer` to `ArrayBufferLike` (which includes
  // `SharedArrayBuffer`) — a real cast, not a bug workaround, since every
  // array this package uploads is backed by a plain `ArrayBuffer`. The
  // explicit (buffer, byteOffset, byteLength) form also matters under Dawn:
  // some bindings ignore a TypedArray view's own byteOffset.
  device.queue.writeBuffer(buffer, 0, bytes.buffer as ArrayBuffer, bytes.byteOffset, bytes.byteLength);
}

/** Allocate a STORAGE buffer for a shader to write into (also COPY_SRC so it can be staged out afterward), reusing a pooled buffer of the same size when available. `bytesPerElement` is 4 for f32 (the default) and 2 for f16. */
export function allocateOutputBuffer(device: GPUDevice, elementCount: number, bytesPerElement = 4): SizedBuffer {
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  return acquireBuffer(device, paddedByteLength(elementCount * bytesPerElement), usage);
}

/**
 * Allocate a STORAGE buffer for a shader to write into that ALSO carries
 * `COPY_DST` — the fuller usage set `GPUTensor`/`toWebGPU` (device.ts) use,
 * needed for an output buffer that will be wrapped as a `GPUTensor` and kept
 * GPU-resident (attention.ts's chained ops): a plain `allocateOutputBuffer`
 * result can be staged OUT via `COPY_SRC` but not written into again later,
 * which a GPU-resident intermediate potentially needs if it's ever reused as
 * an upload target.
 */
export function allocateGPUResidentBuffer(device: GPUDevice, elementCount: number, bytesPerElement = 4): SizedBuffer {
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  return acquireBuffer(device, paddedByteLength(elementCount * bytesPerElement), usage);
}

/** A read-only view of a caller-owned buffer (e.g. a `GPUTensor`'s) as a dispatch binding — never released/destroyed by the dispatching op; ownership stays with whoever holds it. */
export function bindingOf(t: { readonly buffer: GPUBuffer }): SizedBuffer {
  return { buffer: t.buffer, byteLength: t.buffer.size, usage: GPUBufferUsage.STORAGE };
}

/**
 * Copy a GPU buffer to a MAP_READ staging buffer, submit, await the map, and
 * return a plain (detached-from-GPU-memory) `Float32Array` copy. The
 * `.slice(0)` on the mapped range matters: `getMappedRange()` returns a view
 * backed by the mapping, which becomes invalid the instant `unmap()` runs.
 *
 * The staging buffer itself is NOT pooled: `MAP_READ` buffers have host-side
 * mapping state that's finicky to reuse safely across an `unmap()`/next-`mapAsync()`
 * cycle without an explicit ordering guarantee, and staging reads are already
 * the least frequent operation in this package's hot paths (one per readback,
 * not one per intermediate) — pooling storage/output buffers is where the
 * actual per-dispatch allocation cost was.
 */
export async function readBackFloat32(
  device: GPUDevice,
  source: SizedBuffer,
): Promise<Float32Array> {
  return new Float32Array(await readBackBytes(device, source.buffer, source.byteLength));
}

/**
 * The dtype-agnostic core of {@link readBackFloat32} (also used by
 * `GPUTensor.toFloat32Array`/`toUint16Array` and the f16 GEMM entry point):
 * stage `byteLength` bytes (a multiple of 4) of `source` out through a fresh
 * MAP_READ buffer and return a detached `ArrayBuffer` copy.
 */
export async function readBackBytes(device: GPUDevice, source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return out;
}

// ---- shader module + pipeline cache ------------------------------------

const pipelineCaches = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();

/** Compiled-pipeline cache keyed by shader source (plus entry point, for the rare case two kernels share a device but not an entry point name) — a `createShaderModule`+`createComputePipeline` pair only ever happens once per distinct source per device. */
export function getOrCreateComputePipeline(
  device: GPUDevice,
  code: string,
  entryPoint = "main",
): GPUComputePipeline {
  let cache = pipelineCaches.get(device);
  if (!cache) {
    cache = new Map();
    pipelineCaches.set(device, cache);
  }
  const key = entryPoint === "main" ? code : `${entryPoint} ${code}`;
  let pipeline = cache.get(key);
  if (!pipeline) {
    const module = device.createShaderModule({ code });
    pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });
    cache.set(key, pipeline);
  }
  return pipeline;
}

/** Number of distinct (source, entryPoint) pipelines compiled so far for `device` — test-only introspection to assert caching actually happened, not a production API. */
export function pipelineCacheSize(device: GPUDevice): number {
  return pipelineCaches.get(device)?.size ?? 0;
}

/** Dispatch a compute shader with `bindings` bound at group 0 in binding-index order, `workgroupCount` groups along X only (every kernel in this package is a flat 1-D or 2-D-encoded-as-1-D dispatch). Uses/populates the shared pipeline cache. */
export function dispatchCompute(
  device: GPUDevice,
  code: string,
  bindings: readonly SizedBuffer[],
  workgroupCountX: number,
): void {
  const pipeline = getOrCreateComputePipeline(device, code);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map((b, i) => ({ binding: i, resource: { buffer: b.buffer } })),
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, workgroupCountX));
  pass.end();
  device.queue.submit([encoder.finish()]);
}

/** `Math.ceil(total / groupSize)`, minimum 1 (WebGPU rejects a 0-workgroup dispatch on some backends). */
export function workgroupsFor(total: number, groupSize: number): number {
  return Math.max(1, Math.ceil(total / groupSize));
}
