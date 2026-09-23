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
 *
 * Runtime wins ported from laya-js's WebGPU backend (issue #126):
 *
 *  3. Explicit bind-group layouts, a bind-group cache and a uniform arena
 *     with dynamic offsets (`dispatchKernel`): kernels no longer allocate a
 *     16-byte uniform buffer and a fresh `GPUBindGroup` per dispatch. Every
 *     kernel's `@group(0)` bindings are read from its WGSL and turned into an
 *     explicit layout (shared by kernels with the same binding signature);
 *     uniforms are written into a per-device ring buffer and bound with a
 *     dynamic offset, so a bind group depends only on the storage buffers
 *     and the uniform SIZE, and is cached by those. The size is part of the
 *     key on purpose: two kernels with the same signature but different
 *     uniform struct sizes would otherwise share a bind group whose uniform
 *     binding is too small for one of them (laya-js hit exactly that
 *     "binding is too small" validation error).
 *  4. No busy-polling while waiting for a readback under Dawn
 *     (`readBackBytes`): Dawn's Node binding resolves `mapAsync` by polling,
 *     keeping a core at 100% for the whole GPU wait. When enabled (default:
 *     only when there is no `navigator.gpu`, i.e. a native binding, and the
 *     expected wait is over 15 ms) the readback first sleeps for ~80% of the
 *     last observed wait of the same work, and polls only for the rest.
 *  5. Every `queue.writeBuffer` goes through {@link writeBytes}, which always
 *     passes `(arrayBuffer, byteOffset, byteLength)`: Bun's Dawn binding
 *     ignores a TypedArray view's own `byteOffset` and uploads the wrong
 *     bytes (test/bun/write-buffer-offset.bun.ts reproduces it).
 *  6. An opt-in GPU timestamp profiler (`startProfiling`/`stopProfiling`,
 *     needs the `timestamp-query` feature).
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
  writeBytes(device, buffer, 0, bytes);
}

/**
 * The ONLY `queue.writeBuffer` call in this package (test/write-buffer.test.ts
 * enforces that): writes `data`'s bytes to `buffer` at `bufferOffset`,
 * always as `(arrayBuffer, byteOffset, byteLength)`. Passing the view itself
 * is not safe: Bun's Dawn binding ignores a TypedArray view's `byteOffset`
 * and uploads bytes from the START of the underlying `ArrayBuffer` (reproduced
 * by test/bun/write-buffer-offset.bun.ts; laya-js hit it as corrupted
 * weights). `byteLength` must be a multiple of 4 (WebGPU's rule).
 */
export function writeBytes(device: GPUDevice, buffer: GPUBuffer, bufferOffset: number, data: ArrayBufferView): void {
  // `as ArrayBuffer`: @webgpu/types wants a plain-ArrayBuffer-backed source,
  // TS widens `.buffer` to `ArrayBufferLike`; every array uploaded here is
  // backed by a plain ArrayBuffer.
  device.queue.writeBuffer(buffer, bufferOffset, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
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

// ---- per-device runtime state --------------------------------------------

/** Counters for tests and benchmarks (see {@link gpuRuntimeStats}). */
export interface GPURuntimeStats {
  /** Compute dispatches encoded through {@link dispatchKernel}. */
  dispatches: number;
  /** `createBindGroup` calls (bind-group cache misses). */
  bindGroupsCreated: number;
  /** Dispatches that reused a cached bind group. */
  bindGroupCacheHits: number;
  /** Readbacks through {@link readBackBytes}. */
  readbacks: number;
  /** Readbacks that slept before polling `mapAsync` (see {@link configureGPURuntime}). */
  readbackSleeps: number;
}

/** Per-device runtime options (see {@link configureGPURuntime}). */
export interface GPURuntimeOptions {
  /**
   * Before awaiting a readback's `mapAsync`, sleep (`setTimeout`) for ~80%
   * of the last observed wait for the same work, and only poll for the
   * rest. Dawn's Node/Bun binding resolves `mapAsync` by polling, which
   * keeps a CPU core at 100% for the whole GPU wait (and on a thermally
   * limited machine that power comes out of the GPU's budget). Browsers
   * don't busy-poll, so there it only adds latency. Default: on when there
   * is no `navigator.gpu` (a native binding such as Dawn), off otherwise.
   */
  sleepWhileWaiting?: boolean;
  /**
   * Only sleep when the expected wait exceeds this many milliseconds
   * (default 15). Measured on an M2 under Dawn (docs/spikes/webgpu-runtime.md):
   * for a ~35 ms wait, sleeping costs ~1% latency and saves 2-3x process
   * CPU; for a ~9 ms wait it cost 25-45% latency — the post-sleep poll
   * resolves later than a continuous poll would — for a smaller saving.
   */
  sleepThresholdMs?: number;
}

interface ProfilerState {
  querySet: GPUQuerySet;
  resolve: GPUBuffer;
  /** Dispatches per resolve batch (the query set holds 2 timestamps each). */
  capacity: number;
  labels: string[];
  pending: { staging: GPUBuffer; labels: string[] }[];
}

interface RuntimeState {
  kernels: Map<string, ComputeKernel>;
  layouts: Map<string, { layout: GPUBindGroupLayout; id: number }>;
  nextLayoutId: number;
  bufferIds: WeakMap<GPUBuffer, number>;
  nextBufferId: number;
  bindGroups: Map<string, GPUBindGroup>;
  arena: { buffer: GPUBuffer; id: number; cursor: number } | undefined;
  uniformAlign: number;
  /** FNV-1a hash of the dispatches (kernel, grid, uniforms) since the last readback — the key for the readback wait estimate. */
  workHash: number;
  dispatchesSinceRead: number;
  waitMs: Map<number, number>;
  sleeping: Promise<void> | null;
  options: GPURuntimeOptions;
  profiler: ProfilerState | null;
  stats: GPURuntimeStats;
}

const states = new WeakMap<GPUDevice, RuntimeState>();

function stateFor(device: GPUDevice): RuntimeState {
  let st = states.get(device);
  if (!st) {
    st = {
      kernels: new Map(),
      layouts: new Map(),
      nextLayoutId: 1,
      bufferIds: new WeakMap(),
      nextBufferId: 1,
      bindGroups: new Map(),
      arena: undefined,
      uniformAlign: Math.max(256, device.limits?.minUniformBufferOffsetAlignment ?? 256),
      workHash: FNV_OFFSET,
      dispatchesSinceRead: 0,
      waitMs: new Map(),
      sleeping: null,
      options: {},
      profiler: null,
      stats: { dispatches: 0, bindGroupsCreated: 0, bindGroupCacheHits: 0, readbacks: 0, readbackSleeps: 0 },
    };
    states.set(device, st);
  }
  return st;
}

/** Set per-device runtime options (merged into the current ones). */
export function configureGPURuntime(device: GPUDevice, options: GPURuntimeOptions): void {
  Object.assign(stateFor(device).options, options);
}

/** Snapshot of `device`'s runtime counters, plus the current bind-group cache size. Test/benchmark introspection. */
export function gpuRuntimeStats(device: GPUDevice): GPURuntimeStats & { bindGroupCacheSize: number; sleepWhileWaiting: boolean } {
  const st = stateFor(device);
  return { ...st.stats, bindGroupCacheSize: st.bindGroups.size, sleepWhileWaiting: sleepWhileWaiting(st) };
}

const FNV_OFFSET = 0x811c9dc5;
function mix(h: number, v: number): number {
  return Math.imul(h ^ (v >>> 0), 0x01000193) >>> 0;
}

// ---- readback ---------------------------------------------------------------

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

/** Default {@link GPURuntimeOptions.sleepThresholdMs}: shorter expected waits are just polled. */
const DEFAULT_SLEEP_THRESHOLD_MS = 15;
/** A map that resolves within this long of waking was already done before the sleep ended. */
const OVERSLEEP_POLL_MS = 0.5;
/** Sleep this fraction of the estimated wait; the remainder is polled. An overestimate therefore shrinks by 20% per readback (the measured wait includes the sleep). */
const SLEEP_FRACTION = 0.8;

function sleepWhileWaiting(st: RuntimeState): boolean {
  if (st.options.sleepWhileWaiting !== undefined) return st.options.sleepWhileWaiting;
  const nav = (globalThis as { navigator?: { gpu?: unknown } }).navigator;
  return !nav?.gpu;
}

/**
 * The dtype-agnostic core of {@link readBackFloat32} (also used by
 * `GPUTensor.toFloat32Array`/`toUint16Array` and the f16 GEMM entry point):
 * stage `byteLength` bytes (a multiple of 4) of `source` out through a fresh
 * MAP_READ buffer and return a detached `ArrayBuffer` copy.
 *
 * Under Dawn (see {@link GPURuntimeOptions.sleepWhileWaiting}) it sleeps for
 * most of the expected GPU time first instead of letting `mapAsync` busy-poll
 * the whole wait. The estimate is the last measured wait for the same work
 * (a hash of every dispatch's kernel, grid and uniforms since the previous
 * readback); unseen work is simply polled, as before.
 */
export async function readBackBytes(device: GPUDevice, source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
  const st = stateFor(device);
  st.stats.readbacks++;
  const work = st.dispatchesSinceRead ? st.workHash : 0;
  st.dispatchesSinceRead = 0;
  st.workHash = FNV_OFFSET;
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const t0 = performance.now();
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  const est = work ? st.waitMs.get(work) : undefined;
  let target: number | undefined;
  let woke = 0;
  if (est !== undefined && est > (st.options.sleepThresholdMs ?? DEFAULT_SLEEP_THRESHOLD_MS) && sleepWhileWaiting(st)) {
    st.stats.readbackSleeps++;
    target = est * SLEEP_FRACTION;
    const p = new Promise<void>((r) => setTimeout(r, Math.max(0, (target as number) - (performance.now() - t0))));
    st.sleeping = p;
    await p;
    if (st.sleeping === p) st.sleeping = null;
    woke = performance.now();
  } else if (st.sleeping) {
    // A concurrent readback is already sleeping through this queue's work.
    await st.sleeping;
  }
  await staging.mapAsync(GPUMapMode.READ);
  if (work) {
    const now = performance.now();
    // If the map was already done when the sleep ended, we overslept: the
    // measured wait is the timer's lateness, not the GPU's. Record the
    // sleep target instead, so an overestimate shrinks by 20% per readback
    // (recording the overslept wait would feed timer lateness back into
    // the estimate and ratchet the latency up — measured: +60% on a 9 ms
    // GEMM readback under Node before this rule).
    const overslept = target !== undefined && now - woke < OVERSLEEP_POLL_MS;
    if (st.waitMs.size > 256) st.waitMs.clear();
    st.waitMs.set(work, overslept ? (target as number) : now - t0);
  }
  const out = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return out;
}

// ---- kernels: explicit layouts parsed from WGSL -------------------------------

/** How a kernel declares one `@group(0)` binding. */
export type BindingKind = "read" | "read_write" | "uniform";

const BINDING_RE = /@group\(\s*0\s*\)\s*@binding\(\s*(\d+)\s*\)\s*var\s*<\s*(storage|uniform)\s*(?:,\s*(read_write|read))?\s*>/g;

/**
 * The `@group(0)` bindings a WGSL source declares, in binding order —
 * or `undefined` when they don't fit this runtime's model (other groups,
 * gaps, more than one uniform, textures/samplers), in which case the kernel
 * falls back to `layout: "auto"` and cannot take a uniform. Every kernel in
 * this package fits. Pure; exported for tests.
 */
export function parseWGSLBindings(code: string): BindingKind[] | undefined {
  if (/@group\(\s*[1-9]/.test(code)) return undefined;
  const out: BindingKind[] = [];
  let declared = 0;
  for (const m of code.matchAll(BINDING_RE)) {
    const i = Number(m[1]);
    if (out[i] !== undefined) return undefined;
    out[i] = m[2] === "uniform" ? "uniform" : m[3] === "read_write" ? "read_write" : "read";
    declared++;
  }
  // Every `@binding` must be one we understood (no textures/samplers), with no gaps.
  if ((code.match(/@binding\(/g) ?? []).length !== declared || out.length !== declared) return undefined;
  if (out.filter((k) => k === "uniform").length > 1) return undefined;
  return out;
}

/** A compiled kernel: pipeline + the bind-group layout its dispatches use. */
export interface ComputeKernel {
  readonly label: string;
  readonly pipeline: GPUComputePipeline;
  /** Explicit layout (dynamic-offset uniform), or `undefined` for an `"auto"`-layout fallback kernel. */
  readonly layout: GPUBindGroupLayout | undefined;
  /** Bind groups are cached per layout id; kernels with the same binding signature share one. */
  readonly layoutId: number;
  readonly bindings: readonly BindingKind[] | undefined;
  /** Stable per-device id (part of the readback wait-estimate key). */
  readonly id: number;
}

/**
 * Compile (once per device and source) a compute kernel. Explicit layouts
 * are shared across kernels with the same binding signature (e.g. `rrwu`),
 * which is what makes the bind-group cache effective across kernels — and
 * why its key must include the uniform size (see {@link dispatchKernel}).
 */
export function getKernel(device: GPUDevice, code: string, label = "kernel", entryPoint = "main"): ComputeKernel {
  const st = stateFor(device);
  const key = entryPoint === "main" ? code : `${entryPoint} ${code}`;
  const hit = st.kernels.get(key);
  if (hit) return hit;
  const bindings = parseWGSLBindings(code);
  const module = device.createShaderModule({ code, label });
  let kernel: ComputeKernel;
  if (bindings) {
    const sig = bindings.map((b) => (b === "read" ? "r" : b === "read_write" ? "w" : "u")).join("");
    let entry = st.layouts.get(sig);
    if (!entry) {
      const layout = device.createBindGroupLayout({
        label: `layout:${sig}`,
        entries: bindings.map((b, i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer:
            b === "uniform"
              ? { type: "uniform" as const, hasDynamicOffset: true }
              : { type: b === "read" ? ("read-only-storage" as const) : ("storage" as const) },
        })),
      });
      entry = { layout, id: st.nextLayoutId++ };
      st.layouts.set(sig, entry);
    }
    const pipeline = device.createComputePipeline({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [entry.layout] }),
      compute: { module, entryPoint },
    });
    kernel = { label, pipeline, layout: entry.layout, layoutId: entry.id, bindings, id: st.kernels.size + 1 };
  } else {
    const pipeline = device.createComputePipeline({ label, layout: "auto", compute: { module, entryPoint } });
    kernel = { label, pipeline, layout: undefined, layoutId: st.nextLayoutId++, bindings: undefined, id: st.kernels.size + 1 };
  }
  st.kernels.set(key, kernel);
  return kernel;
}

const checked = new WeakMap<GPUDevice, Map<string, string | null>>();

/**
 * {@link getKernel}, but the FIRST compile of `code` on a device is wrapped
 * in a validation error scope and awaited, throwing the shader error
 * instead of letting later dispatches silently no-op (WebGPU reports
 * shader errors asynchronously). The outcome is remembered per device.
 */
export async function getKernelChecked(device: GPUDevice, code: string, label = "kernel"): Promise<ComputeKernel> {
  let results = checked.get(device);
  if (!results) checked.set(device, (results = new Map()));
  let err = results.get(code);
  if (err === undefined) {
    device.pushErrorScope("validation");
    getKernel(device, code, label);
    const e = await device.popErrorScope();
    err = e ? e.message : null;
    results.set(code, err);
  }
  if (err !== null) throw new Error(`${label}: shader failed validation: ${err}`);
  return getKernel(device, code, label);
}

/** Compiled-pipeline cache keyed by shader source (plus entry point) — a `createShaderModule`+`createComputePipeline` pair only ever happens once per distinct source per device. */
export function getOrCreateComputePipeline(device: GPUDevice, code: string, entryPoint = "main"): GPUComputePipeline {
  return getKernel(device, code, "kernel", entryPoint).pipeline;
}

/** Number of distinct (source, entryPoint) pipelines compiled so far for `device` — test-only introspection to assert caching actually happened, not a production API. */
export function pipelineCacheSize(device: GPUDevice): number {
  return states.get(device)?.kernels.size ?? 0;
}

// ---- dispatch -----------------------------------------------------------------

/** Bytes in each device's uniform ring (`UNIFORM | COPY_DST`). */
const UNIFORM_ARENA_BYTES = 1 << 16;
/** Bind groups cached per device before the cache is simply cleared (keys embed never-reused buffer ids, so stale entries only cost memory). */
const MAX_CACHED_BIND_GROUPS = 4096;

function bufferId(st: RuntimeState, b: GPUBuffer): number {
  let id = st.bufferIds.get(b);
  if (id === undefined) st.bufferIds.set(b, (id = st.nextBufferId++));
  return id;
}

/**
 * Write `data` (padded to a 16-byte multiple) into the device's uniform ring
 * and return its offset. Wrapping back to 0 is safe because every dispatch
 * writes and submits before the next one encodes: `writeBuffer` is ordered
 * on the queue after every earlier submit that read the old contents.
 */
function writeUniform(device: GPUDevice, st: RuntimeState, data: ArrayBufferView): { offset: number; size: number; arenaId: number; buffer: GPUBuffer } {
  const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
  if (size > UNIFORM_ARENA_BYTES) throw new RangeError(`dispatchKernel: uniform of ${data.byteLength} bytes is too large`);
  let bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  if (bytes.byteLength !== size) {
    const padded = new Uint8Array(size);
    padded.set(bytes);
    bytes = padded;
  }
  if (!st.arena) {
    const buffer = device.createBuffer({ label: "uniform-arena", size: UNIFORM_ARENA_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    st.arena = { buffer, id: bufferId(st, buffer), cursor: 0 };
  }
  const arena = st.arena;
  let offset = arena.cursor;
  if (offset + size > UNIFORM_ARENA_BYTES) offset = 0;
  arena.cursor = offset + Math.ceil(size / st.uniformAlign) * st.uniformAlign;
  writeBytes(device, arena.buffer, offset, bytes);
  return { offset, size, arenaId: arena.id, buffer: arena.buffer };
}

export interface DispatchOptions {
  /** Uniform data for the kernel's `var<uniform>` binding (padded to 16 bytes). */
  uniform?: ArrayBufferView;
  /** Name reported by the timestamp profiler (default: the kernel's label). */
  label?: string;
}

/**
 * Encode and submit one compute dispatch of `code` (compiled and cached via
 * {@link getKernel}). `buffers` are the kernel's STORAGE bindings in binding
 * order (the uniform binding, if any, is skipped — it's filled from
 * `options.uniform`); `groups` is the workgroup grid.
 *
 * The bind group is cached by (layout, storage buffers, uniform ring,
 * uniform SIZE) and the uniform travels as a dynamic offset, so a repeated
 * dispatch over the same buffers creates nothing. Leaving the size out of
 * that key is a real bug, not a nicety: two kernels with the same binding
 * signature but different uniform struct sizes would share a bind group
 * whose uniform range is too small for the larger one ("binding is too
 * small"); test/runtime.test.ts pins it.
 */
export function dispatchKernel(
  device: GPUDevice,
  code: string,
  buffers: readonly (GPUBuffer | SizedBuffer)[],
  groups: readonly [number, number?, number?],
  options: DispatchOptions = {},
): void {
  const st = stateFor(device);
  const kernel = getKernel(device, code, options.label);
  const storage = buffers.map((b) => ("buffer" in b ? b.buffer : b));
  const uniformIndex = kernel.bindings ? kernel.bindings.indexOf("uniform") : -1;
  const expected = kernel.bindings ? kernel.bindings.length - (uniformIndex >= 0 ? 1 : 0) : storage.length;
  if (storage.length !== expected) {
    throw new RangeError(`dispatchKernel(${kernel.label}): ${storage.length} storage buffers for ${expected} storage bindings`);
  }
  if ((uniformIndex >= 0) !== (options.uniform !== undefined)) {
    throw new TypeError(
      uniformIndex >= 0
        ? `dispatchKernel(${kernel.label}): the kernel declares a uniform but none was passed`
        : `dispatchKernel(${kernel.label}): a uniform was passed but the kernel ${kernel.bindings ? "declares none" : "uses an auto layout"}`,
    );
  }

  let hash = mix(st.workHash, kernel.id);
  let key = String(kernel.layoutId);
  for (const b of storage) key += `,${bufferId(st, b)}`;
  let u: ReturnType<typeof writeUniform> | undefined;
  if (options.uniform) {
    u = writeUniform(device, st, options.uniform);
    key += `|u${u.arenaId}:${u.size}`;
    const words = new Uint32Array(options.uniform.buffer, options.uniform.byteOffset, options.uniform.byteLength >> 2);
    for (const w of words) hash = mix(hash, w);
  }
  let bindGroup = st.bindGroups.get(key);
  if (bindGroup) {
    st.stats.bindGroupCacheHits++;
  } else {
    const entries: GPUBindGroupEntry[] = [];
    let s = 0;
    const n = kernel.bindings ? kernel.bindings.length : storage.length;
    for (let i = 0; i < n; i++) {
      entries.push(
        i === uniformIndex && u
          ? { binding: i, resource: { buffer: u.buffer, offset: 0, size: u.size } }
          : { binding: i, resource: { buffer: storage[s++] as GPUBuffer } },
      );
    }
    bindGroup = device.createBindGroup({ layout: kernel.layout ?? kernel.pipeline.getBindGroupLayout(0), entries });
    if (st.bindGroups.size >= MAX_CACHED_BIND_GROUPS) st.bindGroups.clear();
    st.bindGroups.set(key, bindGroup);
    st.stats.bindGroupsCreated++;
  }

  const [x, y = 1, z = 1] = groups;
  hash = mix(mix(mix(hash, x), y), z);
  const encoder = device.createCommandEncoder();
  const prof = st.profiler;
  let passDesc: GPUComputePassDescriptor | undefined;
  if (prof) {
    if (prof.labels.length >= prof.capacity) drainProfiler(device, prof, encoder);
    const i = prof.labels.length;
    passDesc = { timestampWrites: { querySet: prof.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } };
    prof.labels.push(options.label ?? kernel.label);
  }
  const pass = encoder.beginComputePass(passDesc);
  pass.setPipeline(kernel.pipeline);
  if (u) pass.setBindGroup(0, bindGroup, [u.offset]);
  else pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.max(1, x), Math.max(1, y), Math.max(1, z));
  pass.end();
  device.queue.submit([encoder.finish()]);
  st.workHash = hash;
  st.dispatchesSinceRead++;
  st.stats.dispatches++;
}

/** Dispatch a compute shader with `bindings` bound at group 0 in binding-index order, `workgroupCount` groups along X only. A thin wrapper over {@link dispatchKernel} for kernels without a uniform. */
export function dispatchCompute(
  device: GPUDevice,
  code: string,
  bindings: readonly SizedBuffer[],
  workgroupCountX: number,
): void {
  dispatchKernel(device, code, bindings, [workgroupCountX], { label: "compute" });
}

/** `Math.ceil(total / groupSize)`, minimum 1 (WebGPU rejects a 0-workgroup dispatch on some backends). */
export function workgroupsFor(total: number, groupSize: number): number {
  return Math.max(1, Math.ceil(total / groupSize));
}

// ---- GPU timestamp profiler ---------------------------------------------------

/** GPU time spent in one kernel label while profiling. */
export interface KernelTiming {
  kernel: string;
  /** Total GPU milliseconds across `count` dispatches. */
  ms: number;
  count: number;
}

/** Resolve the queries recorded so far into a fresh staging buffer (read at stop) and start the query set over. */
function drainProfiler(device: GPUDevice, prof: ProfilerState, encoder: GPUCommandEncoder): void {
  const n = prof.labels.length;
  if (!n) return;
  const staging = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  encoder.resolveQuerySet(prof.querySet, 0, 2 * n, prof.resolve, 0);
  encoder.copyBufferToBuffer(prof.resolve, 0, staging, 0, n * 16);
  prof.pending.push({ staging, labels: prof.labels });
  prof.labels = [];
}

/**
 * Start timing every dispatch on `device` on the GPU (each dispatch's
 * compute pass is bracketed by `timestampWrites`). Needs a device created
 * with the `timestamp-query` feature (`detectWebGPU({ timestampQuery: true })`);
 * throws otherwise. Browsers may quantize timestamps (Chrome does, unless
 * started with `--enable-unsafe-webgpu`). Dispatches add a little overhead
 * while profiling. Call {@link stopProfiling} for the results.
 */
export function startProfiling(device: GPUDevice, options: { capacity?: number } = {}): void {
  if (!device.features.has("timestamp-query")) {
    throw new Error("startProfiling: the device lacks the timestamp-query feature (request it, e.g. detectWebGPU({ timestampQuery: true }))");
  }
  const st = stateFor(device);
  if (st.profiler) return;
  const capacity = options.capacity ?? 256;
  st.profiler = {
    querySet: device.createQuerySet({ type: "timestamp", count: 2 * capacity }),
    resolve: device.createBuffer({ size: 16 * capacity, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
    capacity,
    labels: [],
    pending: [],
  };
}

/** Stop profiling and return GPU time per kernel label, slowest first. Returns `[]` when not profiling. */
export async function stopProfiling(device: GPUDevice): Promise<KernelTiming[]> {
  const st = stateFor(device);
  const prof = st.profiler;
  if (!prof) return [];
  st.profiler = null;
  const encoder = device.createCommandEncoder();
  drainProfiler(device, prof, encoder);
  device.queue.submit([encoder.finish()]);
  const totals = new Map<string, { ns: number; count: number }>();
  for (const { staging, labels } of prof.pending) {
    await staging.mapAsync(GPUMapMode.READ);
    const ts = new BigUint64Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    labels.forEach((label, i) => {
      const t = totals.get(label) ?? { ns: 0, count: 0 };
      const begin = ts[2 * i] as bigint;
      const end = ts[2 * i + 1] as bigint;
      t.ns += end > begin ? Number(end - begin) : 0;
      t.count++;
      totals.set(label, t);
    });
  }
  prof.querySet.destroy();
  prof.resolve.destroy();
  return [...totals].map(([kernel, t]) => ({ kernel, ms: t.ns / 1e6, count: t.count })).sort((a, b) => b.ms - a.ms);
}
