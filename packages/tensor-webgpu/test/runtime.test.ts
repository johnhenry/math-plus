/**
 * gpu-runtime.ts's laya-js ports (issue #126), on a real adapter
 * (test/helpers.ts: Dawn in-process or headless Chrome):
 *
 *  - explicit layouts parsed from WGSL (pure unit test),
 *  - the bind-group cache + uniform ring with dynamic offsets, including the
 *    REGRESSION test for a cache key without the uniform size ("binding is
 *    too small"),
 *  - sleep-while-waiting readbacks (default on under Dawn only),
 *  - the timestamp profiler.
 */
import assert from "node:assert/strict";
import path from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { parseWGSLBindings } from "../src/gpu-runtime.ts";
import { bundleForBrowser, closeHarness, getHarness, SRC } from "./helpers.ts";

after(closeHarness);

const bundle = (): string => bundleForBrowser([path.join(SRC, "gemm.ts"), path.join(SRC, "attention.ts")]);

test("parseWGSLBindings: reads group-0 storage/uniform bindings in order; undefined for anything it can't lay out explicitly", () => {
  assert.deepEqual(
    parseWGSLBindings(`
      @group(0) @binding(1) var<storage, read_write> o: array<f32>;
      @group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
      @group(0)@binding(2) var<uniform> p: P;`),
    ["read", "read_write", "uniform"],
  );
  assert.deepEqual(parseWGSLBindings("@group(0) @binding(0) var<storage> a: array<f32>;"), ["read"], "storage defaults to read");
  assert.deepEqual(parseWGSLBindings("fn main() {}"), []);
  assert.equal(parseWGSLBindings("@group(0) @binding(1) var<storage, read> a: array<f32>;"), undefined, "gap at 0");
  assert.equal(parseWGSLBindings("@group(1) @binding(0) var<storage, read> a: array<f32>;"), undefined, "other groups");
  assert.equal(parseWGSLBindings("@group(0) @binding(0) var t: texture_2d<f32>;"), undefined, "textures");
  assert.equal(
    parseWGSLBindings("@group(0) @binding(0) var<uniform> a: P; @group(0) @binding(1) var<uniform> b: P;"),
    undefined,
    "two uniforms",
  );
});

test("bind-group cache: a repeated dispatch over the same (pooled) buffers creates no bind group and no uniform buffer", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ first: number; second: number; hits: number; buffersSecond: number; ok: boolean }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    let bindGroups = 0, buffers = 0;
    const cbg = device.createBindGroup.bind(device);
    device.createBindGroup = (d) => { bindGroups++; return cbg(d); };
    const cb = device.createBuffer.bind(device);
    device.createBuffer = (d) => { buffers++; return cb(d); };
    const a = new Float32Array(24 * 20).map((_, i) => (i % 7) - 3);
    const b = new Float32Array(20 * 28).map((_, i) => (i % 5) - 2);
    const c1 = await runGemmWGSL(device, a, b, 24, 20, 28);
    const first = bindGroups;
    buffers = 0;
    const c2 = await runGemmWGSL(device, a, b, 24, 20, 28);
    const ok = c1.every((x, i) => x === c2[i]);
    return { first, second: bindGroups - first, hits: gpuRuntimeStats(device).bindGroupCacheHits, buffersSecond: buffers, ok };
    `,
    bundle(),
  );
  assert.equal(r.first, 1);
  assert.equal(r.second, 0, "same-shape second call must reuse the cached bind group");
  assert.equal(r.hits, 1);
  assert.equal(r.buffersSecond, 1, "only the (unpooled) MAP_READ staging buffer is created; the uniform lives in the ring");
  assert.ok(r.ok);
});

test("REGRESSION: the bind-group cache key includes the uniform size — same layout and buffers, 16- vs 32-byte uniforms, no 'binding is too small'", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  // Two kernels with the SAME binding signature ("wu") over the SAME storage
  // buffer; only the uniform struct size differs. Keyed without the size,
  // the second dispatch would reuse the first one's bind group, whose
  // uniform range (16 bytes) is smaller than the 32-byte struct: a
  // validation error and a silently skipped dispatch.
  const r = await harness.run<{ values: number[]; error: string | null; bindGroups: number }>(
    `
    const K16 = \`struct P { a: u32, b: u32, c: u32, d: u32 };
    @group(0) @binding(0) var<storage, read_write> out: array<u32>;
    @group(0) @binding(1) var<uniform> p: P;
    @compute @workgroup_size(1) fn main() { out[0] = p.a + p.d; }\`;
    const K32 = \`struct P { a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32 };
    @group(0) @binding(0) var<storage, read_write> out: array<u32>;
    @group(0) @binding(1) var<uniform> p: P;
    @compute @workgroup_size(1) fn main() { out[1] = p.a + p.h; }\`;
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const out = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    device.pushErrorScope("validation");
    dispatchKernel(device, K16, [out], [1], { uniform: new Uint32Array([1, 0, 0, 2]) });
    dispatchKernel(device, K32, [out], [1], { uniform: new Uint32Array([10, 0, 0, 0, 0, 0, 0, 20]) });
    dispatchKernel(device, K16, [out], [1], { uniform: new Uint32Array([100, 0, 0, 200]) });
    const e = await device.popErrorScope();
    const values = Array.from(new Uint32Array(await readBackBytes(device, out, 16)));
    return { values, error: e ? e.message : null, bindGroups: gpuRuntimeStats(device).bindGroupsCreated };
    `,
    bundle(),
  );
  assert.equal(r.error, null);
  assert.deepEqual(r.values.slice(0, 2), [300, 30]);
  assert.equal(r.bindGroups, 2, "one bind group per uniform size, the third dispatch reuses the first");
});

test("uniform ring: 600 dispatches with distinct uniforms before one readback (wrapping the 64 KiB ring) all see their own values", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ bad: number; created: number; hits: number }>(
    `
    const K = \`struct P { i: u32, v: u32 };
    @group(0) @binding(0) var<storage, read_write> out: array<u32>;
    @group(0) @binding(1) var<uniform> p: P;
    @compute @workgroup_size(1) fn main() { out[p.i] = p.v; }\`;
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const N = 600;
    const out = device.createBuffer({ size: 4 * N, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    for (let i = 0; i < N; i++) dispatchKernel(device, K, [out], [1], { uniform: new Uint32Array([i, 7 * i + 3]) });
    const got = new Uint32Array(await readBackBytes(device, out, 4 * N));
    let bad = 0;
    for (let i = 0; i < N; i++) if (got[i] !== 7 * i + 3) bad++;
    const s = gpuRuntimeStats(device);
    return { bad, created: s.bindGroupsCreated, hits: s.bindGroupCacheHits };
    `,
    bundle(),
  );
  assert.equal(r.bad, 0);
  assert.equal(r.created, 1);
  assert.equal(r.hits, 599);
});

test("readback sleep-while-waiting: on by default under Dawn only; when on, a repeated readback of the same work sleeps first and still returns the right data", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ defaultOn: boolean; sleeps: number[]; waitMs: number; ok: boolean }>(
    `
    // A deliberately slow kernel (a serial fma chain per thread) so the GPU
    // wait is well above the 3 ms sleep threshold on any adapter.
    const K = \`struct P { n: u32, seed: f32 };
    @group(0) @binding(0) var<storage, read_write> out: array<f32>;
    @group(0) @binding(1) var<uniform> p: P;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
      var x = p.seed + f32(g.x);
      for (var i = 0u; i < p.n; i++) { x = fma(x, 0.999999, 1e-6); }
      out[g.x] = x;
    }\`;
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const defaultOn = gpuRuntimeStats(device).sleepWhileWaiting;
    configureGPURuntime(device, { sleepWhileWaiting: true, sleepThresholdMs: 3 });
    const out = device.createBuffer({ size: 4 * 4096, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const u = (n) => { const b = new ArrayBuffer(8); new Uint32Array(b, 0, 1)[0] = n; new Float32Array(b, 4, 1)[0] = 0.5; return new Uint8Array(b); };
    const run = async (n) => { dispatchKernel(device, K, [out], [64], { uniform: u(n) }); const t0 = performance.now(); const d = new Float32Array(await readBackBytes(device, out, 16)); return { ms: performance.now() - t0, d }; };
    // Grow the work until one readback waits > 8 ms.
    let n = 1 << 14, first;
    for (let i = 0; i < 14; i++, n *= 2) { first = await run(n); if (first.ms > 8) break; }
    const sleeps = [gpuRuntimeStats(device).readbackSleeps];
    const second = await run(n); // same kernel, grid and uniforms: has an estimate
    sleeps.push(gpuRuntimeStats(device).readbackSleeps);
    configureGPURuntime(device, { sleepWhileWaiting: false });
    await run(n);
    sleeps.push(gpuRuntimeStats(device).readbackSleeps);
    const ok = second.d.every((x, i) => x === first.d[i]);
    return { defaultOn, sleeps, waitMs: first.ms, ok };
    `,
    bundle(),
  );
  assert.equal(r.defaultOn, harness.kind === "dawn", "default: sleep under Dawn (no navigator.gpu), poll in the browser");
  assert.ok(r.waitMs > 8, `calibrated wait ${r.waitMs} ms`);
  assert.equal(r.sleeps[1]! - r.sleeps[0]!, 1, "the repeated readback slept");
  assert.equal(r.sleeps[2]! - r.sleeps[1]!, 0, "sleepWhileWaiting: false never sleeps");
  assert.ok(r.ok, "same work, same result");
});

test("timestamp profiler: per-label GPU time and counts across a resolve-batch boundary; throws without timestamp-query", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ has: boolean; timings: { kernel: string; ms: number; count: number }[]; noFeature: string }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const has = adapter.features.has("timestamp-query");
    if (!has) return { has, timings: [], noFeature: "" };
    const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
    const plain = await (await navigator.gpu.requestAdapter()).requestDevice();
    let noFeature = "";
    try { startProfiling(plain); } catch (e) { noFeature = e.message; }
    const A = GPUTensor.fromFloat32Array(device, new Float32Array(256 * 256).fill(0.5), [256, 256]);
    const q = GPUTensor.fromFloat32Array(device, new Float32Array(2 * 64 * 32).fill(0.25), [2, 64, 32]);
    startProfiling(device, { capacity: 4 }); // small, so the 10 dispatches below span 3 resolve batches
    const outs = [];
    for (let i = 0; i < 7; i++) outs.push(await runGemm(device, A, A));
    for (let i = 0; i < 3; i++) outs.push(await runQKT(device, q, q, 2, 64, 64, 32));
    const timings = await stopProfiling(device);
    for (const o of [A, q, ...outs]) o.free();
    return { has, timings, noFeature };
    `,
    bundle(),
  );
  // Both harnesses on the reference machine (Dawn/Metal, Chrome/Metal) expose it; a software adapter may not.
  if (!r.has) return t.skip("adapter lacks the timestamp-query feature");
  assert.match(r.noFeature, /timestamp-query/);
  const byKernel = new Map(r.timings.map((x) => [x.kernel, x]));
  const gemm = r.timings.find((x) => x.kernel.startsWith("gemm:"));
  assert.equal(gemm?.count, 7);
  assert.equal(byKernel.get("attention:qkt")?.count, 3);
  assert.ok(r.timings.every((x) => x.ms >= 0));
  assert.ok((gemm?.ms ?? 0) > 0, "a 256^3 GEMM takes measurable GPU time");
});
