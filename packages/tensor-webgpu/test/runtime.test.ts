/**
 * The runtime behaviour this package relies on since issue #146, when every
 * op (GEMM, attention, IR fusion) runs on @johnhenry/backend-webgpu's
 * runtime — on a real adapter (test/helpers.ts: Dawn in-process or
 * headless Chrome). The runtime's own mechanics (explicit layouts, the
 * bind-group cache key, the uniform arena, profiler internals) are tested
 * in laya-js; these pin what the shims and the fusion path depend on:
 *
 *  - one backend per GPUDevice, shared by the deprecated functions and
 *    `createWebGpuDevice({ device })`,
 *  - the bind-group and pipeline caches hit through the shims,
 *  - the fused kernel's per-dispatch uniforms (element count, input
 *    offsets) survive hundreds of dispatches before one readback,
 *  - the deprecated runtime knobs (`configureGPURuntime`, the profiler)
 *    act on that backend.
 */
import assert from "node:assert/strict";
import path from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { bundleForBrowser, closeHarness, getHarness, SRC } from "./helpers.ts";

after(closeHarness);

const bundle = (): string => bundleForBrowser([path.join(SRC, "index.ts")]);

test("one backend per GPUDevice: the deprecated functions and createWebGpuDevice({ device }) share it; a second backend for the device is refused", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ same: boolean; shared: boolean; refused: string; afterDestroy: boolean }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const b = backendFor(device);
    const gpu = await createWebGpuDevice({ device });
    let refused = "";
    try { registerBackend(new b.constructor(device, b.adapterInfo, { f16: false, ownsDevice: false })); } catch (e) { refused = e.message; }
    gpu.destroy(); // a device passed in stays alive and keeps its backend
    return { same: backendFor(device) === b, shared: gpu.backend === b, refused, afterDestroy: backendFor(device) === b };
    `,
    bundle(),
  );
  assert.ok(r.same && r.shared && r.afterDestroy);
  assert.match(r.refused, /already has a backend/);
});

test("bind-group and pipeline caches hit through the shims: a repeated same-shape runGemm creates neither", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ first: number[]; second: number[]; ok: boolean }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const stats = () => { const s = backendFor(device).rt.stats; return [s.bindGroups, s.pipelines, s.buffersCreated]; };
    const A = GPUTensor.fromFloat32Array(device, new Float32Array(24 * 20).map((_, i) => (i % 7) - 3), [24, 20]);
    const B = GPUTensor.fromFloat32Array(device, new Float32Array(20 * 28).map((_, i) => (i % 5) - 2), [20, 28]);
    const s0 = stats();
    const c1 = await runGemm(device, A, B);
    const d1 = await c1.toFloat32Array();
    c1.free();
    const s1 = stats();
    const c2 = await runGemm(device, A, B); // the pool hands back c1's buffer: same bind group
    const d2 = await c2.toFloat32Array();
    const s2 = stats();
    for (const x of [A, B, c2]) x.free();
    return { first: s1.map((v, i) => v - s0[i]), second: s2.map((v, i) => v - s1[i]), ok: d1.every((x, i) => x === d2[i]) };
    `,
    bundle(),
  );
  assert.ok(r.first[0]! >= 1 && r.first[1]! >= 1, "the first call compiles and binds");
  assert.deepEqual(r.second, [0, 0, 0], "second call: no bind group, no pipeline, no buffer");
  assert.ok(r.ok);
});

test("fused kernel: 300 dispatches over views at distinct offsets before one readback each see their own uniforms (n, o0)", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ bad: number; count: number }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const gpu = await createWebGpuDevice({ device });
    const b = gpu.backend;
    const data = new Float32Array(400).map((_, i) => i);
    const x = await gpu.fromHost({ dtype: "f32", shape: [400], data });
    const node = { kind: "binary", op: "add", left: { kind: "binary", op: "mul", left: { kind: "input", index: 0 }, right: { kind: "const", value: 2 } }, right: { kind: "const", value: 1 } };
    const outs = [];
    for (let i = 0; i < 300; i++) outs.push(gpu.fuse(node, [b.slice(x, [i], [i + 1 + (i % 50)])]));
    let bad = 0;
    for (let i = 0; i < outs.length; i++) {
      const got = (await gpu.toHost(outs[i])).data;
      for (let j = 0; j < got.length; j++) if (got[j] !== 2 * (i + j) + 1) bad++;
      if (got.length !== 1 + (i % 50)) bad++;
    }
    return { bad, count: outs.length };
    `,
    bundle(),
  );
  assert.equal(r.count, 300);
  assert.equal(r.bad, 0);
});

test("GPUTensor.fromBuffer: a caller's buffer is viewed without a copy (backend-webgpu's wrapBuffer), feeds ops, is never pooled, and free() destroys it", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ read: number[]; gemm: number[]; pooledDelta: number; mapFailed: boolean; f16: string }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const rt = backendFor(device).rt;
    const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, new Float32Array([1, 2, 3, 4]));
    const a = GPUTensor.fromBuffer(device, buf, [2, 2]);
    const read = [...(await a.toFloat32Array())];
    const id = GPUTensor.fromFloat32Array(device, new Float32Array([1, 0, 0, 1]), [2, 2]);
    const prod = await runGemm(device, a, id);
    const gemm = [...(await prod.toFloat32Array())];
    const pooled0 = rt.stats.pooledBytes;
    a.free();
    const pooledDelta = rt.stats.pooledBytes - pooled0;
    // Destroyed: a copy out of it is a validation error.
    const probe = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    device.pushErrorScope("validation");
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, probe, 0, 16);
    device.queue.submit([enc.finish()]);
    const mapFailed = (await device.popErrorScope()) !== null;
    let f16 = "ok";
    if (!device.features.has("shader-f16")) {
      try { GPUTensor.fromBuffer(device, buf, [4], "f16"); } catch (e) { f16 = e.message; }
    }
    for (const x of [id, prod]) x.free();
    return { read, gemm, pooledDelta, mapFailed, f16 };
    `,
    bundle(),
  );
  assert.deepEqual(r.read, [1, 2, 3, 4]);
  assert.deepEqual(r.gemm, [1, 2, 3, 4], "the wrapped buffer feeds backend ops (A · I)");
  assert.equal(r.pooledDelta, 0, "a caller's buffer never enters the runtime's pool");
  assert.ok(r.mapFailed, "free() destroyed the caller's buffer");
  assert.ok(r.f16 === "ok" || /shader-f16/.test(r.f16), r.f16);
});

test("configureGPURuntime / readback sleep: a 15 ms threshold by default (backend-webgpu's 3 ms one costs latency on typical 2–6 ms readbacks), sleeping on under Dawn and off in browsers as backend-webgpu decides; the deprecated knob sets both", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ defaults: [boolean, number][]; off: boolean; on: boolean; threshold: number }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const rt = backendFor(device).rt;
    // Both ways this package creates a backend: backendFor (sync, deprecated API) and createWebGpuDevice.
    const gpu = await createWebGpuDevice({ device: await (await navigator.gpu.requestAdapter()).requestDevice() });
    const defaults = [[rt.sleepWhileWaiting, rt.sleepThresholdMs], [gpu.backend.rt.sleepWhileWaiting, gpu.backend.rt.sleepThresholdMs]];
    configureGPURuntime(device, { sleepWhileWaiting: false });
    const off = rt.sleepWhileWaiting;
    configureGPURuntime(device, { sleepWhileWaiting: true, sleepThresholdMs: 40 });
    return { defaults, off, on: rt.sleepWhileWaiting, threshold: rt.sleepThresholdMs };
    `,
    bundle(),
  );
  // In the Dawn harness the page's navigator.gpu is a parameter, not globalThis.navigator.gpu, so the backend sees Dawn.
  const sleeps = harness.kind === "dawn";
  for (const [on, ms] of r.defaults) {
    assert.equal(on, sleeps, "sleeping follows backend-webgpu's default (on under Dawn, off for navigator.gpu)");
    assert.equal(ms, 15, "bridge.ts SLEEP_THRESHOLD_MS_DEFAULT");
  }
  assert.equal(r.off, false);
  assert.equal(r.on, true);
  assert.equal(r.threshold, 40);
});

test("timestamp profiler (deprecated startProfiling/stopProfiling): GPU time per backend kernel with counts; throws without timestamp-query", async (t) => {
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
    startProfiling(device);
    const outs = [];
    for (let i = 0; i < 7; i++) outs.push(await runGemm(device, A, A));
    for (let i = 0; i < 3; i++) outs.push(await runQKT(device, q, q, 2, 64, 64, 32));
    const timings = await stopProfiling(device);
    for (const o of [A, q, ...outs]) o.free();
    return { has, timings, noFeature };
    `,
    bundle(),
  );
  if (!r.has) return t.skip("adapter lacks the timestamp-query feature");
  assert.match(r.noFeature, /timestamp-query/);
  const count = (prefix: string) => r.timings.filter((x) => x.kernel.startsWith(prefix)).reduce((a, x) => a + x.count, 0);
  assert.equal(count("gemm"), 10, "7 GEMMs + 3 QKᵀ matmuls");
  assert.equal(count("copy"), 3, "each QKᵀ transposes K once");
  assert.ok(r.timings.every((x) => x.ms >= 0));
  assert.ok(r.timings.filter((x) => x.kernel.startsWith("gemm")).reduce((a, x) => a + x.ms, 0) > 0, "256^3 GEMMs take measurable GPU time");
});
