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

test("configureGPURuntime / readback sleep: off by default (backend-webgpu's 3 ms threshold costs latency on typical readbacks); the deprecated knob sets the backend's flag", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ defaultOn: boolean; off: boolean; on: boolean }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const rt = backendFor(device).rt;
    const defaultOn = rt.sleepWhileWaiting;
    configureGPURuntime(device, { sleepWhileWaiting: false });
    const off = rt.sleepWhileWaiting;
    configureGPURuntime(device, { sleepWhileWaiting: true });
    return { defaultOn, off, on: rt.sleepWhileWaiting };
    `,
    bundle(),
  );
  assert.equal(r.defaultOn, false, "default off for backends this package creates (bridge.ts SLEEP_WHILE_WAITING_DEFAULT)");
  assert.equal(r.off, false);
  assert.equal(r.on, true);
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
