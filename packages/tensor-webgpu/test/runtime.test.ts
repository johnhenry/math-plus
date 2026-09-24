/**
 * The runtime behaviour this package relies on since issue #146, when every
 * op (GEMM, attention, IR fusion) runs on @johnhenry/backend-webgpu's
 * runtime — on a real adapter (test/helpers.ts: Dawn in-process or
 * headless Chrome). The runtime's own mechanics (explicit layouts, the
 * bind-group cache key, the uniform arena, profiler internals) are tested
 * in laya-js; these pin what the facade and the fusion path depend on:
 *
 *  - one backend per GPUDevice, shared by every `createWebGpuDevice({ device })`,
 *  - the bind-group and pipeline caches hit through the facade,
 *  - the fused kernel's per-dispatch uniforms (element count, input
 *    offsets) survive hundreds of dispatches before one readback,
 *  - the readback-sleep defaults of backends created here, and the
 *    runtime's profiler reached through `gpu.backend.rt`.
 */
import assert from "node:assert/strict";
import path from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { bundleForBrowser, closeHarness, getHarness, SRC } from "./helpers.ts";

after(closeHarness);

const bundle = (): string => bundleForBrowser([path.join(SRC, "index.ts")]);

test("one backend per GPUDevice: createWebGpuDevice({ device }) calls share it, it survives destroy() of a device passed in, and a second backend for the device is refused", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ shared: boolean; refused: string; afterDestroy: boolean; deviceAlive: boolean }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const g1 = await createWebGpuDevice({ device });
    const g2 = await createWebGpuDevice({ device });
    let refused = "";
    const extra = await createBackend({ device });
    try { registerBackend(extra); } catch (e) { refused = e.message; } finally { extra.destroy(); }
    g1.destroy(); // a device passed in stays alive and keeps its backend
    const g3 = await createWebGpuDevice({ device });
    const x = await g3.fromHost({ dtype: "f32", shape: [2], data: new Float32Array([1, 2]) });
    const deviceAlive = Array.from((await g3.toHost(g3.backend.add(x, x))).data).join() === "2,4";
    return { shared: g1.backend === g2.backend, refused, afterDestroy: g3.backend === g1.backend, deviceAlive };
    `,
    bundle(),
  );
  assert.ok(r.shared, "one backend per device");
  assert.match(r.refused, /already has a backend/);
  assert.ok(r.afterDestroy, "a device passed in keeps its backend after destroy()");
  assert.ok(r.deviceAlive);
});

test("bind-group and pipeline caches hit through the facade: a repeated same-shape matmul creates neither", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ first: number[]; second: number[]; ok: boolean }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const gpu = await createWebGpuDevice({ device: await adapter.requestDevice() });
    const b = gpu.backend;
    const stats = () => { const s = b.rt.stats; return [s.bindGroups, s.pipelines, s.buffersCreated]; };
    const A = await gpu.fromHost({ dtype: "f32", shape: [24, 20], data: new Float32Array(24 * 20).map((_, i) => (i % 7) - 3) });
    const B = await gpu.fromHost({ dtype: "f32", shape: [20, 28], data: new Float32Array(20 * 28).map((_, i) => (i % 5) - 2) });
    const s0 = stats();
    const c1 = b.matmul(A, B);
    const d1 = (await gpu.toHost(c1)).data;
    gpu.dispose(c1);
    const s1 = stats();
    const c2 = b.matmul(A, B); // the pool hands back c1's buffer: same bind group
    const d2 = (await gpu.toHost(c2)).data;
    const s2 = stats();
    for (const x of [A, B, c2]) gpu.dispose(x);
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

test("readback sleep: backends created here default to a 15 ms threshold (backend-webgpu's 3 ms one costs latency on typical 2–6 ms readbacks), sleeping on under Dawn and off in browsers as backend-webgpu decides", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const defaults = await harness.run<[boolean, number][]>(
    `
    // A device passed in (from the page's adapter). A device createWebGpuDevice() requests itself: facade.test.ts.
    const passed = await createWebGpuDevice({ device: await (await navigator.gpu.requestAdapter()).requestDevice() });
    return [[passed.backend.rt.sleepWhileWaiting, passed.backend.rt.sleepThresholdMs]];
    `,
    bundle(),
  );
  assert.equal(defaults.length, 1);
  for (const [i, [on, ms]] of defaults.entries()) {
    // In the Dawn harness the page's navigator.gpu is a parameter, not globalThis.navigator.gpu, so the backend sees Dawn.
    assert.equal(on, harness.kind === "dawn", `#${i}: sleeping follows backend-webgpu's default (on under Dawn, off for navigator.gpu)`);
    assert.equal(ms, 15, `#${i}: bridge.ts SLEEP_THRESHOLD_MS_DEFAULT`);
  }
});

test("timestamp profiler through gpu.backend.rt: GPU time per backend kernel with counts; throws without timestamp-query", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ has: boolean; timings: { kernel: string; ms: number; count: number }[]; noFeature: string }>(
    `
    const cap = await detectWebGPU({ gpu: navigator.gpu, timestampQuery: true });
    if (!cap.available) throw new Error(cap.reason);
    const has = cap.device.features.has("timestamp-query");
    if (!has) return { has, timings: [], noFeature: "" };
    const gpu = await createWebGpuDevice({ device: cap.device });
    const plain = await createWebGpuDevice({ device: await (await navigator.gpu.requestAdapter()).requestDevice() });
    let noFeature = "";
    try { plain.backend.rt.startProfiling(); } catch (e) { noFeature = e.message; }
    const b = gpu.backend;
    const A = await gpu.fromHost({ dtype: "f32", shape: [256, 256], data: new Float32Array(256 * 256).fill(0.5) });
    b.rt.startProfiling();
    const outs = [];
    for (let i = 0; i < 7; i++) outs.push(b.matmul(A, A));
    const timings = await b.rt.stopProfiling();
    for (const o of [A, ...outs]) gpu.dispose(o);
    return { has, timings, noFeature };
    `,
    bundle(),
  );
  if (!r.has) return t.skip("adapter lacks the timestamp-query feature");
  assert.match(r.noFeature, /timestamp-query/);
  const count = (prefix: string) => r.timings.filter((x) => x.kernel.startsWith(prefix)).reduce((a, x) => a + x.count, 0);
  assert.equal(count("gemm"), 7, "7 matmuls");
  assert.ok(r.timings.every((x) => x.ms >= 0));
  assert.ok(r.timings.filter((x) => x.kernel.startsWith("gemm")).reduce((a, x) => a + x.ms, 0) > 0, "256^3 GEMMs take measurable GPU time");
});
