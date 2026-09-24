/**
 * Regression tests for issue #100's WebGPU perf fixes (findings 2-4):
 * shader/pipeline caching, buffer pooling, and GPU-resident chaining —
 * still holding through the device facade on @johnhenry/backend-webgpu's
 * runtime (whose pool also recycles the MAP_READ staging buffers). These
 * don't re-check numeric correctness (gemm.test.ts and
 * flash-attention.test.ts cross-check against NumPy) — they check that the
 * PERFORMANCE behavior the issue asked for is actually happening, by
 * monkey-patching `GPUDevice` creation methods inside the page and counting
 * calls, on the same harness every other test in this package uses
 * (test/helpers.ts).
 */
import assert from "node:assert/strict";
import path from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { bundleForBrowser, closeHarness, getHarness, SRC } from "./helpers.ts";

after(closeHarness);

function randomMatrix(size: number, seed: number): Float32Array {
  let s = seed >>> 0;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

const bundle = (): string => bundleForBrowser([path.join(SRC, "facade.ts")]);

/** Page-side: a facade on a fresh device, and a host-array GEMM through it (upload, `matmul`, readback, dispose). */
const PAGE_SETUP = `
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const gpu = await createWebGpuDevice({ device });
  const hostGemm = async (a, b, m, k, n) => {
    const A = await gpu.backend.fromHost({ dtype: "f32", shape: [m, k], data: a });
    const B = await gpu.backend.fromHost({ dtype: "f32", shape: [k, n], data: b });
    const C = gpu.backend.matmul(A, B);
    const out = (await gpu.toHost(C)).data;
    for (const x of [A, B, C]) gpu.dispose(x);
    return out;
  };
`;

test("matmul: a second call with a different shape in the same kernel variant reuses the cached shader module + compute pipeline", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const a1 = randomMatrix(4 * 3, 1);
  const b1 = randomMatrix(3 * 5, 2);
  const a2 = randomMatrix(6 * 7, 3);
  const b2 = randomMatrix(7 * 9, 4);
  const result = await harness.run<{
    shaderModuleCalls: number;
    pipelineCalls: number;
    cacheSizeAfter: number;
  }>(
    `${PAGE_SETUP}
    let shaderModuleCalls = 0;
    let pipelineCalls = 0;
    const origCreateShaderModule = device.createShaderModule.bind(device);
    device.createShaderModule = (desc) => { shaderModuleCalls++; return origCreateShaderModule(desc); };
    const origCreateComputePipeline = device.createComputePipeline.bind(device);
    device.createComputePipeline = (desc) => { pipelineCalls++; return origCreateComputePipeline(desc); };

    await hostGemm(new Float32Array(${JSON.stringify(Array.from(a1))}), new Float32Array(${JSON.stringify(Array.from(b1))}), 4, 3, 5);
    // A DIFFERENT shape in the same variant (backend-webgpu's tiled kernel,
    // K and N both not multiples of 4 -> scalar loads) -- m/n/k travel via
    // uniforms, not baked into the WGSL text, so the pipeline cache hits.
    await hostGemm(new Float32Array(${JSON.stringify(Array.from(a2))}), new Float32Array(${JSON.stringify(Array.from(b2))}), 6, 7, 9);

    return { shaderModuleCalls, pipelineCalls, cacheSizeAfter: gpu.backend.rt.stats.pipelines };
    `,
    bundle(),
  );
  assert.equal(result.shaderModuleCalls, 1, "createShaderModule should only run once across two calls");
  assert.equal(result.pipelineCalls, 1, "createComputePipeline should only run once across two calls");
  assert.equal(result.cacheSizeAfter, 1, "pipeline cache should hold exactly one entry for the one tiled variant both shapes use");
});

test("matmul: a second call with the SAME shape reuses pooled buffers (staging included) instead of allocating new ones", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const a1 = randomMatrix(4 * 3, 5);
  const b1 = randomMatrix(3 * 5, 6);
  const a2 = randomMatrix(4 * 3, 7);
  const b2 = randomMatrix(3 * 5, 8);
  const result = await harness.run<{ createBufferCallsAfterFirst: number; createBufferCallsAfterSecond: number }>(
    `${PAGE_SETUP}
    let createBufferCalls = 0;
    const origCreateBuffer = device.createBuffer.bind(device);
    device.createBuffer = (desc) => { createBufferCalls++; return origCreateBuffer(desc); };

    await hostGemm(new Float32Array(${JSON.stringify(Array.from(a1))}), new Float32Array(${JSON.stringify(Array.from(b1))}), 4, 3, 5);
    const createBufferCallsAfterFirst = createBufferCalls;
    // SAME shape as the first call -- every buffer it needs (A, B, out, the
    // uniform arena, the MAP_READ staging buffer) went back to
    // backend-webgpu's pools when the first call disposed its tensors.
    await hostGemm(new Float32Array(${JSON.stringify(Array.from(a2))}), new Float32Array(${JSON.stringify(Array.from(b2))}), 4, 3, 5);
    const createBufferCallsAfterSecond = createBufferCalls;

    return { createBufferCallsAfterFirst, createBufferCallsAfterSecond };
    `,
    bundle(),
  );
  assert.ok(result.createBufferCallsAfterFirst > 0, "the first call should allocate real buffers");
  assert.equal(
    result.createBufferCallsAfterSecond - result.createBufferCallsAfterFirst,
    0,
    "a same-shape second call should allocate nothing -- A/B/out/uniforms/staging all come from the pools",
  );
});

test("attention chain (QKᵀ -> softmax -> ·V) and a GEMM chain stay GPU-resident: no MAP_READ staging buffer until the caller explicitly reads a result back", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const q = randomMatrix(4 * 4, 66);
  const k = randomMatrix(4 * 4, 77);
  const v = randomMatrix(4 * 4, 88);
  const c = randomMatrix(4 * 3, 99);
  const result = await harness.run<{ duringChain: number; afterReadback: number }[]>(
    `${PAGE_SETUP}
    const b = gpu.backend;
    const up = (data, shape) => gpu.backend.fromHost({ dtype: "f32", shape, data: new Float32Array(data) });
    const q = await up(${JSON.stringify(Array.from(q))}, [1, 4, 4]);
    const k = await up(${JSON.stringify(Array.from(k))}, [1, 4, 4]);
    const v = await up(${JSON.stringify(Array.from(v))}, [1, 4, 4]);
    const c = await up(${JSON.stringify(Array.from(c))}, [4, 3]);
    await b.sync();

    let mapReadBuffers = 0;
    const origCreateBuffer = device.createBuffer.bind(device);
    device.createBuffer = (desc) => {
      if ((desc.usage & GPUBufferUsage.MAP_READ) !== 0) mapReadBuffers++;
      return origCreateBuffer(desc);
    };
    const out = [];
    for (const chain of [
      () => b.matmul(b.softmax(b.matmul(q, b.transpose(k, [0, 2, 1])), -1), v),
      () => b.matmul(b.matmul(b.reshape(q, [4, 4]), b.reshape(k, [4, 4])), c),
    ]) {
      const before = mapReadBuffers;
      const r = b.scope(chain);
      const duringChain = mapReadBuffers - before;
      await gpu.toHost(r); // only NOW read the final result back
      out.push({ duringChain, afterReadback: mapReadBuffers - before });
      gpu.dispose(r);
    }
    return out;
    `,
    bundle(),
  );
  for (const [i, r] of result.entries()) {
    assert.equal(r.duringChain, 0, `chain ${i}: no CPU staging/readback buffer while chaining GPU-resident ops`);
    assert.ok(r.afterReadback <= 1, `chain ${i}: at most one staging buffer once the caller reads the final result (the pool may already hold one)`);
  }
  assert.equal(result[0]!.afterReadback, 1, "the first readback creates exactly one staging buffer");
});
