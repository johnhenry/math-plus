/**
 * Regression tests for issue #100's WebGPU perf fixes (findings 2-4):
 * shader/pipeline caching, buffer pooling, and GPU-resident attention
 * chaining. These don't re-check numeric correctness (gemm.test.ts and
 * attention.test.ts already cross-check every kernel against a CPU
 * reference for the NEW GPUTensor-based signatures) — they check that the
 * PERFORMANCE behavior the issue asked for is actually happening, by
 * monkey-patching `GPUDevice` creation methods inside the page and counting
 * calls, the same headless-Chrome-over-CDP harness every other test in this
 * package uses (test/helpers.ts).
 */
import assert from "node:assert/strict";
import path from "node:path";
import test, { after } from "node:test";
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

test("runGemmWGSL: a second call with a different shape in the same kernel variant reuses the cached shader module + compute pipeline", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const a1 = randomMatrix(4 * 3, 1);
  const b1 = randomMatrix(3 * 5, 2);
  const a2 = randomMatrix(6 * 7, 3);
  const b2 = randomMatrix(7 * 9, 4);
  const bundle = bundleForBrowser([path.join(SRC, "gemm.ts")]);
  const result = await harness.run<{
    shaderModuleCalls: number;
    pipelineCalls: number;
    cacheSizeAfter: number;
  }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    let shaderModuleCalls = 0;
    let pipelineCalls = 0;
    const origCreateShaderModule = device.createShaderModule.bind(device);
    device.createShaderModule = (desc) => { shaderModuleCalls++; return origCreateShaderModule(desc); };
    const origCreateComputePipeline = device.createComputePipeline.bind(device);
    device.createComputePipeline = (desc) => { pipelineCalls++; return origCreateComputePipeline(desc); };

    const a1 = new Float32Array(${JSON.stringify(Array.from(a1))});
    const b1 = new Float32Array(${JSON.stringify(Array.from(b1))});
    await runGemmWGSL(device, a1, b1, 4, 3, 5);

    // A DIFFERENT shape in the same variant (tiled kernel, K and N both not
    // multiples of 4 -> scalar loads) -- m/n/k travel via a uniform buffer,
    // not baked into the WGSL text, so the pipeline cache should still hit.
    // (Shaders ARE specialized by kernel/dtype/alignment -- planGemm's unit
    // test in gemm.test.ts pins which shape changes switch variants.)
    const a2 = new Float32Array(${JSON.stringify(Array.from(a2))});
    const b2 = new Float32Array(${JSON.stringify(Array.from(b2))});
    await runGemmWGSL(device, a2, b2, 6, 7, 9);

    return { shaderModuleCalls, pipelineCalls, cacheSizeAfter: pipelineCacheSize(device) };
    `,
    bundle,
  );
  assert.equal(result.shaderModuleCalls, 1, "createShaderModule should only run once across two calls");
  assert.equal(result.pipelineCalls, 1, "createComputePipeline should only run once across two calls");
  assert.equal(result.cacheSizeAfter, 1, "pipeline cache should hold exactly one entry for the one tiled variant both shapes use");
});

test("runGemmWGSL: a second call with the SAME shape reuses pooled buffers instead of allocating new ones", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const a1 = randomMatrix(4 * 3, 5);
  const b1 = randomMatrix(3 * 5, 6);
  const a2 = randomMatrix(4 * 3, 7);
  const b2 = randomMatrix(3 * 5, 8);
  const bundle = bundleForBrowser([path.join(SRC, "gemm.ts")]);
  const result = await harness.run<{ createBufferCallsAfterFirst: number; createBufferCallsAfterSecond: number }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    let createBufferCalls = 0;
    const origCreateBuffer = device.createBuffer.bind(device);
    device.createBuffer = (desc) => { createBufferCalls++; return origCreateBuffer(desc); };

    const a1 = new Float32Array(${JSON.stringify(Array.from(a1))});
    const b1 = new Float32Array(${JSON.stringify(Array.from(b1))});
    await runGemmWGSL(device, a1, b1, 4, 3, 5);
    const createBufferCallsAfterFirst = createBufferCalls;

    // SAME shape as the first call -- every buffer runGemmWGSL needs
    // (A, B, out, dims uniform) was released back to the pool at the end of
    // the first call, so this second call of the identical shape should
    // pull all four out of the pool rather than calling createBuffer again.
    const a2 = new Float32Array(${JSON.stringify(Array.from(a2))});
    const b2 = new Float32Array(${JSON.stringify(Array.from(b2))});
    await runGemmWGSL(device, a2, b2, 4, 3, 5);
    const createBufferCallsAfterSecond = createBufferCalls;

    return { createBufferCallsAfterFirst, createBufferCallsAfterSecond };
    `,
    bundle,
  );
  assert.ok(result.createBufferCallsAfterFirst > 0, "the first call should allocate real buffers");
  // Each call also creates exactly one MAP_READ staging buffer for its final
  // readback (readBackFloat32, gpu-runtime.ts) -- deliberately NOT pooled
  // (see that function's doc comment), so a same-shape second call should
  // allocate exactly ONE new buffer (the staging buffer), not the four
  // storage/uniform buffers runGemmWGSL itself needs -- those come out of
  // the pool this time.
  assert.equal(
    result.createBufferCallsAfterSecond - result.createBufferCallsAfterFirst,
    1,
    "a same-shape second call should allocate exactly one new buffer (the unpooled MAP_READ staging buffer) -- A/B/out/dims should all be reused from the pool",
  );
});

test("attention chain (QK^T -> softmax -> weighted-sum) stays GPU-resident: no MAP_READ staging buffer is created until the caller explicitly reads a result back", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const batch = 1;
  const seqQ = 4;
  const seqK = 4;
  const dim = 4;
  const q = randomMatrix(batch * seqQ * dim, 66);
  const k = randomMatrix(batch * seqK * dim, 77);
  const v = randomMatrix(batch * seqK * dim, 88);
  const bundle = bundleForBrowser([path.join(SRC, "attention.ts")]);
  const result = await harness.run<{ mapReadBuffersDuringChain: number; mapReadBuffersAfterReadback: number }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    let mapReadBuffers = 0;
    const origCreateBuffer = device.createBuffer.bind(device);
    device.createBuffer = (desc) => {
      // GPUBufferUsage.MAP_READ === 0x0001
      if ((desc.usage & GPUBufferUsage.MAP_READ) !== 0) mapReadBuffers++;
      return origCreateBuffer(desc);
    };

    const q = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(q))}), [${batch}, ${seqQ}, ${dim}]);
    const k = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(k))}), [${batch}, ${seqK}, ${dim}]);
    const v = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(v))}), [${batch}, ${seqK}, ${dim}]);

    const scores = await runQKT(device, q, k, ${batch}, ${seqQ}, ${seqK}, ${dim});
    const weights = await runSoftmax(device, scores, ${batch * seqQ}, ${seqK});
    const outT = await runWeightedSum(device, weights, v, ${batch}, ${seqQ}, ${seqK}, ${dim});
    const mapReadBuffersDuringChain = mapReadBuffers;

    // Only NOW read the final result back to the CPU.
    await outT.toFloat32Array();
    const mapReadBuffersAfterReadback = mapReadBuffers;

    q.free(); k.free(); v.free(); scores.free(); weights.free(); outT.free();
    return { mapReadBuffersDuringChain, mapReadBuffersAfterReadback };
    `,
    bundle,
  );
  assert.equal(
    result.mapReadBuffersDuringChain,
    0,
    "no CPU staging/readback buffer should be created while chaining GPU-resident attention ops",
  );
  assert.equal(
    result.mapReadBuffersAfterReadback,
    1,
    "exactly one staging buffer should appear once the caller explicitly reads the final result back",
  );
});

test("runQKT/runSoftmax/runWeightedSum never call GPUTensor.fromFloat32Array internally (inputs/intermediates are never re-uploaded from a host copy)", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const batch = 1;
  const seqQ = 3;
  const seqK = 3;
  const dim = 2;
  const q = randomMatrix(batch * seqQ * dim, 1);
  const k = randomMatrix(batch * seqK * dim, 2);
  const v = randomMatrix(batch * seqK * dim, 3);
  const bundle = bundleForBrowser([path.join(SRC, "attention.ts")]);
  const result = await harness.run<number>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    let fromFloat32ArrayCalls = 0;
    const orig = GPUTensor.fromFloat32Array;
    GPUTensor.fromFloat32Array = (...args) => { fromFloat32ArrayCalls++; return orig(...args); };

    const q = orig(device, new Float32Array(${JSON.stringify(Array.from(q))}), [${batch}, ${seqQ}, ${dim}]);
    const k = orig(device, new Float32Array(${JSON.stringify(Array.from(k))}), [${batch}, ${seqK}, ${dim}]);
    const v = orig(device, new Float32Array(${JSON.stringify(Array.from(v))}), [${batch}, ${seqK}, ${dim}]);
    fromFloat32ArrayCalls = 0; // only count calls made DURING the chain below

    const scores = await runQKT(device, q, k, ${batch}, ${seqQ}, ${seqK}, ${dim});
    const weights = await runSoftmax(device, scores, ${batch * seqQ}, ${seqK});
    const outT = await runWeightedSum(device, weights, v, ${batch}, ${seqQ}, ${seqK}, ${dim});

    q.free(); k.free(); v.free(); scores.free(); weights.free(); outT.free();
    return fromFloat32ArrayCalls;
    `,
    bundle,
  );
  assert.equal(result, 0, "the attention primitives should never re-upload a GPU-resident intermediate from a host array");
});

test("runGemm stays GPU-resident: chaining two GEMMs creates no MAP_READ staging buffer and no host re-upload until the caller reads back", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const a = randomMatrix(8 * 12, 21);
  const b = randomMatrix(12 * 16, 22);
  const c = randomMatrix(16 * 4, 23);
  const bundle = bundleForBrowser([path.join(SRC, "gemm.ts")]);
  const result = await harness.run<{ mapReadDuringChain: number; uploadsDuringChain: number; mapReadAfterReadback: number }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const A = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(a))}), [8, 12]);
    const B = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(b))}), [12, 16]);
    const C = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(c))}), [16, 4]);

    let mapRead = 0;
    let uploads = 0;
    const origCreateBuffer = device.createBuffer.bind(device);
    device.createBuffer = (desc) => { if ((desc.usage & GPUBufferUsage.MAP_READ) !== 0) mapRead++; return origCreateBuffer(desc); };
    const origFrom = GPUTensor.fromFloat32Array;
    GPUTensor.fromFloat32Array = (...args) => { uploads++; return origFrom(...args); };

    const AB = await runGemm(device, A, B);
    const ABC = await runGemm(device, AB, C);
    const mapReadDuringChain = mapRead;
    const uploadsDuringChain = uploads;
    await ABC.toFloat32Array();
    const mapReadAfterReadback = mapRead;
    GPUTensor.fromFloat32Array = origFrom;
    for (const x of [A, B, C, AB, ABC]) x.free();
    return { mapReadDuringChain, uploadsDuringChain, mapReadAfterReadback };
    `,
    bundle,
  );
  assert.equal(result.mapReadDuringChain, 0, "no staging/readback buffer while chaining GPU-resident GEMMs");
  assert.equal(result.uploadsDuringChain, 0, "the intermediate must not round-trip through the host");
  assert.equal(result.mapReadAfterReadback, 1, "exactly one staging buffer once the caller reads the final result");
});
