import assert from "node:assert/strict";
import path from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { bundleForBrowser, closeHarness, getHarness, SRC } from "./helpers.ts";

after(closeHarness);

function randomData(size: number, seed: number): Float32Array {
  let s = seed >>> 0;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

// ---- CPU reference implementations (the oracle each GPU primitive is checked against) ----

function referenceQKT(
  q: Float32Array,
  k: Float32Array,
  batch: number,
  seqQ: number,
  seqK: number,
  dim: number,
): Float32Array {
  const out = new Float32Array(batch * seqQ * seqK);
  for (let b = 0; b < batch; b++) {
    for (let i = 0; i < seqQ; i++) {
      for (let j = 0; j < seqK; j++) {
        let acc = 0;
        for (let d = 0; d < dim; d++) {
          acc += q[(b * seqQ + i) * dim + d]! * k[(b * seqK + j) * dim + d]!;
        }
        out[(b * seqQ + i) * seqK + j] = acc;
      }
    }
  }
  return out;
}

function referenceSoftmax(x: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let max = -Infinity;
    for (let j = 0; j < cols; j++) max = Math.max(max, x[base + j]!);
    let sum = 0;
    for (let j = 0; j < cols; j++) sum += Math.exp(x[base + j]! - max);
    for (let j = 0; j < cols; j++) out[base + j] = Math.exp(x[base + j]! - max) / sum;
  }
  return out;
}

function referenceWeightedSum(
  weights: Float32Array,
  v: Float32Array,
  batch: number,
  seqQ: number,
  seqK: number,
  dim: number,
): Float32Array {
  const out = new Float32Array(batch * seqQ * dim);
  for (let b = 0; b < batch; b++) {
    for (let i = 0; i < seqQ; i++) {
      for (let d = 0; d < dim; d++) {
        let acc = 0;
        for (let j = 0; j < seqK; j++) {
          acc += weights[(b * seqQ + i) * seqK + j]! * v[(b * seqK + j) * dim + d]!;
        }
        out[(b * seqQ + i) * dim + d] = acc;
      }
    }
  }
  return out;
}

function assertClose(actual: readonly number[], expected: Float32Array, label: string): void {
  assert.equal(actual.length, expected.length, `${label}: length mismatch`);
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs((actual[i] as number) - (expected[i] as number));
    const tol = 1e-3 * Math.max(1, Math.abs(expected[i] as number));
    assert.ok(diff <= tol, `${label}: mismatch at ${i}: got ${actual[i]}, expected ${expected[i]} (diff ${diff})`);
  }
}

test("runQKT: matches a CPU reference (batched Q @ K^T)", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const batch = 2;
  const seqQ = 5;
  const seqK = 7;
  const dim = 4;
  const q = randomData(batch * seqQ * dim, 11);
  const k = randomData(batch * seqK * dim, 22);
  const expected = referenceQKT(q, k, batch, seqQ, seqK, dim);
  const bundle = bundleForBrowser([path.join(SRC, "attention.ts")]);
  const result = await harness.run<number[]>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const q = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(q))}), [${batch}, ${seqQ}, ${dim}]);
    const k = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(k))}), [${batch}, ${seqK}, ${dim}]);
    const outT = await runQKT(device, q, k, ${batch}, ${seqQ}, ${seqK}, ${dim});
    const out = await outT.toFloat32Array();
    q.free();
    k.free();
    outT.free();
    return Array.from(out);
    `,
    bundle,
  );
  assertClose(result, expected, "runQKT");
});

test("runSoftmax: matches a CPU reference and each row sums to 1", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const rows = 6;
  const cols = 9;
  // Include large values to exercise the numerically-stable max-subtraction path.
  const x = randomData(rows * cols, 33).map((v) => v * 50);
  const expected = referenceSoftmax(x, rows, cols);
  const bundle = bundleForBrowser([path.join(SRC, "attention.ts")]);
  const result = await harness.run<number[]>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const x = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(x))}), [${rows}, ${cols}]);
    const outT = await runSoftmax(device, x, ${rows}, ${cols});
    const out = await outT.toFloat32Array();
    x.free();
    outT.free();
    return Array.from(out);
    `,
    bundle,
  );
  assertClose(result, expected, "runSoftmax");
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let c = 0; c < cols; c++) sum += result[r * cols + c] as number;
    assert.ok(Math.abs(sum - 1) < 1e-3, `row ${r} should sum to 1, got ${sum}`);
  }
});

test("runWeightedSum: matches a CPU reference (batched weights @ V)", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const batch = 2;
  const seqQ = 4;
  const seqK = 6;
  const dim = 3;
  const weights = randomData(batch * seqQ * seqK, 44);
  const v = randomData(batch * seqK * dim, 55);
  const expected = referenceWeightedSum(weights, v, batch, seqQ, seqK, dim);
  const bundle = bundleForBrowser([path.join(SRC, "attention.ts")]);
  const result = await harness.run<number[]>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const weights = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(weights))}), [${batch}, ${seqQ}, ${seqK}]);
    const v = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(v))}), [${batch}, ${seqK}, ${dim}]);
    const outT = await runWeightedSum(device, weights, v, ${batch}, ${seqQ}, ${seqK}, ${dim});
    const out = await outT.toFloat32Array();
    weights.free();
    v.free();
    outT.free();
    return Array.from(out);
    `,
    bundle,
  );
  assertClose(result, expected, "runWeightedSum");
});

test("attention primitives compose: QK^T -> softmax -> weighted-sum matches a CPU-only reference SDPA (unscaled)", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const batch = 1;
  const seqQ = 4;
  const seqK = 4;
  const dim = 4;
  const q = randomData(batch * seqQ * dim, 66);
  const k = randomData(batch * seqK * dim, 77);
  const v = randomData(batch * seqK * dim, 88);

  const scores = referenceQKT(q, k, batch, seqQ, seqK, dim);
  const weights = referenceSoftmax(scores, batch * seqQ, seqK);
  const expected = referenceWeightedSum(weights, v, batch, seqQ, seqK, dim);

  const bundle = bundleForBrowser([path.join(SRC, "attention.ts")]);
  const result = await harness.run<number[]>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const q = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(q))}), [${batch}, ${seqQ}, ${dim}]);
    const k = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(k))}), [${batch}, ${seqK}, ${dim}]);
    const v = GPUTensor.fromFloat32Array(device, new Float32Array(${JSON.stringify(Array.from(v))}), [${batch}, ${seqK}, ${dim}]);
    // Stay GPU-resident end-to-end (issue #100): scores/weights are never
    // read back to the CPU here, only the final weighted-sum output is.
    const scores = await runQKT(device, q, k, ${batch}, ${seqQ}, ${seqK}, ${dim});
    const weights = await runSoftmax(device, scores, ${batch * seqQ}, ${seqK});
    const outT = await runWeightedSum(device, weights, v, ${batch}, ${seqQ}, ${seqK}, ${dim});
    const out = await outT.toFloat32Array();
    q.free();
    k.free();
    v.free();
    scores.free();
    weights.free();
    outT.free();
    return Array.from(out);
    `,
    bundle,
  );
  assertClose(result, expected, "composed SDPA");
});
