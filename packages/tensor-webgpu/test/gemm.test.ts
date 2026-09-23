import assert from "node:assert/strict";
import path from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { bundleForBrowser, closeHarness, getHarness, SRC } from "./helpers.ts";

after(closeHarness);

/** Naive JS reference GEMM — the CPU oracle this test cross-checks the WGSL kernel against. */
function referenceGemm(a: Float32Array, b: Float32Array, m: number, k: number, n: number): Float32Array {
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let p = 0; p < k; p++) acc += a[i * k + p]! * b[p * n + j]!;
      out[i * n + j] = acc;
    }
  }
  return out;
}

function randomMatrix(size: number, seed: number): Float32Array {
  // Small deterministic LCG — no dependency on @johnhenry/math-plus-tensor-core's Rng needed for this.
  let s = seed >>> 0;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

async function runCase(m: number, k: number, n: number): Promise<void> {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    return; // caller's t.skip handles messaging
  }
  const a = randomMatrix(m * k, m * 1000 + k);
  const b = randomMatrix(k * n, k * 1000 + n);
  const expected = referenceGemm(a, b, m, k, n);

  const bundle = bundleForBrowser([path.join(SRC, "gemm.ts")]);
  const result = await harness.run<number[]>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const a = new Float32Array(${JSON.stringify(Array.from(a))});
    const b = new Float32Array(${JSON.stringify(Array.from(b))});
    const out = await runGemmWGSL(device, a, b, ${m}, ${k}, ${n});
    return Array.from(out);
    `,
    bundle,
  );

  assert.equal(result.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs((result[i] as number) - (expected[i] as number));
    // f32 accumulation over k terms — rtol matches this repo's f32 tolerance
    // convention (packages/tensor-core/test/differential.test.ts's TOLERANCES).
    const tol = 1e-3 * Math.max(1, Math.abs(expected[i] as number));
    assert.ok(diff <= tol, `mismatch at ${i}: got ${result[i]}, expected ${expected[i]} (diff ${diff})`);
  }
}

test("runGemmWGSL: matches a CPU reference for a small square matmul", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  await runCase(4, 3, 5);
});

test("runGemmWGSL: matches a CPU reference for a non-square matmul", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  await runCase(17, 33, 9);
});

test("runGemmWGSL: matches a CPU reference for a larger matmul (multiple workgroups in both dimensions)", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  // 128 x 128 x 128 — big enough to exercise multiple workgroups in both
  // dimensions without making the test slow. Correctness only: the actual
  // WASM-vs-WebGPU performance crossover is measured separately (and not
  // found within a practical range on this machine) by
  // scripts/measure-gemm-threshold.ts — see docs/spikes/webgpu-baseline.md —
  // not gated as a per-PR test per the issue's "GPU performance tests gate
  // behind real hardware, not per-PR".
  await runCase(128, 128, 128);
});
