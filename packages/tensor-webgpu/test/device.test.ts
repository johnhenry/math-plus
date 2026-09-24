import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { chooseGemmBackend, GEMM_ELEMENT_THRESHOLD, GEMM_WORK_THRESHOLD } from "../src/threshold.ts";
import { closeHarness, getHarness } from "./helpers.ts";

after(closeHarness);

// ---- pure logic (no GPU needed) --------------------------------------------

test("chooseGemmBackend: small matmuls stay on wasm", () => {
  assert.equal(chooseGemmBackend(8, 8, 8), "wasm");
  assert.equal(chooseGemmBackend(128, 128, 128), "wasm", "won by WebGPU before the SIMD WASM GEMM (#130); not any more");
  assert.equal(chooseGemmBackend(160, 160, 160), "wasm", "Chrome still loses here (0.87x); only Dawn wins");
  assert.equal(chooseGemmBackend(191, 191), "wasm");
});

test("chooseGemmBackend: pins the measured m·n >= 192² AND m·n·k >= 2^22 rule (docs/spikes/webgpu-tiled-gemm.md)", () => {
  // Pins the measured values so recalibrating is a deliberate, visible change
  // to this test, not a silent drift. History: Infinity (v1 naive kernel),
  // then m·n >= 128² against the scalar WASM GEMM, now this two-part rule
  // against tensor-wasm's SIMD GEMM (#130), measured with the thermal-aware
  // method on an Apple M2 under Dawn and headless Chrome.
  assert.equal(GEMM_ELEMENT_THRESHOLD, 192 * 192);
  assert.equal(GEMM_WORK_THRESHOLD, 2 ** 22);
  assert.equal(chooseGemmBackend(192, 192, 192), "webgpu");
  assert.equal(chooseGemmBackend(4096, 4096, 4096), "webgpu");
  assert.equal(chooseGemmBackend(16, 3072, 1024), "webgpu", "a Linear x[16,1024]·Wᵀ: m·n = 49152, 50M multiply-adds");
  // k matters: enough output elements but too little work per element stays on WASM (measured 0.27-0.92x)...
  assert.equal(chooseGemmBackend(192, 192, 64), "wasm");
  assert.equal(chooseGemmBackend(192, 192, 16), "wasm");
  // ...and a lot of work on too few output elements does too (Chrome: 128x4096x128 0.81x, 96x4096x96 0.57x).
  assert.equal(chooseGemmBackend(128, 128, 4096), "wasm");
  assert.equal(chooseGemmBackend(1, 3072, 1024), "wasm", "single-token Linear: a Dawn-only win (1.17x), a Chrome loss (0.70x)");
  // k omitted: the m·n test alone (the pre-k signature's behavior, at the new threshold).
  assert.equal(chooseGemmBackend(192, 192), "webgpu");
  assert.equal(chooseGemmBackend(96, 384), "webgpu", "m·n-based: a 96x384 output has 192*192 elements");
});

// ---- real headless WebGPU (skips if unavailable) ---------------------------

test("detectWebGPU: headless Chrome under Xvfb resolves a real GPUAdapter", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available in this environment: ${harness.reason}`);
    return;
  }
  const result = await harness.run<{ hasDevice: boolean; limits: { maxBufferSize: number } }>(`
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    return { hasDevice: !!device, limits: { maxBufferSize: adapter.limits.maxBufferSize } };
  `);
  assert.equal(result.hasDevice, true);
  assert.ok(result.limits.maxBufferSize > 0);
});
