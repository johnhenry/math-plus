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
  assert.equal(chooseGemmBackend(160, 160, 160), "wasm", "a Dawn-only win (1.78x); both browsers lose (0.80x / 0.62x)");
  assert.equal(chooseGemmBackend(192, 192, 192), "wasm", "routed to WebGPU before 0.3.0; both browsers now lose (0.88x / 0.89x)");
  assert.equal(chooseGemmBackend(255, 256), "wasm");
});

test("chooseGemmBackend: pins the measured m·n >= 256² AND m·n·k >= 2^24 rule (docs/spikes/webgpu-tiled-gemm.md)", () => {
  // Pins the measured values so recalibrating is a deliberate, visible change
  // to this test, not a silent drift. History: Infinity (v1 naive kernel),
  // m·n >= 128² against the scalar WASM GEMM, m·n >= 192² AND m·n·k >= 2^22
  // against tensor-wasm's SIMD GEMM (#130), now this rule, measured through
  // the facade (backend-webgpu's matmul/linear) with the thermal-aware
  // method on an Apple M2 under Dawn, headless Chrome and a visible Chromium.
  assert.equal(GEMM_ELEMENT_THRESHOLD, 256 * 256);
  assert.equal(GEMM_WORK_THRESHOLD, 2 ** 24);
  assert.equal(chooseGemmBackend(256, 256, 256), "webgpu", "measured 2.78x / 1.41x / 1.53x (Dawn / headless / visible)");
  assert.equal(chooseGemmBackend(4096, 4096, 4096), "webgpu");
  assert.equal(chooseGemmBackend(64, 3072, 1024), "webgpu", "a Linear x[64,1024]·Wᵀ: 2.86-3.29x");
  // Shapes the previous rule sent to WebGPU that lost in a browser:
  assert.equal(chooseGemmBackend(192, 192, 256), "wasm", "visible Chromium 0.95x");
  assert.equal(chooseGemmBackend(192, 192, 1024), "wasm", "visible Chromium 0.96x");
  assert.equal(chooseGemmBackend(16, 3072, 1024), "wasm", "16-row Linear: visible Chromium 0.96x");
  // Wins in all three environments that the rule still leaves on WASM (conservative, documented):
  assert.equal(chooseGemmBackend(128, 128, 1024), "wasm");
  assert.equal(chooseGemmBackend(192, 192, 4096), "wasm");
  // k matters: enough output elements but too little work stays on WASM.
  assert.equal(chooseGemmBackend(256, 256, 64), "wasm", "4.2 M multiply-adds < 2^24");
  assert.equal(chooseGemmBackend(1, 3072, 1024), "wasm", "single-token Linear: loses everywhere (0.46-0.55x)");
  // k omitted: the m·n test alone (the pre-k signature's behavior, at the new threshold).
  assert.equal(chooseGemmBackend(256, 256), "webgpu");
  assert.equal(chooseGemmBackend(128, 512), "webgpu", "m·n-based: a 128x512 output has 256*256 elements");
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
