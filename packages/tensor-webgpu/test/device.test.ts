import assert from "node:assert/strict";
import test, { after } from "node:test";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { toWebGPU, GPUTensor } from "../src/device.ts";
import { chooseGemmBackend, GEMM_ELEMENT_THRESHOLD } from "../src/threshold.ts";
import { closeHarness, getHarness } from "./helpers.ts";

after(closeHarness);

// ---- pure logic (no GPU needed) --------------------------------------------

test("chooseGemmBackend: small matmuls stay on wasm", () => {
  assert.equal(chooseGemmBackend(8, 8), "wasm");
  assert.equal(chooseGemmBackend(100, 100), "wasm");
  assert.equal(chooseGemmBackend(127, 127), "wasm");
});

test("chooseGemmBackend: crosses to webgpu at the measured 128x128 crossover (docs/spikes/webgpu-tiled-gemm.md)", () => {
  // Pins the measured value so recalibrating is a deliberate, visible change
  // to this test, not a silent drift. v1's naive kernel measured no
  // crossover at all (Infinity, docs/spikes/webgpu-baseline.md); the tiled /
  // skinny / subgroup-matrix kernels cross at n = 96 (Dawn) / n = 128
  // (headless Chrome) end to end on an Apple M2 — the constant takes the
  // more conservative of the two.
  assert.equal(GEMM_ELEMENT_THRESHOLD, 128 * 128);
  assert.equal(chooseGemmBackend(128, 128), "webgpu");
  assert.equal(chooseGemmBackend(64, 256), "webgpu", "m*n-based: a 64x256 output has 128*128 elements");
  assert.equal(chooseGemmBackend(4096, 4096), "webgpu");
});

test("toWebGPU: rejects dtypes other than f32/f16 without needing a real device", async () => {
  const t = Tensor.zeros([4], { dtype: "f64" });
  await assert.rejects(
    () => toWebGPU(t, undefined as unknown as GPUDevice),
    /f32 and f16 only/,
  );
});

test("toWebGPU: rejects a non-contiguous view without needing a real device", async () => {
  const t = Tensor.zeros([4, 4], { dtype: "f32" }).transpose();
  await assert.rejects(
    () => toWebGPU(t, undefined as unknown as GPUDevice),
    /contiguous/,
  );
});

test("GPUTensor.fromFloat32Array: rejects a shape/data length mismatch without needing a real device", () => {
  assert.throws(
    () => GPUTensor.fromFloat32Array(undefined as unknown as GPUDevice, new Float32Array(3), [2, 2]),
    /does not match data length/,
  );
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
