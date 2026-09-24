/**
 * The @johnhenry/tensor-backend conformance suite (MLX/NumPy-generated core
 * op cases plus the "general numerics" cases; f32, and f16/bf16 where the
 * device supports them) run against the Backend behind
 * `createWebGpuDevice().backend` — the contract gate RFC 0001 asks every
 * math-plus device package to run (issue #146). Runs in-process on Dawn
 * (Node and Bun); skips, never fails, where no WebGPU adapter exists.
 */
import { loadOpCases, runConformance, type TestApi } from "@johnhenry/tensor-backend/conformance";
import { createWebGpuDevice, webGpuUnavailableReason } from "../src/index.ts";
import { testFns } from "../../tensor-cpu/test/helpers.ts";
import { makeTest } from "../../../test/harness.ts";

// Each file imports bun:test itself (see test/harness.ts).
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const harness = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
const { describe, it, itUnless } = testFns(harness);

const skip = await webGpuUnavailableReason();
if (skip) {
  describe("tensor-backend conformance (tensor-webgpu)", () => itUnless(skip, "conformance", () => {}));
} else {
  const api = { describe, it: it as unknown as TestApi["it"] };
  describe("tensor-webgpu facade", () => runConformance(async () => (await createWebGpuDevice()).backend, loadOpCases(), api));
}
