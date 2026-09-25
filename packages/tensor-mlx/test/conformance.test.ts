/**
 * The @johnhenry/tensor-backend conformance suite (MLX/NumPy-generated op
 * cases: the core ops plus the "general numerics" ops, f32 + f16 + bf16) run
 * against the Backend behind `MlxDevice.backend` — the contract gate RFC 0001
 * proposes every math-plus device package runs. Optional ops go through
 * tensor-backend's compose helpers, the same path `MlxArray` uses.
 * Skips (never fails) off darwin/arm64 or without libmlxc.
 */
import { loadOpCases, runConformance, type OpCase, type TestApi } from "@johnhenry/tensor-backend/conformance";
import { createMlxDevice } from "../src/index.ts";
import { mlxSkip, testFns } from "./helpers.ts";
import { makeTest } from "../../../test/harness.ts";

// Each file imports bun:test itself (see test/harness.ts).
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const harness = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
const { describe, it, itUnless } = testFns(harness);

/**
 * f64 is CPU-only on MLX (no Apple GPU has double-precision hardware),
 * so f64 cases are real, permanent failures on the GPU device, not a gap
 * -- excluded here, covered instead by the `device: cpu` run below, which
 * already includes them (no filtering there). Mirrors backend-mlx's own
 * conformance.test.ts.
 */
const isF64Case = (c: OpCase): boolean => c.outputs.some((o) => o.dtype === "f64") || c.inputs.some((i) => i?.dtype === "f64");
const nonF64Cases = () => loadOpCases().then((cases) => cases.filter((c) => !isF64Case(c)));

if (mlxSkip) {
  describe("tensor-backend conformance (tensor-mlx)", () => itUnless(mlxSkip, "conformance", () => {}));
} else {
  const api = { describe, it: it as unknown as TestApi["it"] };
  describe("device: gpu", () => runConformance(() => createMlxDevice().backend, nonF64Cases(), api));
  describe("device: cpu", () => runConformance(() => createMlxDevice({ device: "cpu" }).backend, loadOpCases(), api));
}
