/**
 * The @johnhenry/tensor-backend conformance suite (49 MLX/NumPy-generated op
 * cases, f32 + f16) run against the Backend behind `MlxDevice.backend` —
 * the contract gate RFC 0001 proposes every math-plus device package runs.
 * Skips (never fails) off darwin/arm64 or without libmlxc.
 */
import { loadOpCases, runConformance, type TestApi } from "@johnhenry/tensor-backend/conformance";
import { createMlxDevice } from "../src/index.ts";
import { mlxSkip, testFns } from "./helpers.ts";

// Each file imports bun:test itself (see testFns in helpers.ts).
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { describe, it, itUnless } = testFns(bunTest);

if (mlxSkip) {
  describe("tensor-backend conformance (tensor-mlx)", () => itUnless(mlxSkip, "conformance", () => {}));
} else {
  const api = { describe, it: it as unknown as TestApi["it"] };
  describe("device: gpu", () => runConformance(() => createMlxDevice().backend, loadOpCases(), api));
  describe("device: cpu", () => runConformance(() => createMlxDevice({ device: "cpu" }).backend, loadOpCases(), api));
}
