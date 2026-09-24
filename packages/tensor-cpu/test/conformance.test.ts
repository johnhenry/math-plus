/**
 * The @johnhenry/tensor-backend conformance suite (MLX/NumPy-generated core
 * op cases, the "general numerics" cases and — since tensor-backend 0.3 —
 * the quantized-weight cases, f32; the f16/bf16 passes are
 * no-ops because this backend reports `supports(...) === false` for them)
 * run against `createCpuBackend()` — twice: once with every optional op
 * native, once with the numerics ops hidden so the `compose.ts` default
 * compositions are checked on this backend too (`cumsum` stays native: it
 * has no composition). This backend has no native quantized ops, so the
 * quantized cases run through compose.ts's dequantizing fallback. Pure
 * TypeScript: nothing to skip.
 */
import assert from "node:assert/strict";
import { loadOpCases, runConformance, withoutOptionalOps, type TestApi } from "@johnhenry/tensor-backend/conformance";
import { hasNativeQuantized, NUMERICS_OPS } from "@johnhenry/tensor-backend";
import { createCpuBackend } from "../src/index.ts";
import { testFns } from "./helpers.ts";
import { makeTest } from "../../../test/harness.ts";

// Each file imports bun:test itself (see test/harness.ts).
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const harness = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
const { describe, it } = testFns(harness);

const api = { describe, it: it as unknown as TestApi["it"] };
describe("native ops", () => runConformance(() => createCpuBackend(), loadOpCases(), api));
describe("default compositions (numerics ops hidden)", () =>
  runConformance(() => withoutOptionalOps(createCpuBackend(), NUMERICS_OPS.filter((o) => o !== "cumsum")), loadOpCases(), api),
);

it("runs tensor-backend 0.3's quantized cases, through the compose fallback (no native quantized ops here)", async () => {
  const cases = await loadOpCases();
  const quantized = cases.filter((c) => c.op === "quantizedLinear" || c.op === "quantizedEmbedding");
  assert.ok(quantized.length >= 26, `expected tensor-backend 0.3's ops-quantized.json cases, got ${quantized.length}`);
  assert.equal(hasNativeQuantized(createCpuBackend()), false);
});
