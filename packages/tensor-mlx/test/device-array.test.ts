/**
 * The shared DeviceArray suite (tensor-cpu's test/device-array-suite.ts,
 * the one behavioural suite of the chainable array API every math-plus
 * device shares) over `createMlxDevice()` on Metal: f32/f16/bf16/i32/bool
 * against the NumPy oracle. Skips (never fails) off darwin/arm64, without
 * libmlxc, or without a python3 that imports numpy.
 */
import { createMlxDevice } from "../src/index.ts";
import { deviceArraySuite } from "../../tensor-cpu/test/device-array-suite.ts";
import { mlxSkip, testFns } from "./helpers.ts";
import { makeTest } from "../../../test/harness.ts";

// Each file imports bun:test itself (see test/harness.ts).
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const harness = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);

deviceArraySuite(testFns(harness), {
  label: "tensor-mlx",
  skip: mlxSkip,
  device: mlxSkip ? null : createMlxDevice(),
  other: () => createMlxDevice(),
});
