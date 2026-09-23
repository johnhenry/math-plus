/**
 * Node-side coverage for the native FFI module (issue #55 Phase 2): under
 * Node there is no `Deno` global, so the availability contract's "graceful
 * undefined, never a throw" half is what CAN be tested here. The
 * correctness half (native results vs wasm on real FFI calls) runs under
 * Deno via scripts/deno-native-test.ts -- see package.json's test:deno.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { matrix, NativeKernels } from "../src/native.ts";

test("NativeKernels.load() returns undefined (never throws) outside Deno", () => {
  assert.equal(NativeKernels.load(), undefined);
  assert.equal(NativeKernels.load({ libraryPath: "/nonexistent.so" }), undefined);
});

test("matrix() validates dimensions without touching FFI", () => {
  const m = matrix(new Float32Array(6), 2, 3);
  assert.deepEqual([m.rows, m.cols, m.rowStride, m.colStride], [2, 3, 3, 1]);
  assert.throws(() => matrix(new Float32Array(5), 2, 3), RangeError);
});
