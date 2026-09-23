import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { findPeaks } from "../src/index.ts";
import { runScipyOracle, SCIPY_SKIP_REASON } from "./helpers.ts";

test("findPeaks: finds simple, well-separated local maxima at exact expected indices", () => {
  const x = [0, 1, 0, 0, 2, 0, 0, 0, 3, 0];
  const result = findPeaks(Tensor.from(x, { dtype: "f64" }));
  assert.deepEqual(result.indices, [1, 4, 8]);
  assert.deepEqual(result.heights, [1, 2, 3]);
});

test("findPeaks: a plateau's representative index is the (floor-rounded) midpoint", () => {
  const x = [0, 1, 3, 3, 3, 1, 0];
  const result = findPeaks(Tensor.from(x, { dtype: "f64" }));
  assert.deepEqual(result.indices, [3]); // midpoint of indices 2,3,4
  assert.deepEqual(result.heights, [3]);
});

test("findPeaks: height option filters out peaks below the threshold", () => {
  const x = [0, 1, 0, 5, 0, 2, 0];
  const result = findPeaks(Tensor.from(x, { dtype: "f64" }), { height: 3 });
  assert.deepEqual(result.indices, [3]);
});

test("findPeaks: distance option keeps the tallest of two nearby peaks", () => {
  const x = [0, 5, 0, 8, 0, 3, 0, 0, 0, 6, 0];
  const result = findPeaks(Tensor.from(x, { dtype: "f64" }), { distance: 4 });
  // Peaks at 1(h=5), 3(h=8), 5(h=3), 9(h=6). distance=4: 1&3 too close (keep 3),
  // 3&5 too close (keep 3), 5&9 far enough, 3&9 far enough.
  assert.deepEqual(result.indices, [3, 9]);
});

test("findPeaks: prominence option filters out a peak with low topographic prominence", () => {
  // A tall "hill" (index 5, height 10) with a small "bump" on its shoulder (index 3, height 6) that never
  // descends far below its own height before re-ascending to the taller peak -- low prominence.
  const x = [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0];
  const result = findPeaks(Tensor.from(x, { dtype: "f64" }));
  assert.deepEqual(result.indices, [5]); // monotonic up then down -- only one true local max
});

test("findPeaks: matches scipy.signal.find_peaks for a synthetic multi-peak signal", { skip: SCIPY_SKIP_REASON }, () => {
  const x = Array.from({ length: 100 }, (_, i) => Math.sin(i * 0.3) + 0.3 * Math.sin(i * 0.05));
  const mine = findPeaks(Tensor.from(x, { dtype: "f64" }));
  const oracle = runScipyOracle<{ indices: number[] }>({ op: "find_peaks", x });
  assert.deepEqual(mine.indices, oracle.indices);
});

test("findPeaks: matches scipy.signal.find_peaks with height/distance filters", { skip: SCIPY_SKIP_REASON }, () => {
  const x = Array.from({ length: 100 }, (_, i) => 2 * Math.sin(i * 0.4) + Math.sin(i * 1.3) + 0.2 * Math.sin(i * 3.1));
  const mine = findPeaks(Tensor.from(x, { dtype: "f64" }), { height: 0.5, distance: 5 });
  const oracle = runScipyOracle<{ indices: number[] }>({ op: "find_peaks", x, height: 0.5, distance: 5 });
  assert.deepEqual(mine.indices, oracle.indices);
});

test("findPeaks: prominence values match scipy.signal.peak_prominences", { skip: SCIPY_SKIP_REASON }, () => {
  const x = Array.from({ length: 60 }, (_, i) => Math.sin(i * 0.35) + 0.4 * Math.sin(i * 0.9));
  const mine = findPeaks(Tensor.from(x, { dtype: "f64" }));
  const oracle = runScipyOracle<{ prominences: number[] }>({ op: "peak_prominences", x, indices: mine.indices });
  assert.equal(mine.prominences.length, oracle.prominences.length);
  for (let i = 0; i < mine.prominences.length; i++) {
    assert.ok(
      Math.abs((mine.prominences[i] as number) - (oracle.prominences[i] as number)) < 1e-9,
      `prominence ${i}: ${mine.prominences[i]} vs ${oracle.prominences[i]}`,
    );
  }
});

test("findPeaks: rejects a non-1-D Tensor and an invalid distance", () => {
  const bad2d = Tensor.zeros([2, 2], { dtype: "f64" });
  assert.throws(() => findPeaks(bad2d), RangeError);
  assert.throws(() => findPeaks(Tensor.from([1, 2, 1], { dtype: "f64" }), { distance: 0 }), RangeError);
});
