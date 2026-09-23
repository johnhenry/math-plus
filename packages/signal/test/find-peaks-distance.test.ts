/**
 * Regression tests for issue #101's second finding: `filterByDistance` (the
 * `distance`-option peak-selection step inside `findPeaks`) used to rescan
 * the *entire* remaining candidate array for every accepted peak -- O(n^2),
 * measured at n=40,000 candidates -> 4594ms. It's now a doubly-linked-list
 * walk bounded by `distance` (see the doc comment on `filterByDistance` in
 * ../src/find-peaks.ts), amortized O(n) on top of the O(n log n) height
 * sort.
 *
 * `filterByDistance` itself is a private helper (not part of the package's
 * public surface), so these tests drive it indirectly through the public
 * `findPeaks(signal, { distance })` API:
 *  - a differential test against a from-scratch copy of the OLD O(n^2)
 *    reference algorithm, applied to the same (unfiltered) candidate list
 *    `findPeaks` itself would have fed into `filterByDistance`, across many
 *    random signals/distances;
 *  - a timing regression test with tens of thousands of candidate peaks.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { findPeaks } from "../src/index.ts";

/** A faithful copy of the pre-fix O(n^2) `filterByDistance` (issue #101) — the differential-test oracle. */
function bruteForceFilterByDistance(indices: readonly number[], heights: readonly number[], distance: number): boolean[] {
  const n = indices.length;
  const keep = new Array<boolean>(n).fill(true);
  const order = indices.map((_, i) => i).sort((a, b) => (heights[b] as number) - (heights[a] as number));
  for (const i of order) {
    if (!keep[i]) continue;
    for (let j = 0; j < n; j++) {
      if (j === i || !keep[j]) continue;
      if (Math.abs((indices[j] as number) - (indices[i] as number)) < distance) keep[j] = false;
    }
  }
  return keep;
}

function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test("findPeaks distance filtering: matches a from-scratch O(n^2) reference across random signals/distances", () => {
  const rand = mulberry32(1337);
  for (let trial = 0; trial < 40; trial++) {
    const length = 50 + Math.floor(rand() * 400);
    const x = Array.from({ length }, () => rand() * 100);
    const signal = Tensor.from(x, { dtype: "f64" });

    // Every local maximum, unfiltered -- exactly what production `findPeaks`
    // internally hands to `filterByDistance` when a `distance` option is given.
    const unfiltered = findPeaks(signal);
    const distance = 1 + Math.floor(rand() * 30);

    const expectedKeep = bruteForceFilterByDistance(unfiltered.indices, unfiltered.heights, distance);
    const expectedIndices = unfiltered.indices.filter((_, i) => expectedKeep[i]);

    const actual = findPeaks(signal, { distance });
    assert.deepEqual(
      actual.indices,
      expectedIndices,
      `trial ${trial} (length=${length}, distance=${distance}): indices mismatch`,
    );
  }
});

test("findPeaks distance filtering: near-linear on tens of thousands of candidate peaks (was 4594ms at n=40,000 pre-fix)", () => {
  // A sharp zigzag -- x[2k] a random height, x[2k+1] a deep valley -- makes
  // every even sample its own local maximum, giving ~n/2 distinct candidate
  // peaks for `filterByDistance` to process.
  const rand = mulberry32(42);
  const n = 80_000; // ~40,000 candidate peaks
  const x = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    x[i] = i % 2 === 0 ? rand() * 1000 : -1000;
  }
  const signal = Tensor.from(x, { dtype: "f64" });

  const start = performance.now();
  const result = findPeaks(signal, { distance: 50 });
  const elapsedMs = performance.now() - start;

  assert.ok(result.indices.length > 0, "expected at least one surviving peak");
  // Generous threshold (was 4594ms at n=40,000 pre-fix; near-linear now) --
  // sized to avoid flakiness on a loaded CI box while still catching an
  // O(n^2) regression by a wide margin.
  assert.ok(elapsedMs < 1000, `findPeaks with distance took ${elapsedMs}ms for ~${n / 2} candidate peaks, expected well under 1000ms`);
});
