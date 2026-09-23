/**
 * correlate2D (issue #84) — true 2-D cross-correlation via FFT, upstream
 * for the generalized Wang tile laboratory's autocorrelation surface.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { correlate2D } from "../src/index.ts";
import { runScipyOracle, SCIPY_SKIP_REASON } from "./helpers.ts";

function randomMatrix(rows: number, cols: number, seedStart: number): number[][] {
  let seed = seedStart;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed / 0x7fffffff) * 4 - 2;
  };
  return Array.from({ length: rows }, () => Array.from({ length: cols }, rng));
}

function toTensor(matrix: number[][]): Tensor {
  return Tensor.from(matrix.flat(), { dtype: "f64" }).reshape([matrix.length, (matrix[0] as number[]).length]);
}

test("correlate2D matches scipy.signal.correlate2d(mode='full') on a random 4x5 vs 3x3 input", { skip: SCIPY_SKIP_REASON }, () => {
  const a = randomMatrix(4, 5, 42);
  const b = randomMatrix(3, 3, 917);
  const got = correlate2D(toTensor(a), toTensor(b)).toArray() as number[][];
  const { y } = runScipyOracle<{ y: number[][] }>({ op: "correlate2d", a, b });
  assert.equal(got.length, y.length, "row count");
  for (let i = 0; i < got.length; i++) {
    for (let j = 0; j < (got[i] as number[]).length; j++) {
      const gv = (got[i] as number[])[j] as number;
      const wv = (y[i] as number[])[j] as number;
      assert.ok(Math.abs(gv - wv) < 1e-9, `[${i}][${j}] ${gv} vs ${wv}`);
    }
  }
});

test("correlate2D matches scipy.signal.correlate2d for a non-square 6x2 vs 2x4 input (asymmetric on both axes)", { skip: SCIPY_SKIP_REASON }, () => {
  const a = randomMatrix(6, 2, 314);
  const b = randomMatrix(2, 4, 271);
  const got = correlate2D(toTensor(a), toTensor(b)).toArray() as number[][];
  const { y } = runScipyOracle<{ y: number[][] }>({ op: "correlate2d", a, b });
  for (let i = 0; i < got.length; i++) {
    for (let j = 0; j < (got[i] as number[]).length; j++) {
      assert.ok(Math.abs((got[i] as number[])[j]! - (y[i] as number[])[j]!) < 1e-9, `[${i}][${j}]`);
    }
  }
});

test("correlate2D output shape is [Ma+Mb-1, Na+Nb-1], the 'full' mode convention", () => {
  const a = toTensor(randomMatrix(4, 5, 1));
  const b = toTensor(randomMatrix(3, 3, 2));
  const result = correlate2D(a, b);
  assert.deepEqual([...result.shape], [6, 7]);
});

test("correlate2D autocorrelation: a field correlated with itself peaks at the zero-shift position (center of 'full' output)", () => {
  // A field with one dominant bright cell -- its autocorrelation should
  // peak at the shift that aligns the field with itself (zero shift),
  // which for an NxN 'full' autocorrelation lands at output index
  // [N-1, N-1] (the center of a (2N-1)x(2N-1) output).
  const field = [
    [0, 0, 0, 0],
    [0, 5, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  const a = toTensor(field);
  const auto = correlate2D(a, a).toArray() as number[][];
  assert.equal(auto.length, 7);
  let maxVal = -Infinity;
  let maxPos: [number, number] = [0, 0];
  for (let i = 0; i < auto.length; i++) {
    for (let j = 0; j < (auto[i] as number[]).length; j++) {
      const v = (auto[i] as number[])[j] as number;
      if (v > maxVal) {
        maxVal = v;
        maxPos = [i, j];
      }
    }
  }
  assert.deepEqual(maxPos, [3, 3], "peak at the zero-shift (center) position");
  assert.ok(Math.abs(maxVal - 25) < 1e-9, "peak value is the single bright cell squared (5*5)");
});

test("correlate2D rejects non-2-D input", () => {
  const oneD = Tensor.from([1, 2, 3], { dtype: "f64" });
  const twoD = toTensor([[1, 2]]);
  assert.throws(() => correlate2D(oneD, twoD), RangeError);
  assert.throws(() => correlate2D(twoD, oneD), RangeError);
});
