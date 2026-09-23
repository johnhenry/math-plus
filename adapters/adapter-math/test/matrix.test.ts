import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { Vector, VectorUtils } from "@johnhenry/math";
import { fromMatrix, fromVector, toMatrix, toVector } from "../src/index.ts";

test("fromMatrix accepts a plain number[][] and produces a 2-D f32 Tensor", () => {
  const t = fromMatrix([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  assert.deepEqual([...t.shape], [2, 3]);
  assert.equal(t.dtype, "f32");
  assert.deepEqual(t.toArray(), [
    [1, 2, 3],
    [4, 5, 6],
  ]);
});

test("fromMatrix accepts @johnhenry/math's Matrix<number> (Vector<Vector<number>>)", () => {
  const rows = Vector.fromArray([Vector.fromArray([1, 2]), Vector.fromArray([3, 4])]);
  const t = fromMatrix(rows);
  assert.deepEqual([...t.shape], [2, 2]);
  assert.deepEqual(t.toArray(), [
    [1, 2],
    [3, 4],
  ]);
});

test("fromMatrix throws on a ragged (non-rectangular) input", () => {
  assert.throws(() => fromMatrix([[1, 2], [3]]), RangeError);
});

test("fromMatrix respects an explicit dtype", () => {
  const t = fromMatrix([[1, 2]], { dtype: "f64" });
  assert.equal(t.dtype, "f64");
});

test("fromVector accepts a plain number[] and a @johnhenry/math Vector<number>", () => {
  const t1 = fromVector([1, 2, 3]);
  assert.deepEqual([...t1.shape], [3]);
  assert.deepEqual(t1.toArray(), [1, 2, 3]);

  const t2 = fromVector(Vector.fromArray([4, 5, 6]));
  assert.deepEqual(t2.toArray(), [4, 5, 6]);
});

test("toMatrix inverts fromMatrix, including through a non-contiguous (transposed) view", () => {
  const original = [
    [1, 2, 3],
    [4, 5, 6],
  ];
  const t = fromMatrix(original);
  assert.deepEqual(toMatrix(t), original);

  const transposed = t.transpose();
  assert.deepEqual(toMatrix(transposed), [
    [1, 4],
    [2, 5],
    [3, 6],
  ]);
});

test("toVector inverts fromVector", () => {
  const t = fromVector([7, 8, 9]);
  assert.deepEqual(toVector(t), [7, 8, 9]);
});

test("toMatrix throws on a non-2-D tensor (1-D and 3-D)", () => {
  assert.throws(() => toMatrix(Tensor.from([1, 2, 3])), RangeError);
  assert.throws(() => toMatrix(Tensor.zeros([2, 2, 2])), RangeError);
});

test("toVector throws on a non-1-D tensor", () => {
  assert.throws(() => toVector(Tensor.zeros([2, 2])), RangeError);
});

test("toMatrix/toVector throw a clear error on bigint dtypes rather than silently truncating", () => {
  const t = Tensor.zeros([2, 2], { dtype: "i64" });
  assert.throws(() => toMatrix(t), TypeError);
  const v = Tensor.zeros([2], { dtype: "i64" });
  assert.throws(() => toVector(v), TypeError);
});

test("round-trips through @johnhenry/math's own matrix machinery (VectorUtils.transpose) via toMatrix/fromMatrix", () => {
  const original = VectorUtils.constantMatrix(2, 3, 0);
  original[0][0] = 1;
  original[0][1] = 2;
  original[1][2] = 5;

  const t = fromMatrix(original);
  const mathTransposed = VectorUtils.transpose(original);
  const tensorTransposed = toMatrix(t.transpose());
  assert.deepEqual(tensorTransposed, [...mathTransposed].map((row) => [...row]));
});
