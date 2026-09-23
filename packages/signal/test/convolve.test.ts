import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { convolve, convolve1D } from "../src/index.ts";

function close(a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-9): void {
  assert.equal(a.length, b.length, `length ${a.length} vs ${b.length}`);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs((a[i] as number) - (b[i] as number)) < tol, `index ${i}: ${a[i]} vs ${b[i]}`);
}

test("convolve1D full: matches a hand-computed example", () => {
  // [1,2,3] * [0,1,0.5] -- classic textbook example.
  const out = convolve1D(new Float64Array([1, 2, 3]), new Float64Array([0, 1, 0.5]));
  close(out, [0, 1, 2.5, 4, 1.5]);
});

test("convolve1D: full/same/valid mode lengths and content match NumPy's np.convolve convention", () => {
  const a = new Float64Array([1, 2, 3, 4]);
  const b = new Float64Array([1, 1, 1]);
  const full = convolve1D(a, b, "full");
  close(full, [1, 3, 6, 9, 7, 4]); // length 4+3-1=6
  const same = convolve1D(a, b, "same");
  close(same, [3, 6, 9, 7]); // length max(4,3)=4, centered
  const valid = convolve1D(a, b, "valid");
  close(valid, [6, 9]); // length max-min+1 = 4-3+1=2
});

test("convolve: 1-D Tensor delegates correctly", () => {
  const a = Tensor.from([1, 2, 3], { dtype: "f64" });
  const b = Tensor.from([0, 1, 0.5], { dtype: "f64" });
  const out = convolve(a, b);
  close(out.toArray() as number[], [0, 1, 2.5, 4, 1.5]);
});

test("convolve: batched 2-D [N,T] input, axis=1 (default), each row convolved independently", () => {
  const input = Tensor.from([1, 2, 3, 4, 10, 20, 30, 40], { dtype: "f64" }).reshape([2, 4]);
  const kernel = Tensor.from([1, 1], { dtype: "f64" });
  const out = convolve(input, kernel, { mode: "valid" });
  assert.deepEqual([...out.shape], [2, 3]);
  const rows = out.toArray() as number[][];
  close(rows[0] as number[], [3, 5, 7]); // [1+2,2+3,3+4]
  close(rows[1] as number[], [30, 50, 70]);
});

test("convolve: batched 2-D input, axis=0 (columns are the time axis)", () => {
  // Same data as above but transposed: [T, N] with axis=0 as time.
  const input = Tensor.from([1, 10, 2, 20, 3, 30, 4, 40], { dtype: "f64" }).reshape([4, 2]);
  const kernel = Tensor.from([1, 1], { dtype: "f64" });
  const out = convolve(input, kernel, { mode: "valid", axis: 0 });
  assert.deepEqual([...out.shape], [3, 2]);
  const rows = out.toArray() as number[][];
  close(rows.map((r) => r[0] as number), [3, 5, 7]);
  close(rows.map((r) => r[1] as number), [30, 50, 70]);
});

test("convolve: rejects a non-1-D kernel and a >2-D input", () => {
  const input = Tensor.from([1, 2, 3], { dtype: "f64" });
  const badKernel = Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2]);
  assert.throws(() => convolve(input, badKernel), RangeError);

  const kernel = Tensor.from([1], { dtype: "f64" });
  const bad3d = Tensor.zeros([2, 2, 2], { dtype: "f64" });
  assert.throws(() => convolve(bad3d, kernel), RangeError);
});
