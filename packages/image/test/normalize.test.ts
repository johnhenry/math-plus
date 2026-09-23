import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { normalize } from "../src/index.ts";

test("normalize: known input/mean/std produces exact expected output", () => {
  // 1x1 pixel, 2 channels: [10, 20]. mean=[5,10], std=[2,5] -> [(10-5)/2, (20-10)/5] = [2.5, 2]
  const input = Tensor.from([10, 20], { dtype: "f64" }).reshape([1, 1, 2]);
  const out = normalize(input, { mean: [5, 10], std: [2, 5] });
  const actual = out.toArray() as number[][][];
  assert.ok(Math.abs((actual[0]?.[0]?.[0] as number) - 2.5) < 1e-9);
  assert.ok(Math.abs((actual[0]?.[0]?.[1] as number) - 2) < 1e-9);
});

test("normalize: per-channel mean/std applies to the correct channel, not mixed up", () => {
  // 1x2 image, 3 channels (RGB-like), same value repeated per pixel: [1,2,3] for both pixels.
  const input = Tensor.from([1, 2, 3, 1, 2, 3], { dtype: "f64" }).reshape([1, 2, 3]);
  const out = normalize(input, { mean: [1, 2, 3], std: [1, 1, 1] });
  const actual = out.toArray() as number[][][];
  // (x - mean[c]) / std[c] with x == mean[c] for every channel -> all zeros
  for (const pixel of actual[0] as number[][]) {
    for (const v of pixel) assert.ok(Math.abs(v) < 1e-9);
  }
});

test("normalize: batched [N,H,W,C] input normalizes every image", () => {
  const input = Tensor.from([0, 0, 10, 10], { dtype: "f64" }).reshape([2, 1, 1, 2]);
  const out = normalize(input, { mean: [0, 0], std: [1, 1] });
  const actual = out.toArray() as number[][][][];
  assert.deepEqual(actual[0]?.[0]?.[0], [0, 0]);
  assert.deepEqual(actual[1]?.[0]?.[0], [10, 10]);
});

test("normalize: rejects a mean/std length that doesn't match the channel count", () => {
  const input = Tensor.from([1, 2, 3], { dtype: "f64" }).reshape([1, 1, 3]);
  assert.throws(() => normalize(input, { mean: [0, 0], std: [1, 1, 1] }), RangeError);
});

test("normalize: rejects a zero std", () => {
  const input = Tensor.from([1, 2], { dtype: "f64" }).reshape([1, 1, 2]);
  assert.throws(() => normalize(input, { mean: [0, 0], std: [1, 0] }), RangeError);
});

test("normalize: rejects a non-3D/4D tensor and a non-float dtype", () => {
  const bad2d = Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2]);
  assert.throws(() => normalize(bad2d, { mean: [0, 0], std: [1, 1] }), RangeError);

  const intDtype = Tensor.from([1, 2], { dtype: "i32" }).reshape([1, 1, 2]);
  assert.throws(() => normalize(intDtype, { mean: [0, 0], std: [1, 1] }), TypeError);
});
