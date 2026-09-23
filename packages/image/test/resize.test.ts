import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { resize } from "../src/index.ts";

function nested(t: Tensor): unknown[] {
  return t.toArray();
}

test("resize nearest: 2x2 -> 4x4, single channel, exact expected values", () => {
  // 2x2 image, 1 channel: [[1,2],[3,4]]
  const input = Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2, 1]);
  const out = resize(input, { height: 4, width: 4 }, { method: "nearest" });
  assert.deepEqual([...out.shape], [4, 4, 1]);
  // nearest with sy=floor(oy*2/4), sx=floor(ox*2/4) -> rows/cols 0,0,1,1
  const expectedRows = [
    [1, 1, 2, 2],
    [1, 1, 2, 2],
    [3, 3, 4, 4],
    [3, 3, 4, 4],
  ];
  const actual = nested(out) as number[][][];
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      assert.equal(actual[y]?.[x]?.[0], expectedRows[y]?.[x], `(${y},${x})`);
    }
  }
});

test("resize nearest: downscale 4x4 -> 2x2", () => {
  const input = Tensor.from(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    { dtype: "f64" },
  ).reshape([4, 4, 1]);
  const out = resize(input, { height: 2, width: 2 }, { method: "nearest" });
  assert.deepEqual([...out.shape], [2, 2, 1]);
  // sy = floor(oy*4/2) = oy*2, sx = ox*2 -> picks rows/cols 0,2
  const actual = nested(out) as number[][][];
  assert.equal(actual[0]?.[0]?.[0], 1);
  assert.equal(actual[0]?.[1]?.[0], 3);
  assert.equal(actual[1]?.[0]?.[0], 9);
  assert.equal(actual[1]?.[1]?.[0], 11);
});

test("resize bilinear: exact hand-computed interpolation for a simple gradient", () => {
  // 2x2 image, 1 channel: [[0,10],[0,10]] -- a horizontal gradient, resize to 1x4.
  const input = Tensor.from([0, 10, 0, 10], { dtype: "f64" }).reshape([2, 2, 1]);
  const out = resize(input, { height: 1, width: 4 }, { method: "bilinear" });
  const actual = (nested(out) as number[][][])[0] as number[][];
  // half-pixel-center mapping, scaleX = 2/4 = 0.5:
  // ox=0: srcX = 0.5*0.5-0.5 = -0.25 -> clamped to 0 -> value 0
  // ox=1: srcX = 1.5*0.5-0.5 = 0.25 -> interpolate 0 and 10 at wx=0.25 -> 2.5
  // ox=2: srcX = 2.5*0.5-0.5 = 0.75 -> interpolate at wx=0.75 -> 7.5
  // ox=3: srcX = 3.5*0.5-0.5 = 1.25 -> clamped to 1 -> value 10
  assert.ok(Math.abs((actual[0]?.[0] as number) - 0) < 1e-9);
  assert.ok(Math.abs((actual[1]?.[0] as number) - 2.5) < 1e-9);
  assert.ok(Math.abs((actual[2]?.[0] as number) - 7.5) < 1e-9);
  assert.ok(Math.abs((actual[3]?.[0] as number) - 10) < 1e-9);
});

test("resize: batched [N,H,W,C] input preserves N and resizes each image independently", () => {
  const img1 = [1, 2, 3, 4]; // 2x2, 1ch
  const img2 = [10, 20, 30, 40];
  const input = Tensor.from([...img1, ...img2], { dtype: "f64" }).reshape([2, 2, 2, 1]);
  const out = resize(input, { height: 4, width: 4 }, { method: "nearest" });
  assert.deepEqual([...out.shape], [2, 4, 4, 1]);
  const actual = nested(out) as number[][][][];
  assert.equal(actual[0]?.[0]?.[0]?.[0], 1);
  assert.equal(actual[1]?.[0]?.[0]?.[0], 10);
});

test("resize: multi-channel — each channel resized independently, no cross-channel bleed", () => {
  // 2x2, 2 channels: pixel (0,0)=[1,100], (0,1)=[2,200], (1,0)=[3,300], (1,1)=[4,400]
  const input = Tensor.from([1, 100, 2, 200, 3, 300, 4, 400], { dtype: "f64" }).reshape([2, 2, 2]);
  const out = resize(input, { height: 2, width: 2 }, { method: "nearest" });
  const actual = nested(out) as number[][][];
  assert.deepEqual(actual[0]?.[0], [1, 100]);
  assert.deepEqual(actual[1]?.[1], [4, 400]);
});

test("resize: rejects a non-3D/4D tensor and a non-float dtype", () => {
  const bad2d = Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2]);
  assert.throws(() => resize(bad2d, { height: 2, width: 2 }), RangeError);

  const intDtype = Tensor.from([1, 2, 3, 4], { dtype: "i32" }).reshape([2, 2, 1]);
  assert.throws(() => resize(intDtype, { height: 2, width: 2 }), TypeError);
});
