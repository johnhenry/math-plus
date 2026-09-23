import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { ComplexNumber } from "@johnhenry/math-plus-scalar-types";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { ComplexTensor } from "../src/index.ts";

test("fromComplexArray -> toComplexArray round-trips exactly", () => {
  const values = [new ComplexNumber(1, 2), new ComplexNumber(-3, 4.5), new ComplexNumber(0, -1)];
  const ct = ComplexTensor.fromComplexArray(values);
  assert.equal(ct.shape.length, 1);
  assert.equal(ct.size, 3);
  const round = ct.toComplexArray();
  for (let i = 0; i < values.length; i++) {
    assert.ok((values[i] as ComplexNumber).equals(round[i] as ComplexNumber));
  }
});

test("at() returns a boxed ComplexNumber matching real/imag at the same index", () => {
  const ct = ComplexTensor.fromComplexArray([new ComplexNumber(1, 2), new ComplexNumber(3, 4)]);
  const z = ct.at(1);
  assert.equal(z.re, 3);
  assert.equal(z.im, 4);
});

test("item() requires size 1 and returns a boxed ComplexNumber", () => {
  const ct = ComplexTensor.fromComplexArray([new ComplexNumber(5, -6)]);
  const z = ct.item();
  assert.equal(z.re, 5);
  assert.equal(z.im, -6);

  const multi = ComplexTensor.fromComplexArray([new ComplexNumber(1, 1), new ComplexNumber(2, 2)]);
  assert.throws(() => multi.item(), RangeError);
});

test("fromReal wraps a real Tensor with a zero-imaginary part, no copy of real", () => {
  const real = Tensor.from([1, 2, 3], { dtype: "f64" });
  const ct = ComplexTensor.fromReal(real);
  assert.equal(ct.real, real); // same object, not a copy
  assert.deepEqual([...(ct.imag.toArray() as number[])], [0, 0, 0]);
});

test("fromParts throws on shape or dtype mismatch", () => {
  const a = Tensor.from([1, 2, 3], { dtype: "f64" });
  const bWrongShape = Tensor.from([1, 2], { dtype: "f64" });
  assert.throws(() => ComplexTensor.fromParts(a, bWrongShape), RangeError);

  const bWrongDtype = Tensor.from([1, 2, 3], { dtype: "f32" });
  assert.throws(() => ComplexTensor.fromParts(a, bWrongDtype), TypeError);
});

test("zeros creates a ComplexTensor of the given shape, all zero", () => {
  const ct = ComplexTensor.zeros([2, 2]);
  assert.deepEqual([...ct.shape], [2, 2]);
  for (const v of ct.toComplexArray()) {
    assert.equal(v.re, 0);
    assert.equal(v.im, 0);
  }
});
