import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import {
  ComplexNumber,
  Decimal,
  Fraction,
  Interval,
  Quaternion,
  Rational,
  complexToParts,
  partsToComplex,
} from "../src/index.ts";

test("re-exports @johnhenry/math scalar types", () => {
  const z = new ComplexNumber(3, 4);
  assert.equal(z.magnitude(), 5);
  const r = new Rational(2n, 4n);
  assert.equal(r.toString(), "1/2");
  const d = Decimal.fromString("1.5");
  assert.equal(d.add(Decimal.fromString("2.5")).toNumber(), 4);
});

test("Fraction is an alias of Rational", () => {
  assert.equal(Fraction, Rational);
});

// @johnhenry/math's Interval outward-rounds every non-exact op by ~1 ULP per
// side (johnhenry/math#57) to preserve interval arithmetic's containment
// guarantee -- so even an exact integer result like 1*3=3 comes back a hair
// WIDER than [3, 10], not equal to it. Assert containment of the
// mathematically exact bounds, not exact equality.
function containsExactly(interval: Interval, lo: number, hi: number): boolean {
  return interval.lo <= lo && interval.hi >= hi;
}

test("Interval is re-exported and does real rigorous interval arithmetic", () => {
  const a = new Interval(1, 2);
  const b = new Interval(3, 5);
  const sum = a.add(b);
  assert.ok(containsExactly(sum, 4, 7));
  const product = a.multiply(b);
  assert.ok(containsExactly(product, 3, 10)); // 1*3 .. 2*5
  assert.ok(a.contains(1.5));
  assert.ok(!a.contains(2.5));
});

test("Interval: a point interval bounds a known f64 computation, useful as a rounding-error oracle", () => {
  // sin(pi/2) computed at f64 precision, wrapped as a degenerate interval,
  // still contains the exact value -- the basic property any f32-vs-f64
  // precision-bound check builds on (see tensor-webgpu's fusion tests for
  // the real usage against an actual GPU f32 result).
  const exact = Math.sin(Math.PI / 2);
  const bounded = Interval.point(exact);
  assert.ok(bounded.contains(exact));
  assert.equal(bounded.width, 0);
});

test("Quaternion is re-exported and toRotationMatrix()/rotateVector() round-trip a known 90-degree rotation", () => {
  // 90 degrees about the Z axis.
  const q = Quaternion.fromAxisAngle([0, 0, 1], Math.PI / 2);
  const rotated = q.rotateVector([1, 0, 0]);
  assert.ok(Math.abs((rotated[0] as number) - 0) < 1e-9);
  assert.ok(Math.abs((rotated[1] as number) - 1) < 1e-9);
  assert.ok(Math.abs((rotated[2] as number) - 0) < 1e-9);

  const matrix = q.toRotationMatrix();
  const viaMatrix = [
    matrix[0]![0]! * 1 + matrix[0]![1]! * 0 + matrix[0]![2]! * 0,
    matrix[1]![0]! * 1 + matrix[1]![1]! * 0 + matrix[1]![2]! * 0,
    matrix[2]![0]! * 1 + matrix[2]![1]! * 0 + matrix[2]![2]! * 0,
  ];
  assert.ok(Math.abs(viaMatrix[0]! - (rotated[0] as number)) < 1e-9);
  assert.ok(Math.abs(viaMatrix[1]! - (rotated[1] as number)) < 1e-9);
  assert.ok(Math.abs(viaMatrix[2]! - (rotated[2] as number)) < 1e-9);
});

test("Quaternion: Identity leaves a vector unchanged", () => {
  const rotated = Quaternion.Identity.rotateVector([1, 2, 3]);
  assert.deepEqual(rotated, [1, 2, 3]);
});

test("complex boxed<->flat round-trip (ComplexTensor edge format)", () => {
  const boxed = [new ComplexNumber(1, 2), new ComplexNumber(3, -4)];
  const parts = complexToParts(boxed);
  assert.deepEqual([...parts.real], [1, 3]);
  assert.deepEqual([...parts.imag], [2, -4]);
  const round = partsToComplex(parts);
  assert.ok(boxed[0]!.equals(round[0]!));
  assert.ok(boxed[1]!.equals(round[1]!));
});
