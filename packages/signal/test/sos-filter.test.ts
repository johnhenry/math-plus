import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { sosFilter } from "../src/index.ts";
import { runScipyOracle, SCIPY_SKIP_REASON } from "./helpers.ts";

function close(a: ArrayLike<number>, b: ArrayLike<number>, tol: number): void {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs((a[i] as number) - (b[i] as number)) < tol, `index ${i}: ${a[i]} vs ${b[i]}`);
}

test("sosFilter: a single section with b=[1,0,0], a=[1,0,0] (identity) passes the signal through unchanged", () => {
  const x = Tensor.from([1, 2, 3, 4, 5], { dtype: "f64" });
  const out = sosFilter([[1, 0, 0, 1, 0, 0]], x);
  close(out.toArray() as number[], [1, 2, 3, 4, 5], 1e-12);
});

test("sosFilter: a simple one-pole lowpass (y[n] = x[n] + 0.5*y[n-1]) matches a hand-computed recurrence", () => {
  // a1=-0.5 -> y[n] - 0.5*y[n-1] = x[n], i.e. y[n] = x[n] + 0.5*y[n-1].
  const x = [1, 0, 0, 0, 0];
  const out = sosFilter([[1, 0, 0, 1, -0.5, 0]], Tensor.from(x, { dtype: "f64" }));
  const expected: number[] = [];
  let y = 0;
  for (const xi of x) {
    y = xi + 0.5 * y;
    expected.push(y);
  }
  close(out.toArray() as number[], expected, 1e-12);
});

test("sosFilter: cascading two sections is equivalent to filtering with the FIRST then the SECOND independently", () => {
  const x = Tensor.from([1, 0.5, -0.3, 0.8, -1, 0.2], { dtype: "f64" });
  const section1: [number, number, number, number, number, number] = [1, 0.2, 0, 1, -0.3, 0];
  const section2: [number, number, number, number, number, number] = [1, -0.1, 0, 1, 0.4, 0];
  const cascaded = sosFilter([section1, section2], x);
  const stepwise = sosFilter([section2], sosFilter([section1], x));
  close(cascaded.toArray() as number[], stepwise.toArray() as number[], 1e-12);
});

test("sosFilter: normalizes a non-unit a0 coefficient", () => {
  const x = Tensor.from([1, 2, 3], { dtype: "f64" });
  const normalized = sosFilter([[1, 0, 0, 1, 0, 0]], x);
  const unnormalized = sosFilter([[2, 0, 0, 2, 0, 0]], x); // same filter, scaled by 2/2=1
  close(normalized.toArray() as number[], unnormalized.toArray() as number[], 1e-12);
});

test("sosFilter: rejects an a0 of zero", () => {
  const x = Tensor.from([1, 2, 3], { dtype: "f64" });
  assert.throws(() => sosFilter([[1, 0, 0, 0, 0, 0]], x), RangeError);
});

test("sosFilter: matches scipy.signal.sosfilt for an arbitrary (non-butter) SOS cascade", { skip: SCIPY_SKIP_REASON }, () => {
  const sos: [number, number, number, number, number, number][] = [
    [0.2, 0.1, -0.05, 1, -0.3, 0.1],
    [1, -0.4, 0.2, 1, 0.15, -0.2],
  ];
  const x = Array.from({ length: 25 }, (_, i) => Math.sin(i * 0.4) + 0.2 * Math.cos(i * 1.3));
  const mine = sosFilter(sos, Tensor.from(x, { dtype: "f64" })).toArray() as number[];
  const oracle = runScipyOracle<{ y: number[] }>({ op: "sosfilt", sos, x });
  close(mine, oracle.y, 1e-9);
});
