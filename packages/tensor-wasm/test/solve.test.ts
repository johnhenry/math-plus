/**
 * solveInto (issue #39, the first native-kernel candidate named in
 * docs/PLAN.md §9 item 1). Differential-tested against adapter-math's
 * existing reference-speed `linalg.solve` (itself already verified against
 * @johnhenry/math's MatrixMath.solve) -- see linalg.ts's own doc comment: this
 * native kernel sits ALONGSIDE that reference path, not replacing it.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { linalg } from "@johnhenry/math-plus-adapter-math";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { Kernels } from "../src/index.ts";

function closeArrays(a: ArrayLike<number>, b: ArrayLike<number>, tol: number): void {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs((a[i] as number) - (b[i] as number)) < tol, `index ${i}: ${a[i]} vs ${b[i]}`);
  }
}

/** Small deterministic LCG -- no dependency on tensor-core's Rng needed for this. */
function randomWellConditionedSystem(n: number, seed: number): { a: Float32Array; b: Float32Array } {
  let s = seed >>> 0;
  const next = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s / 0xffffffff) * 2 - 1;
  };
  const a = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) a[i * n + j] = next();
    // Strengthen the diagonal so the system stays well-conditioned --
    // avoids a differential test that's flaky because of genuine
    // near-singularity, not because either implementation is wrong.
    a[i * n + i] = (a[i * n + i] as number) + n * 3;
  }
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) b[i] = next() * 10;
  return { a, b };
}

test("solveInto matches a hand-computed 2x2 system", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array([2, 1, 1, 3]), [2, 2]);
  const b = kernels.fromArray(new Float32Array([3, 5]), [2]);
  const out = kernels.zeros([2]);
  kernels.solveInto(out, a, b);
  closeArrays(out.toFloat32Array(), [0.8, 1.4], 1e-5);
});

test("solveInto on a transposed view reads via strides — no copy needed", async () => {
  const kernels = await Kernels.load();
  // Target system: A = [[2,1],[4,3]], x = [3,-1] -> b = A@x = [5,9].
  // Physically store A^T = [[2,4],[1,3]] (row-major flat: [2,4,1,3]);
  // .transposed() reads it back as A via swapped strides, no copy.
  const aT = kernels.fromArray(new Float32Array([2, 4, 1, 3]), [2, 2]).transposed();
  const b = kernels.fromArray(new Float32Array([5, 9]), [2]);
  const out = kernels.zeros([2]);
  kernels.solveInto(out, aT, b);
  closeArrays(out.toFloat32Array(), [3, -1], 1e-4);
});

test("solveInto rejects a non-square matrix and a size mismatch", async () => {
  const kernels = await Kernels.load();
  const nonSquare = kernels.zeros([2, 3]);
  const b2 = kernels.zeros([2]);
  const out2 = kernels.zeros([2]);
  assert.throws(() => kernels.solveInto(out2, nonSquare, b2), RangeError);

  const a3 = kernels.zeros([3, 3]);
  const bWrong = kernels.zeros([2]);
  assert.throws(() => kernels.solveInto(out2, a3, bWrong), RangeError);
});

test("solveInto agrees with adapter-math's linalg.solve across several random well-conditioned systems", async () => {
  const kernels = await Kernels.load();
  for (let trial = 0; trial < 5; trial++) {
    const n = 3 + (trial % 3); // sizes 3, 4, 5, 3, 4
    const { a, b } = randomWellConditionedSystem(n, 1000 + trial);

    const wasmA = kernels.fromArray(a, [n, n]);
    const wasmB = kernels.fromArray(b, [n]);
    const wasmOut = kernels.zeros([n]);
    kernels.solveInto(wasmOut, wasmA, wasmB);

    const aTensor = Tensor.from([...a], { dtype: "f64" }).reshape([n, n]);
    const bTensor = Tensor.from([...b], { dtype: "f64" });
    const reference = linalg.solve(aTensor, bTensor).toArray() as number[];

    // f32 (WASM) vs f64 (reference) -- a looser tolerance than a same-precision
    // comparison, matching this repo's precedent elsewhere for cross-precision checks.
    closeArrays(wasmOut.toFloat32Array(), reference, 1e-2);
  }
});
