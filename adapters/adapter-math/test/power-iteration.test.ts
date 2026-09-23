/**
 * Tests for powerIteration (issue #84) — matrix-free dominant (Perron)
 * eigenvalue. Cross-checked against eigGeneral (small matrices, this
 * repo's own oracle) and against numpy.linalg.eigvals (the family's own
 * eig_oracle.py, same skip-don't-fail convention as eig-general.test.ts).
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { linalg } from "../src/index.ts";

const ORACLE_SCRIPT = new URL("../scripts/eig_oracle.py", import.meta.url).pathname;

function findOraclePython(): string | undefined {
  for (const candidate of [process.env.MATH_PLUS_ORACLE_PYTHON, "python3"].filter((c): c is string => Boolean(c))) {
    try {
      execFileSync(candidate, ["-c", "import numpy"], { stdio: "ignore" });
      return candidate;
    } catch {
      // try next
    }
  }
  return undefined;
}

const PYTHON = findOraclePython();
const skip = PYTHON ? false : "no python with numpy found (set MATH_PLUS_ORACLE_PYTHON)";

function runOracle(matrix: number[][]): Array<[number, number]> {
  const out = execFileSync(PYTHON as string, [ORACLE_SCRIPT], { input: JSON.stringify({ matrix }), encoding: "utf8" });
  return (JSON.parse(out) as { eigenvalues: Array<[number, number]> }).eigenvalues;
}

function matvecOf(matrix: number[][]): (v: readonly number[]) => number[] {
  return (v) => matrix.map((row) => row.reduce((s, a, j) => s + a * (v[j] as number), 0));
}

test("powerIteration matches numpy.linalg.eigvals' largest-magnitude eigenvalue on a random 5x5 matrix", { skip }, () => {
  let seed = 12345;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed / 0x7fffffff) * 2 - 1;
  };
  // Symmetric (M = (R+Rᵀ)/2), which GUARANTEES real eigenvalues, plus a
  // large diagonal shift so the dominant one is well-separated (power
  // iteration's convergence rate is |lambda2/lambda1|) -- a plain random
  // (non-symmetric) matrix risks a complex-conjugate dominant pair, which
  // this test isn't checking (confirmed empirically: an earlier
  // diagonally-dominant-but-non-symmetric attempt hit exactly that).
  const n = 5;
  const raw = Array.from({ length: n }, () => Array.from({ length: n }, rng));
  const matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) =>
      i === j ? 5 * n : ((raw[i] as number[])[j] as number) + ((raw[j] as number[])[i] as number),
    ),
  );
  const oracleEigs = runOracle(matrix);
  const dominant = oracleEigs.reduce((best, e) => (Math.hypot(...e) > Math.hypot(...best) ? e : best));
  assert.ok(Math.abs(dominant[1]) < 1e-6, "the constructed matrix's dominant eigenvalue should be real");

  const result = linalg.powerIteration(matvecOf(matrix), n);
  assert.ok(result.converged, `did not converge in ${result.iterations} iterations`);
  assert.ok(Math.abs(result.eigenvalue - dominant[0]) < 1e-6, `${result.eigenvalue} vs oracle ${dominant[0]}`);
});

test("powerIteration matches eigGeneral's dominant eigenvalue for a symmetric matrix with a known eigenvector (I + J, J=all-ones)", () => {
  // I+J has eigenvalues 1+3=4 (eigenvector (1,1,1)/sqrt(3)) and 1+0=1
  // (multiplicity 2) -- hand-derivable since J=ones(3,3) is rank 1 with
  // eigenvalues {3,0,0}. Verified against eigGeneral before writing this
  // as the primary assertion.
  const matrix = [
    [2, 1, 1],
    [1, 2, 1],
    [1, 1, 2],
  ];
  const t = Tensor.from(matrix.flat(), { dtype: "f64" }).reshape([3, 3]);
  const eigs = linalg.eigGeneral(t);
  const dominant = eigs.reduce((best, e) => (Math.abs(e.re) > Math.abs(best.re) ? e : best));
  assert.ok(Math.abs(dominant.re - 4) < 1e-9);

  const result = linalg.powerIteration(matvecOf(matrix), 3);
  assert.ok(result.converged);
  assert.ok(Math.abs(result.eigenvalue - 4) < 1e-8);
  const expected = 1 / Math.sqrt(3);
  for (const component of result.eigenvector) {
    assert.ok(Math.abs(Math.abs(component) - expected) < 1e-6, `eigenvector component ${component}`);
  }
});

test("powerIteration correctly captures a NEGATIVE dominant eigenvalue (not just its magnitude), via the Rayleigh quotient", () => {
  // [[0,1],[-2,-3]] has eigenvalues -1 and -2 -- dominant BY MAGNITUDE is
  // -2. Verified against eigGeneral before writing this.
  const matrix = [
    [0, 1],
    [-2, -3],
  ];
  const t = Tensor.from(matrix.flat(), { dtype: "f64" }).reshape([2, 2]);
  const eigs = linalg.eigGeneral(t);
  const eigVals = eigs.map((e) => e.re).sort((a, b) => a - b);
  assert.deepEqual(
    eigVals.map((v) => Math.round(v * 1e6) / 1e6),
    [-2, -1],
  );

  const result = linalg.powerIteration(matvecOf(matrix), 2, { maxIterations: 2000 });
  assert.ok(result.converged);
  assert.ok(Math.abs(result.eigenvalue - -2) < 1e-6, `expected -2, got ${result.eigenvalue}`);
});

test("powerIteration: an explicit initial vector and a tighter tolerance both work", () => {
  const matrix = [
    [2, 1, 1],
    [1, 2, 1],
    [1, 1, 2],
  ];
  const result = linalg.powerIteration(matvecOf(matrix), 3, { initial: [1, 0, 0], tolerance: 1e-13, maxIterations: 500 });
  assert.ok(result.converged);
  assert.ok(Math.abs(result.eigenvalue - 4) < 1e-10);
});

test("powerIteration: a converged run's eigenvector actually satisfies A*v ~ lambda*v", () => {
  const matrix = [
    [4, 1, 0],
    [1, 3, 1],
    [0, 1, 2],
  ];
  // A tighter tolerance than the default: the Rayleigh-quotient eigenvalue
  // estimate converges quadratically near the fixed point, but the
  // eigenvector itself only linearly (rate |lambda2/lambda1|) -- so at the
  // DEFAULT tolerance (1e-10 on the eigenvalue), the eigenvector residual
  // can still be ~1e-6, not yet at the eigenvalue's own precision.
  // Confirmed empirically (a standalone script) before loosening this
  // test's residual tolerance to 1e-4 accordingly.
  const result = linalg.powerIteration(matvecOf(matrix), 3, { tolerance: 1e-13, maxIterations: 500 });
  assert.ok(result.converged);
  const av = matvecOf(matrix)(result.eigenvector);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs((av[i] as number) - result.eigenvalue * (result.eigenvector[i] as number)) < 1e-6, `component ${i}`);
  }
});

test("powerIteration: rejects a non-positive n and a matvec returning the wrong length", () => {
  assert.throws(() => linalg.powerIteration(() => [], 0), RangeError);
  assert.throws(() => linalg.powerIteration(() => [1, 2], 3), RangeError);
});

test("powerIteration: throws on a matvec that collapses the iterate to zero (e.g. the zero matrix)", () => {
  assert.throws(() => linalg.powerIteration(() => [0, 0, 0], 3), /zero vector/);
});
