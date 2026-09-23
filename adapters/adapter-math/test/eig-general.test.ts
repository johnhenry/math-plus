/**
 * Differential tests for eigGeneral (issue #68) against NumPy's
 * numpy.linalg.eigvals -- same skip-don't-fail convention as the rest of
 * this family's Python-oracle tests (docs/TESTING.md). Compared as SETS
 * (order-independent), matching this repo's established "compare
 * properties, not incidental order" convention (the SymPy oracle work
 * follows the same principle for solve()/solveSystem()).
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

function oracleEigenvalues(matrix: number[][]): Array<[number, number]> {
  const out = execFileSync(PYTHON as string, [ORACLE_SCRIPT], {
    input: JSON.stringify({ matrix }),
    encoding: "utf8",
  });
  return (JSON.parse(out) as { eigenvalues: Array<[number, number]> }).eigenvalues;
}

const TOL = 1e-4;

/** Greedy nearest-match set comparison: every @johnhenry/math eigenvalue must have
 * an unclaimed NumPy eigenvalue within tolerance, and vice versa (both
 * directions catch both "extra" and "missing" eigenvalues). */
function assertSameEigenvalueSet(
  mallory: Array<{ re: number; im: number }>,
  numpy: Array<[number, number]>,
  label: string,
): void {
  assert.equal(mallory.length, numpy.length, `${label}: count mismatch (${mallory.length} vs ${numpy.length})`);
  const remaining = [...numpy];
  for (const m of mallory) {
    const idx = remaining.findIndex(([re, im]) => Math.abs(re - m.re) < TOL && Math.abs(im - m.im) < TOL);
    assert.ok(
      idx >= 0,
      `${label}: mallory eigenvalue ${m.re}${m.im >= 0 ? "+" : ""}${m.im}i has no matching NumPy eigenvalue in ${JSON.stringify(remaining)}`,
    );
    remaining.splice(idx, 1);
  }
}

function toMatrix(rows: number, cols: number, flat: number[]): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < rows; i++) out.push(flat.slice(i * cols, (i + 1) * cols));
  return out;
}

test("eigGeneral matches numpy.linalg.eigvals, all-real-eigenvalue case", { skip }, () => {
  const flat = [4, 1, 2, 3];
  const a = Tensor.from(flat).reshape([2, 2]);
  const got = linalg.eigGeneral(a);
  assertSameEigenvalueSet(got, oracleEigenvalues(toMatrix(2, 2, flat)), "2x2 real");
});

test("eigGeneral matches numpy.linalg.eigvals, a genuine complex-conjugate pair (rotation-like block)", { skip }, () => {
  const flat = [0.5, -2, 1, 0.5]; // trace=1, det=0.25+2=2.25 -> discriminant = 1 - 9 < 0
  const a = Tensor.from(flat).reshape([2, 2]);
  const got = linalg.eigGeneral(a);
  assert.ok(Math.abs(got[0]!.im) > 0.1, "expected a genuinely complex eigenvalue, got a real one");
  assertSameEigenvalueSet(got, oracleEigenvalues(toMatrix(2, 2, flat)), "2x2 complex");
});

test("eigGeneral matches numpy.linalg.eigvals, a larger random non-symmetric matrix", { skip }, () => {
  const n = 5;
  const rng = (() => {
    let s = 12345;
    return () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return (s / 0x7fffffff) * 4 - 2;
    };
  })();
  const flat = Array.from({ length: n * n }, () => rng());
  const a = Tensor.from(flat).reshape([n, n]);
  const got = linalg.eigGeneral(a);
  assert.equal(got.length, n);
  assertSameEigenvalueSet(got, oracleEigenvalues(toMatrix(n, n, flat)), "5x5 random");
});

test("eigGeneral cross-checks with eigSymmetric on a symmetric input (both paths should agree)", { skip }, () => {
  const flat = [4, 2, 1, 2, 5, 3, 1, 3, 6];
  const a = Tensor.from(flat).reshape([3, 3]);
  const general = linalg.eigGeneral(a);
  const symmetric = linalg.eigSymmetric(a);
  const symValues = [...(symmetric.values.contiguous().data as Float64Array)].map((v) => [v, 0] as [number, number]);
  assertSameEigenvalueSet(general, symValues, "3x3 symmetric cross-check");
  // And both should agree with NumPy independently.
  assertSameEigenvalueSet(general, oracleEigenvalues(toMatrix(3, 3, flat)), "3x3 symmetric vs numpy");
});

test("eigGeneral rejects a non-square matrix", () => {
  const a = Tensor.from([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  assert.throws(() => linalg.eigGeneral(a), RangeError);
});

test("eigGeneral on a 1x1 matrix returns its single entry", () => {
  const a = Tensor.from([7]).reshape([1, 1]);
  const got = linalg.eigGeneral(a);
  assert.equal(got.length, 1);
  assert.equal(got[0]!.re, 7);
  assert.equal(got[0]!.im, 0);
});

test("eigGeneral on an upper-triangular matrix returns the diagonal exactly (eigenvalues of a triangular matrix)", () => {
  const a = Tensor.from([3, 1, 4, 0, 5, 9, 0, 0, 2]).reshape([3, 3]);
  const got = linalg.eigGeneral(a).map((z) => z.re).sort((x, y) => x - y);
  assert.deepEqual(
    got.map((v) => Math.round(v * 1e6) / 1e6),
    [2, 3, 5],
  );
});
