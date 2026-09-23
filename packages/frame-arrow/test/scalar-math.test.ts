/**
 * fn.* elementary math functions (issue #38: growing fn.* beyond fn.month —
 * see expr.ts's SCALAR_MATH_FUNCS doc comment for why the names are spelled
 * to match @johnhenry/math's Symbolic FuncName 1:1).
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { Float64, Int64, Table, Utf8, vectorFromArray } from "apache-arrow";
import { col, fn, Frame, SCALAR_MATH_FUNCS, type ScalarMathFuncName } from "../src/index.ts";

test("fn.* elementary functions compute per-row and agree with Math.* / textbook formulas", () => {
  const frame = Frame.fromArrow(new Table({ x: vectorFromArray([0.5, 1.5, 2.5], new Float64()) }));

  const cases: Array<[ScalarMathFuncName, (x: number) => number]> = [
    ["sin", Math.sin],
    ["cos", Math.cos],
    ["tan", Math.tan],
    ["exp", Math.exp],
    ["sqrt", Math.sqrt],
    ["asin", (x) => Math.asin(x / 3)], // keep asin's domain [-1,1] happy for x=2.5
    ["sinh", Math.sinh],
    ["cosh", Math.cosh],
    ["tanh", Math.tanh],
    ["abs", Math.abs],
    ["floor", Math.floor],
    ["ceil", Math.ceil],
    ["round", Math.round],
    ["sign", Math.sign],
    ["trunc", Math.trunc],
    ["cbrt", Math.cbrt],
    ["log10", Math.log10],
    ["log2", Math.log2],
    ["expm1", Math.expm1],
    ["log1p", Math.log1p],
    ["sigmoid", (x) => 1 / (1 + Math.exp(-x))],
    ["relu", (x) => (x > 0 ? x : 0)],
    ["cot", (x) => 1 / Math.tan(x)],
    ["sec", (x) => 1 / Math.cos(x)],
    ["csc", (x) => 1 / Math.sin(x)],
  ];

  for (const [name, expected] of cases) {
    const applied = name === "asin" ? col("x").div(3) : col("x");
    const result = frame.withColumns({ out: fn[name](applied) }).toRows().map((r) => r.out as number);
    const xs = [0.5, 1.5, 2.5];
    for (let i = 0; i < xs.length; i++) {
      const want = expected(xs[i] as number);
      assert.ok(
        Math.abs((result[i] as number) - want) < 1e-9,
        `fn.${name}(${xs[i]}): got ${result[i]}, want ${want}`,
      );
    }
  }
});

test("fn.ln matches Math.log (spelled 'ln' to match @johnhenry/math's Symbolic FuncName, not tensor-compile's 'log')", () => {
  const frame = Frame.fromArrow(new Table({ x: vectorFromArray([1, Math.E, 10], new Float64()) }));
  const result = frame.withColumns({ y: fn.ln(col("x")) }).toRows().map((r) => r.y);
  assert.deepEqual(result, [0, 1, Math.log(10)]);
});

test("fn.erf is within its Abramowitz & Stegun 7.1.26 error bound (a known non-canonical copy, see eval-expr.ts / #122)", () => {
  const frame = Frame.fromArrow(new Table({ x: vectorFromArray([-1, 0, 1, 2], new Float64()) }));
  const result = frame.withColumns({ y: fn.erf(col("x")) }).toRows().map((r) => r.y as number);
  // erf is odd, erf(0) = 0, and it should be within the approximation's documented error bound of the true values.
  assert.ok(Math.abs(result[1] as number) < 1.5e-7);
  assert.ok(Math.abs((result[0] as number) + (result[2] as number)) < 1e-6); // erf(-1) ~= -erf(1)
  assert.ok(Math.abs((result[2] as number) - 0.8427007929) < 1.5e-7);
  assert.ok(Math.abs((result[3] as number) - 0.9953222650) < 1.5e-7);
});

test("fn.* propagates null through elementwise math, same as arithmetic combinators", () => {
  const frame = Frame.fromArrow(new Table({ x: vectorFromArray([4, null, 9], new Float64()) }));
  const result = frame.withColumns({ y: fn.sqrt(col("x")) }).toRows().map((r) => r.y);
  assert.deepEqual(result, [2, null, 3]);
});

test("fn.* coerces a bigint (Int64) column to number before applying the math function", () => {
  const frame = Frame.fromArrow(new Table({ x: vectorFromArray([1n, 4n, 9n], new Int64()) }));
  const result = frame.withColumns({ y: fn.sqrt(col("x")) }).toRows().map((r) => r.y);
  assert.deepEqual(result, [1, 2, 3]);
});

test("fn.* on a non-numeric column throws instead of silently writing NaN (mirrors arithOp's guard, issue #102)", () => {
  const frame = Frame.fromArrow(new Table({ name: vectorFromArray(["alice", "bob"], new Utf8()) }));
  assert.throws(() => frame.withColumns({ bad: fn.sin(col("name")) }).toRows(), /requires a numeric operand/);
});

test("SCALAR_MATH_FUNCS lists exactly 41 names, matching @johnhenry/math's Symbolic FuncName count 1:1", () => {
  assert.equal(SCALAR_MATH_FUNCS.length, 41);
  assert.equal(new Set(SCALAR_MATH_FUNCS).size, 41); // no duplicates
  for (const name of SCALAR_MATH_FUNCS) {
    assert.equal(typeof fn[name], "function");
  }
});
