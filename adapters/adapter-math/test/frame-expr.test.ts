import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Float64, Table, vectorFromArray } from "apache-arrow";
import { Symbolic } from "@johnhenry/math";
import { col, Frame } from "@johnhenry/math-plus-frame-arrow";
import { compileFrameExpr, UnsupportedFrameExprError } from "../src/index.ts";

function frameOf(values: Record<string, number[]>): Frame {
  const columns: Record<string, ReturnType<typeof vectorFromArray>> = {};
  for (const [name, vals] of Object.entries(values)) {
    columns[name] = vectorFromArray(vals, new Float64());
  }
  return Frame.fromArrow(new Table(columns));
}

test("compileFrameExpr: value matches Symbolic.evaluate for a formula with arithmetic + a function call", () => {
  const expr = Symbolic.parse("sin(x) * y + sqrt(x^2 + 1)");
  // pow (x^2) isn't representable in frame-arrow, so pre-expand it away for this bridge.
  const withoutPow = Symbolic.parse("sin(x) * y + sqrt(x*x + 1)");
  const frame = frameOf({ x: [0.1, -0.5, 1.7], y: [1.0, -2.0, 0.3] });
  const compiled = compileFrameExpr(withoutPow);
  const rows = frame.withColumns({ out: compiled }).toRows();
  for (let i = 0; i < rows.length; i++) {
    const x = (frame.toRows()[i] as { x: number }).x;
    const y = (frame.toRows()[i] as { y: number }).y;
    const expected = Symbolic.evaluate(expr, { x, y });
    assert.ok(Math.abs((rows[i]?.out as number) - expected) < 1e-9, `row ${i}`);
  }
});

test("compileFrameExpr: each free variable becomes a col() reference of the same name", () => {
  const expr = Symbolic.parse("a - b");
  const frame = frameOf({ a: [10, 20], b: [3, 7] });
  const compiled = compileFrameExpr(expr);
  const rows = frame.withColumns({ diff: compiled }).toRows();
  assert.deepEqual(rows.map((r) => r.diff), [7, 13]);
});

test("compileFrameExpr: differentiate-then-compile-a-second-column workflow (the issue #38 payoff)", () => {
  const formula = Symbolic.parse("sin(x) * x");
  const derivative = Symbolic.differentiate(formula, "x");
  const frame = frameOf({ x: [0.3, 1.2, -0.7] });
  const withBoth = frame.withColumns({
    y: compileFrameExpr(formula),
    dy_dx: compileFrameExpr(derivative),
  });
  const rows = withBoth.toRows();
  const xs = [0.3, 1.2, -0.7];
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i] as number;
    assert.ok(Math.abs((rows[i]?.y as number) - Math.sin(x) * x) < 1e-9, `y row ${i}`);
    const expectedDeriv = Symbolic.evaluate(derivative, { x });
    assert.ok(Math.abs((rows[i]?.dy_dx as number) - expectedDeriv) < 1e-9, `dy_dx row ${i}`);
  }
});

test("compileFrameExpr: comparisons translate to frame-arrow CompareExpr and agree with Symbolic.evaluate's 1/0", () => {
  const expr = Symbolic.parse("x > 2");
  const frame = frameOf({ x: [1, 2, 3] });
  const compiled = compileFrameExpr(expr);
  const rows = frame.withColumns({ out: compiled }).toRows();
  assert.deepEqual(rows.map((r) => r.out), [false, false, true]);
});

test("compileFrameExpr: unary minus works via mul(-1)", () => {
  const expr = Symbolic.parse("-x");
  const frame = frameOf({ x: [5, -3] });
  const rows = frame.withColumns({ out: compileFrameExpr(expr) }).toRows();
  assert.deepEqual(rows.map((r) => r.out), [-5, 3]);
});

test("compileFrameExpr: throws UnsupportedFrameExprError on pow (no exponentiation combinator in frame-arrow)", () => {
  const expr = Symbolic.parse("x^2");
  assert.throws(() => compileFrameExpr(expr), UnsupportedFrameExprError);
});

test("compileFrameExpr: throws UnsupportedFrameExprError on two-argument functions (atan2/hypot/min/max/gcd/lcm)", () => {
  assert.throws(() => compileFrameExpr(Symbolic.parse("atan2(y, x)")), UnsupportedFrameExprError);
  assert.throws(() => compileFrameExpr(Symbolic.parse("hypot(x, y)")), UnsupportedFrameExprError);
  assert.throws(() => compileFrameExpr(Symbolic.parse("gcd(x, y)")), UnsupportedFrameExprError);
});

test("compileFrameExpr: throws UnsupportedFrameExprError on piecewise (no conditional-select combinator)", () => {
  const expr = Symbolic.parse("piecewise(x < 0, -x, x)");
  assert.throws(() => compileFrameExpr(expr), UnsupportedFrameExprError);
});

test("compileFrameExpr: throws UnsupportedFrameExprError on sum/product (bound-range reductions)", () => {
  assert.throws(() => compileFrameExpr(Symbolic.parse("sum(i, 1, 10, i)")), UnsupportedFrameExprError);
  assert.throws(() => compileFrameExpr(Symbolic.parse("product(i, 1, 5, i)")), UnsupportedFrameExprError);
});

test("compileFrameExpr: transcendental functions all bridge through and agree with Symbolic.evaluate", () => {
  const sources = ["exp(x)", "ln(x)", "sqrt(x)", "sigmoid(x)", "erf(x)", "tanh(x)", "abs(x)"];
  const frame = frameOf({ x: [0.5, 1.5, 2.5] });
  for (const source of sources) {
    const expr = Symbolic.parse(source);
    const rows = frame.withColumns({ out: compileFrameExpr(expr) }).toRows();
    for (let i = 0; i < rows.length; i++) {
      const x = [0.5, 1.5, 2.5][i] as number;
      const expected = Symbolic.evaluate(expr, { x });
      assert.ok(Math.abs((rows[i]?.out as number) - expected) < 1e-6, `${source} row ${i}`);
    }
  }
});
