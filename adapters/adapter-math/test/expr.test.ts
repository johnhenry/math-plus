import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { Symbolic } from "@johnhenry/math";
import { compileExpr, UnsupportedExprError } from "../src/index.ts";

function num(t: unknown): number {
  return t as number;
}

test("compileExpr: value matches Symbolic.evaluate across a batch of tensor inputs", () => {
  const expr = Symbolic.parse("sin(x) * y + sqrt(x^2 + 1)");
  const compiled = compileExpr(expr, { variables: ["x", "y"] });

  const xs = [0.1, -0.5, 1.7, 2.3];
  const ys = [1.0, -2.0, 0.3, 4.5];
  const result = compiled.forward(Tensor.from(xs), Tensor.from(ys));
  const actual = result.toArray().map(num);

  const expected = xs.map((x, i) => Symbolic.evaluate(expr, { x, y: ys[i] as number }));
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Math.abs((actual[i] as number) - (expected[i] as number)) < 1e-4, `index ${i}`);
  }
});

test("compileExpr: default variable order is Symbolic.freeVariables (alphabetical)", () => {
  const expr = Symbolic.parse("b - a");
  const compiled = compileExpr(expr); // no explicit variables -> ["a", "b"]
  const result = compiled.forward(Tensor.from([10]), Tensor.from([3]));
  assert.equal(num(result.toArray()[0]), 3 - 10);
});

test("compileExpr: gradient (via CompiledFn.forwardWithGrad) matches Symbolic.differentiate evaluated at the same point — an independent oracle, not just a hand-derived formula check", () => {
  const cases = ["sin(x)*cos(x)", "exp(-x^2)", "erf(x) + tanh(x)", "1/(1+exp(-x))", "ln(x^2+1)"];
  const points = [0.3, -0.8, 1.1];
  for (const source of cases) {
    const expr = Symbolic.parse(source);
    const dExpr = Symbolic.differentiate(expr, "x");
    const compiled = compileExpr(expr, { variables: ["x"] });
    for (const x of points) {
      const { localGrads } = compiled.forwardWithGrad(Tensor.from([x]));
      const analytic = num((localGrads[0] as Tensor).toArray()[0]);
      const oracle = Symbolic.evaluate(dExpr, { x });
      assert.ok(
        Math.abs(analytic - oracle) < 1e-4,
        `${source} at x=${x}: compileExpr grad=${analytic}, Symbolic.differentiate oracle=${oracle}`,
      );
    }
  }
});

test("compileExpr: piecewise compiles to a chain of selects and matches Symbolic.evaluate branch-by-branch", () => {
  // A textbook piecewise: |x| via cases, without calling abs().
  // Syntax: piecewise(cond1, expr1, ..., otherwise) — alternating (cond, expr) pairs plus a trailing fallback.
  const expr = Symbolic.parse("piecewise(x < 0, -x, x)");
  const compiled = compileExpr(expr, { variables: ["x"] });
  for (const x of [-3, -0.5, 0, 0.5, 3]) {
    const actual = num(compiled.forward(Tensor.from([x])).toArray()[0]);
    const expected = Symbolic.evaluate(expr, { x });
    assert.equal(actual, expected, `x=${x}`);
  }
});

test("compileExpr: comparisons evaluate to 1/0 matching Symbolic.evaluate", () => {
  const expr = Symbolic.parse("x > 2");
  const compiled = compileExpr(expr, { variables: ["x"] });
  for (const x of [1, 2, 3]) {
    assert.equal(num(compiled.forward(Tensor.from([x])).toArray()[0]), Symbolic.evaluate(expr, { x }));
  }
});

test("compileExpr: pow, atan2, hypot, min, max all bridge through and agree with Symbolic.evaluate", () => {
  const cases: Array<[string, Record<string, number>]> = [
    ["x^y", { x: 2, y: 5 }],
    ["atan2(y, x)", { x: 3, y: 4 }],
    ["hypot(x, y)", { x: 3, y: 4 }],
    ["min(x, y)", { x: 3, y: 4 }],
    ["max(x, y)", { x: 3, y: 4 }],
  ];
  for (const [source, env] of cases) {
    const expr = Symbolic.parse(source);
    const vars = Symbolic.freeVariables(expr);
    const compiled = compileExpr(expr, { variables: vars });
    const inputs = vars.map((name) => Tensor.from([env[name] as number]));
    const actual = num(compiled.forward(...inputs).toArray()[0]);
    const expected = Symbolic.evaluate(expr, env);
    // f32 storage round-trip (Tensor defaults to f32) caps precision well below f64's ~1e-15.
    assert.ok(Math.abs(actual - expected) < 1e-5, `${source}: actual=${actual} expected=${expected}`);
  }
});

test("compileExpr: throws UnsupportedExprError on sum/product (bound-range reductions), not a silent fallback", () => {
  const sumExpr = Symbolic.parse("sum(i, 1, 10, i)");
  assert.throws(() => compileExpr(sumExpr, { variables: [] }), UnsupportedExprError);

  const productExpr = Symbolic.parse("product(i, 1, 5, i)");
  assert.throws(() => compileExpr(productExpr, { variables: [] }), UnsupportedExprError);
});

test("compileExpr: throws UnsupportedExprError on gcd/lcm (integer-domain, no elementwise-tensor meaning)", () => {
  const gcdExpr = Symbolic.parse("gcd(x, y)");
  assert.throws(() => compileExpr(gcdExpr, { variables: ["x", "y"] }), UnsupportedExprError);
});

test("compileExpr: throws @johnhenry/math's UndeclaredVariableError (not a silent NaN) when the declared variable list omits a referenced variable", () => {
  const expr = Symbolic.parse("x + y");
  assert.throws(() => compileExpr(expr, { variables: ["x"] }));
});

test("compileExpr: fused evaluation over a large tensor batch stays elementwise-correct (vectorized symbolic expressions)", () => {
  const expr = Symbolic.parse("sigmoid(w*x + b)");
  const compiled = compileExpr(expr, { variables: ["x", "w", "b"] });
  const n = 10_000;
  const xs = Array.from({ length: n }, (_, i) => (i - n / 2) / 500);
  const result = compiled.forward(
    Tensor.from(xs),
    Tensor.full([n], 2),
    Tensor.full([n], -0.5),
  );
  const arr = result.toArray().map(num);
  for (let i = 0; i < n; i += 997) {
    const expected = 1 / (1 + Math.exp(-(2 * (xs[i] as number) - 0.5)));
    assert.ok(Math.abs((arr[i] as number) - expected) < 1e-4, `index ${i}`);
  }
});
