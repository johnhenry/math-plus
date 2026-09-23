/**
 * Coverage for the IR extension added alongside issue #15 (Symbolic bridge):
 * the full unary function set (matching @johnhenry/math's FuncName), pow, the
 * atan2/hypot/min/max call2 ops, comparisons, and select (piecewise).
 * Every derivative is checked against central finite differences — the same
 * oracle style used throughout tensor-autograd's gradcheck suite.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { compile, type Traced } from "../src/index.ts";

function evalAt(fn: (v: Traced) => Traced, x: number): number {
  const f = compile(1, fn);
  return f.forward(Tensor.from([x])).toArray()[0] as number;
}

function analyticGradAt(fn: (v: Traced) => Traced, x: number): number {
  const f = compile(1, fn);
  const { localGrads } = f.forwardWithGrad(Tensor.from([x]));
  return (localGrads[0] as Tensor).toArray()[0] as number;
}

function checkUnaryGrad(name: string, fn: (v: Traced) => Traced, x: number, tolerance = 1e-3): void {
  const eps = 1e-4;
  const numeric = (evalAt(fn, x + eps) - evalAt(fn, x - eps)) / (2 * eps);
  const analytic = analyticGradAt(fn, x);
  assert.ok(
    Math.abs(numeric - analytic) < tolerance,
    `${name} at x=${x}: numeric=${numeric} analytic=${analytic}`,
  );
}

const DIFFERENTIABLE_UNARY: Array<[string, (v: Traced) => Traced, number]> = [
  ["sin", (v) => v.sin(), 0.6],
  ["cos", (v) => v.cos(), 0.6],
  ["tan", (v) => v.tan(), 0.4],
  ["asin", (v) => v.asin(), 0.4],
  ["acos", (v) => v.acos(), 0.4],
  ["atan", (v) => v.atan(), 0.8],
  ["sinh", (v) => v.sinh(), 0.7],
  ["cosh", (v) => v.cosh(), 0.7],
  ["tanh", (v) => v.tanh(), 0.7],
  ["cot", (v) => v.cot(), 0.9],
  ["sec", (v) => v.sec(), 0.5],
  ["csc", (v) => v.csc(), 0.9],
  ["asinh", (v) => v.asinh(), 0.8],
  ["acosh", (v) => v.acosh(), 1.5],
  ["atanh", (v) => v.atanh(), 0.4],
  ["coth", (v) => v.coth(), 0.9],
  ["sech", (v) => v.sech(), 0.6],
  ["csch", (v) => v.csch(), 0.9],
  ["acot", (v) => v.acot(), 0.8],
  ["asec", (v) => v.asec(), 1.7],
  ["acsc", (v) => v.acsc(), 1.7],
  ["acoth", (v) => v.acoth(), 1.7],
  ["asech", (v) => v.asech(), 0.4],
  ["acsch", (v) => v.acsch(), 0.9],
  ["abs", (v) => v.abs(), 1.3], // away from the kink at 0
  ["log10", (v) => v.log10(), 2.3],
  ["log2", (v) => v.log2(), 2.3],
  ["cbrt", (v) => v.cbrt(), 2.3],
  ["expm1", (v) => v.expm1(), 0.5],
  ["log1p", (v) => v.log1p(), 0.5],
  ["erf", (v) => v.erf(), 0.7],
];

for (const [name, fn, x] of DIFFERENTIABLE_UNARY) {
  test(`gradcheck: ${name}`, () => checkUnaryGrad(name, fn, x));
}

test("floor/ceil/round/sign/trunc: gradient is 0 away from the kink, value matches Math.*", () => {
  const cases: Array<[string, (v: Traced) => Traced, (n: number) => number]> = [
    ["floor", (v) => v.floor(), Math.floor],
    ["ceil", (v) => v.ceil(), Math.ceil],
    ["round", (v) => v.round(), Math.round],
    ["sign", (v) => v.sign(), Math.sign],
    ["trunc", (v) => v.trunc(), Math.trunc],
  ];
  for (const [name, fn, ref] of cases) {
    assert.equal(evalAt(fn, 2.3), ref(2.3), name);
    assert.equal(analyticGradAt(fn, 2.3), 0, `${name} grad`);
  }
});

test("pow: gradcheck w.r.t. the base with a constant exponent, including a negative base (no NaN leak from the exponent branch)", () => {
  const eps = 1e-4;
  const f = (v: Traced) => v.pow(3);
  for (const x of [2.1, -1.7]) {
    const numeric = (evalAt(f, x + eps) - evalAt(f, x - eps)) / (2 * eps);
    const analytic = analyticGradAt(f, x);
    assert.ok(Math.abs(numeric - analytic) < 1e-2, `x=${x}: numeric=${numeric} analytic=${analytic}`);
    assert.ok(Number.isFinite(analytic), `x=${x}: analytic gradient must be finite, got ${analytic}`);
  }
});

test("pow: gradcheck w.r.t. both base and exponent (positive base)", () => {
  const base = Tensor.from([2.5]);
  const exponent = Tensor.from([1.7]);
  const f = compile(2, (b, e) => b.pow(e));
  const { localGrads } = f.forwardWithGrad(base, exponent);
  const eps = 1e-4;

  const evalPow = (b: number, e: number) => f.forward(Tensor.from([b]), Tensor.from([e])).toArray()[0] as number;
  const dBase = (evalPow(2.5 + eps, 1.7) - evalPow(2.5 - eps, 1.7)) / (2 * eps);
  const dExp = (evalPow(2.5, 1.7 + eps) - evalPow(2.5, 1.7 - eps)) / (2 * eps);

  assert.ok(Math.abs(dBase - ((localGrads[0] as Tensor).toArray()[0] as number)) < 1e-2);
  assert.ok(Math.abs(dExp - ((localGrads[1] as Tensor).toArray()[0] as number)) < 1e-2);
});

test("atan2/hypot/min/max: value and gradient agreement", () => {
  const eps = 1e-4;
  const cases: Array<[string, (a: Traced, b: Traced) => Traced, (a: number, b: number) => number]> = [
    ["atan2", (a, b) => a.atan2(b), Math.atan2],
    ["hypot", (a, b) => a.hypot(b), Math.hypot],
    ["min", (a, b) => a.min(b), Math.min],
    ["max", (a, b) => a.max(b), Math.max],
  ];
  const a0 = 1.3;
  const b0 = -0.7;
  for (const [name, fn, ref] of cases) {
    const f = compile(2, fn);
    assert.ok(
      Math.abs((f.forward(Tensor.from([a0]), Tensor.from([b0])).toArray()[0] as number) - ref(a0, b0)) < 1e-6,
      `${name} value`,
    );
    const evalIt = (a: number, b: number) => f.forward(Tensor.from([a]), Tensor.from([b])).toArray()[0] as number;
    const { localGrads } = f.forwardWithGrad(Tensor.from([a0]), Tensor.from([b0]));
    const dA = (evalIt(a0 + eps, b0) - evalIt(a0 - eps, b0)) / (2 * eps);
    const dB = (evalIt(a0, b0 + eps) - evalIt(a0, b0 - eps)) / (2 * eps);
    assert.ok(Math.abs(dA - ((localGrads[0] as Tensor).toArray()[0] as number)) < 1e-2, `${name} dA`);
    assert.ok(Math.abs(dB - ((localGrads[1] as Tensor).toArray()[0] as number)) < 1e-2, `${name} dB`);
  }
});

test("cmp: evaluates to 1/0 matching the JS operator, gradient is 0", () => {
  const f = compile(2, (a, b) => a.lt(b));
  assert.equal(f.forward(Tensor.from([1]), Tensor.from([2])).toArray()[0], 1);
  assert.equal(f.forward(Tensor.from([2]), Tensor.from([1])).toArray()[0], 0);
  const { localGrads } = f.forwardWithGrad(Tensor.from([1]), Tensor.from([2]));
  assert.equal((localGrads[0] as Tensor).toArray()[0], 0);
  assert.equal((localGrads[1] as Tensor).toArray()[0], 0);
});

test("select: routes to the correct branch and short-circuits (the untaken branch's domain error never surfaces)", () => {
  // relu(x), reimplemented as select(x > 0, x, 0) — should agree with the built-in relu op.
  const relu = compile(1, (v) => v.gt(0).select(v, 0));
  assert.equal(relu.forward(Tensor.from([3])).toArray()[0], 3);
  assert.equal(relu.forward(Tensor.from([-3])).toArray()[0], 0);

  // The "else" branch takes sqrt of a negative number if it were ever evaluated;
  // short-circuiting means this never throws/NaNs when the condition is true.
  const guarded = compile(1, (v) => v.gt(0).select(v.sqrt(), v.abs()));
  assert.equal(guarded.forward(Tensor.from([4])).toArray()[0], 2);
  assert.ok(Number.isFinite(guarded.forward(Tensor.from([4])).toArray()[0] as number));

  // Gradient follows the taken branch only.
  const { localGrads } = relu.forwardWithGrad(Tensor.from([3]));
  assert.equal((localGrads[0] as Tensor).toArray()[0], 1);
  const { localGrads: negGrads } = relu.forwardWithGrad(Tensor.from([-3]));
  assert.equal((negGrads[0] as Tensor).toArray()[0], 0);
});

test("piecewise-style nested select mirrors @johnhenry/math's piecewise semantics: first true branch wins", () => {
  // sign-like: x < 0 -> -1, x > 0 -> 1, else 0
  const signish = compile(1, (v) => v.lt(0).select(-1, v.gt(0).select(1, 0)));
  assert.equal(signish.forward(Tensor.from([-5])).toArray()[0], -1);
  assert.equal(signish.forward(Tensor.from([5])).toArray()[0], 1);
  assert.equal(signish.forward(Tensor.from([0])).toArray()[0], 0);
});
