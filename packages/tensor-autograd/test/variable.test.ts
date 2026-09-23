import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { DualNumber } from "@johnhenry/math";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { Variable, constant, enableGrad, grad, noGrad, variable } from "../src/index.ts";
import { assertGradientMatches, randomTensor } from "./gradcheck.ts";

// ---- basic tape mechanics ---------------------------------------------------

test("backward accumulates into leaf .grad, non-leaves stay null", () => {
  const x = variable(Tensor.from([2, 3], { dtype: "f64" }));
  const y = variable(Tensor.from([4, 5], { dtype: "f64" }));
  const z = x.mul(y).sum(); // scalar
  z.backward();
  assert.deepEqual(x.grad?.toArray(), [4, 5]); // d(sum(x*y))/dx = y
  assert.deepEqual(y.grad?.toArray(), [2, 3]); // d(sum(x*y))/dy = x
  const intermediate = x.mul(y);
  assert.equal(intermediate.grad, null); // non-leaf: only requiresGrad leaves accumulate
});

test("backward() with no argument requires a scalar output", () => {
  const x = variable(Tensor.from([1, 2, 3], { dtype: "f64" }));
  const y = x.mul(x); // shape [3], not scalar
  assert.throws(() => y.backward(), RangeError);
  assert.doesNotThrow(() => y.backward(Tensor.ones([3], { dtype: "f64" })));
});

test("grad ACCUMULATES across repeated backward() calls; zeroGrad() resets it", () => {
  const x = variable(Tensor.from([1, 2], { dtype: "f64" }));
  x.mul(x).sum().backward(); // grad = 2x = [2,4]
  assert.deepEqual(x.grad?.toArray(), [2, 4]);
  x.mul(x).sum().backward(); // accumulates again
  assert.deepEqual(x.grad?.toArray(), [4, 8]);
  x.zeroGrad();
  assert.equal(x.grad, null);
});

test("detach() cuts the tape — no gradient flows through it", () => {
  const x = variable(Tensor.from([2, 3], { dtype: "f64" }));
  const y = x.mul(x).detach(); // value correct, but no history
  const z = y.mul(constant(Tensor.from([1, 1], { dtype: "f64" }))).sum();
  z.backward();
  assert.equal(x.grad, null); // never touched
  assert.deepEqual(y.value.toArray(), [4, 9]); // value still right
});

test("constant() leaves never accumulate gradients even when part of the graph", () => {
  const w = variable(Tensor.from([2], { dtype: "f64" }));
  const c = constant(Tensor.from([10], { dtype: "f64" }));
  const y = w.mul(c).sum();
  y.backward();
  assert.deepEqual(w.grad?.toArray(), [10]);
  assert.equal(c.grad, null);
});

test("noGrad() suppresses tape construction; enableGrad() re-enables inside it", () => {
  const x = variable(Tensor.from([2], { dtype: "f64" }));
  const untracked = noGrad(() => x.mul(x));
  assert.equal(untracked.node, null); // no tape recorded
  const reTracked = noGrad(() => enableGrad(() => x.mul(x)));
  assert.notEqual(reTracked.node, null);
});

// ---- functional grad API ----------------------------------------------------

test("grad.of computes d(f)/dx for a scalar function", () => {
  const df = grad.of((v: Variable) => v.mul(v).sum()); // f(x) = sum(x^2), df/dx = 2x
  const g = df(Tensor.from([3, 4], { dtype: "f64" }));
  assert.deepEqual(g.toArray(), [6, 8]);
});

test("grad.valueAndGrad returns both in one forward pass", () => {
  const vg = grad.valueAndGrad((v: Variable) => v.mul(v).sum());
  const { value, grad: g } = vg(Tensor.from([3, 4], { dtype: "f64" }));
  assert.equal(value.item(), 25); // 9+16
  assert.deepEqual(g.toArray(), [6, 8]);
});

test("grad.of throws when the function never uses its input differentiably", () => {
  const df = grad.of(() => constant(Tensor.from([1], { dtype: "f64" })));
  assert.throws(() => df(Tensor.from([1], { dtype: "f64" })), Error);
});

// ---- finite-difference gradient checks (issue #8 acceptance criterion) ------

test("gradcheck: add/sub with broadcasting", () => {
  const a = randomTensor([3, 4]);
  const b = randomTensor([4]);
  assertGradientMatches((x) => x.add(constant(b)).sum(), a);
  assertGradientMatches((x) => x.sub(constant(b)).sum(), a);
  // broadcasting the OTHER way: gradient checked w.r.t. the smaller operand
  assertGradientMatches((x) => constant(a).add(x).sum(), b);
});

test("gradcheck: mul/div with broadcasting", () => {
  const a = randomTensor([3, 4], "f64", 1);
  const b = randomTensor([4], "f64", 1).add(Tensor.full([4], 3, { dtype: "f64" })); // keep away from 0
  assertGradientMatches((x) => x.mul(constant(b)).sum(), a);
  assertGradientMatches((x) => x.div(constant(b)).sum(), a);
});

test("gradcheck: matmul", () => {
  const a = randomTensor([3, 4]);
  const b = randomTensor([4, 2]);
  assertGradientMatches((x) => x.matmul(constant(b)).sum(), a);
  assertGradientMatches((x) => constant(a).matmul(x).sum(), b);
});

test("gradcheck: sum/mean, full and per-axis", () => {
  const a = randomTensor([3, 4]);
  assertGradientMatches((x) => x.sum(), a);
  assertGradientMatches((x) => x.mean(), a);
  assertGradientMatches((x) => x.sum(0).sum(), a);
  assertGradientMatches((x) => x.mean(1).sum(), a);
});

test("gradcheck: relu (away from the kink at 0)", () => {
  const a = randomTensor([5], "f64", 1).add(Tensor.full([5], 5, { dtype: "f64" })); // shift positive, away from 0
  assertGradientMatches((x) => x.relu().sum(), a);
});

test("gradcheck: sigmoid", () => {
  assertGradientMatches((x) => x.sigmoid().sum(), randomTensor([5]));
});

test("gradcheck: gelu", () => {
  assertGradientMatches((x) => x.gelu().sum(), randomTensor([5]));
});

test("gradcheck: softmax", () => {
  assertGradientMatches((x) => x.softmax().sum(), randomTensor([4]));
  assertGradientMatches((x) => x.softmax(0).sum(), randomTensor([3, 4]));
});

test("gradcheck: a composed multi-op graph (small MLP-shaped function)", () => {
  const w = randomTensor([4, 3], "f64", 0.5);
  const b = randomTensor([3], "f64", 0.5);
  const fn = (x: Variable) => x.matmul(constant(w)).add(constant(b)).relu().sum();
  assertGradientMatches(fn, randomTensor([2, 4], "f64", 0.5));
});

// ---- DualNumber cross-check (independent oracle, issue #8's second leg) -----

test("gradient of a scalar chain matches @johnhenry/math's DualNumber forward-mode", () => {
  // f(x) = sigmoid(x)*x + x^2 — an arbitrary composition exercising several ops at once.
  const fn = (v: Variable) => v.sigmoid().mul(v).add(v.mul(v)).sum();

  // DualNumber has no built-in sigmoid; build it from primitives it does
  // have (add/multiply/divide/negate/exp) — an independent implementation
  // path from tensor-core's Tensor.sigmoid(), which is the point of the
  // cross-check.
  const dualSigmoid = (d: DualNumber): DualNumber =>
    DualNumber.constant(1).divide(DualNumber.constant(1).add(DualNumber.exp(d.negate())));

  for (const x0 of [-2, -0.5, 0.3, 1.7, 3]) {
    const tape = grad.of(fn)(Tensor.from([x0], { dtype: "f64" })).item() as number;
    const dual = DualNumber.derivative(
      (d: DualNumber) => dualSigmoid(d).multiply(d).add(d.multiply(d)),
      x0,
    );
    assert.ok(
      Math.abs(tape - dual) < 1e-6,
      `tape grad ${tape} vs DualNumber grad ${dual} at x=${x0}`,
    );
  }
});

test("gradcheck: unsqueeze, sqrt, log", () => {
  assertGradientMatches((x) => x.unsqueeze(0).sum(), randomTensor([4]));
  assertGradientMatches((x) => x.unsqueeze(1).sum(), randomTensor([3, 4]));
  assertGradientMatches(
    (x) => x.sqrt().sum(),
    randomTensor([5], "f64", 1).add(Tensor.full([5], 5, { dtype: "f64" })), // keep positive
  );
  assertGradientMatches(
    (x) => x.log().sum(),
    randomTensor([5], "f64", 1).add(Tensor.full([5], 5, { dtype: "f64" })), // keep positive
  );
});

test("unsqueeze backward reduces via sum when the axis was broadcast wider than 1", () => {
  // x: [3] -> unsqueeze(0) -> [1,3] -> broadcast-multiply against [4,3] -> sum
  // forces the inserted axis to receive a gradient of shape [4,1], not [1,1].
  const x = variable(Tensor.from([1, 2, 3], { dtype: "f64" }));
  const wide = constant(Tensor.ones([4, 3], { dtype: "f64" }));
  const y = x.unsqueeze(0).mul(wide).sum();
  y.backward();
  assert.deepEqual(x.grad?.toArray(), [4, 4, 4]); // summed over the 4 broadcast rows
});
