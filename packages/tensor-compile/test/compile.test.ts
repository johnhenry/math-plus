import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { Variable } from "@johnhenry/math-plus-tensor-autograd";
import { compile } from "../src/index.ts";

function flat(t: Tensor): number[] {
  return Array.from(t.contiguous().data as Float32Array | Float64Array);
}

function assertClose(actual: readonly number[], expected: readonly number[], tolerance = 1e-5): void {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs((actual[i] as number) - (expected[i] as number));
    assert.ok(diff <= tolerance, `index ${i}: |${actual[i]} - ${expected[i]}| = ${diff} > ${tolerance}`);
  }
}

test("fused forward matches unfused eager computation: (a+b)*c -> relu", () => {
  const a = Tensor.from([1, -2, 3, -4]);
  const b = Tensor.from([0.5, 0.5, -6, 10]);
  const c = Tensor.from([2, 2, 2, 2]);

  const fused = compile(3, (x, y, z) => x.add(y).mul(z).relu());
  const actual = flat(fused.forward(a, b, c));

  const expected = flat(a.add(b).mul(c).relu());
  assertClose(actual, expected);
});

test("fused forward broadcasts inputs like eager Tensor ops", () => {
  const a = Tensor.from([1, 2, 3, 4]).reshape([2, 2]);
  const b = Tensor.from([10, 20]); // broadcasts along the trailing axis
  const fused = compile(2, (x, y) => x.mul(y).sigmoid());
  const actual = flat(fused.forward(a, b));
  const expected = flat(a.mul(b).sigmoid());
  assertClose(actual, expected);
});

test("every unary op agrees with its Tensor counterpart (or, for exp — which tensor-core doesn't expose — with plain Math.exp)", () => {
  const x = Tensor.from([0.25, 1.5, 2.0, 0.1]);
  const xs = flat(x);
  type TracedT = import("../src/index.ts").Traced;
  const cases: Array<[string, (t: TracedT) => TracedT, (v: number) => number]> = [
    ["neg", (t) => t.neg(), (v) => -v],
    ["relu", (t) => t.relu(), (v) => Math.max(0, v)],
    ["sigmoid", (t) => t.sigmoid(), (v) => 1 / (1 + Math.exp(-v))],
    ["exp", (t) => t.exp(), (v) => Math.exp(v)],
    ["log", (t) => t.log(), (v) => Math.log(v)],
    ["sqrt", (t) => t.sqrt(), (v) => Math.sqrt(v)],
  ];
  for (const [, traced, reference] of cases) {
    const fused = compile(1, (v) => traced(v));
    assertClose(flat(fused.forward(x)), xs.map(reference), 1e-4);
  }
  // gelu compared separately against tensor-core's own (already-verified) implementation.
  const fusedGelu = compile(1, (v) => v.gelu());
  assertClose(flat(fusedGelu.forward(x)), flat(x.gelu()), 1e-4);
});

test("asVariableOp gradient matches the equivalent unfused Variable graph", () => {
  const aT = Tensor.from([1, -2, 3, -4]);
  const bT = Tensor.from([0.5, 0.5, -6, 10]);
  const cT = Tensor.from([2, 2, 2, 2]);

  const fused = compile(3, (x, y, z) => x.add(y).mul(z).relu());
  const op = fused.asVariableOp();

  const a1 = Variable.variable(aT);
  const b1 = Variable.variable(bT);
  const c1 = Variable.variable(cT);
  const fusedOut = op(a1, b1, c1);
  fusedOut.sum().backward();

  const a2 = Variable.variable(aT);
  const b2 = Variable.variable(bT);
  const c2 = Variable.variable(cT);
  const unfusedOut = a2.add(b2).mul(c2).relu();
  unfusedOut.sum().backward();

  assertClose(flat(fusedOut.value), flat(unfusedOut.value));
  assertClose(flat(a1.grad as Tensor), flat(a2.grad as Tensor));
  assertClose(flat(b1.grad as Tensor), flat(b2.grad as Tensor));
  assertClose(flat(c1.grad as Tensor), flat(c2.grad as Tensor));
});

test("asVariableOp gradient matches under broadcasting", () => {
  const aT = Tensor.from([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  const bT = Tensor.from([10, 20, 30]); // broadcasts against the trailing axis

  const fused = compile(2, (x, y) => x.mul(y).sigmoid());
  const op = fused.asVariableOp();

  const a1 = Variable.variable(aT);
  const b1 = Variable.variable(bT);
  op(a1, b1).sum().backward();

  const a2 = Variable.variable(aT);
  const b2 = Variable.variable(bT);
  a2.mul(b2).sigmoid().sum().backward();

  assert.deepEqual([...(a1.grad as Tensor).shape], [...aT.shape]);
  assert.deepEqual([...(b1.grad as Tensor).shape], [...bT.shape]);
  assertClose(flat(a1.grad as Tensor), flat(a2.grad as Tensor));
  assertClose(flat(b1.grad as Tensor), flat(b2.grad as Tensor));
});

test("compile() is strictly opt-in: eager Tensor/Variable usage is unaffected whether or not it's imported/called", () => {
  const a = Tensor.from([1, 2, 3]);
  const b = Tensor.from([4, 5, 6]);
  const beforeAnyCompile = flat(a.add(b).relu());

  // Import and use compile() in between...
  const fused = compile(2, (x, y) => x.add(y).relu());
  fused.forward(a, b);

  const afterCompile = flat(a.add(b).relu());
  assertClose(beforeAnyCompile, afterCompile, 0); // bit-identical: eager path never touched by tracing

  const v1 = Variable.variable(a);
  const v2 = Variable.variable(b);
  const out = v1.add(v2).relu();
  out.sum().backward();
  assert.ok(v1.grad, "eager Variable autograd still works normally after compile() has been used elsewhere");
});

test("rejects a call with the wrong number of inputs", () => {
  const fused = compile(2, (x, y) => x.add(y));
  assert.throws(() => fused.forward(Tensor.from([1, 2])));
});

test("rejects mismatched dtypes", () => {
  const fused = compile(2, (x, y) => x.add(y));
  assert.throws(() =>
    fused.forward(Tensor.from([1, 2], { dtype: "f32" }), Tensor.from([1, 2], { dtype: "f64" })),
  );
});
