/**
 * Validates the DualNumber oracle plumbing itself (issue #17) — against
 * finite differences first (cheap, no other package needed), then wired to
 * tensor-autograd's reverse-mode tape (the actual consumer this oracle
 * exists for: three independent gradient computations — reverse tape,
 * forward-mode dual numbers, finite differences — agreeing on the same
 * function is much stronger evidence than any one of them alone).
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { DualNumber } from "@johnhenry/math";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { grad, Variable } from "@johnhenry/math-plus-tensor-autograd";
import { dualGrad, dualGradN, type ScalarDualFn } from "../src/test-utils.ts";

function finiteDifference(fn: (x: number) => number, x: number, h = 1e-6): number {
  return (fn(x + h) - fn(x - h)) / (2 * h);
}

const CASES: Array<{ name: string; dual: ScalarDualFn; plain: (x: number) => number; points: number[] }> = [
  {
    name: "x^2 + 3x",
    dual: (x) => x.multiply(x).add(x.multiply(DualNumber.constant(3))),
    plain: (x) => x * x + 3 * x,
    points: [-2, 0, 1.5, 10],
  },
  {
    name: "sin(x)",
    dual: (x) => DualNumber.sin(x),
    plain: Math.sin,
    points: [0, 0.5, Math.PI / 2, 3],
  },
  {
    name: "exp(x) * cos(x)",
    dual: (x) => DualNumber.exp(x).multiply(DualNumber.cos(x)),
    plain: (x) => Math.exp(x) * Math.cos(x),
    points: [-1, 0, 1, 2],
  },
  {
    name: "sqrt(x^2 + 1)",
    dual: (x) => DualNumber.sqrt(x.multiply(x).add(DualNumber.constant(1))),
    plain: (x) => Math.sqrt(x * x + 1),
    points: [-3, 0, 0.25, 5],
  },
];

for (const { name, dual, plain, points } of CASES) {
  test(`dualGrad matches finite differences: ${name}`, () => {
    for (const x of points) {
      const forward = dualGrad(dual, x);
      const numeric = finiteDifference(plain, x);
      const bound = 1e-5 + 1e-5 * Math.abs(numeric);
      assert.ok(Math.abs(forward - numeric) <= bound, `${name} at x=${x}: dual ${forward} vs finite-diff ${numeric}`);
    }
  });
}

test("dualGradN: multivariate gradient via @johnhenry/math's gradient driver", () => {
  // f(x, y) = x^2*y + y^3 ; df/dx = 2xy ; df/dy = x^2 + 3y^2
  const g = dualGradN((xs) => {
    const [x, y] = xs as [DualNumber, DualNumber];
    return x.multiply(x).multiply(y).add(y.multiply(y).multiply(y));
  }, [2, 3]);
  assert.ok(Math.abs((g[0] as number) - 12) < 1e-9, `df/dx: ${g[0]}`);
  assert.ok(Math.abs((g[1] as number) - 31) < 1e-9, `df/dy: ${g[1]}`);
});

test("wired to tensor-autograd: dualGrad agrees with Variable's reverse-mode tape — two independent gradient algorithms, same answer", () => {
  // f(x) = sigmoid(x)*x + x^2, same composition tensor-autograd's own
  // internal DualNumber cross-check uses (test/variable.test.ts) — this
  // test demonstrates the "ships as test-utils, wired to tensor-autograd"
  // half of issue #17, in the only direction the architecture allows
  // (adapter -> core, never core -> adapter).
  const dualSigmoid = (d: DualNumber): DualNumber =>
    DualNumber.constant(1).divide(DualNumber.constant(1).add(DualNumber.exp(d.negate())));

  const tapeFn = (v: Variable) => v.sigmoid().mul(v).add(v.mul(v)).sum();
  const dualFn: ScalarDualFn = (d) => dualSigmoid(d).multiply(d).add(d.multiply(d));

  for (const x0 of [-2, -0.5, 0.3, 1.7, 3]) {
    const tape = grad.of(tapeFn)(Tensor.from([x0], { dtype: "f64" })).item() as number;
    const dual = dualGrad(dualFn, x0);
    assert.ok(Math.abs(tape - dual) < 1e-6, `tape grad ${tape} vs dualGrad ${dual} at x=${x0}`);
  }
});

test("wired to tensor-autograd, multivariate: dualGradN agrees with grad.of on a two-input function", () => {
  // f(x, y) = (x*y + y^2).sum()
  const tapeFn = (x: Variable, y: Variable) => x.mul(y).add(y.mul(y)).sum();
  const points: Array<[number, number]> = [
    [1.3, -0.7],
    [-2, 3.1],
  ];
  for (const [x0, y0] of points) {
    const xT = Variable.variable(Tensor.from([x0], { dtype: "f64" }));
    const yT = Variable.variable(Tensor.from([y0], { dtype: "f64" }));
    tapeFn(xT, yT).backward();
    const tapeGrad = [xT.grad?.item() as number, yT.grad?.item() as number];

    const dual = dualGradN((xs) => {
      const [x, y] = xs as [DualNumber, DualNumber];
      return x.multiply(y).add(y.multiply(y));
    }, [x0, y0]);

    assert.ok(Math.abs(tapeGrad[0] - (dual[0] as number)) < 1e-6, `dx at (${x0},${y0})`);
    assert.ok(Math.abs(tapeGrad[1] - (dual[1] as number)) < 1e-6, `dy at (${x0},${y0})`);
  }
});
