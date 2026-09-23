/**
 * Issue #122: tensor-compile's `erf` and both GELU modes come from tensor-core's
 * canonical src/special.ts — so the compiled/traced path and eager `Tensor`
 * must agree BIT-FOR-BIT (same scalar functions, no second approximation),
 * and the fused gradient must match the eager `Variable.gelu()` backward in
 * both modes. Accuracy against SciPy/PyTorch is tensor-core's job
 * (tensor-core/test/special-oracle.test.ts); this file guards the wiring.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { Variable } from "@johnhenry/math-plus-tensor-autograd";
import { compile, Traced } from "../src/index.ts";

const XS = [-9, -4.5, -2, -1.4142135623730951, -1, -0.3, 0, 1e-8, 0.3, 1, 1.7, 3, 6.5];

function flat(t: Tensor): number[] {
  return Array.from(t.contiguous().data as Float64Array);
}

test("Traced.gelu() defaults to exact erf-GELU (IR op 'gelu'); { approximate: 'tanh' } selects 'gelu_tanh'", () => {
  const x = Traced.input(0);
  assert.deepEqual(x.gelu().node, { kind: "unary", op: "gelu", arg: { kind: "input", index: 0 } });
  assert.deepEqual(x.gelu({ approximate: "none" }).node, { kind: "unary", op: "gelu", arg: { kind: "input", index: 0 } });
  assert.deepEqual(x.gelu({ approximate: "tanh" }).node, { kind: "unary", op: "gelu_tanh", arg: { kind: "input", index: 0 } });
  // @ts-expect-error -- runtime validation for JS callers
  assert.throws(() => x.gelu({ approximate: "erf" }), /approximate must be "none" or "tanh"/);
});

test("compiled erf / gelu (both modes) are bit-identical to eager Tensor.erf() / Tensor.gelu()", () => {
  const t = Tensor.from(XS, { dtype: "f64" });
  assert.deepEqual(flat(compile(1, (v) => v.erf()).forward(t)), flat(t.erf()));
  assert.deepEqual(flat(compile(1, (v) => v.gelu()).forward(t)), flat(t.gelu()));
  assert.deepEqual(
    flat(compile(1, (v) => v.gelu({ approximate: "tanh" })).forward(t)),
    flat(t.gelu({ approximate: "tanh" })),
  );
});

test("fused gelu gradient matches eager Variable.gelu() backward, both modes", () => {
  for (const approximate of ["none", "tanh"] as const) {
    const xT = Tensor.from(XS, { dtype: "f64" });
    const fused = compile(1, (v) => v.gelu({ approximate })).asVariableOp();
    const a = Variable.variable(xT);
    fused(a).sum().backward();
    const b = Variable.variable(xT);
    b.gelu({ approximate }).sum().backward();
    const ga = flat(a.grad as Tensor);
    const gb = flat(b.grad as Tensor);
    for (let i = 0; i < XS.length; i++) {
      const diff = Math.abs(ga[i]! - gb[i]!);
      assert.ok(diff <= 1e-12 * Math.max(1, Math.abs(gb[i]!)), `${approximate} x=${XS[i]}: fused ${ga[i]} vs eager ${gb[i]}`);
    }
  }
});

test("the default and tanh GELU genuinely differ (the #122 behaviour change is real, not a relabel)", () => {
  const t = Tensor.from([1], { dtype: "f64" });
  const exact = flat(compile(1, (v) => v.gelu()).forward(t))[0]!;
  const tanh = flat(compile(1, (v) => v.gelu({ approximate: "tanh" })).forward(t))[0]!;
  assert.ok(Math.abs(exact - tanh) > 1e-5, `exact ${exact} vs tanh ${tanh}`);
});
