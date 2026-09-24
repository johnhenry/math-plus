/**
 * Tensor-level surface of the canonical erf/erfc/GELU, which lives in
 * @johnhenry/math-plus-special (its own oracle-free and SciPy tests are there).
 * tensor-core re-exports those functions and applies them elementwise; these
 * tests pin that the Tensor methods ARE those functions and the dtype/option
 * surface around them. PyTorch parity for Tensor.gelu is special-oracle.test.ts.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import * as special from "@johnhenry/math-plus-special";
import { erf, erfc, ERF_F32_PARAMS, geluErf, geluTanh, Tensor } from "../src/index.ts";

test("tensor-core re-exports the canonical functions from @johnhenry/math-plus-special (no second copy)", () => {
  assert.equal(erf, special.erf);
  assert.equal(erfc, special.erfc);
  assert.equal(geluErf, special.geluErf);
  assert.equal(ERF_F32_PARAMS, special.ERF_F32_PARAMS);
});

test("Tensor.erf()/erfc() are the canonical scalar functions applied elementwise", () => {
  const xs: number[] = [];
  for (let i = -30 * 16; i <= 30 * 16; i++) xs.push(i / 16);
  xs.push(1 - Number.EPSILON / 2, 1 + Number.EPSILON, 1e-300, -1e-300);
  const t = Tensor.from(xs, { dtype: "f64" });
  assert.deepEqual(Array.from(t.erf().data as Float64Array), xs.map(erf));
  assert.deepEqual(Array.from(t.erfc().data as Float64Array), xs.map(erfc));
});

test("Tensor.gelu option surface: default exact, explicit modes, validation, dtype guard", () => {
  const x = Tensor.from([-2, -0.5, 0, 0.5, 2], { dtype: "f64" });
  const data = (t: Tensor) => Array.from(t.data as Float64Array);
  assert.deepEqual(data(x.gelu()), [-2, -0.5, 0, 0.5, 2].map(geluErf));
  assert.deepEqual(data(x.gelu({ approximate: "none" })), [-2, -0.5, 0, 0.5, 2].map(geluErf));
  assert.deepEqual(data(x.gelu({ approximate: "tanh" })), [-2, -0.5, 0, 0.5, 2].map(geluTanh));
  // @ts-expect-error -- runtime validation for JS callers
  assert.throws(() => x.gelu({ approximate: "exact" }), /approximate must be "none" or "tanh"/);
  assert.throws(() => Tensor.from([1, 2], { dtype: "i64" }).gelu(), TypeError);
  assert.throws(() => Tensor.from([1, 2], { dtype: "i64" }).erf(), TypeError);
  // f32 in, f32 out.
  const f = Tensor.from([0.5, -1.5], { dtype: "f32" });
  assert.equal(f.gelu().dtype, "f32");
  assert.equal(f.erf().dtype, "f32");
  assert.equal(f.erfc().dtype, "f32");
});
