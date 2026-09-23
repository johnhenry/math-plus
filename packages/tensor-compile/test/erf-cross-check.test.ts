/**
 * Differential test (issue #34, cross-repo interop investigation; updated for
 * #122): does tensor-compile's `erf` agree with @johnhenry/math's
 * `SpecialFunctions.erf` (an independently-written, separately-sourced
 * implementation, in the sibling johnhenry/math repo)?
 *
 * Since #122 tensor-compile's `erf` IS tensor-core's canonical double-precision
 * `erf` (src/special.ts, ~1e-15 relative, SciPy-verified in
 * tensor-core/test/special-oracle.test.ts) rather than its own Abramowitz &
 * Stegun 7.1.26 copy. `SpecialFunctions.erf` is the LESS accurate side of
 * this comparison (measured max |diff| ≈ 1.4e-7 over [-6, 6], near x ≈ 0.5),
 * so the 1e-6 tolerance below is bounding @johnhenry/math's own error, and
 * this file is now an interop sanity check, not the accuracy oracle.
 *
 * `@johnhenry/math` is a devDependency ONLY here -- tensor-compile's own
 * shipped runtime dependency graph is unchanged (see ir.ts's own doc
 * comment: "tensor-compile stays dependency-free of @johnhenry/math").
 *
 * tensor-webgpu's WGSL `math_plus_erf` is an f32 lowering of the same
 * canonical algorithm; it is verified against the canonical f64 `erf` on a
 * live adapter in packages/tensor-webgpu/test/fusion.test.ts.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { SpecialFunctions } from "@johnhenry/math";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { compile, type Traced } from "../src/index.ts";

function tensorCompileErf(x: number): number {
  const f = compile(1, (v: Traced) => v.erf());
  return f.forward(Tensor.from([x])).toArray()[0] as number;
}

test("tensor-compile's erf matches @johnhenry/math's SpecialFunctions.erf within a documented tolerance", () => {
  // @johnhenry/math's SpecialFunctions.erf doesn't document a bound; its
  // measured error against the canonical erf is ~1.4e-7 (see header), so
  // 1e-6 leaves headroom without hiding a real disagreement.
  const TOLERANCE = 1e-6;
  const xs = [
    -4, -3, -2.5, -2, -1.5, -1, -0.75, -0.5, -0.25, -0.1, -0.01, 0, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4,
  ];
  for (const x of xs) {
    const ours = tensorCompileErf(x);
    const reference = SpecialFunctions.erf(x);
    assert.ok(
      Math.abs(ours - reference) < TOLERANCE,
      `erf(${x}): tensor-compile=${ours} @johnhenry/math=${reference} diff=${Math.abs(ours - reference)}`,
    );
  }
});

test("tensor-compile's erf is odd (erf(-x) === -erf(x)) matching @johnhenry/math's own value at the same points", () => {
  for (const x of [0.3, 1.1, 2.7]) {
    assert.ok(Math.abs(tensorCompileErf(-x) + tensorCompileErf(x)) < 1e-9, `erf should be odd at x=${x}`);
    assert.ok(Math.abs(SpecialFunctions.erf(-x) + SpecialFunctions.erf(x)) < 1e-9, `@johnhenry/math erf should be odd at x=${x}`);
  }
});

test("both erf implementations approach +-1 in the tails, in agreement with each other", () => {
  for (const x of [5, 6, -5, -6]) {
    const ours = tensorCompileErf(x);
    const reference = SpecialFunctions.erf(x);
    const expectedSign = Math.sign(x);
    assert.ok(Math.abs(ours - expectedSign) < 1e-5, `tensor-compile erf(${x})=${ours} should be near ${expectedSign}`);
    assert.ok(Math.abs(reference - expectedSign) < 1e-9, `@johnhenry/math erf(${x})=${reference} should be near ${expectedSign}`);
    assert.ok(Math.abs(ours - reference) < 1e-5, `erf(${x}): tensor-compile and @johnhenry/math disagree in the tail`);
  }
});
