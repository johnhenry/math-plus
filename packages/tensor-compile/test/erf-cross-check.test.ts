/**
 * Differential test (issue #34, cross-repo interop investigation): does
 * tensor-compile's own `erf` (Abramowitz & Stegun 7.1.26, `|error| <= 1.5e-7`
 * per its own doc comment in src/ir.ts) actually agree with @johnhenry/math's
 * `SpecialFunctions.erf` (an independently-written, separately-sourced
 * implementation, in the sibling johnhenry/math repo)?
 *
 * `@johnhenry/math` is a devDependency ONLY here -- tensor-compile's own
 * shipped runtime dependency graph is unchanged (see ir.ts's own doc
 * comment: "tensor-compile stays dependency-free of @johnhenry/math"). This
 * test exists purely to build confidence that the two independently-sourced
 * approximations actually agree, not to introduce a real coupling.
 *
 * Covers tensor-webgpu's WGSL `erf` too, without needing a GPU: its
 * `math_plus_erf` (packages/tensor-webgpu/src/fusion-wgsl.ts) is the exact
 * same formula as this one, generated from the same source per that file's
 * own doc comment ("same math, different backend") -- so a mismatch here
 * would mean a mismatch there too, and an agreement here transfers.
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
  // Both implementations are approximations with their own stated error
  // bounds -- @johnhenry/math's SpecialFunctions.erf doesn't document a bound
  // as explicitly as tensor-compile's Abramowitz & Stegun comment does, but
  // tensor-compile's OWN 1.5e-7 bound already dominates any tolerance tight
  // enough to matter for elementwise-tensor use; 1e-6 gives headroom for
  // floating-point accumulation differences between the two call paths
  // (@johnhenry/math's own polynomial/series approach vs Abramowitz & Stegun)
  // without hiding a real disagreement.
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
