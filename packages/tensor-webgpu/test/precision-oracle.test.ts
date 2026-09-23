/**
 * Demonstrates `@johnhenry/math-plus-scalar-types`' `Interval` (bridged from @johnhenry/math
 * in issue #36) as an f32-vs-f64 rounding-error BOUND oracle -- a stronger
 * claim than the tolerance-based comparisons `fusion.test.ts` already does
 * ("these two numbers happen to be close"): propagate a per-step f32
 * relative-rounding-error bound through the SAME operation sequence
 * (`add -> mul -> sigmoid`, matching fusion.test.ts's own "a realistic fused
 * chain" case) using Interval's real arithmetic (`add`/`multiply`/`negate`/
 * `exp`/`divide`), then assert the ACTUAL GPU f32 result falls inside the
 * resulting interval, not just "close to" the f64 CPU value.
 *
 * The per-step widening (`f32Round`) is a standard first-order
 * error-propagation approximation, not a formally verified numerical bound:
 * each elementary f32 operation can round by up to half a ULP, i.e. a
 * relative error up to `2^-24`; widening by `2^-23` (one full ULP) per
 * step is a deliberately generous, conservative margin, not a tight one.
 */
import assert from "node:assert/strict";
import path from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Interval } from "@johnhenry/math-plus-scalar-types";
import { Traced } from "@johnhenry/math-plus-tensor-compile";
import { bundleForBrowser, closeHarness, getHarness, SRC } from "./helpers.ts";

after(closeHarness);

/** Conservative bound on f32's per-operation relative rounding error (one full ULP at the result's own magnitude). */
const F32_ULP = Math.pow(2, -23);

/** Widen `x` by one f32-rounding-error ULP in both directions, at `x`'s own current magnitude. */
function f32Round(x: Interval): Interval {
  const bound = Math.max(Math.abs(x.lo), Math.abs(x.hi)) * F32_ULP;
  return new Interval(x.lo - bound, x.hi + bound);
}

/** `(a + b) * c`, then `sigmoid`, propagated through Interval arithmetic with an f32 rounding bound injected after each elementary step -- mirrors `Traced.input(0).add(Traced.input(1)).mul(Traced.input(2)).sigmoid()`, the exact chain run on the real GPU below. */
function boundedFusedChain(a: number, b: number, c: number): Interval {
  const sum = f32Round(Interval.point(a).add(Interval.point(b)));
  const product = f32Round(sum.multiply(Interval.point(c)));
  // sigmoid(x) = 1 / (1 + exp(-x))
  const negProduct = product.negate();
  const expNeg = f32Round(negProduct.exp());
  const denom = f32Round(Interval.point(1).add(expNeg));
  const one = Interval.point(1);
  return f32Round(one.divide(denom));
}

test("Interval as a precision oracle: the real GPU f32 result of add->mul->sigmoid falls inside the propagated f32-rounding-error bound", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }

  const cases: Array<[number, number, number]> = [
    [0.5, -0.25, 2],
    [1.5, 1.5, -0.5],
    [-2, 0.75, 1.25],
  ];

  const expr = Traced.input(0).add(Traced.input(1)).mul(Traced.input(2)).sigmoid();
  const bundle = bundleForBrowser([path.join(SRC, "elementwise.ts")]);

  for (const [a, b, c] of cases) {
    const bound = boundedFusedChain(a, b, c);

    const result = await harness.run<number[]>(
      `
      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice();
      const node = ${JSON.stringify(expr.node)};
      const inputs = [new Float32Array([${a}]), new Float32Array([${b}]), new Float32Array([${c}])];
      const out = await runElementwiseWGSL(device, node, inputs, 1);
      return Array.from(out);
      `,
      bundle,
    );
    const gpuValue = result[0] as number;

    assert.ok(
      bound.contains(gpuValue),
      `(${a}+${b})*${c} -> sigmoid: GPU f32 result ${gpuValue} is outside the propagated bound [${bound.lo}, ${bound.hi}]`,
    );
    // Sanity: the bound should be a real, non-degenerate interval (proves the
    // widening actually did something), not a zero-width point masquerading
    // as a check that always trivially passes.
    assert.ok(bound.width > 0, `bound for (${a}, ${b}, ${c}) should have nonzero width`);
  }
});
