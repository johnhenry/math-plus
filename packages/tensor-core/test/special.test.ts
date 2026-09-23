/**
 * Oracle-free properties of the canonical erf/erfc/GELU (src/special.ts, #122):
 * special values, symmetry identities, the f32-lowering truncation bound that
 * tensor-webgpu's WGSL relies on, the gelu option surface, and the derivative.
 * Accuracy against SciPy/PyTorch lives in special-oracle.test.ts.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import {
  ERF_F32_PARAMS,
  ERF_SERIES_CUTOFF,
  erf,
  erfc,
  erfcContinuedFraction,
  erfSeries,
  gelu,
  geluDerivative,
  geluErf,
  geluTanh,
  Tensor,
} from "../src/index.ts";

test("erf/erfc special values", () => {
  assert.ok(Object.is(erf(0), 0));
  assert.ok(Object.is(erf(-0), -0));
  assert.equal(erfc(0), 1);
  assert.equal(erf(Infinity), 1);
  assert.equal(erf(-Infinity), -1);
  assert.equal(erfc(Infinity), 0);
  assert.equal(erfc(-Infinity), 2);
  assert.ok(Number.isNaN(erf(NaN)));
  assert.ok(Number.isNaN(erfc(NaN)));
  assert.equal(erfc(28), 0); // true value ~6e-343, below the smallest subnormal
  assert.ok(erfc(26) > 0);
  // Tiny x: erf(x) = 2x/√π to first order, no underflow to 0.
  assert.equal(erf(1e-300), 1.1283791670955126e-300);
});

test("erf is odd and erf + erfc = 1 across the series/continued-fraction seam", () => {
  for (let x = -8; x <= 8; x += 0.0625) {
    assert.ok(Object.is(erf(-x), -erf(x)) || erf(x) === 0, `erf odd at ${x}`);
    assert.ok(Math.abs(erf(x) + erfc(x) - 1) <= 2 * Number.EPSILON, `erf + erfc at ${x}`);
    assert.ok(Math.abs(erfc(-x) - (2 - erfc(x))) <= 2 * Number.EPSILON, `erfc reflection at ${x}`);
  }
  // Continuity at the |x| = 1 switch between algorithms.
  const below = erf(ERF_SERIES_CUTOFF - Number.EPSILON / 2);
  const at = erf(ERF_SERIES_CUTOFF);
  assert.ok(Math.abs(at - below) < 4 * Number.EPSILON, `seam jump ${at - below}`);
  assert.ok(erf(ERF_SERIES_CUTOFF) >= below);
});

test("erf is monotonically non-decreasing on a fine grid", () => {
  let prev = -1;
  for (let x = -7; x <= 7; x += 1 / 512) {
    const v = erf(x);
    assert.ok(v >= prev, `erf decreased at ${x}: ${prev} -> ${v}`);
    prev = v;
  }
});

test("ERF_F32_PARAMS (the WGSL f32 lowering) truncates the canonical algorithm by < 2^-24 relative", () => {
  const halfUlpF32 = 2 ** -24;
  // Series leg: |x| < cutoff (tiny |x| excluded -- fewer terms only help there).
  for (let x = 1e-6; x < ERF_SERIES_CUTOFF; x += 1 / 1024) {
    const rel = Math.abs(erfSeries(x, ERF_F32_PARAMS.seriesTerms) - erf(x)) / erf(x);
    assert.ok(rel < halfUlpF32, `series truncation at ${x}: ${rel}`);
  }
  // Continued-fraction leg: cutoff <= z <= f32 underflow point.
  for (let z = ERF_SERIES_CUTOFF; z <= ERF_F32_PARAMS.underflow; z += 1 / 256) {
    const rel = Math.abs(erfcContinuedFraction(z, ERF_F32_PARAMS.cfDepth) - erfc(z)) / erfc(z);
    assert.ok(rel < halfUlpF32, `continued-fraction truncation at ${z}: ${rel}`);
  }
  // From `underflow` on, erfc rounds to 0 in f32 anyway -- flushing loses nothing.
  assert.equal(Math.fround(erfc(ERF_F32_PARAMS.underflow)), 0);
});

test("gelu: default is exact ('none'), 'tanh' is the tanh approximation, and they differ", () => {
  for (const x of [-3, -1, -0.5, 0.25, 1, 2.5]) {
    assert.equal(gelu(x), geluErf(x));
    assert.equal(gelu(x, "none"), geluErf(x));
    assert.equal(gelu(x, "tanh"), geluTanh(x));
    // exact GELU is x·Φ(x); cross-check against erf directly away from cancellation.
    assert.ok(Math.abs(geluErf(x) - 0.5 * x * (1 + erf(x / Math.SQRT2))) <= 1e-15 * Math.max(1, Math.abs(x)));
  }
  // The two formulas genuinely differ (max gap ~4.7e-4 near |x| ≈ 2.7).
  let maxGap = 0;
  for (let x = -6; x <= 6; x += 1 / 256) maxGap = Math.max(maxGap, Math.abs(geluErf(x) - geluTanh(x)));
  assert.ok(maxGap > 1e-4 && maxGap < 1e-3, `max |exact - tanh| = ${maxGap}`);
});

test("exact gelu keeps relative accuracy in the far-left tail instead of cancelling to 0", () => {
  // 0.5·x·(1 + erf(x/√2)) is exactly 0 here in f64; the true value is ~ -6e-89.
  assert.equal(0.5 * -20 * (1 + erf(-20 / Math.SQRT2)), -0);
  const v = geluErf(-20);
  assert.ok(v < 0 && v > -1e-87 && v < -1e-89, `gelu(-20) = ${v}`);
  assert.ok(Object.is(geluErf(-40), -0));
});

test("geluDerivative matches central finite differences in both modes", () => {
  for (const approximate of ["none", "tanh"] as const) {
    for (let x = -6; x <= 6; x += 0.37) {
      const h = 1e-6;
      const fd = (gelu(x + h, approximate) - gelu(x - h, approximate)) / (2 * h);
      const d = geluDerivative(x, approximate);
      assert.ok(Math.abs(fd - d) < 1e-8, `${approximate} d/dx at ${x}: analytic ${d} vs fd ${fd}`);
    }
  }
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
