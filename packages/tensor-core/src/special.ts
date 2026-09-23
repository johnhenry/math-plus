/**
 * The ONE canonical `erf`/`erfc`/GELU implementation for Math Plus (issue #122).
 *
 * Every other `erf` in this monorepo derives from this module rather than
 * carrying its own approximation (AGENTS.md's canonical-implementation rule):
 *
 * - `Tensor.erf()`/`Tensor.erfc()`/`Tensor.gelu()` (tensor-core) call it directly.
 * - `@johnhenry/math-plus-tensor-compile`'s IR evaluator (`ir.ts`,
 *   `unaryValueAndDeriv`'s `erf`/`gelu`/`gelu_tanh` cases) imports it.
 * - `@johnhenry/math-plus-tensor-autograd`'s `Variable.gelu()` backward uses
 *   `Tensor.erfc()` for the exact-GELU derivative.
 * - `@johnhenry/math-plus-tensor-webgpu`'s WGSL `math_plus_erf`/`math_plus_erfc`
 *   (`fusion-wgsl.ts`) lower THIS algorithm (same regions, same series, same
 *   continued fraction) to f32, truncated with {@link ERF_F32_PARAMS} — whose
 *   truncation error is itself verified against this f64 implementation
 *   (test/special.test.ts), so the f32 variant is derived from, not
 *   independent of, this one.
 *
 * Why tensor-core: it is the root of the tensor graph (zero runtime
 * dependencies), and every consumer above already depends on it — so no
 * package gains a new dependency edge. `@johnhenry/math-plus-scalar-types`
 * was the other candidate, but it pulls in `@johnhenry/math`, which
 * tensor-compile/tensor-webgpu deliberately do not depend on.
 *
 * Algorithm (derived from laya-js's `packages/backend-cpu/src/erf.ts`, same
 * author, Apache-2.0; tightened here — see "Accuracy"):
 *
 * - `|x| < 1`: Maclaurin series `erf(x) = 2/√π · Σ (-1)^n x^(2n+1) / (n!(2n+1))`
 *   (≤ ~20 terms; alternating but with no significant cancellation below 1).
 * - `|x| >= 1`: `erfc(|x|)` from the even contraction of Laplace's continued
 *   fraction, evaluated backward at a fixed, `z`-dependent depth, so the
 *   tail has no `1 - erf` cancellation. `exp(-z²)` is computed as
 *   `exp(-s²)·exp(-(z-s)(z+s))` with `s` = `z` rounded to 12 fractional
 *   bits (so `s²` is exact) — the fdlibm trick that keeps the rounding of
 *   `z²` from costing `~2z²` ulps of relative error in the tail.
 * - `erf = 1 - erfc` for `|x| >= 1` (no cancellation: `erfc(1) ≈ 0.157`).
 *   `erfc = 1 - erf` for `-1 < x < 1` (at most ~3 bits lost, near x = 1).
 *
 * Accuracy (verified against SciPy's `special.erf`/`special.erfc` over
 * [-6, 6] plus both tails in test/special-oracle.test.ts, and against 50-digit
 * mpmath during development): `erf` ≤ 7e-16 relative everywhere; `erfc`
 * ≤ 3.5e-15 relative wherever the result is a normal f64 (it then underflows
 * gradually through the subnormals to 0 at x ≈ 27.3); exact GELU keeps
 * ~1e-15 relative accuracy all the way down its left tail (x ≈ -38.5),
 * where the textbook `0.5·x·(1 + erf(x/√2))` — and PyTorch — cancel to 0
 * or lose digits.
 * The previous Abramowitz & Stegun 7.1.26 formula this replaces had
 * ~1.5e-7 ABSOLUTE error, i.e. no correct digits of `erfc` beyond x ≈ 4.
 *
 * Out of scope (disclosed, not silently missing): complex arguments,
 * `erfinv`/`erfcinv`, scaled `erfcx`. `@johnhenry/math`'s own
 * `SpecialFunctions.erf` (separate repo) is not replaced by this module;
 * tensor-compile keeps a cross-check test against it.
 */

const TWO_OVER_SQRT_PI = 1.1283791670955126; // 2/√π
const INV_SQRT_PI = 0.5641895835477563; // 1/√π
const INV_SQRT2 = 0.7071067811865476; // 1/√2
const INV_SQRT_2PI = 0.3989422804014327; // 1/√(2π)
const SQRT_2_OVER_PI = 0.7978845608028654; // √(2/π) — tanh-GELU constant
const GELU_TANH_COEFF = 0.044715;

/** Below this `|x|` erf uses the Maclaurin series; at or above it, the erfc continued fraction. */
export const ERF_SERIES_CUTOFF = 1;

/** `erfc(z)` is exactly 0 in f64 above this (the true value is below the smallest subnormal). */
const ERFC_UNDERFLOW = 27.3;

/**
 * Truncation parameters for the f32 lowering of this algorithm (tensor-webgpu's
 * WGSL `math_plus_erf`/`math_plus_erfc`). `seriesTerms` Maclaurin terms for
 * `|x| < ERF_SERIES_CUTOFF`, a fixed `cfDepth`-deep continued fraction above
 * it, and `erfc` flushed to 0 above `underflow` (erfc(10.1) ≈ 2.8e-46 is
 * already below half of f32's smallest subnormal, ~1.4e-45, so it rounds to 0). Chosen so the TRUNCATION error alone
 * is < 2^-24 (f32 half-ulp) relative — asserted by test/special.test.ts by
 * running this module's own f64 `erfSeries`/`erfcContinuedFraction` at
 * exactly these parameters.
 */
export const ERF_F32_PARAMS = Object.freeze({ seriesTerms: 10, cfDepth: 28, underflow: 10.1 });

/**
 * Maclaurin-series `erf(x)`; accurate for `|x| < ERF_SERIES_CUTOFF`. `maxTerms`
 * caps the number of terms after the leading `x` (exported only so the f32
 * lowering's truncation can be verified against the f64 original).
 */
export function erfSeries(x: number, maxTerms = 60): number {
  if (x === 0) return x; // keep erf(-0) === -0
  const x2 = x * x;
  let term = x;
  let sum = x;
  for (let n = 1; n <= maxTerms; n++) {
    term *= -x2 / n;
    const c = term / (2 * n + 1);
    sum += c;
    if (Math.abs(c) <= 1e-17 * Math.abs(sum)) break;
  }
  return TWO_OVER_SQRT_PI * sum;
}

/** Continued-fraction depth that converges to f64 precision at `z` (measured against 40-digit mpmath). */
function cfDepthFor(z: number): number {
  if (z < 1.5) return 100;
  if (z < 2) return 60;
  if (z < 3) return 36;
  return 24;
}

/** `exp(-z²)` without the `~2z²`-ulp relative error of rounding `z*z` first (fdlibm's split). */
function expNegSquare(z: number): number {
  const s = Math.round(z * 4096) / 4096; // ≤ 12 fractional bits: s*s is exact for |z| < 2^20
  return Math.exp(-s * s) * Math.exp(-(z - s) * (z + s));
}

/** `exp(-x²/2)` with the same split — lets GELU/Φ avoid rounding `x/√2` inside the exponent. */
function expNegHalfSquare(x: number): number {
  const s = Math.round(x * 4096) / 4096;
  return Math.exp(-0.5 * (s * s)) * Math.exp(-0.5 * (x - s) * (x + s));
}

/** The continued fraction proper, given `expTerm = exp(-z²)` computed by the caller. */
function erfcTail(z: number, expTerm: number, depth: number): number {
  const t = 2 * z * z + 1;
  let f = 0;
  for (let n = depth; n >= 1; n--) {
    f = ((2 * n - 1) * (2 * n)) / (t + 4 * n - f);
  }
  return (expTerm * INV_SQRT_PI * 2 * z) / (t - f);
}

/**
 * `erfc(z)` for `z >= ERF_SERIES_CUTOFF` via the even contraction of Laplace's
 * continued fraction:
 * `erfc(z) = exp(-z²)/√π · 2z / (t - 1·2/(t+4 - 3·4/(t+8 - 5·6/(t+12 - ...))))`,
 * `t = 2z² + 1`, evaluated backward from `depth`. `depth` defaults to a
 * `z`-dependent value that reaches f64 precision.
 */
export function erfcContinuedFraction(z: number, depth = cfDepthFor(z)): number {
  if (z > ERFC_UNDERFLOW) return 0;
  return erfcTail(z, expNegSquare(z), depth);
}

/**
 * Standard normal CDF `Φ(x) = 0.5·erfc(-x/√2)`. In the left tail the
 * exponent is taken from `x` directly (`exp(-x²/2)`), so the rounding of
 * `x/√2` only perturbs the slowly-varying continued-fraction factor, not the
 * exponential — full relative accuracy down to underflow.
 */
function normalCdf(x: number): number {
  const z = -x * INV_SQRT2;
  if (z >= ERF_SERIES_CUTOFF) {
    if (z > ERFC_UNDERFLOW) return 0;
    return 0.5 * erfcTail(z, expNegHalfSquare(x), cfDepthFor(z));
  }
  return 0.5 * erfc(z);
}

/** The error function `erf(x) = 2/√π ∫₀ˣ e^(-t²) dt`, to ~1e-15 relative. `erf(±∞) = ±1`, `erf(NaN) = NaN`. */
export function erf(x: number): number {
  if (x !== x) return NaN;
  const ax = Math.abs(x);
  if (ax < ERF_SERIES_CUTOFF) return erfSeries(x);
  const c = erfcContinuedFraction(ax);
  return x > 0 ? 1 - c : c - 1;
}

/** The complementary error function `erfc(x) = 1 - erf(x)`, to ~3.5e-15 relative (no cancellation in the right tail). */
export function erfc(x: number): number {
  if (x !== x) return NaN;
  if (x >= ERF_SERIES_CUTOFF) return erfcContinuedFraction(x);
  if (x <= -ERF_SERIES_CUTOFF) return 2 - erfcContinuedFraction(-x);
  return 1 - erfSeries(x);
}

/** Which GELU formula: `"none"` = exact `x·Φ(x)` (PyTorch's default), `"tanh"` = the tanh approximation. Same names as `torch.nn.functional.gelu(approximate=...)`. */
export type GeluApproximate = "none" | "tanh";

/**
 * Exact GELU `x·Φ(x) = 0.5·x·(1 + erf(x/√2))`, computed as `0.5·x·erfc(-x/√2)`
 * (see `normalCdf`) so it keeps full RELATIVE accuracy for large negative `x`,
 * where `1 + erf(...)` would cancel to 0 long before the true value underflows.
 */
export function geluErf(x: number): number {
  return x * normalCdf(x);
}

/** Tanh-approximation GELU `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` (PyTorch's `approximate="tanh"`). */
export function geluTanh(x: number): number {
  return 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + GELU_TANH_COEFF * x * x * x)));
}

/** Scalar GELU in either mode; `approximate` defaults to `"none"` (exact erf), matching PyTorch. */
export function gelu(x: number, approximate: GeluApproximate = "none"): number {
  return approximate === "tanh" ? geluTanh(x) : geluErf(x);
}

/**
 * `d/dx gelu(x)` for either mode. Exact: `Φ(x) + x·φ(x)`. Tanh: the exact
 * derivative of the tanh approximation (NOT of exact GELU) — the gradient of
 * whatever forward the caller actually computed.
 */
export function geluDerivative(x: number, approximate: GeluApproximate = "none"): number {
  if (approximate === "tanh") {
    const t = Math.tanh(SQRT_2_OVER_PI * (x + GELU_TANH_COEFF * x * x * x));
    const dInner = SQRT_2_OVER_PI * (1 + 3 * GELU_TANH_COEFF * x * x);
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dInner;
  }
  return normalCdf(x) + x * INV_SQRT_2PI * expNegHalfSquare(x);
}

/** Validates a user-supplied `approximate` option (JS callers can pass anything). */
export function checkGeluApproximate(approximate: unknown): GeluApproximate {
  if (approximate === undefined) return "none";
  if (approximate === "none" || approximate === "tanh") return approximate;
  throw new TypeError(`gelu: approximate must be "none" or "tanh", got ${JSON.stringify(approximate)}`);
}
