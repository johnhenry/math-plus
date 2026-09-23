/**
 * Scalar `erf`/`erfc` for exact (non-tanh) GELU (issue #123).
 *
 * OVERLAP NOTE: issue #122 is adding a canonical `erf` (and exact GELU) to
 * `@johnhenry/math-plus-tensor-core`. This file exists only because that
 * work had not landed when #123 needed exact GELU; once tensor-core ships
 * `Tensor.erf()`, delete this file and route `Variable.gelu({ approximate:
 * "none" })` through it (AGENTS.md's canonical-implementation rule). It is
 * deliberately NOT exported from the package's public entry point, so
 * removing it is not a breaking change.
 *
 * Accuracy (checked against Python's `math.erf`/`math.erfc` in
 * test/erf.test.ts): erf ~1e-15 relative everywhere; erfc ~1e-16 ABSOLUTE
 * (relative error grows to ~1e-12 just below x = 2.5, where it is computed
 * as 1 - erf) — ample for GELU, whose error is absolute in x * erfc.
 *   - |x| < 2.5: the all-positive-terms series
 *     erf(x) = 2/sqrt(pi) * exp(-x^2) * sum_n 2^n x^(2n+1) / (1*3*...*(2n+1))
 *     (no alternating-sign cancellation, unlike the Maclaurin series).
 *   - x >= 2.5: erfc via its classic continued fraction
 *     erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...)))),
 *     evaluated with modified Lentz.
 */
const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI);
const ONE_OVER_SQRT_PI = 1 / Math.sqrt(Math.PI);
const SERIES_LIMIT = 2.5;

function erfSeries(x: number): number {
  // x >= 0 and < SERIES_LIMIT
  const x2 = x * x;
  let term = x;
  let sum = x;
  for (let n = 1; n < 500; n++) {
    term *= (2 * x2) / (2 * n + 1);
    sum += term;
    if (term <= sum * 1e-17) break;
  }
  return TWO_OVER_SQRT_PI * Math.exp(-x2) * sum;
}

function erfcContinuedFraction(x: number): number {
  // x >= SERIES_LIMIT. f = x + a1/(x + a2/(x + ...)), a_n = n/2; modified Lentz.
  const tiny = 1e-300;
  let f = x;
  let c = x;
  let d = 0;
  for (let n = 1; n < 1000; n++) {
    const a = n / 2;
    d = x + a * d;
    if (d === 0) d = tiny;
    c = x + a / c;
    if (c === 0) c = tiny;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return (ONE_OVER_SQRT_PI * Math.exp(-x * x)) / f;
}

export function erf(x: number): number {
  if (Number.isNaN(x)) return Number.NaN;
  const ax = Math.abs(x);
  const r = ax < SERIES_LIMIT ? erfSeries(ax) : 1 - erfcContinuedFraction(ax);
  return x < 0 ? -r : r;
}

export function erfc(x: number): number {
  if (Number.isNaN(x)) return Number.NaN;
  if (x < 0) return 2 - erfc(-x);
  return x < SERIES_LIMIT ? 1 - erfSeries(x) : erfcContinuedFraction(x);
}

const SQRT1_2 = Math.SQRT1_2;
const ONE_OVER_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

/** Exact GELU `x * Phi(x)`, written as `0.5 x erfc(-x/sqrt2)` so the negative tail doesn't cancel. */
export function geluExact(x: number): number {
  return 0.5 * x * erfc(-x * SQRT1_2);
}

/** d/dx of exact GELU: `Phi(x) + x * phi(x)`. */
export function geluExactDerivative(x: number): number {
  return 0.5 * erfc(-x * SQRT1_2) + x * ONE_OVER_SQRT_2PI * Math.exp(-0.5 * x * x);
}
