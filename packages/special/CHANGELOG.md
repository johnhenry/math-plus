# @johnhenry/math-plus-special

## 0.1.0

### Minor Changes

- 739e3be: New zero-dependency package `@johnhenry/math-plus-special`: the canonical double-precision `erf`/`erfc`/GELU (issue #122), moved out of tensor-core so frame-arrow can depend on it without a static dependency on the tensor track. tensor-core now depends on it and re-exports every name unchanged, so its public API is unchanged. frame-arrow's `fn.erf()` now uses the canonical `erf` (~1e-15 relative) instead of its own Abramowitz & Stegun 7.1.26 copy (~1.5e-7 absolute error), so its results change in the 7th decimal place.
