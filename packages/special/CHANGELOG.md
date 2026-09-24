# @johnhenry/math-plus-special

## 0.1.1

### Patch Changes

- d122aef: Published declarations no longer import `./x.ts` (closes #157). tsc's `rewriteRelativeImportExtensions` rewrites `.ts` specifiers to `.js` in emitted JS but not in emitted `.d.ts`, so `dist/*.d.ts` referenced files that aren't in the package, and Deno's type check failed on them. Every package's build now runs `scripts/rewrite-dts-extensions.mjs` after `tsc`:

  - relative `.ts` / `.mts` / `.cts` specifiers in `dist/**/*.d.ts` become `.js` / `.mjs` / `.cjs` (as in laya-js),
  - each `dist/*.js` with a declaration file starts with `// @ts-self-types="./x.d.ts"` (after the `#!` line of a bin), so Deno finds the types when it loads `dist/` as plain files or from a URL instead of through `npm:`. Source maps are shifted by the inserted line.

  No runtime change. The manifest drift test checks every built `dist/` for both.

## 0.1.0

### Minor Changes

- 739e3be: New zero-dependency package `@johnhenry/math-plus-special`: the canonical double-precision `erf`/`erfc`/GELU (issue #122), moved out of tensor-core so frame-arrow can depend on it without a static dependency on the tensor track. tensor-core now depends on it and re-exports every name unchanged, so its public API is unchanged. frame-arrow's `fn.erf()` now uses the canonical `erf` (~1e-15 relative) instead of its own Abramowitz & Stegun 7.1.26 copy (~1.5e-7 absolute error), so its results change in the 7th decimal place.
