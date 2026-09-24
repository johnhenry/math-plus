# @johnhenry/math-plus-frame-arrow

## 0.0.4

### Patch Changes

- 71fa3f0: Widen the optional tensor-core peer range to include `^0.2.0`, so tensor-core's 0.2.0 release doesn't force a major bump on its peer dependents.

## 0.0.3

### Patch Changes

- 739e3be: New zero-dependency package `@johnhenry/math-plus-special`: the canonical double-precision `erf`/`erfc`/GELU (issue #122), moved out of tensor-core so frame-arrow can depend on it without a static dependency on the tensor track. tensor-core now depends on it and re-exports every name unchanged, so its public API is unchanged. frame-arrow's `fn.erf()` now uses the canonical `erf` (~1e-15 relative) instead of its own Abramowitz & Stegun 7.1.26 copy (~1.5e-7 absolute error), so its results change in the 7th decimal place.
- Updated dependencies [739e3be]
  - @johnhenry/math-plus-special@0.1.0

## 0.0.2

### Patch Changes

- 80123df: Republish with an npm provenance attestation. The previous versions were published from a local machine on 2026-09-23 without provenance (the CI `NPM_TOKEN` secret had expired). No code changes.

## 0.0.1

### Patch Changes

- 5d7172b: Lower `engines.node` from `>=26.0.0` to `>=24.0.0`. Nothing in these packages needs Node 26: the full test suite passes on Node 24.9, and CI now tests Node 24. Every suite also runs under Bun 1.2.17 (`npm run test:bun`).

  `@johnhenry/math-plus-frame-parquet`: `scanParquet`/`scanParquetLazy` now accept an absolute file path without wildcards under Bun. Bun 1.2's `fs.promises.glob` returns no matches for such a path, so a pattern with no glob metacharacters is now resolved with `stat` on every runtime.

- e9b691d: Widen internal peer-dependency ranges to `^0.0.0 || ^0.1.0` so the 0.1.0 releases of tensor-core and safetensors stay in range (Changesets would otherwise force a major bump on every peer dependent).

## 1.0.0

### Patch Changes

- Updated dependencies [262a154]
  - @johnhenry/math-plus-tensor-core@0.2.0

## 0.1.0

### Minor Changes

- 7b0ced4: Fixes johnhenry/math-plus#86: `Frame.fromCSV()`, the reader counterpart to the existing `.toCSV()` writer -- RFC-4180 parsing (quoted fields, doubled-quote escapes, commas/newlines inside quotes, CRLF or LF) plus per-column dtype inference (bool/int64/float64/utf8, widening to the narrowest type every non-empty cell agrees on; large integers stay exact via `BigInt`, unlike a `Number()`-based parser). Ragged rows and unterminated quotes throw a clear error rather than silently mishandling the input.

  Also widens the `@johnhenry/math-plus-tensor-core` peerDependency from an exact pin (`0.1.0`) to a caret range (`^0.1.0`), a backward-compatible relaxation: the exact pin made `npm install` `ERESOLVE` for any consumer (e.g. `mallory-graph`) already depending on tensor-core via its own `^0.1.0` range, even though frame-arrow's actual coupling to tensor-core is a small, stable, dynamically-imported surface with no static or type-level dependency at all.
