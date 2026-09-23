# @johnhenry/math-plus-frame-parquet

## 0.0.2

### Patch Changes

- 80123df: Republish with an npm provenance attestation. The previous versions were published from a local machine on 2026-09-23 without provenance (the CI `NPM_TOKEN` secret had expired). No code changes.
- Updated dependencies [80123df]
  - @johnhenry/math-plus-frame-arrow@0.0.2

## 0.0.1

### Patch Changes

- 5d7172b: Lower `engines.node` from `>=26.0.0` to `>=24.0.0`. Nothing in these packages needs Node 26: the full test suite passes on Node 24.9, and CI now tests Node 24. Every suite also runs under Bun 1.2.17 (`npm run test:bun`).

  `@johnhenry/math-plus-frame-parquet`: `scanParquet`/`scanParquetLazy` now accept an absolute file path without wildcards under Bun. Bun 1.2's `fs.promises.glob` returns no matches for such a path, so a pattern with no glob metacharacters is now resolved with `stat` on every runtime.

- Updated dependencies [5d7172b]
- Updated dependencies [e9b691d]
  - @johnhenry/math-plus-frame-arrow@0.0.1

## 0.0.3

### Patch Changes

- @johnhenry/math-plus-frame-arrow@1.0.0

## 0.0.2

### Patch Changes

- Updated dependencies [7b0ced4]
  - @johnhenry/math-plus-frame-arrow@0.1.0
