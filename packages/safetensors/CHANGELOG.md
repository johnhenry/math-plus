# @johnhenry/math-plus-safetensors

## 0.1.3

### Patch Changes

- d122aef: Published declarations no longer import `./x.ts` (closes #157). tsc's `rewriteRelativeImportExtensions` rewrites `.ts` specifiers to `.js` in emitted JS but not in emitted `.d.ts`, so `dist/*.d.ts` referenced files that aren't in the package, and Deno's type check failed on them. Every package's build now runs `scripts/rewrite-dts-extensions.mjs` after `tsc`:

  - relative `.ts` / `.mts` / `.cts` specifiers in `dist/**/*.d.ts` become `.js` / `.mjs` / `.cjs` (as in laya-js),
  - each `dist/*.js` with a declaration file starts with `// @ts-self-types="./x.d.ts"` (after the `#!` line of a bin), so Deno finds the types when it loads `dist/` as plain files or from a URL instead of through `npm:`. Source maps are shifted by the inserted line.

  No runtime change. The manifest drift test checks every built `dist/` for both.

## 0.1.2

### Patch Changes

- 71fa3f0: Widen the optional tensor-core peer range to include `^0.2.0`, so tensor-core's 0.2.0 release doesn't force a major bump on its peer dependents.

## 0.1.1

### Patch Changes

- 80123df: Republish with an npm provenance attestation. The previous versions were published from a local machine on 2026-09-23 without provenance (the CI `NPM_TOKEN` secret had expired). No code changes.

## 0.1.0

### Minor Changes

- 648d5e0: New package `@johnhenry/math-plus-safetensors`: safetensors reader/writer for any JS runtime. Reference-equivalent header validation (offset tiling, end-of-file coverage, duplicate names, typed `SafetensorsError` codes), typed views (F16 as `Float16Array`, BF16 bits), `toFloat32`/`toF32` for every dtype, a writer byte-identical to Python's `safetensors.serialize`, lazy `openSafetensors()` over Blob/File, HTTP Range requests (full-download fallback), Node/Bun file paths and FileHandles, and `@johnhenry/math-plus-safetensors/tensor` interop with tensor-core (optional peer).

### Patch Changes

- e9b691d: Widen internal peer-dependency ranges to `^0.0.0 || ^0.1.0` so the 0.1.0 releases of tensor-core and safetensors stay in range (Changesets would otherwise force a major bump on every peer dependent).
