# @johnhenry/math-plus-tensor-mlx

## 0.2.1

### Patch Changes

- d122aef: Published declarations no longer import `./x.ts` (closes #157). tsc's `rewriteRelativeImportExtensions` rewrites `.ts` specifiers to `.js` in emitted JS but not in emitted `.d.ts`, so `dist/*.d.ts` referenced files that aren't in the package, and Deno's type check failed on them. Every package's build now runs `scripts/rewrite-dts-extensions.mjs` after `tsc`:

  - relative `.ts` / `.mts` / `.cts` specifiers in `dist/**/*.d.ts` become `.js` / `.mjs` / `.cjs` (as in laya-js),
  - each `dist/*.js` with a declaration file starts with `// @ts-self-types="./x.d.ts"` (after the `#!` line of a bin), so Deno finds the types when it loads `dist/` as plain files or from a URL instead of through `npm:`. Source maps are shifted by the inserted line.

  No runtime change. The manifest drift test checks every built `dist/` for both.

- Updated dependencies [d122aef]
  - @johnhenry/math-plus-tensor-core@0.2.1
  - @johnhenry/math-plus-tensor-cpu@0.2.1

## 0.2.0

### Minor Changes

- 2091213: **Breaking (0.x minor): uploads are async**, per RFC 0001 §12 Q2 and PLAN.md non-goal 5. tensor-mlx now depends on `@johnhenry/backend-mlx@^0.3.0` and `@johnhenry/tensor-backend@^0.2.0`.

  - `device.fromTensor(t)` and `device.fromHost(h)` return `Promise<MlxArray>`. Validation errors (non-contiguous tensor, unsupported dtype) still throw synchronously. Downloads (`toTensor`/`toHost`) were already async.
  - **General-numerics ops** on `MlxArray`, called through tensor-backend's compose helpers (native mlx-c kernels in backend-mlx): `pow`, `abs`, `sqrt`, `rsqrt`, `tanh`, `sigmoid`, `erf`; `equal`, `notEqual`, `less`, `lessEqual`, `greater`, `greaterEqual` (bool results); `logicalAnd`, `logicalOr`, `logicalNot` (bool inputs); `argmax`/`argmin` (i32, `(axis?, { keepDims? })`); `cumsum(axis?)`. `neg`, `mean` and `min` now use the native kernels instead of compositions. The dtype rules are unchanged: no implicit promotion, float-only math refuses integers, and there is no implicit truthiness.
  - Number operands (`x.add(2)`, `x.less(0.5)`) are built on the device from the array instead of being uploaded, so ops stay synchronous. i32 constants are exact over the whole i32 range, and out-of-range values throw.
  - **Deno 2** (backend-mlx's new `Deno.dlopen` loader): the suites run under Deno 2.9.7 (`npm run test:deno`), and the package is now **published to JSR** (`jsr:@johnhenry/math-plus-tensor-mlx`, closes #147). The platform package comes with the `npm:@johnhenry/backend-mlx` import, or set `LAYA_MLXC_PATH`.
  - Tests: the NumPy differential suite now covers every new op in f32, f16 and bf16 (and i32/bool where defined, compared exactly), and the tensor-backend conformance suite runs the numerics cases in f32/f16/bf16. That is 213 tests, 0 skipped, on Node, Bun and Deno.

  Migration: add `await` to every upload, for example `const x = await device.fromTensor(t)`. Upload several at once with `await Promise.all(ts.map((t) => device.fromTensor(t)))`. An upload can no longer be chained directly (`device.fromTensor(t).add(1)`): await it first. Nothing else changes for existing code.

### Patch Changes

- 8ec9c91: The tensor-core `Tensor` <-> tensor-backend `HostTensor` bridge (`hostFromTensor(t, label?)`, `tensorFromHost(h)`, `DEVICE_DTYPES`, `isDeviceDType`) moved from tensor-mlx into `@johnhenry/math-plus-tensor-cpu`. tensor-mlx and tensor-webgpu both use it, so the mapping has one implementation (#146). tensor-mlx re-exports it unchanged, with its errors still labelled `tensor-mlx:`. The f64 hint now reads "the device has no float64" instead of "MLX on Metal has no float64".
- Updated dependencies [8ec9c91]
  - @johnhenry/math-plus-tensor-cpu@0.2.0

## 0.1.2

### Patch Changes

- Updated dependencies [f68063d]
  - @johnhenry/math-plus-tensor-core@0.2.0

## 0.1.1

### Patch Changes

- 80123df: Republish with an npm provenance attestation. The previous versions were published from a local machine on 2026-09-23 without provenance (the CI `NPM_TOKEN` secret had expired). No code changes.
- Updated dependencies [80123df]
  - @johnhenry/math-plus-tensor-core@0.1.1

## 0.1.0

### Minor Changes

- 65e6f69: New experimental package (#125): native Apple Silicon (MLX/Metal) arrays on Node and Bun, built on the published `@johnhenry/backend-mlx` / `@johnhenry/tensor-backend` contract (no FFI of its own). Explicit `device.fromTensor(t)` / `await arr.toTensor()` transfers (zero-copy host views, one copy per transfer, non-contiguous and f64/i64 inputs rejected), no global default device, no implicit dtype promotion; elementwise/broadcast ops, `sum`/`mean`/`max`/`min`/`softmax`, `matmul`, `layerNorm`, `cast` incl. f16/bf16, lazy graph with explicit `eval`. Tested against a NumPy oracle and the tensor-backend conformance suite (skip-don't-fail off darwin/arm64). Prototype for RFC 0001 (`docs/rfcs/0001-device-backends.md`, Proposed). Not published to JSR.

### Patch Changes

- Updated dependencies [866f3ef]
- Updated dependencies [5d7172b]
- Updated dependencies [648d5e0]
- Updated dependencies [6beb547]
- Updated dependencies [c00998a]
  - @johnhenry/math-plus-tensor-core@0.1.0
