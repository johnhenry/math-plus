# @johnhenry/math-plus-tensor-cpu

## 0.2.1

### Patch Changes

- d122aef: Published declarations no longer import `./x.ts` (closes #157). tsc's `rewriteRelativeImportExtensions` rewrites `.ts` specifiers to `.js` in emitted JS but not in emitted `.d.ts`, so `dist/*.d.ts` referenced files that aren't in the package, and Deno's type check failed on them. Every package's build now runs `scripts/rewrite-dts-extensions.mjs` after `tsc`:

  - relative `.ts` / `.mts` / `.cts` specifiers in `dist/**/*.d.ts` become `.js` / `.mjs` / `.cjs` (as in laya-js),
  - each `dist/*.js` with a declaration file starts with `// @ts-self-types="./x.d.ts"` (after the `#!` line of a bin), so Deno finds the types when it loads `dist/` as plain files or from a URL instead of through `npm:`. Source maps are shifted by the inserted line.

  No runtime change. The manifest drift test checks every built `dist/` for both.

- Updated dependencies [d122aef]
  - @johnhenry/math-plus-tensor-core@0.2.1

## 0.2.0

### Minor Changes

- 8ec9c91: The tensor-core `Tensor` <-> tensor-backend `HostTensor` bridge (`hostFromTensor(t, label?)`, `tensorFromHost(h)`, `DEVICE_DTYPES`, `isDeviceDType`) moved from tensor-mlx into `@johnhenry/math-plus-tensor-cpu`. tensor-mlx and tensor-webgpu both use it, so the mapping has one implementation (#146). tensor-mlx re-exports it unchanged, with its errors still labelled `tensor-mlx:`. The f64 hint now reads "the device has no float64" instead of "MLX on Metal has no float64".

## 0.1.0

### Minor Changes

- f68063d: New package (#144, RFC 0001 §12 Q3): the CPU reference `Backend` for the `@johnhenry/tensor-backend@^0.2.0` contract. `createCpuBackend()` implements every required op, `geglu`/`meanPool`, and every optional "general numerics" op natively (only `compile` is absent). All computation runs on tensor-core's kernels (`@johnhenry/math-plus-tensor-core/kernels`) and `Tensor.matmul`, so GEMM, softmax, LayerNorm, RoPE, attention and erf/GELU have one implementation in math-plus. f32 compute; f16/bf16 host data is widened to f32 on `fromHost`. Passes tensor-backend's conformance suite (native and composed) under Node and Bun, plus a NumPy differential suite. Drop-in compatible with laya-js's `@johnhenry/backend-cpu@0.2.0` and at least as fast on encoder-shaped work (see README).

### Patch Changes

- Updated dependencies [f68063d]
  - @johnhenry/math-plus-tensor-core@0.2.0
