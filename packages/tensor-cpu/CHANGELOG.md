# @johnhenry/math-plus-tensor-cpu

## 0.3.0

### Minor Changes

- a075fe4: Full dtype parity with the widened `@johnhenry/tensor-backend` contract: `supports()` is now true for every dtype except f16/bf16 (unchanged — this stays an f32-based reference backend by design). `u8 i8 u16 i16 u32 f64` flow through the existing flat numeric kernels; `u64`/`i64` (bigint storage) delegate to `@johnhenry/math-plus-tensor-core`'s own bigint arithmetic — the same delegation pattern `matmul` already used for f32 GEMM — since JS throws mixing bigint with plain-number arithmetic and the kernels operate on plain numbers.

  Along the way, fixed real bugs the new dtypes exposed: `#unaryF`, `mean`, and `scale` all unconditionally downcast their output to f32, silently truncating f64 precision; `cumsum`/`sum` now correctly preserve the input's own dtype (wrapping on overflow within its own width, matching real MLX behavior) instead of forcing i32 for any non-f32 input; `concat` now requires an exact dtype match for the new dtypes instead of silently applying the old 3-dtype promotion rule to values it was never designed for.

  Verified with the real NumPy oracle: 168/168 tests, 0 skipped.

### Patch Changes

- 7eee743: Track the 13-dtype backend releases: `@johnhenry/tensor-backend` `^0.4.0` (whose `DType` has the full dtype set these facades now implement; `^0.3.0` resolved to the published 0.3.0 with five dtypes, which broke type-checking), `@johnhenry/backend-webgpu` `^0.6.0` (u32 support the WebGPU facade relies on; 0.4.x crashed on u32 cases) and `@johnhenry/backend-mlx` `^0.5.0` (wide-dtype MLX). Each facade and its backend now share one `tensor-backend` 0.4.

## 0.2.2

### Patch Changes

- 7ba9603: One chainable device-array API for every math-plus device (the open follow-up of #159/#162).

  - **tensor-cpu** now hosts it, next to the shared `Tensor` <-> `HostTensor` bridge: `ArrayDevice<B>` wraps any `@johnhenry/tensor-backend` `Backend`, and `DeviceArray` is the chainable array (`add` … `pow`, the unary math, comparisons and logic, `sum` … `cumsum`, `softmax`, `matmul`, `layerNorm`, `cast`, `reshape`, `transpose`, explicit async `toTensor()`/`toHost()`, `eval`, `dispose`, plus `handle` and `device.wrap(handle)` to reach the backend ops it does not wrap). The optional contract ops go through tensor-backend's compose helpers, so any backend works. New CPU device facade: `createCpuDevice()` → `CpuDevice` / `CpuArray` (f32/i32/bool; f16/bf16 are refused instead of widened silently). Additive, so a patch: `^0.2` ranges (including laya-js's `@johnhenry/backend-cpu@0.3.2`) keep matching.
  - **tensor-mlx**: `MlxDevice` extends `ArrayDevice` and `MlxArray` is a `DeviceArray` subclass that adds nothing — the public API and its tests are unchanged; `supports()`, `wrap()` and `handle` are new.
  - **tensor-webgpu (breaking, hence 0.4.0)**: `gpu.fromTensor()` / `gpu.fromHost()` now resolve to a chainable `WebGpuArray` (the same `DeviceArray`) instead of a raw `WebGpuTensor`, and their validation errors (non-contiguous tensor, a dtype the device does not support — f16 without `shader-f16` is now refused, not widened) throw synchronously instead of rejecting. `gpu.backend` is unchanged. Migration: raw-tensor code calls `gpu.backend.fromHost(h)`, or passes `x.handle` to `gpu.backend.*` and `gpu.wrap()`s the result. `gpu.toTensor`/`toHost`/`dispose`/`scope` and `fuse`/`compile` take arrays or raw tensors (arrays in → array out). See the README's "Changed in 0.4.0".

  One behavioural suite (`packages/tensor-cpu/test/device-array-suite.ts`, NumPy oracle `scripts/device_array_oracle.py`, moved from tensor-mlx) runs over the CPU, MLX (Metal) and WebGPU (Dawn) devices in every dtype each supports.

- 7ba9603: Track `@johnhenry/tensor-backend@^0.3.0` (additive: optional quantized-weight ops and their compose helpers), `@johnhenry/backend-mlx@^0.4.0` and `@johnhenry/backend-webgpu@^0.4.0`, so an install with laya-js's `@johnhenry/backend-cpu@0.3.2` (which re-exports tensor-cpu and declares `^0.3.0`) resolves one tensor-backend copy. tensor-cpu's backend passes tensor-backend 0.3's conformance suite, including its 26 quantized cases through the compose fallback (it has no native quantized ops).

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
