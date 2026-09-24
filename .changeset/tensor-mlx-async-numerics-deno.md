---
"@johnhenry/math-plus-tensor-mlx": minor
---

**Breaking (0.x minor): uploads are async**, per RFC 0001 §12 Q2 and PLAN.md non-goal 5. tensor-mlx now depends on `@johnhenry/backend-mlx@^0.3.0` and `@johnhenry/tensor-backend@^0.2.0`.

- `device.fromTensor(t)` and `device.fromHost(h)` return `Promise<MlxArray>`. Validation errors (non-contiguous tensor, unsupported dtype) still throw synchronously. Downloads (`toTensor`/`toHost`) were already async.
- **General-numerics ops** on `MlxArray`, called through tensor-backend's compose helpers (native mlx-c kernels in backend-mlx): `pow`, `abs`, `sqrt`, `rsqrt`, `tanh`, `sigmoid`, `erf`; `equal`, `notEqual`, `less`, `lessEqual`, `greater`, `greaterEqual` (bool results); `logicalAnd`, `logicalOr`, `logicalNot` (bool inputs); `argmax`/`argmin` (i32, `(axis?, { keepDims? })`); `cumsum(axis?)`. `neg`, `mean` and `min` now use the native kernels instead of compositions. The dtype rules are unchanged: no implicit promotion, float-only math refuses integers, and there is no implicit truthiness.
- Number operands (`x.add(2)`, `x.less(0.5)`) are built on the device from the array instead of being uploaded, so ops stay synchronous. i32 constants are exact over the whole i32 range, and out-of-range values throw.
- **Deno 2** (backend-mlx's new `Deno.dlopen` loader): the suites run under Deno 2.9.7 (`npm run test:deno`), and the package is now **published to JSR** (`jsr:@johnhenry/math-plus-tensor-mlx`, closes #147). The platform package comes with the `npm:@johnhenry/backend-mlx` import, or set `LAYA_MLXC_PATH`.
- Tests: the NumPy differential suite now covers every new op in f32, f16 and bf16 (and i32/bool where defined, compared exactly), and the tensor-backend conformance suite runs the numerics cases in f32/f16/bf16. That is 213 tests, 0 skipped, on Node, Bun and Deno.

Migration: add `await` to every upload, for example `const x = await device.fromTensor(t)`. Upload several at once with `await Promise.all(ts.map((t) => device.fromTensor(t)))`. An upload can no longer be chained directly (`device.fromTensor(t).add(1)`): await it first. Nothing else changes for existing code.
