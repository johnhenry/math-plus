---
"@johnhenry/math-plus-tensor-core": minor
---

Real f16/bf16 support. `cast()` to/from `f16`/`bf16` now converts values (round-to-nearest-even, bit-for-bit equal to NumPy's `astype(float16)`) instead of truncating to integers and reinterpreting raw bits — a correctness fix. `from`/`full`/`arange`/`random.*` encode and `at`/`item`/`toArray` decode half dtypes; `.npy` read/write supports `<f2`. New exports `encodeHalf`/`decodeHalf`/`isHalfDType`. Arithmetic/comparison/reduction/sort/matmul kernels now throw a clear `TypeError` on half dtypes (they previously computed on bit patterns); `cast("f32")` first.
