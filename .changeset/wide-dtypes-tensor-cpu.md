---
"@johnhenry/math-plus-tensor-cpu": minor
---

Full dtype parity with the widened `@johnhenry/tensor-backend` contract: `supports()` is now true for every dtype except f16/bf16 (unchanged — this stays an f32-based reference backend by design). `u8 i8 u16 i16 u32 f64` flow through the existing flat numeric kernels; `u64`/`i64` (bigint storage) delegate to `@johnhenry/math-plus-tensor-core`'s own bigint arithmetic — the same delegation pattern `matmul` already used for f32 GEMM — since JS throws mixing bigint with plain-number arithmetic and the kernels operate on plain numbers.

Along the way, fixed real bugs the new dtypes exposed: `#unaryF`, `mean`, and `scale` all unconditionally downcast their output to f32, silently truncating f64 precision; `cumsum`/`sum` now correctly preserve the input's own dtype (wrapping on overflow within its own width, matching real MLX behavior) instead of forcing i32 for any non-f32 input; `concat` now requires an exact dtype match for the new dtypes instead of silently applying the old 3-dtype promotion rule to values it was never designed for.

Verified with the real NumPy oracle: 168/168 tests, 0 skipped.
