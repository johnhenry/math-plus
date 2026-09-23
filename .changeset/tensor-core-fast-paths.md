---
"@johnhenry/math-plus-tensor-core": patch
---

Contiguous fast paths, blocked GEMM, and fused softmax/variance (no API change; fixes #120). Contiguous inputs to elementwise ops (same-shape, scalar, trailing-block "bias" and per-row broadcasts), unary ops, `cast`, `contiguous`, comparisons, and `sum`/`mean`/`min`/`max` now run flat typed-array loops instead of the per-element offset generator; `matmul` runs a register-blocked GEMM over packed f64 panels (~5–7× faster at 256²/1024² on an Apple M2); `softmax` and `variance`/`std` are fused single kernels with no temporaries. All fast paths are bit-identical to the general strided path, which still handles views, broadcasts, and i64/u64. Also fixes `broadcastShapes` turning a zero-size dim into 1 when broadcast against a size-1 dim (`[0, 4]` with `[4]` now gives `[0, 4]`, matching NumPy).
