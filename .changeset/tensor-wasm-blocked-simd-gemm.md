---
"@johnhenry/math-plus-tensor-wasm": minor
---

`matmulInto` now runs a cache-blocked, register-tiled GEMM (issue #121): the scalar module's `gemm_f32` uses a portable 4x8 micro-kernel and the SIMD128 module gains `gemm_f32_simd128` (f32x4, used for every `matmulInto` call when SIMD is available, any strides). ~37 GFLOP/s at 1024³ vs ~1.7 for the old naive loop on the same machine (21.6x; docs/spikes/wasm-simd.md); SIMD and scalar results are bit-identical. `matmulInto` now rejects a wrongly-shaped `out` with a RangeError instead of writing past its buffer, and `Kernels.load()` verifies that instantiating the SIMD module leaves the shared linear memory untouched (falling back to scalar-only, memory restored, if not). The native cdylib gets the same blocked kernel plus an opt-in `accelerate` cargo feature (macOS, `cblas_sgemm`). Same JS API.
