---
"@johnhenry/math-plus-tensor-webgpu": patch
---

Re-measure the WebGPU-vs-WASM GEMM crossover against tensor-wasm's SIMD128 GEMM (#130) with the thermal-aware method (docs/BENCHMARKING.md). `chooseGemmBackend(m, n, k?)` now takes an optional `k` and picks WebGPU only when `m·n >= GEMM_ELEMENT_THRESHOLD` (now `192 * 192`, was `128 * 128`) **and**, when `k` is given, `m·n·k >= GEMM_WORK_THRESHOLD` (new export, `2 ** 22`). On an Apple M2 the square crossover moved to n = 160 (Dawn) / n = 192 (headless Chrome), and small-k or small-output products at those sizes still lose to WASM. Numbers: docs/spikes/webgpu-tiled-gemm.md.
