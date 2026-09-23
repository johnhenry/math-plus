---
"@johnhenry/math-plus-tensor-mlx": minor
---

New experimental package (#125): native Apple Silicon (MLX/Metal) arrays on Node and Bun, built on the published `@johnhenry/backend-mlx` / `@johnhenry/tensor-backend` contract (no FFI of its own). Explicit `device.fromTensor(t)` / `await arr.toTensor()` transfers (zero-copy host views, one copy per transfer, non-contiguous and f64/i64 inputs rejected), no global default device, no implicit dtype promotion; elementwise/broadcast ops, `sum`/`mean`/`max`/`min`/`softmax`, `matmul`, `layerNorm`, `cast` incl. f16/bf16, lazy graph with explicit `eval`. Tested against a NumPy oracle and the tensor-backend conformance suite (skip-don't-fail off darwin/arm64). Prototype for RFC 0001 (`docs/rfcs/0001-device-backends.md`, Proposed). Not published to JSR.
