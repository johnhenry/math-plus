---
"@johnhenry/math-plus-tensor-cpu": minor
---

New package (#144, RFC 0001 §12 Q3): the CPU reference `Backend` for the `@johnhenry/tensor-backend@^0.2.0` contract. `createCpuBackend()` implements every required op, `geglu`/`meanPool`, and every optional "general numerics" op natively (only `compile` is absent). All computation runs on tensor-core's kernels (`@johnhenry/math-plus-tensor-core/kernels`) and `Tensor.matmul`, so GEMM, softmax, LayerNorm, RoPE, attention and erf/GELU have one implementation in math-plus. f32 compute; f16/bf16 host data is widened to f32 on `fromHost`. Passes tensor-backend's conformance suite (native and composed) under Node and Bun, plus a NumPy differential suite. Drop-in compatible with laya-js's `@johnhenry/backend-cpu@0.2.0` and at least as fast on encoder-shaped work (see README).
