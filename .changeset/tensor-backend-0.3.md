---
"@johnhenry/math-plus-tensor-cpu": patch
"@johnhenry/math-plus-tensor-mlx": patch
"@johnhenry/math-plus-tensor-webgpu": patch
---

Track `@johnhenry/tensor-backend@^0.3.0` (additive: optional quantized-weight ops and their compose helpers), `@johnhenry/backend-mlx@^0.4.0` and `@johnhenry/backend-webgpu@^0.4.0`, so an install with laya-js's `@johnhenry/backend-cpu@0.3.2` (which re-exports tensor-cpu and declares `^0.3.0`) resolves one tensor-backend copy. tensor-cpu's backend passes tensor-backend 0.3's conformance suite, including its 26 quantized cases through the compose fallback (it has no native quantized ops).
