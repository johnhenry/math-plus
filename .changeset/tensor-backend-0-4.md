---
"@johnhenry/math-plus-tensor-cpu": patch
"@johnhenry/math-plus-tensor-mlx": patch
"@johnhenry/math-plus-tensor-webgpu": patch
---

Track the 13-dtype backend releases: `@johnhenry/tensor-backend` `^0.4.0` (whose `DType` has the full dtype set these facades now implement; `^0.3.0` resolved to the published 0.3.0 with five dtypes, which broke type-checking), `@johnhenry/backend-webgpu` `^0.6.0` (u32 support the WebGPU facade relies on; 0.4.x crashed on u32 cases) and `@johnhenry/backend-mlx` `^0.5.0` (wide-dtype MLX). Each facade and its backend now share one `tensor-backend` 0.4.
