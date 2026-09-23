---
"@johnhenry/math-plus-tensor-webgpu": patch
---

Fused WGSL `tanh` and `gelu_tanh` clamp tanh's argument to ±15 (exactly 1.0 in f32 beyond that): Metal via Dawn computes tanh through `exp` and returned NaN once it overflowed. Found by running the #122 accuracy tests on the new in-process Dawn harness.
