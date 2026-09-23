---
"@johnhenry/math-plus-frame-arrow": patch
"@johnhenry/math-plus-safetensors": patch
"@johnhenry/math-plus-tensor-autograd": patch
---

Widen internal peer-dependency ranges to `^0.0.0 || ^0.1.0` so the 0.1.0 releases of tensor-core and safetensors stay in range (Changesets would otherwise force a major bump on every peer dependent).
