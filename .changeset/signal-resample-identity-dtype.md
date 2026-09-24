---
"@johnhenry/math-plus-signal": patch
---

`resamplePoly`: the identity path (`up === down` after GCD reduction) no longer returns a dtype-mislabeled tensor. An f32 (or f16/integer) input used to come back labeled `f64` over its original `Float32Array` storage; it now returns a genuine f64 copy, like every other `resamplePoly` output. Inputs are cast to f64 up front, which also decodes f16/bf16 values instead of reading their bit patterns (#113).
