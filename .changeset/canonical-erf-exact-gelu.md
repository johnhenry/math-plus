---
"@johnhenry/math-plus-tensor-core": minor
"@johnhenry/math-plus-tensor-autograd": minor
"@johnhenry/math-plus-tensor-compile": minor
"@johnhenry/math-plus-tensor-webgpu": minor
---

**BREAKING (pre-1.0, so a minor bump): `gelu()` now defaults to EXACT erf-GELU, not the tanh approximation.** One canonical double-precision `erf`/`erfc` for the monorepo (#122).

- **Behaviour change — read this if you call `gelu()`:** `Tensor.gelu()`, `Variable.gelu()`, `Traced.gelu()` and the WGSL `gelu` lowering now compute exact `x·Φ(x) = 0.5·x·(1 + erf(x/√2))` by default, matching PyTorch's `nn.GELU()` / `F.gelu(x)` and what BERT/ModernBERT were trained with. Previously they always used the tanh approximation. Outputs move by up to ~4.7e-4 (at |x| ≈ 2.7). To keep the old numbers, pass `{ approximate: "tanh" }` (same option name and values as `torch.nn.functional.gelu(approximate=...)`). `Variable.gelu()`'s backward differentiates whichever mode ran.
- **@johnhenry/math-plus-tensor-core:** new `src/special.ts`, the single canonical implementation: `erf`, `erfc` (~1e-15 / ~3.5e-15 relative; Maclaurin series below |x| = 1, Laplace continued fraction above, with an fdlibm-style split `exp(-z²)`), `gelu(x, approximate)`, `geluErf`, `geluTanh`, `geluDerivative`, plus `ERF_F32_PARAMS` for f32 lowerings. New `Tensor.erf()` / `Tensor.erfc()` (op-table parity with the compiled IR's `erf`) and `Tensor.gelu({ approximate })`. Exact GELU is computed as `0.5·x·erfc(-x/√2)` with the exponent taken from `x` directly, so it keeps full relative accuracy down its left tail instead of cancelling to 0.
- **@johnhenry/math-plus-tensor-compile:** the Abramowitz & Stegun 7.1.26 `erf` copy (~1.5e-7 absolute) is gone; `erf` and both GELU modes evaluate tensor-core's canonical functions, so compiled and eager results are bit-identical. IR op `"gelu"` now means exact GELU; new `UnaryOp` `"gelu_tanh"` for the tanh form (`Traced.gelu({ approximate: "tanh" })`). Code that switches exhaustively over `UnaryOp` must add a `"gelu_tanh"` case.
- **@johnhenry/math-plus-tensor-webgpu:** WGSL `math_plus_erf` is now an f32 lowering of the canonical algorithm (loop counts from `ERF_F32_PARAMS`, measured ~1.4e-7 absolute on a real adapter) with `math_plus_erfc` / `math_plus_gelu` alongside; IR `"gelu"` lowers to exact GELU, `"gelu_tanh"` to the tanh form.

Not changed: `@johnhenry/math-plus-frame-arrow`'s `fn.erf` keeps its local A&S 7.1.26 copy (frame-arrow has no static dependency on tensor-core by design), so it now differs from the tensor-compile path by up to ~1.5e-7 — documented in `eval-expr.ts`. `@johnhenry/math`'s `SpecialFunctions.erf` (separate repo) is unchanged.
