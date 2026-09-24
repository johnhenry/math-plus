# @johnhenry/math-plus-special

The one canonical double-precision `erf` / `erfc` / GELU for Math Plus. Zero
dependencies, pure TypeScript.

```ts
import { erf, erfc, gelu, geluDerivative } from "@johnhenry/math-plus-special";

erf(0.5);                    // 0.5204998778130465
erfc(10);                    // 2.0884875837625446e-45 (no 1 - erf cancellation)
gelu(-30);                   // exact x·Φ(x), keeps relative accuracy in the left tail
gelu(1.2, "tanh");           // PyTorch's approximate="tanh"
```

## Why this is its own package

AGENTS.md's canonical-implementation rule: one construct, one implementation.
This module began in `@johnhenry/math-plus-tensor-core` (#122), which still
re-exports every name here and applies them in `Tensor.erf()` / `erfc()` /
`gelu()`. But `@johnhenry/math-plus-frame-arrow` deliberately has no static
dependency on the tensor track (tensor-core is only its optional, lazily
imported peer), so its `fn.erf()` carried a second, lower-accuracy copy. Both
now depend on this zero-dependency leaf instead.

Consumers: tensor-core (re-export + `Tensor` methods), tensor-compile's IR
evaluator, tensor-webgpu's WGSL lowering (`ERF_F32_PARAMS`), tensor-autograd's
GELU backward (via `Tensor.erfc()`), frame-arrow's `fn.erf`.

## Exports

| Name | What |
|---|---|
| `erf(x)`, `erfc(x)` | ~1e-15 / ~3.5e-15 relative (erfc wherever the result is a normal f64) |
| `gelu(x, approximate = "none")`, `geluErf`, `geluTanh` | Exact `x·Φ(x)` or the tanh approximation, PyTorch's names |
| `geluDerivative(x, approximate)` | Derivative of whichever forward was computed |
| `checkGeluApproximate(v)`, `type GeluApproximate` | Option validation for JS callers |
| `erfSeries`, `erfcContinuedFraction`, `ERF_SERIES_CUTOFF`, `ERF_F32_PARAMS` | The algorithm's pieces, exported so the f32 WGSL lowering can be verified against the f64 original |

Algorithm and accuracy notes are in `src/index.ts`. Tests: oracle-free
properties (`test/special.test.ts`) and a SciPy differential oracle
(`test/special-oracle.test.ts`, `scripts/special_oracle.py`; same
`$MATH_PLUS_SCIPY_ORACLE_PYTHON` / `$MATH_PLUS_ORACLE_PYTHON` / `python3`
resolution and skip-don't-fail rule as the rest of the repo).

## Not included

Complex arguments, `erfinv` / `erfcinv`, scaled `erfcx`. `@johnhenry/math`'s own
`SpecialFunctions.erf` (a separate repo) is not replaced by this package.
