# @johnhenry/math-plus-adapter-math

The bridge between `@johnhenry/math` (Vector/Matrix, `Symbolic` CAS — the
science side) and `@johnhenry/math-plus-tensor-core` (`Tensor` — the
engineering side). Three jobs live here: Matrix/Vector ↔ Tensor conversion,
compiling `Symbolic` expressions into the tensor and dataframe worlds, and
the tensor-native numerics (`linalg`, stats, FFT) that sit on top of the
conversion layer.

## Install

```bash
npm install @johnhenry/math-plus-adapter-math
```

Depends on `@johnhenry/math`, `tensor-core`, `tensor-compile`, and
`frame-arrow` (all `@johnhenry/math-plus-*`).

## Quick start

```js
import { fromMatrix, toMatrix, compileExpr, compileFrameExpr, linalg } from "@johnhenry/math-plus-adapter-math";
import { Symbolic } from "@johnhenry/math";
import { Tensor } from "@johnhenry/math-plus-tensor-core";

// Matrix/Vector <-> Tensor — plain arrays or @johnhenry/math values
const t = fromMatrix([[1, 2, 3], [4, 5, 6]]); // 2x3 f32 Tensor
toMatrix(t.transpose());                      // [[1,4],[2,5],[3,6]] — views handled

// Compile a CAS expression to run over tensor batches
const expr = Symbolic.parse("sin(x) * y + sqrt(x^2 + 1)");
const compiled = compileExpr(expr, { variables: ["x", "y"] });
compiled.forward(Tensor.from([0.1, 1.7]), Tensor.from([1.0, 0.3]));
const { localGrads } = compiled.forwardWithGrad(Tensor.from([0.3]), Tensor.from([1]));
// gradients — tested against Symbolic.differentiate as an independent oracle

// Same Symbolic expression as a frame-arrow computed column
const formula = Symbolic.parse("sin(x) * x");
const derivative = Symbolic.differentiate(formula, "x");
frame.withColumns({
  y: compileFrameExpr(formula),        // free variables become col() refs
  dy_dx: compileFrameExpr(derivative), // differentiate symbolically, run columnar
});

// Tensor-native linear algebra
linalg.solve(a, b);
linalg.svd(a);
```

## API surface

- **Conversion:** `fromMatrix` / `fromVector` (plain arrays *or*
  `@johnhenry/math` `Matrix`/`Vector`; optional `{ dtype }`), `toMatrix` /
  `toVector` (correct through non-contiguous views). Ragged input throws
  `RangeError`.
- **`compileExpr(expr, opts?)`:** `Symbolic` → tensor-compile IR. Returns a
  `CompiledFn` with `forward(...tensors)` and `forwardWithGrad(...)`;
  default variable order is `Symbolic.freeVariables` (alphabetical) —
  pass `variables` to pin positions. `piecewise` compiles to select
  chains. Unsupported nodes throw `UnsupportedExprError`.
- **`compileFrameExpr(expr)`:** `Symbolic` → frame-arrow `Expr`. Function
  names map 1:1 (frame-arrow's `fn.*` was spelled to match `Symbolic`);
  unknown columns surface as frame-arrow's own error.
- **`linalg`:** `lu`/`solve`/`det`/`inv`/`rref`/`rank`/`nullSpace`, `qr`,
  `cholesky`, `eigSymmetric` (Jacobi) and `eigGeneral` (complex
  eigenvalues, returned as `@johnhenry/math` `ComplexNumber`s),
  `powerIteration`, `svd`, `leastSquares`, `pseudoInverse`, norms and
  `conditionNumber`.
- **Stats:** `mean`/`median`/`percentile`/`variance`/`standardDeviation`
  (+ population variants), `correlation`, `linearRegression`,
  `Distributions`, `HypothesisTests`, `SpecialFunctions`.
- **FFT:** `fft`/`ifft`/`fftPadded`/`convolve`/`realSignal` over
  `Float64Array` pairs.
- **Graphs:** `toCSR`/`toDense` for `@johnhenry/math` graph values.
- **`./test-utils`:** `dualGrad`/`dualGradN` — the `DualNumber`
  forward-mode gradient oracle, exported so *other* packages can check
  their gradients against an independent implementation.

## Traps

- **`compileExpr` renames `ln` → `log`** internally to match
  tensor-compile's IR; `compileFrameExpr` needs no rename table — if you
  extend one bridge, don't assume the other spells functions the same way.
- **Variable order is alphabetical unless you pass `variables`.**
  `Symbolic.parse("b - a")` compiled without options takes `(a, b)` — in
  that order.
- **Conversion defaults to `f32`.** Round-tripping f64 data through
  `fromMatrix` without `{ dtype: "f64" }` quietly drops precision.

Part of the [math-plus](https://github.com/johnhenry/math-plus) family —
docs at [opensource.johnhenry.me/math/](https://opensource.johnhenry.me/math/).
