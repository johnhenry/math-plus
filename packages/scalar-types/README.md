# @johnhenry/math-plus-scalar-types

Thin re-export of [`@johnhenry/math`](https://github.com/johnhenry/math)'s
scalar types (`ComplexNumber`, `Rational`, `Decimal`, `Interval`,
`Quaternion`) plus the tensor-boundary converters the math-plus family uses
at its API edges. If the scalar layer ever changes, only this package moves.

The rule it encodes: **boxed scalars live only at tensor API edges**
(`at()`/`item()`/constructors), never in tensor storage or kernels.

## Install

```bash
npm install @johnhenry/math-plus-scalar-types
```

## Quick start

```js
import {
  ComplexNumber, Rational, Decimal, Interval, Quaternion,
  complexToParts, partsToComplex,
} from "@johnhenry/math-plus-scalar-types";

new ComplexNumber(3, 4).magnitude(); // 5
new Rational(2n, 4n).toString();     // "1/2"
Decimal.fromString("1.5").add(Decimal.fromString("2.5")).toNumber(); // 4

// The ComplexTensor edge format: boxed <-> flat split storage
const parts = complexToParts([new ComplexNumber(1, 2), new ComplexNumber(3, -4)]);
parts.real; // Float64Array [1, 3]
parts.imag; // Float64Array [2, -4]
partsToComplex(parts); // back to ComplexNumber[]
```

## API surface

| Export | What it is |
|---|---|
| `ComplexNumber`, `Rational`, `Decimal` | Re-exports from `@johnhenry/math` |
| `Fraction` | Alias of `Rational` (math.js vocabulary, for adapter familiarity) |
| `Interval` (issue #36) | Outward-rounding interval arithmetic — see trap below |
| `Quaternion` (issue #37) | `fromAxisAngle`, `rotateVector`, `Identity`, ... |
| `ComplexParts` | `{ real: Float64Array; imag: Float64Array }` — fft's `ComplexTensor` split-storage edge format |
| `complexToParts` / `partsToComplex` | Boxed ↔ flat converters; length mismatch throws `RangeError` |

`Interval` and `Quaternion` deliberately have no tensor converters yet — no
natural tensor-of-values shape / no concrete consumer.

## Traps

- **Don't equality-compare `Interval`s after arithmetic.** `@johnhenry/math`
  outward-rounds every non-exact op by ~1 ULP per side (johnhenry/math#57)
  to preserve the containment guarantee — even 1×3 comes back a hair wider
  than exact. Assert *containment* of exact bounds, not equality. (This is
  also what makes `Interval` useful here: as a rounding-error oracle to bound
  f32 GPU results against an f64 reference — see tensor-webgpu's fusion
  tests.)
- **`ComplexParts` is `Float64Array`** — lossless for f64 tensor paths; any
  f32 path truncates at the boundary.
- **`Decimal` is built from strings** (`Decimal.fromString("1.5")`), not
  number literals.

## Provenance

Part of the [math-plus](https://github.com/johnhenry/math-plus) monorepo —
the bridge to the `@johnhenry/math` scalar layer (docs/PLAN.md §B.1,
non-goals 3 and 9). Family docs: <https://opensource.johnhenry.me/math/>.
