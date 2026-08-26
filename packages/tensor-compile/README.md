# @johnhenry/math-plus-tensor-compile

Elementwise expression IR + fusion for math-plus (issue #11): trace a
function once over symbolic inputs, then execute it fused — one pass over the
data instead of one intermediate tensor per op. The same IR is the lowering
target for `@johnhenry/math-plus-tensor-webgpu`'s WGSL codegen and the
planned `Symbolic` bridge.

Strictly **opt-in**: eager `Tensor`/`Variable` usage is unaffected whether or
not this package is imported.

## Install

```bash
npm install @johnhenry/math-plus-tensor-compile
```

## Quick start

```js
import { compile } from "@johnhenry/math-plus-tensor-compile";
import { Tensor } from "@johnhenry/math-plus-tensor-core";

// Trace once...
const fused = compile(3, (x, y, z) => x.add(y).mul(z).relu());

// ...execute fused; broadcasting works like eager Tensor ops
const out = fused.forward(a, b, c); // matches a.add(b).mul(c).relu()

// Differentiable version, pluggable into the autograd tape
const op = fused.asVariableOp();
const result = op(va, vb, vc); // a Variable; backward() matches the unfused graph
```

## API surface

- `compile(numInputs, fn)` → `CompiledFn` with:
  - `forward(...tensors)` — value only; routes through `evalValue`, which
    skips the per-node gradient bookkeeping (~15x faster in the measured
    6-node/200k-element case: 571 ms → 37 ms, issue #99).
  - `forwardWithGrad(...tensors)` — `{ value, localGrads }`; each
    `localGrads[k]` has the broadcast *output* shape, not yet reduced or
    chained — `asVariableOp()` does both.
  - `asVariableOp()` — a `(...Variable) => Variable` op for the tape.
- `Traced` builder: binary `add sub mul div pow atan2 hypot min max`,
  comparisons + short-circuiting `select(then, else)`, and a large unary set
  (full trig incl. reciprocals, full hyperbolic, `exp`/`log*`, `erf`,
  `relu`/`sigmoid`/`gelu`, `floor`/`ceil`/`round`/`sign`/`trunc`...).
- Lower-level: `evalValue`, `evalWithGrad`, `IRNode`/`UnaryOp`/`BinaryOp`/
  `CmpOp` types.

## Traps

- **Float dtypes only (`f32`/`f64`), and all inputs must share one dtype** —
  no implicit promotion, matching tensor-core's M1 rule.
- **v1 is elementwise/broadcast only** — no reductions, no matmul. That's
  deliberate: the IR stays small enough to be a credible WebGPU/Symbolic
  lowering target.
- `select()` genuinely short-circuits: the untaken branch's domain error
  never surfaces (piecewise semantics, first true branch wins).
- Step functions (`floor`/`ceil`/`round`/`sign`/`trunc`) and comparisons
  have zero gradient everywhere they're defined — correct, but easy to
  forget when a "trained" parameter mysteriously never moves.
- `pow` with a negative base is gradchecked w.r.t. the base only; the
  exponent gradient assumes a positive base.
- `npm test` here **includes a wall-clock bench assertion** (unlike
  tensor-wasm, whose bench is excluded from CI) — a slow machine can fail
  the perf test without a correctness bug.

## Tests

`npm test` — includes an `evalValue`-vs-`evalWithGrad` parity suite over
every op (issue #99: add an op to one evaluator and the tests force you to
add it to both), a seeded random-graph fuzzer (issue #48, shared with
tensor-webgpu's WGSL cross-check), and an `erf` cross-check against
`@johnhenry/math`'s `SpecialFunctions.erf` (issue #34).

## Provenance

Part of the [math-plus](https://github.com/johnhenry/math-plus) monorepo;
family docs at <https://opensource.johnhenry.me/math/>.
