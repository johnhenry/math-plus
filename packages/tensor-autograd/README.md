# @johnhenry/math-plus-tensor-autograd

Reverse-mode automatic differentiation over `@johnhenry/math-plus-tensor-core`
tensors: a define-by-run tape (`Variable`), a small `nn.*` layer/loss set, and
`optim.*` optimizers (SGD/Adam/AdamW/RMSprop + StepLR), with a batteries-light
`trainer`. PyTorch's mental model, tensor-core's storage rules.

`Variable` wraps a `Tensor` rather than extending it — deliberately, so
tensor-core never depends on autograd. Non-differentiable ops (argmax, sort,
comparisons) simply aren't `Variable` methods: call them on `.value`. "No
in-place ops on tracked tensors" is satisfied structurally — the mutating
method doesn't exist.

## Install

```bash
npm install @johnhenry/math-plus-tensor-autograd
```

## Quick start

```js
import { grad, variable } from "@johnhenry/math-plus-tensor-autograd";
import { Tensor } from "@johnhenry/math-plus-tensor-core";

// Tape + backward
const x = variable(Tensor.from([2, 3], { dtype: "f64" }));
const y = variable(Tensor.from([4, 5], { dtype: "f64" }));
x.mul(y).sum().backward();
x.grad.toArray(); // [4, 5] — d(sum(x*y))/dx = y

// Functional: value and gradient in one pass
const vg = grad.valueAndGrad((v) => v.mul(v).sum());
const { value, grad: g } = vg(Tensor.from([3, 4], { dtype: "f64" }));
value.item();   // 25
g.toArray();    // [6, 8]
```

Training a model:

```js
import { nn, optim, trainer } from "@johnhenry/math-plus-tensor-autograd";
import { random, Tensor } from "@johnhenry/math-plus-tensor-core";

const model = new nn.Linear(1, 1, { rng: random.seed(3) });
const opt = new optim.SGD(model.parameters(), { lr: 0.01 });
const t = trainer.configure({ model, optimizer: opt, lossFn: nn.mseLoss, epochs: 2000 });
const { lossHistory } = await t.fit({ x: X, y: Y }); // X/Y are f64 Tensors
```

## API surface

- `Variable` / `variable` / `constant`; ops: `add sub mul div matmul
  unsqueeze sqrt log sum mean relu sigmoid gelu softmax`; `backward`,
  `zeroGrad`, `detach`.
- `grad.of` / `grad.valueAndGrad`; `noGrad` / `enableGrad` / `isGradEnabled`.
- `nn`: `Parameter`, `Module`, `Linear`, `Embedding`, `LayerNorm`,
  `Sequential`, `Dropout`; losses `mseLoss`, `huberLoss`,
  `binaryCrossEntropy` (logits-based), `crossEntropy`.
- `optim`: `SGD` (momentum/nesterov), `AdamW`, `Adam`, `RMSprop`, `StepLR`.
- `io`: `writeCheckpoint` / `loadCheckpoint`.
- `trainer`: `configure`, `Trainer`, `Batch` (the shape
  `@johnhenry/math-plus-data`'s `collate.xy()` produces).
- `sumToShape` — the broadcast-reduction helper every backward uses.

## Traps

- **Gradients accumulate** across repeated `backward()` calls — `zeroGrad()`
  resets (`.grad` back to `null`, not zeros). Only leaves accumulate; only a
  scalar output may call `backward()` without an explicit `gradOutput`.
- **`nn.Linear` initializes at f64** — feed f64 inputs or hit tensor-core's
  no-promotion `TypeError`. (`@johnhenry/math-plus-data`'s collate defaults
  to f32; pass `{ dtype: "f64" }` there.)
- **`trainer.fit(dataLoader)` ignores `config.epochs`** — one pass, because
  an arbitrary `AsyncIterable` isn't guaranteed re-iterable. Epochs apply
  only to the full-batch `fit({ x, y })` overload; put epochs in the data
  pipeline (`dataset.epochs(n)`) otherwise.
- `binaryCrossEntropy` uses the BCEWithLogits reformulation so saturated
  (|z| ≳ 37) logits give finite loss and gradients, not NaN (issue #85).
- `io.writeCheckpoint` is a custom `"MPCK"` container, **not** NumPy `.npz`;
  `loadStateDict` is strict both ways (missing *and* unexpected keys throw).
- SGD `nesterov` without nonzero `momentum` throws (issue #89).
- Telemetry is opt-in: `backward()` emits a trace span and `optim.step()` a
  `optim/gradNorm` metric only when a `@johnhenry/math-plus-telemetry` sink
  is installed — the grad norm isn't even computed otherwise.

## Tests

`npm test` — includes a cross-oracle check against `@johnhenry/math`'s
forward-mode `DualNumber`, saturation regressions (#85), scheduler exactness
(#72), and a sparse-Embedding-backward perf guard.

## Provenance

Part of the [math-plus](https://github.com/johnhenry/math-plus) monorepo;
family docs at <https://opensource.johnhenry.me/math/>.
