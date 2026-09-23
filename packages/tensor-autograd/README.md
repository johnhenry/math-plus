# @johnhenry/math-plus-tensor-autograd

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fmath-plus-tensor-autograd.svg)](https://www.npmjs.com/package/@johnhenry/math-plus-tensor-autograd)
[![license](https://img.shields.io/npm/l/%40johnhenry%2Fmath-plus-tensor-autograd.svg)](../../LICENSE)

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
const { lossHistory } = await t.fit({ x: X, y: Y }); // X/Y are f32 Tensors (the parameters' default dtype)
```

Transformer blocks, PyTorch-compatible down to the state-dict keys:

```js
import { nn, noGrad, constant } from "@johnhenry/math-plus-tensor-autograd";
import { loadSafetensorsInto } from "@johnhenry/math-plus-tensor-autograd/safetensors";

const layer = new nn.TransformerEncoderLayer(768, 12, {
  dimFeedforward: 3072, normFirst: true, batchFirst: true, activation: "gelu",
  dtype: "f16", // storage only: computes in the input's f32
});
// A PyTorch nn.TransformerEncoderLayer's state_dict(), saved with safetensors:
await loadSafetensorsInto(layer, "encoder_layer.safetensors");
const y = noGrad(() => layer.forward(constant(x), { srcKeyPaddingMask })); // x: f32 [B, L, 768]
```

## API surface

- `Variable` / `variable` / `constant`; ops: `add sub mul div matmul
  unsqueeze sqrt log exp tanh sum mean relu sigmoid gelu softmax
  maskedFill cast`, views `reshape permute transpose slice narrow`, and
  `Variable.concat`; `backward`, `zeroGrad`, `detach`. `matmul` is batched
  (ndim >= 2, broadcasting batch axes).
- `grad.of` / `grad.valueAndGrad`; `noGrad` / `enableGrad` / `isGradEnabled`.
- `nn`: `Parameter`, `Module` (`parameters`, `namedParameters`,
  `namedModules`, `stateDict`, `loadStateDict(dict, { strict,
  legacyLinearLayout })`), `Linear`, `Embedding`, `LayerNorm` (`bias:
  false`), `Sequential`, `Dropout`; losses `mseLoss`, `huberLoss`,
  `binaryCrossEntropy` (logits-based), `crossEntropy`.
- `nn` transformer blocks: `scaledDotProductAttention`,
  `MultiheadAttention` (+ optional `rotary`), `RotaryEmbedding` /
  `applyRotaryEmbedding` (split-half RoPE), `TransformerEncoderLayer`
  (`normFirst`, `bias`, `relu`/`gelu`), `GeGLU` / `geglu`. Every one is
  differential-tested against PyTorch, forward and backward.
- Parameter `dtype` option on every layer: `"f32"` (default), `"f64"`,
  or storage-only `"f16"`/`"bf16"`.
- `optim`: `SGD` (momentum/nesterov), `AdamW`, `Adam`, `RMSprop`, `StepLR`.
- `io`: `writeCheckpoint` / `loadCheckpoint` (MPCK v2; still reads v1),
  `LEGACY_LINEAR_LAYOUT`.
- `@johnhenry/math-plus-tensor-autograd/safetensors` subpath (needs the
  optional peer `@johnhenry/math-plus-safetensors`): `saveSafetensors`,
  `stateDictFromSafetensors`, `loadSafetensors` (path/Blob/URL/bytes, lazy),
  `loadSafetensorsInto`.
- `trainer`: `configure`, `Trainer`, `Batch` (the shape
  `@johnhenry/math-plus-data`'s `collate.xy()` produces).
- `sumToShape` — the broadcast-reduction helper every backward uses.

## Traps

- **Gradients accumulate** across repeated `backward()` calls — `zeroGrad()`
  resets (`.grad` back to `null`, not zeros). Only leaves accumulate; only a
  scalar output may call `backward()` without an explicit `gradOutput`.
- **Parameters default to f32** (since #123; f64 before) — feed inputs of
  the parameters' dtype or hit tensor-core's no-promotion `TypeError`. Pass
  `{ dtype: "f64" }` to a layer for f64. `f16`/`bf16` parameters are
  storage-only: they're upcast to the input's dtype on the fly, fine for
  inference, but `optim.*` can't update them (tensor-core can't compute in
  half) — train in f32/f64.
- **`nn.Linear.weight` is `[out, in]`** (PyTorch's layout, since #123;
  `[in, out]` before). See "Migrating to the `[out, in]` Linear" below.
- **Two bool-mask conventions, both PyTorch's:** `scaledDotProductAttention`'s
  `attnMask` is `true` = *may attend*; `MultiheadAttention` /
  `TransformerEncoderLayer` masks are `true` = *hidden*. A fully-masked
  row is NaN (as in PyTorch).
- **`Variable.gelu()` defaults to the tanh approximation** (tensor-core's
  `gelu`); PyTorch's default is exact — pass `{ approximate: "none" }`.
  `GeGLU`, `geglu` and `TransformerEncoderLayer`'s `"gelu"` use the exact
  form, like PyTorch. `geglu` applies GELU to the *first* half (ModernBERT
  order; diffusers' `GEGLU` uses the second).
- Transformer layers have **no dropout** (they equal PyTorch in `eval()`),
  `MultiheadAttention` has no `kdim`/`vdim`/`add_bias_kv` and returns only
  the output (no attention weights), and `RotaryEmbedding` rotates the whole
  head dim with no scaling variants.
- **`trainer.fit(dataLoader)` ignores `config.epochs`** — one pass, because
  an arbitrary `AsyncIterable` isn't guaranteed re-iterable. Epochs apply
  only to the full-batch `fit({ x, y })` overload; put epochs in the data
  pipeline (`dataset.epochs(n)`) otherwise.
- `binaryCrossEntropy` uses the BCEWithLogits reformulation so saturated
  (|z| ≳ 37) logits give finite loss and gradients, not NaN (issue #85).
- `io.writeCheckpoint` is a custom `"MPCK"` container, **not** NumPy `.npz`
  (and can't hold bf16 — use the safetensors subpath). `loadStateDict` is
  strict both ways by default (missing *and* unexpected keys throw; `{
  strict: false }` loads the intersection), always checks shapes, and casts
  each tensor to the parameter's dtype (like PyTorch's `copy_`).
- SGD `nesterov` without nonzero `momentum` throws (issue #89).
- Telemetry is opt-in: `backward()` emits a trace span and `optim.step()` a
  `optim/gradNorm` metric only when a `@johnhenry/math-plus-telemetry` sink
  is installed — the grad norm isn't even computed otherwise.

## Migrating to the `[out, in]` Linear (0.3)

- **Checkpoints:** nothing to do. MPCK files written by <= 0.2 are version
  1; `io.loadCheckpoint` tags them and `loadStateDict` transposes every
  `nn.Linear` weight and casts f64 values to your parameters' dtype. For an
  old state dict that reached you some other way, pass
  `loadStateDict(dict, { legacyLinearLayout: true })`.
- **dtype:** layers now default to f32. Keep the old numerics with
  `new nn.Linear(i, o, { dtype: "f64" })` (same for `Embedding`,
  `LayerNorm`), or feed f32 inputs (e.g. `collate.xy()`'s default).
- **Code that reads weights directly:** `linear.weight.value` is now
  `[out, in]`; `x.matmul(W)` becomes `x.matmul(W.transpose())`.
- **Type changes:** `LayerNorm.bias` is `Parameter | null`; `Variable.div`
  also accepts a number; `Variable.matmul` accepts batched operands.

## Tests

`npm test` — includes PyTorch differential tests (forward and backward,
inputs and parameters, parameter names checked via `load_state_dict(strict=
True)`) for every view op and transformer layer — `scripts/torch_oracle.py`,
resolved via `$MATH_PLUS_TORCH_ORACLE_PYTHON`, else
`$MATH_PLUS_ORACLE_PYTHON`, else `python3`; skips (never fails) without
torch — safetensors interop both ways with PyTorch, legacy-checkpoint
loading, and a cross-oracle check against `@johnhenry/math`'s
forward-mode `DualNumber`, saturation regressions (#85), scheduler exactness
(#72), and a sparse-Embedding-backward perf guard.

## Provenance

Part of the [math-plus](https://github.com/johnhenry/math-plus) monorepo;
family docs at <https://opensource.johnhenry.me/math/>.
