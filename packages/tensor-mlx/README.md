# @johnhenry/math-plus-tensor-mlx

> **Experimental.** This is the prototype for
> [RFC 0001](../../docs/rfcs/0001-device-backends.md), which is still
> *Proposed*. The API may change or be removed depending on how the RFC is
> decided. It is not re-exported from any other math-plus package.

Native Apple Silicon arrays for math-plus, running on MLX's Metal GPU (or
its CPU backend) from **Node and Bun**. You move data in from a
tensor-core `Tensor` and back out with explicit calls. Nothing is copied
implicitly.

This package contains no native code and no FFI of its own. Everything runs
through [`@johnhenry/backend-mlx`](https://www.npmjs.com/package/@johnhenry/backend-mlx),
a binding to Apple's mlx-c that uses koffi on Node and `bun:ffi` on Bun.
backend-mlx implements the
[`@johnhenry/tensor-backend`](https://www.npmjs.com/package/@johnhenry/tensor-backend)
contract. This package wraps that contract in a math-plus-style,
method-chaining API.

## Install

```bash
npm install @johnhenry/math-plus-tensor-mlx @johnhenry/math-plus-tensor-core
bun add @johnhenry/math-plus-tensor-mlx @johnhenry/math-plus-tensor-core
```

- **Requires macOS on Apple Silicon (darwin/arm64).** On darwin/arm64, npm
  also installs the optional dependency `@johnhenry/backend-mlx-darwin-arm64`
  (a 64 MB download, 207 MB unpacked: libmlxc, libmlx and mlx.metallib
  from MLX 0.32.2).
- On any other platform the package still installs and imports.
  `createMlxDevice()` throws, and `mlxUnavailableReason()` tells you why.
- For other ways to supply `libmlxc.dylib` (`$LAYA_MLXC_PATH`, a local
  build or Homebrew), see backend-mlx's README.

## Quick start

```ts
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { createMlxDevice } from "@johnhenry/math-plus-tensor-mlx";

const mlx = createMlxDevice();                 // you create the device; there is no global default
const x = mlx.fromTensor(Tensor.from([1, 2, 3, 4]).reshape([2, 2])); // explicit upload, one copy
const w = mlx.fromTensor(Tensor.from([0.5, -1, 2, 0]).reshape([2, 2]));

const y = mlx.scope(() => x.matmul(w).add(1).softmax(-1)); // lazy MLX graph; intermediates are freed
const t = await y.toTensor();                  // explicit download: evaluates, then copies into a Tensor
y.dispose(); x.dispose(); w.dispose();
```

## API

**Device.** `createMlxDevice({ device?: "gpu" | "cpu", libPath?, finalizers? })`
returns an `MlxDevice`.

- Transfers in: `fromTensor(t)` and `fromHost(hostTensor)`.
- Graph control: `eval(...arrays)` evaluates the given arrays; with no
  arguments it synchronizes the stream.
- Lifetime: `scope(fn)` frees every array created inside `fn` except the
  arrays it returns, directly or one level deep in an array or object.
- Ops: `where(cond, a, b)`.
- Introspection: `liveArrays()`, `memory()`, `destroy()`, `name`
  (`"mlx"`), `kind`, and `info` (which library loaded, `node` or `bun`, and
  the mlx-c ABI).
- `backend` exposes the raw `@johnhenry/tensor-backend` `Backend`, for
  code written against that contract and for its conformance suite.

`mlxUnavailableReason()` returns a string saying why MLX can't run here,
or `null` if it can.

**`MlxArray`.**

- Properties: `shape`, `dtype`, `ndim`, `size`, `device`, `disposed`.
- Transfers out and lifetime: `toTensor()` (async), `toHost()` (async,
  returns a `HostTensor`), `eval()` (returns `this`), `dispose()`
  (idempotent).
- Elementwise, with NumPy broadcasting: `add`, `sub`, `mul`, `div`,
  `maximum`, `minimum`. Each takes an array or a number. `neg`, `exp`,
  `log`, `relu`, and `gelu` (exact erf).
- Reductions: `sum`, `mean`, `max` and `min`, each taking
  `(axis?, { keepDims? })`. With no axis they reduce over every element.
  `softmax(axis = -1)`.
- Linear algebra and NN: `matmul` (batched, with broadcast leading dims)
  and `layerNorm(weight?, bias?, eps = 1e-5)` over the last axis.
- Shape and dtype: `cast(dtype)`, `reshape(shape)`, `transpose(axes?)`.

**Host helpers**, which need no MLX: `hostFromTensor(t)` and
`tensorFromHost(h)` return zero-copy views across the tensor-core and
`HostTensor` boundary. Also exported: `DEVICE_DTYPES` and `isDeviceDType`.

## Rules

- **Transfers are explicit.** Data enters only through `fromTensor` or
  `fromHost`, and leaves only through `toTensor` or `toHost`.
  - Passing a tensor-core `Tensor`, or an array from another `MlxDevice`,
    to an op throws. Nothing is uploaded behind your back (PLAN.md
    non-goal 5).
  - `fromTensor` makes exactly one copy, from the tensor's own storage into
    MLX unified memory. It reads a `subarray` view of that storage, so
    there is no intermediate packing.
  - A non-contiguous tensor, such as a transposed view, is rejected. Call
    `.contiguous()` yourself first.
  - `toTensor` makes one copy out, and the `Tensor` wraps that buffer.
- **dtypes.** The device holds `f32`, `f16`, `bf16`, `i32` and `bool`,
  under the same names as tensor-core.
  - There is no implicit promotion. Binary ops need matching dtypes.
    Number operands take the array's dtype, so `f16.mul(0.5)` stays f16.
  - `exp`, `log`, `gelu`, `mean`, `softmax` and `layerNorm` need a float
    dtype.
  - `cast()` is the only way to change dtype, even though MLX itself would
    promote.
  - f16 and bf16 use tensor-core's storage layout: raw IEEE bits in a
    `Uint16Array`. On upload, f16 is re-viewed as a `Float16Array` without
    a copy.
  - `f64` is refused (Metal has no float64): `cast("f32")` first. The
    other integer dtypes, including `i64`, are also refused; cast them to
    `i32` yourself.
- **Lazy execution.** Every op appends a node to MLX's graph and returns
  immediately.
  - Shape errors still throw at the call site. Only the arithmetic is
    deferred, and it runs at `eval()`, `toTensor()` or `toHost()`.
  - `compile` (MLX's fusing compiler) is not exposed. It stays opt-in (see
    the RFC).
- **Lifetime.** MLX memory is freed by `dispose()` or `scope()`, not by
  the JS garbage collector. The FinalizationRegistry in backend-mlx is only
  a safety net.
  - A pending graph keeps its own inputs alive.
  - Using a disposed array throws.

## Tests

```bash
npm test -w @johnhenry/math-plus-tensor-mlx     # node:test
npm run test:bun -w @johnhenry/math-plus-tensor-mlx
```

- **`test/differential.test.ts`** compares every op against a **NumPy
  oracle**, `scripts/numpy_oracle.py`, which is resolved as
  `$MATH_PLUS_ORACLE_PYTHON`, else `python3` (see docs/TESTING.md).
  - Float ops run in f32 with tight tolerances, and in f16 on f16-rounded
    inputs within 2e-2.
  - Casts are compared bit-exact with NumPy's `astype`.
- **`test/conformance.test.ts`** runs `@johnhenry/tensor-backend`'s shared
  conformance suite (49 cases, f32 and f16) against `device.backend` on the
  GPU and CPU devices.
- **`test/bridge.test.ts`** covers the transfer and lifetime rules above.
  Its host-view half runs on every platform.

The suites **skip, never fail**, when MLX is unavailable (not darwin/arm64,
or no libmlxc) or when numpy is missing. On an Apple Silicon machine with
numpy, a real run must report **0 skipped** (89 tests on Node and Bun).

GPU etiquette on shared machines: wrap GPU test runs in the `~/gpu.lock`
convention, for example
`until shlock -p $$ -f ~/gpu.lock; do sleep 3; done; npm test -w @johnhenry/math-plus-tensor-mlx; rm -f ~/gpu.lock`.

## Limitations (what this does not do)

- **darwin/arm64 only.** Deno is unsupported: backend-mlx has no
  `Deno.dlopen` adapter. For that reason the package is deliberately **not
  published to JSR** (`JSR_EXCLUDED_DIRS` in
  `scripts/sync-jsr-configs.mjs`).
- **The op set is the tensor-backend contract's.** Missing ops are
  composed from it: `minimum` is `-maximum(-a, -b)`, `min` is `-max(-x)`,
  and `mean` is `sum · (1/n)`. The package has no `sqrt`, `pow`,
  comparisons, `argmax`, slicing, indexing, `concat`, random numbers or
  linalg beyond `matmul`. The RFC proposes growing the contract upstream,
  not forking it.
- **No autograd.** `@johnhenry/math-plus-tensor-autograd` works on
  tensor-core `Tensor`s only. Autograd on device arrays is an RFC open
  question.
- **Reductions take one axis or all axes**, not a list of axes.
- **No `compile`.** The contract's `compile` exists on `device.backend`,
  but it is not wrapped for `MlxArray`.
- **Every upload and download copies once.** There is no zero-copy wrapping
  of JS memory. Reads are synchronous under the hood: the promise from
  `toTensor()` is already settled when it is returned.
- **f16 accumulation follows MLX.** `softmax` asks for f32 accumulation
  (`precise=true`). `sum`, `mean` and `matmul` use whatever MLX's own
  kernels do for f16. This package checks f16 results only to the 2e-2
  tolerance described above, so it does not guarantee f16 accuracy.
- **Reproducibility.** Results can differ from tensor-core's CPU results in
  the last ulps. They are compared within per-op tolerances and are never
  expected to be byte-identical (PLAN.md non-goal: no byte-exact
  cross-platform snapshots).
- `gelu` is the exact erf form, which matches PyTorch's default and MLX. It
  is not tensor-core's current tanh approximation (see #122).

## Provenance

Built for [#125](https://github.com/johnhenry/math-plus/issues/125) as the
prototype for [#124](https://github.com/johnhenry/math-plus/issues/124)'s
RFC. The binding, its prebuilt runtime and the conformance fixtures come
from [laya-js](https://github.com/johnhenry/laya-js) (Apache-2.0). See that
repo's `docs/mlx-binding-decision.md` for measurements (Apple M2: 0.55–0.8 µs
FFI overhead per op, and f16 GEMM at Python-MLX speed). Part of the
[math-plus](https://github.com/johnhenry/math-plus) monorepo; family docs
are at <https://opensource.johnhenry.me/math/>.
