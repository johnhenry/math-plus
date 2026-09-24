# @johnhenry/math-plus-tensor-mlx

> **Experimental.** This is the prototype for
> [RFC 0001](../../docs/rfcs/0001-device-backends.md) (*Accepted with
> changes*, 2026-09-24). The API is 0.x and may still change. It is not
> re-exported from any other math-plus package.

Native Apple Silicon arrays for math-plus, running on MLX's Metal GPU (or
its CPU backend) from **Node, Bun and Deno 2**. You move data in from a
tensor-core `Tensor` and back out with explicit, async calls. Nothing is
copied implicitly.

This package contains no native code and no FFI of its own. Everything runs
through [`@johnhenry/backend-mlx`](https://www.npmjs.com/package/@johnhenry/backend-mlx),
a binding to Apple's mlx-c that uses koffi on Node, `bun:ffi` on Bun and
`Deno.dlopen` on Deno.
backend-mlx implements the
[`@johnhenry/tensor-backend`](https://www.npmjs.com/package/@johnhenry/tensor-backend)
contract. This package wraps that contract in a math-plus-style,
method-chaining API.

## Install

```bash
npm install @johnhenry/math-plus-tensor-mlx @johnhenry/math-plus-tensor-core
bun add @johnhenry/math-plus-tensor-mlx @johnhenry/math-plus-tensor-core
deno add jsr:@johnhenry/math-plus-tensor-mlx jsr:@johnhenry/math-plus-tensor-core
```

- **Requires macOS on Apple Silicon (darwin/arm64).** On darwin/arm64, npm
  also installs the optional dependency `@johnhenry/backend-mlx-darwin-arm64`
  (a 64 MB download, 207 MB unpacked: libmlxc, libmlx and mlx.metallib
  from MLX 0.32.2).
- On any other platform the package still installs and imports.
  `createMlxDevice()` throws, and `mlxUnavailableReason()` tells you why.
- For other ways to supply `libmlxc.dylib` (`$LAYA_MLXC_PATH`, a local
  build or Homebrew), see backend-mlx's README.

### Deno

Deno 2 loads mlx-c through `Deno.dlopen` (backend-mlx 0.3+). Tested on
Deno 2.9.7.

- **Permissions.** Run with `--allow-ffi --allow-read --allow-env` (or
  `-A`). The last two let backend-mlx find the library.
- **The native library.** It comes one of two ways:
  - The platform package `npm:@johnhenry/backend-mlx-darwin-arm64`. tensor-mlx
    imports backend-mlx as `npm:@johnhenry/backend-mlx`, so Deno
    installs the platform package (backend-mlx's optional dependency for
    darwin/arm64) along with it, and backend-mlx finds it in `node_modules`
    or in Deno's npm cache. This was checked with an empty `DENO_DIR` and
    no `node_modules`. If you import `jsr:@johnhenry/backend-mlx`
    directly, add it yourself:
    `deno add npm:@johnhenry/backend-mlx-darwin-arm64`.
  - Or point `LAYA_MLXC_PATH` at a `libmlxc.dylib` file, or at a directory
    containing one (for example a local `build-mlxc.sh` build). This wins
    over the platform package. A set variable that points nowhere is an
    error, not a fallback.
- `mlxUnavailableReason()` says why when neither is found, and
  `device.info.runtime` reports `"deno"`.

## Quick start

```ts
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { createMlxDevice } from "@johnhenry/math-plus-tensor-mlx";

const mlx = createMlxDevice();                 // you create the device; there is no global default
const [x, w] = await Promise.all([             // explicit async uploads, one copy each
  mlx.fromTensor(Tensor.from([1, 2, 3, 4]).reshape([2, 2])),
  mlx.fromTensor(Tensor.from([0.5, -1, 2, 0]).reshape([2, 2])),
]);

const y = mlx.scope(() => x.matmul(w).add(1).softmax(-1)); // lazy MLX graph; intermediates are freed
const best = x.argmax(-1);                     // i32 indices, still on the device
const t = await y.toTensor();                  // explicit async download: evaluates, then copies into a Tensor
y.dispose(); best.dispose(); x.dispose(); w.dispose();
```

## API

**Device.** `createMlxDevice({ device?: "gpu" | "cpu", libPath?, finalizers? })`
returns an `MlxDevice`.

- Transfers in: `await fromTensor(t)` and `await fromHost(hostTensor)`.
  Both return a Promise (since 0.2). Validation errors (non-contiguous
  tensor, unsupported dtype) still throw synchronously.
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
- Arithmetic, with NumPy broadcasting: `add`, `sub`, `mul`, `div`,
  `maximum`, `minimum` and `pow`. Each takes an array or a number.
- Unary math: `neg`, `abs`, `exp`, `log`, `sqrt`, `rsqrt`, `tanh`,
  `sigmoid`, `erf`, `relu` and `gelu` (exact erf).
- Comparisons, returning bool, each taking an array or a number: `equal`,
  `notEqual`, `less`, `lessEqual`, `greater` and `greaterEqual`.
- Logic on bool arrays: `logicalAnd`, `logicalOr` and `logicalNot`.
- Reductions: `sum`, `mean`, `max`, `min`, `argmax` and `argmin`, each
  taking `(axis?, { keepDims? })`. With no axis they reduce over every
  element (`argmax`/`argmin` then index the flattened array, like NumPy).
  `argmax`/`argmin` return i32 and pick the first of equal values.
- Scans and normalisation: `cumsum(axis?)` (inclusive; with no axis, over
  the flattened array) and `softmax(axis = -1)`.
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
    Number operands take the array's dtype, so `f16.mul(0.5)` stays f16,
    and `i32.add(2.9)` adds 2 (truncation, like `numpy.asarray(v, int32)`).
    An i32 operand outside the i32 range throws.
  - Number operands are never uploaded: the constant is built on the
    device from the array itself (a few graph nodes), because uploads are
    async and ops are not.
  - `exp`, `log`, `sqrt`, `rsqrt`, `pow`, `tanh`, `sigmoid`, `erf`,
    `gelu`, `mean`, `softmax` and `layerNorm` need a float dtype.
  - Arithmetic, `neg`, `abs`, reductions, `argmax`/`argmin` and `cumsum`
    refuse bool. Comparisons take any dtype (both sides the same) and
    return bool. `logicalAnd`/`Or`/`Not` need bool inputs:
    `cast("bool")` first, there is no implicit truthiness.
  - `argmax`/`argmin` return i32. `cumsum` keeps its input dtype.
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
npm test -w @johnhenry/math-plus-tensor-mlx          # node:test
npm run test:bun -w @johnhenry/math-plus-tensor-mlx
npm run test:deno -w @johnhenry/math-plus-tensor-mlx # Deno 2 (node:test via Deno's Node compat)
```

`test:deno` runs the same files with `deno test -A
--node-modules-dir=manual`, so it uses the repo's `node_modules` (and the
platform package in it). Deno type-checks them too, against the built
`dist/` of the workspace packages: the build rewrites the `./x.ts`
specifiers tsc leaves in `.d.ts` files, and points each `dist/*.js` at its
declarations with `@ts-self-types` (`scripts/rewrite-dts-extensions.mjs`,
issue #157). Build first.

- **`test/differential.test.ts`** compares every op against a **NumPy
  oracle**, `scripts/numpy_oracle.py`, which is resolved as
  `$MATH_PLUS_ORACLE_PYTHON`, else `python3` (see docs/TESTING.md).
  - Float ops run in f32 with tight tolerances, in f16 on f16-rounded
    inputs within 2e-2, and in bf16 on bf16-rounded inputs within 5e-2.
  - Comparisons, logical ops, `argmax`/`argmin` and i32 arithmetic are
    compared exactly, including the result dtype (bool / i32).
  - Casts are compared bit-exact with NumPy's `astype`.
- **`test/conformance.test.ts`** runs `@johnhenry/tensor-backend`'s shared
  conformance suite (the core op cases plus the general-numerics cases,
  in f32, f16 and bf16) against `device.backend` on the GPU and CPU
  devices.
- **`test/bridge.test.ts`** covers the transfer and lifetime rules above.
  Its host-view half runs on every platform.

The suites **skip, never fail**, when MLX is unavailable (not darwin/arm64,
or no libmlxc) or when numpy is missing. On an Apple Silicon machine with
numpy, a real run must report **0 skipped** (213 tests on Node, Bun and
Deno).

GPU etiquette on shared machines: wrap GPU test runs in the `~/gpu.lock`
convention, for example
`until ( set -o noclobber; echo $$ > ~/gpu.lock ) 2>/dev/null; do sleep 3; done; trap 'rm -f ~/gpu.lock' EXIT; npm test -w @johnhenry/math-plus-tensor-mlx`.

## Limitations (what this does not do)

- **darwin/arm64 only**, on Node, Bun or Deno 2. There is no browser
  build (MLX needs FFI).
- **The op set is the tensor-backend contract's.** The general-numerics
  ops go through tensor-backend's compose helpers, and backend-mlx has a
  native mlx-c kernel for every one of them. Only elementwise `minimum` is
  composed here, as `-maximum(-a, -b)`, because the contract has no
  `minimum`. The package has no slicing, indexing, `concat`, `sort`,
  random numbers or linalg beyond `matmul`, although the contract has some
  of these. New ops go into the contract upstream (RFC 0001 §12 Q7), not
  into a fork.
- **Logical ops take bool only**, although the contract's are
  nonzero-is-true. Cast first.
- **No autograd.** `@johnhenry/math-plus-tensor-autograd` works on
  tensor-core `Tensor`s only. Autograd on device arrays is an RFC open
  question.
- **Reductions take one axis or all axes**, not a list of axes.
- **No `compile`.** The contract's `compile` exists on `device.backend`,
  but it is not wrapped for `MlxArray`.
- **Every upload and download copies once.** There is no zero-copy wrapping
  of JS memory. Both directions are synchronous under the hood (MLX copies
  at call time), so the Promises from `fromTensor()` and `toTensor()` are
  already settled when they are returned. They are async so that a
  transfer always looks like one (PLAN.md non-goal 5). Await a batch of
  uploads together with `Promise.all`.
- **f16/bf16 accumulation follows MLX.** `softmax` asks for f32
  accumulation (`precise=true`). `sum`, `mean`, `cumsum` and `matmul` use
  whatever MLX's own kernels do. This package checks f16 results only to
  2e-2 and bf16 results to 5e-2, so it does not guarantee more accuracy
  than that.
- **NaN handling in comparisons, `argmax`/`argmin` and `min`/`max` is
  MLX's**, and the tests do not cover NaN inputs.
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
