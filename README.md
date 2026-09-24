# Math Plus

[![CI](https://github.com/johnhenry/math-plus/actions/workflows/ci.yml/badge.svg)](https://github.com/johnhenry/math-plus/actions/workflows/ci.yml)
[![license](https://img.shields.io/npm/l/%40johnhenry%2Fmath-plus-tensor-core.svg)](LICENSE)

Full documentation: [opensource.johnhenry.me/math](https://opensource.johnhenry.me/math/)

A JS/TypeScript-native numeric computation runtime — a NumPy + PyTorch + pandas + SciPy equivalent for Node/Deno/browser, built on Rust→WASM kernels, optional WebGPU acceleration, and Apache Arrow for tabular data.

**Status:** actively published. Everything ships independently under `@johnhenry/math-plus-*` on npm (and JSR, mostly) — install only what you need; a project that wants an FFT doesn't pull in a WebGPU backend. See each package's own `CHANGELOG.md` for release history, [docs/PLAN.md](./docs/PLAN.md) for the original implementation plan, and [docs/perplexity-conversation.md](./docs/perplexity-conversation.md) for the source design conversation.

## Contents

- [Which package do I want?](#which-package-do-i-want)
- [Packages](#packages)
- [Adding a new package](#adding-a-new-package)
- [Examples](#examples)
- [Working in this repo](#working-in-this-repo)
- [Family](#family)

## Which package do I want?

| I want to... | Start with |
|---|---|
| Work with n-dimensional arrays | [`tensor-core`](./packages/tensor-core) — everything tensor-shaped builds on it |
| Train something / take gradients | [`tensor-autograd`](./packages/tensor-autograd) |
| Fuse elementwise expressions | [`tensor-compile`](./packages/tensor-compile) |
| Go faster on CPU | [`tensor-wasm`](./packages/tensor-wasm) — read its README first; it's a separate storage type, not a drop-in backend |
| Go faster on GPU | [`tensor-webgpu`](./packages/tensor-webgpu) — a facade over `@johnhenry/backend-webgpu`: browsers, Deno, and Node/Bun via Dawn; read its GEMM-threshold section |
| Run `@johnhenry/tensor-backend` model code on plain CPU (the reference backend) | [`tensor-cpu`](./packages/tensor-cpu) — pure TypeScript, every contract op native, built on tensor-core's kernels |
| Native Apple Silicon GPU from Node/Bun/Deno (experimental) | [`tensor-mlx`](./packages/tensor-mlx) — MLX via `@johnhenry/backend-mlx`; darwin/arm64 only; prototype for [RFC 0001](./docs/rfcs/0001-device-backends.md) |
| The same chainable array code on CPU, MLX or WebGPU | `createCpuDevice()` / `createMlxDevice()` / `await createWebGpuDevice()` — one `DeviceArray` API ([tensor-cpu](./packages/tensor-cpu#the-shared-device-array-api-arraydevice--devicearray)): `await dev.fromTensor(t)`, `x.matmul(w).add(1).softmax()`, `await y.toTensor()` |
| FFTs / filters / peaks | [`fft`](./packages/fft), [`signal`](./packages/signal) |
| Resize/normalize images | [`image`](./packages/image) |
| Load/save model weights (`.safetensors`) | [`safetensors`](./packages/safetensors) — lazy reads from files, Blobs and HTTP ranges |
| Dataframes | [`frame-arrow`](./packages/frame-arrow) (+ [`frame-parquet`](./packages/frame-parquet) for Parquet I/O) |
| Dataset pipelines for training | [`data`](./packages/data) |
| Talk to it from an agent | [`mcp`](./packages/mcp) (`npx math-plus-mcp`) |
| Exchange data with Python | [`interop-python`](./packages/interop-python) (PyPI: `johnhenry-math-plus-interop`) |

## Packages

### Tensors

| Package | Role |
|---|---|
| [`@johnhenry/math-plus-tensor-core`](./packages/tensor-core) | Typed n-D arrays: dtypes, strides/views, broadcasting, `.npy` I/O. Pure JS; its one dependency is `special`. |
| [`@johnhenry/math-plus-tensor-autograd`](./packages/tensor-autograd) | Reverse-mode tape, `nn.*`, `optim.*`, trainer, checkpoints |
| [`@johnhenry/math-plus-tensor-compile`](./packages/tensor-compile) | Expression IR + elementwise fusion (opt-in); the shared lowering target for WGSL |
| [`@johnhenry/math-plus-tensor-wasm`](./packages/tensor-wasm) | Rust→WASM kernels (SIMD, arena allocator, zero-alloc `...Into` ops) + opt-in Deno-native FFI |
| [`@johnhenry/math-plus-tensor-webgpu`](./packages/tensor-webgpu) | WebGPU device facade over `@johnhenry/backend-webgpu` (GEMM, fused attention, the tensor-backend ops) with the shared chainable `DeviceArray` API, plus IR→WGSL fusion. Browsers, Deno, and Node/Bun via Dawn. |
| [`@johnhenry/math-plus-tensor-cpu`](./packages/tensor-cpu) | The CPU reference `Backend` for the `@johnhenry/tensor-backend` contract (RFC 0001 §12 Q3): f32 compute on tensor-core's kernels (`/kernels` subpath), every optional op native, passes the conformance suite. Also the home of the one chainable device-array API (`DeviceArray`) every device facade shares, and `createCpuDevice()`. Pure TypeScript; Node, Bun, Deno, browsers. |
| [`@johnhenry/math-plus-tensor-mlx`](./packages/tensor-mlx) | **Experimental.** MLX (Metal) arrays (the shared `DeviceArray`) on Node, Bun and Deno with explicit `fromTensor`/`toTensor` transfers, over the `@johnhenry/tensor-backend` contract. darwin/arm64 only; configured for JSR, not yet published there. |
| [`@johnhenry/math-plus-safetensors`](./packages/safetensors) | safetensors reader/writer: validated headers, typed views (F16 as `Float16Array`), lazy reads from files/Blobs/HTTP ranges, optional tensor-core interop. Zero deps. |
| [`@johnhenry/math-plus-special`](./packages/special) | The one canonical double-precision `erf`/`erfc`/GELU (SciPy-verified). Zero deps; used by tensor-core (which re-exports it) and frame-arrow's `fn.erf`. |

### Signal & media

| Package | Role |
|---|---|
| [`@johnhenry/math-plus-fft`](./packages/fft) | `ComplexTensor` + `fft`/`ifft`/`rfft`/`irfft`/`fft2`/`fftn` |
| [`@johnhenry/math-plus-signal`](./packages/signal) | `convolve`/`stft`/`welch`/`findPeaks`/`sosFilter`/`butter`/`resamplePoly` (SciPy-equivalent slice) |
| [`@johnhenry/math-plus-image`](./packages/image) | resize/normalize tensor ops for ML/media pipelines |

### Data

| Package | Role |
|---|---|
| [`@johnhenry/math-plus-frame-arrow`](./packages/frame-arrow) | Immutable Arrow-backed `Frame`/`Series` with a lazy expression API |
| [`@johnhenry/math-plus-frame-parquet`](./packages/frame-parquet) | Parquet scan/write with real projection/predicate pushdown |
| [`@johnhenry/math-plus-data`](./packages/data) | Async dataset pipelines: batch/shuffle/epochs/mapConcurrent/prefetch |
| [`@johnhenry/math-plus-scalar-types`](./packages/scalar-types) | Re-export of `@johnhenry/math` scalars + tensor-boundary converters |

### Interop & infrastructure

| Package | Role |
|---|---|
| [`@johnhenry/math-plus-mcp`](./packages/mcp) | MCP server: symbolic CAS + guarded numeric tools for agents (stdio, `npx math-plus-mcp`) |
| [`johnhenry-math-plus-interop`](./packages/interop-python) | **PyPI**, module `math_plus_interop`: Arrow IPC/Parquet/npy helpers for the Python side |
| [`@johnhenry/math-plus-telemetry`](./packages/telemetry) | Shared event schema + sink registry, zero-cost no-op default |
| [`@johnhenry/math-plus-adapter-math`](./adapters/adapter-math) | Bridge to `@johnhenry/math` (Matrix/Vector ↔ Tensor, Symbolic → IR, Graph → CSR) |
| [`@johnhenry/math-plus-adapter-onnx`](./adapters/adapter-onnx) | ONNX Runtime Web wrapper (Tensor marshalling) |
| [`@johnhenry/math-plus-unit`](./scalars/unit) | Unit/dimension scalar type with dimensional-analysis-checked arithmetic |

## Adding a new package

`@johnhenry/math-plus-signal` is the real worked example (`signal: new
package -- convolve/stft/istft/findPeaks/sosFilter/butter/resamplePoly`,
[issue #44](https://github.com/johnhenry/math-plus/issues/44)): a SciPy-shaped
slice of functionality that didn't fit inside `@johnhenry/math-plus-fft`
(which stays a pure Fourier-transform package) or any other existing
cluster, so it got its own npm identity.

**Smallest: a new function on an existing package's exports.** Most new
numeric functionality is one more export from an existing package —
another `nn.*` layer on `tensor-autograd`, another filter on `signal` once
it exists — reusing that package's dtype/broadcasting/oracle machinery. No
new package, no new npm identity, no new semver line, no new row in `##
Which package do I want?` — the whole cost is the export itself.

**A genuinely new package is warranted when the functionality needs its
own install footprint** — a project that wants FFTs shouldn't have to pull
in dataframes — **or crosses into a distinct dependency/runtime shape**
(`tensor-webgpu`'s WebGPU runtime dependency, `interop-python`'s
separate PyPI distribution). `signal` is the harder case: it isn't a new
runtime shape, just a decision that SciPy's `signal` module maps to its
own npm package rather than growing `fft` past what "Fourier transforms"
means.

Every existing package follows the same small, repeatable pattern, so a
new one does too:

1. **`packages/<name>/package.json`** — `name: "@johnhenry/math-plus-<name>"`,
   `version: "0.0.0"`, matching `publishConfig`/`exports` shape — copy an
   existing package's, e.g. `signal`'s or `fft`'s for a numeric package.
2. **`packages/<name>/tsconfig.json` (+ `tsconfig.typecheck.json`)** —
   copy-paste of an existing package's pair.
3. **Root `package.json`'s `build` and `test` script strings** — unlike
   `@johnhenry/math`'s monorepo (a plain `packages/*` glob with nothing
   else to edit), this repo's `build`/`test` scripts enumerate every
   package by `-w @johnhenry/math-plus-<name>` explicitly; a package left
   out of these strings never builds or tests in CI even though
   `workspaces` picks it up for `npm install`. This is the step
   [#47](https://github.com/johnhenry/math-plus/issues/47)'s planned
   manifest-drift check exists to catch automatically.
4. **The one part that isn't boilerplate: `scripts/sync-jsr-configs.mjs`'s
   `PACKAGE_DIRS`.** Every package also publishes to JSR (mostly), and JSR
   config generation is driven by this one hand-maintained list — a
   package present in npm's workspace list but absent here silently never
   gets a `jsr.json`, and nothing fails loudly about it today (this is the
   other half of what #47 is meant to close).

**Tests.** New numeric packages get a differential oracle wherever a
reference implementation exists (NumPy for tensor ops, `scipy.signal` for
`signal`, pyarrow/pandas for frame packages) — see `AGENTS.md`'s "Oracle
discipline" section. Run `npm run example:NN` for the package's example
once one is added to `examples/`.

Add the row to this README's `## Which package do I want?` and `## Packages`
tables. See `@johnhenry/math`'s own "Adding a new package" section for the
simpler contrasting case — a sibling monorepo where the plain `packages/*`
glob needs no manual root-script registration at all.

## Examples

Runnable, one-per-cluster walkthroughs live in [`examples/`](./examples):

```bash
npm install
npm run build        # WASM kernels + TypeScript (needs Rust + lld for the kernels)
npm run examples     # the env-independent set (01-09; CI runs this)
npm run examples:all # includes the WASM example (needs the built artifact)
```

## Working in this repo

```bash
npm install
npm run build        # build:wasm first (rustup target add wasm32-unknown-unknown; lld), then tsc
npm test             # manifest-drift guard + every workspace's tests
```

Differential tests skip (never fail) without their oracles — NumPy/SciPy via `MATH_PLUS_ORACLE_PYTHON`, headless Chrome via `MATH_PLUS_CHROME_PATH`; CI verifies the oracles are importable so a green run can't be a silently-skipped one. The WASM SIMD benchmark is deliberately **not** in `npm test` (mixed CI runner fleets make wall-clock thresholds meaningless — `npm run test:bench -w @johnhenry/math-plus-tensor-wasm` on known hardware instead).

## Family

Part of the **math** family, alongside [`@johnhenry/math`](https://github.com/johnhenry/math).

- **[`@johnhenry/math`](https://github.com/johnhenry/math)** — the
  education/CAS-oriented scalar-math sibling. `math-plus` reuses its scalar
  types (`ComplexNumber`, `Rational`, `Decimal`) at tensor API edges via
  `@johnhenry/math-plus-scalar-types` and `@johnhenry/math-plus-adapter-math`,
  and bridges its `Symbolic` CAS into this repo's tensor compiler. The two
  repos make opposite trade-offs on purpose: `@johnhenry/math`'s boxed,
  generic elements are precisely what this repo's SIMD-friendly tensor
  runtime forbids, and vice versa — neither is a subset of the other.
- **[`@johnhenry/iteration`](https://github.com/johnhenry/math)** — a
  pull-based async iterator/transducer toolkit living in the same `math`
  monorepo. `@johnhenry/math-plus-data`'s dataset pipelines build on the same
  transducer/backpressure ideas, though not as a direct dependency.
