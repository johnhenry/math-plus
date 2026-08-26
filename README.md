# Math Plus

A JS/TypeScript-native numeric computation runtime — a NumPy + PyTorch + pandas + SciPy equivalent for Node/Deno/browser, built on Rust→WASM kernels, optional WebGPU acceleration, and Apache Arrow for tabular data.

Part of the **math** family: the high-performance sibling of [`@johnhenry/math`](https://github.com/johnhenry/math) (education/CAS-oriented scalar math), reusing its scalar types (`ComplexNumber`, `Rational`, `Decimal`) at tensor API edges and bridging its `Symbolic` CAS into the tensor compiler. Data pipelines build on [`@johnhenry/iteration`](https://github.com/johnhenry/math) (a pull-based async iterator/transducer toolkit living in the same monorepo). Family documentation: **[opensource.johnhenry.me/math/](https://opensource.johnhenry.me/math/)**.

**Status:** actively published. Everything ships independently under `@johnhenry/math-plus-*` on npm (and JSR, mostly) — install only what you need; a project that wants an FFT doesn't pull in a WebGPU backend. See each package's own `CHANGELOG.md` for release history, [docs/PLAN.md](./docs/PLAN.md) for the original implementation plan, and [docs/perplexity-conversation.md](./docs/perplexity-conversation.md) for the source design conversation.

## Which package do I want?

| I want to... | Start with |
|---|---|
| Work with n-dimensional arrays | [`tensor-core`](./packages/tensor-core) — everything tensor-shaped builds on it |
| Train something / take gradients | [`tensor-autograd`](./packages/tensor-autograd) |
| Fuse elementwise expressions | [`tensor-compile`](./packages/tensor-compile) |
| Go faster on CPU | [`tensor-wasm`](./packages/tensor-wasm) — read its README first; it's a separate storage type, not a drop-in backend |
| Go faster on GPU | [`tensor-webgpu`](./packages/tensor-webgpu) — browser-only in v1, and read its "honest threshold" section |
| FFTs / filters / peaks | [`fft`](./packages/fft), [`signal`](./packages/signal) |
| Resize/normalize images | [`image`](./packages/image) |
| Dataframes | [`frame-arrow`](./packages/frame-arrow) (+ [`frame-parquet`](./packages/frame-parquet) for Parquet I/O) |
| Dataset pipelines for training | [`data`](./packages/data) |
| Talk to it from an agent | [`mcp`](./packages/mcp) (`npx math-plus-mcp`) |
| Exchange data with Python | [`interop-python`](./packages/interop-python) (PyPI: `johnhenry-math-plus-interop`) |

## Packages

### Tensors

| Package | Role |
|---|---|
| [`@johnhenry/math-plus-tensor-core`](./packages/tensor-core) | Typed n-D arrays: dtypes, strides/views, broadcasting, `.npy` I/O. Pure JS, zero deps. |
| [`@johnhenry/math-plus-tensor-autograd`](./packages/tensor-autograd) | Reverse-mode tape, `nn.*`, `optim.*`, trainer, checkpoints |
| [`@johnhenry/math-plus-tensor-compile`](./packages/tensor-compile) | Expression IR + elementwise fusion (opt-in); the shared lowering target for WGSL |
| [`@johnhenry/math-plus-tensor-wasm`](./packages/tensor-wasm) | Rust→WASM kernels (SIMD, arena allocator, zero-alloc `...Into` ops) + opt-in Deno-native FFI |
| [`@johnhenry/math-plus-tensor-webgpu`](./packages/tensor-webgpu) | WebGPU GEMM, attention primitives, IR→WGSL fusion. Chromium-family browsers only. |

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
