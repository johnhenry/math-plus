---
title: "Math Plus — Implementation Plan"
source: docs/perplexity-conversation.md
generated: 2026-08-10
status: "in progress — v1 shipped in full, v2 underway. This document is the design record; ROADMAP.md and the GitHub issue tracker are the live status source (updated 2026-08-11, see below)."
---

# Math Plus — Implementation Plan

**What this is:** a JS/TypeScript-native numeric computation runtime — a NumPy + PyTorch + TensorFlow + SciPy + pandas equivalent for Node/Deno/browser — built on Rust→WASM kernels, optional WebGPU acceleration, and Apache Arrow for tabular data.

**Naming (decided 2026-08-10):** the source conversation used a placeholder scope `@jh/runtime`. Final naming: **unscoped `mallory-*` npm packages**, joining the existing math family (`mallory-math`) — the `@mallory` npm scope is already owned by another npm user, and unscoped matches `mallory-math`'s convention (all `mallory-*` names below verified available on npm 2026-08-10). Umbrella: `mallory-runtime`; core: `mallory-tensor-*` / `mallory-frame-*`; adapters: `mallory-adapter-<target>`; PyPI: `mallory-interop`.

> **Addendum (later renamed):** this repo and its packages were subsequently renamed to the
> `@johnhenry` npm scope — `mallory-plus` → `math-plus`, every `mallory-<x>` package →
> `@johnhenry/math-plus-<x>`, and `mallory-interop` (PyPI) → `johnhenry-math-plus-interop`. The
> sibling `mallory-math`/`mallory-iteration` packages became `@johnhenry/math`/`@johnhenry/iteration`,
> and `mallory-graph` became `mallory` (see `docs/FAMILY.md` and `docs/RELEASING.md` for why and the
> current names). This decision record is left as originally written below for historical accuracy.

**math family integration (decided 2026-08-10, see §0):** mallory-plus reuses `mallory-math`'s scalar types via a thin `scalar-types` package, adds a `mallory-adapter-math` bridge (Matrix↔Tensor, Symbolic→IR, Graph→CSR, DualNumber test oracle), builds the v2 `data` namespace on `async-itertools@^2` (after fixing its transduce memory leak and adding types/concurrency/cancellation upstream), and treats mallory-graph's `CellGraph` as design prior art for the frame lazy planner. `mallory-math` itself stays independent with zero dependencies.

**Source material:** the full design conversation is at [`docs/perplexity-conversation.md`](./perplexity-conversation.md) (extracted from Perplexity, 6 turns, 268 cited sources). This plan synthesizes that conversation plus follow-up research into a concrete build order. Four independent passes mined the source conversation for: (1) the tensor/autograd core, (2) the Arrow dataframe + Python interop layer, (3) WebGPU + SciPy-equivalent scientific computing, and (4) repo structure, adapters, non-goals, and release strategy. Their findings are consolidated below.

**Explicit framing from the source conversation, which this plan treats as the north star:**
> "The resulting project would not be 'accelerated math.js' or 'Danfo rewritten in Rust.' It would be a modern JS computation substrate that adopts the best parts of those APIs while making fixed-width buffers, explicit devices, WASM kernels, Arrow tables, and opt-in compilation the real foundation."

---

## 0. Positioning, Alternatives & Runtime Targets

**Alternatives considered — the source conversation's Turn 3 dividing line, preserved here so the "why not X" doesn't get re-litigated later:**
- **Pyodide** (CPython + NumPy/pandas compiled to WASM): the right tool for bringing *existing* Python code to a JS client — high semantic compatibility, but proxy/conversion ergonomics at the boundary, heavy startup/download, constrained threading. Explicitly not this project.
- **Server-side Python** (Node/Deno backend calling a Python service over Arrow IPC/Parquet/gRPC — never JSON dataframe records at scale): usually better for large data, native BLAS, multiprocessing. The interop story (§6.2) is deliberately built so users can adopt this hybrid later without a rewrite.
- **TensorFlow.js / Danfo.js / stdlib / math.js**: API references and adapter targets only, never foundations (§2, non-goal 11).
- **This project** is the source's third design: *greenfield, performance-sensitive JS applications* — hot kernels in Rust/WASM, clean JS-native API, no CPython shipped.

**Runtime targets** *(the source names Node/Deno/browser but no support matrix — tiering below is own judgment)*:
- **v1 tier 1:** Node (current LTS), Chromium-family browsers, Deno via npm compatibility.
- **v1 tier 2 (should-work, verified later):** Firefox/Safari WASM paths; Deno-native WebGPU (a better native story than Node's third-party Dawn bindings — see §6.3).
- **Untested/unclaimed:** Bun, Cloudflare Workers — the source mentions them only regarding `@std/math` portability; WASM-heavy packages need explicit verification before claiming support.
- **Distribution open question:** npm is the baseline; dual npm+JSR publishing for first-class Deno reach needs a decision before the first release (own addition, see §9).

---

## 1. Repo Structure

The repo mixes two build systems with no shared package manager: TypeScript (npm) and Rust (Cargo → WASM). Two top-level workspace roots, touching only at defined build seams:

```
mallory-plus/
├── package.json                   # root: npm workspaces (packages/*, adapters/*, scalars/*),
│                                  # turbo pipeline, shared tsconfig, changesets config
├── Cargo.toml                     # [workspace] members = crates/*
├── rust-toolchain.toml
├── crates/                        # Rust workspace — compiled to WASM, never published to npm directly
│   ├── tensor-wasm-kernels/       # allocator, SIMD, fused CPU kernels
│   └── tensor-webgpu-wgsl/        # WGSL kernel sources + Rust-side kernel registry
│   # (a scalar-complex-fraction crate was considered and DEFERRED indefinitely — mallory-math's
│   # boxed scalars fill the role; ComplexTensor kernels use flat typed storage, not boxed scalars)
├── packages/                      # TS workspace — npm-published
│   ├── tensor-core/               # storage, dtypes, shapes, broadcasting, elementwise, reductions, .npy
│   ├── tensor-wasm/               # TS wrapper over crates/tensor-wasm-kernels (build seam)
│   │   └── wasm/                  # generated .wasm + .d.ts glue, gitignored, built by CI
│   ├── tensor-autograd/           # reverse tape, Parameter, nn.*, optim.*
│   ├── tensor-compile/            # expression IR, elementwise fusion, temp-memory planning
│   ├── tensor-webgpu/             # TS wrapper over crates/tensor-webgpu-wgsl (build seam)
│   ├── frame-arrow/               # Arrow-backed Frame/Series, expressions, Arrow IPC
│   ├── frame-parquet/             # scans/writes, projection/predicate pushdown
│   ├── scalar-types/              # thin re-export of mallory-math ComplexNumber/Rational/Decimal
│   │                              # + tensor-boundary converters — the ONE import point for mallory types
│   ├── interop-python/            # Python-side helper package — own pyproject.toml, PyPI: mallory-interop
│   └── runtime/                   # umbrella package — single import surface re-exporting named
│                                  # namespaces (Tensor, ops, linalg, …, Frame, onnx); the source
│                                  # explicitly prefers this over hanging hundreds of methods on Tensor
├── scalars/
│   └── unit/                      # only net-new scalar package — mallory-math has no Unit type
│   # (complex/ and fraction/ were replaced by packages/scalar-types re-exporting mallory-math's
│   # ComplexNumber/Rational/Decimal; terminology: "Rational" adopted, "Fraction" retired to an alias)
├── adapters/                      # core has zero dependency on these — npm names: mallory-adapter-<target>
│   ├── adapter-stdlib/  adapter-mathjs/  adapter-tfjs/  adapter-danfo/
│   ├── adapter-ml-matrix/  adapter-onnx/  adapter-arrow/
│   └── adapter-math/              # mallory-math bridge: Matrix↔Tensor, Symbolic→IR, Graph→CSR,
│                                  # DualNumber forward-mode test oracle
├── apps/                          # example/bench apps: node-bench, browser-webgpu-demo, notebook-repl
└── docs/
    ├── perplexity-conversation.md
    └── PLAN.md
```

Every TS package wrapping Rust output has a gitignored `wasm/` dir populated by `wasm-pack build ../../crates/<crate> --target web --out-dir ../../packages/<pkg>/wasm`, wired as an explicit `build:wasm → build` task dependency in `turbo.json`/`nx.json`.

`interop-python` is excluded from both the npm and Cargo workspaces — own `pyproject.toml`, own PyPI release tag scheme (`mallory-interop-v0.x.y`).

**External ecosystem dependencies** (both johnhenry libraries, both stay independent of this repo):
- `mallory-math@^0.8` — scalar types (via `packages/scalar-types`) and the `adapter-math` bridge target; also supplies scalar reference oracles for differential tests. Zero-dep, education/CAS mission unchanged.
- `mallory-iteration` — foundation of the v2 `data` namespace via a curated facade (see §5 v2). Formerly published as `async-itertools`; now lives at `packages/iteration` in the [johnhenry/math](https://github.com/johnhenry/math) pure-TS monorepo, **restarted at version 0.0.0** (2026-08-13 — the prior "starting at 2.0.0 to keep continuity" plan was reconsidered; the "what's new in 2.0" history inside that package's own readme still documents the real async-itertools-era rewrite, just not tied to the current npm version number). **Wired in 2026-08-13** (issue #22, `packages/data` / `mallory-data`) as a normal versioned npm dependency, once `mallory-iteration@0.0.0` was published — the earlier blocker (npm has no subdirectory git dependencies; a `github:owner/repo#sha&path:/packages/x` spec silently installs the monorepo *root*) is moot.

## 2. Non-Goals (v1, whole project)

Consolidated from the source conversation's two explicit "non-goals" lists plus scattered hard constraints stated elsewhere:

1. No full pandas source/API compatibility (data interchange yes; behavioral clone no).
2. No `eval()`/string-expression API or eval-based execution — expression parsing compiles only into a validated tensor-expression IR.
3. No object-dtype / generic boxed-object tensors. No `Tensor<BigNumber>` or generic `Tensor<T>` — a future `DecimalTensor` would need its own fixed-width storage family. Boxed mallory-math scalars (`ComplexNumber`/`Rational`/`Decimal`) appear only at tensor API edges (`at()`/`item()`/constructors), never in storage or kernels.
4. No Python pickle compatibility (unsafe, ecosystem-specific).
5. No implicit CPU↔GPU or JS↔WASM copying — device transfer is explicit and async-visible (`await x.to("webgpu")`).
6. No magic lazy execution — eager is default, compilation/fusion is opt-in.
7. No in-place autograd mutation until versioning/saved-tensor rules are solid.
8. No arbitrary user-defined WASM kernels in the trusted process without a validated ABI, quotas, and differential tests.
9. No nested-array storage in kernels (fine at input/output edges only).
10. No global mutable default backend.
11. No API-level dependence on TensorFlow.js, Danfo.js, math.js, or stdlib — they're semantic references and optional adapter targets only.
12. No `Proxy`-based pseudo-Python indexing.
13. No targeting Python's deprecated `__dataframe__()` Interchange Protocol — Arrow IPC is the interop bridge, full stop (pandas 4.0 drops the interchange fallback entirely).
14. No unrestricted JS-to-GPU-kernel transpilation (the GPU.js approach) — typed kernel DSL or explicit vetted WGSL only.
15. No legacy `odeint` in `integrate` — one modern `solveIVP` API.

## 3. The Acceleration Test

Decision rule for any contributor deciding Rust/WASM vs. plain TypeScript:

> **Move it to Rust/WASM if it's mostly:** homogeneous numeric data (f32/f64/int/complex) · bulk data-parallel work over typed buffers · a standard dense/sparse/FFT/statistical kernel · called often enough that allocation/loop overhead, SIMD, threading, or GPU dispatch matters · expressible without arbitrary per-element JS callbacks.
>
> **Keep it in TypeScript if it's primarily:** metadata transformation (shape validation, axis normalization, schema management) · expression-plan construction · unit conversion/parsing/formatting/error handling · arbitrary-precision arithmetic · anything driven by user callbacks, comparators, or plain-object data.

If a function can't be described in one sentence using only the left-column nouns, default to TS and revisit only once profiling shows it's hot.

## 4. Release & Versioning Strategy

**Changesets** for the npm/TS workspace (`packages/*`, `scalars/*`, `adapters/*` — npm workspaces, not pnpm, matching mallory-math's and async-itertools' plain-npm tooling), each package independently versioned. This mirrors a pattern already working on this machine for a similarly-shaped multi-package monorepo (`ai.matey`'s changesets + staggered-publish flow), and fits mallory-plus's actual dependency shape: `tensor-autograd` depends on `tensor-core`, `frame-parquet` depends on `frame-arrow`, every adapter depends on one or more core packages — Changesets' `fixed`/`linked` groups can express the two coordinated clusters (`tensor-*`, `frame-*`) if they ever need synchronized bumps.

Where Changesets doesn't reach:
- **Cargo workspace** (`crates/*`): plain Cargo semver, git-tagged (`crates/tensor-wasm-kernels-v0.x.y`), no direct npm publish.
- **Wrapper TS packages** (`tensor-wasm`, `tensor-webgpu`, accelerated `scalars/*`): the actual npm-publish seam. Add a CI check that fails if a wrapper's version bumps without a corresponding crate-version bump reflected in its build metadata — prevents shipping an npm patch against a stale WASM binary.
- **Do not** force lockstep versioning between Rust crates and TS wrappers — different release cadences, forcing sync produces noisy releases for changes that only touch one side.
- **`interop-python`**: own PyPI version bump, outside both the npm and Cargo graphs.

## 5. Phased Roadmap

**Live status (2026-08-11):** the tables below are the original design-time plan, kept as-written for historical context. For what's actually shipped vs. open, see [`ROADMAP.md`](../ROADMAP.md) and the [GitHub issue tracker](https://github.com/johnhenry/math-plus/issues) (repo later renamed from `mallory-plus`) — those are the source of truth going forward, not this document. Short version: **v1 is fully shipped**, v2 is substantially shipped (compile/autograd/frame-arrow/adapter-math all done; frame-parquet in progress).

### v1 — Tensor foundation + parallel dataframe start

| Package | Depends on | Why here | Status |
|---|---|---|---|
| `tensor-core` | — | Root of the graph: storage, dtypes, shapes, broadcasting, elementwise, reductions, `.npy` | ✅ Shipped (#1 #2 #4 #5) |
| `tensor-wasm` | `tensor-core` | Rust kernels/allocator/SIMD under tensor-core's hot paths | ✅ Shipped, incl. SIMD (#6 #7 #3 #13) — SIMD128 measured at a stable ~2.6-3x win for contiguous elementwise add/mul (`docs/spikes/wasm-simd.md`), shipped as a separate feature-detected `.wasm` artifact |
| `tensor-autograd` | `tensor-core` | Reverse tape, `nn.Linear/Embedding/LayerNorm`, `optim.AdamW` | ✅ Shipped (#8 #9 #10, incl. telemetry) |
| `packages/scalar-types` | `mallory-math@^0.8` | Thin re-export of `ComplexNumber`/`Rational`/`Decimal` + tensor-boundary converters — near-zero work; the single import point for mallory-math types so a future swap touches one package | ✅ Shipped |
| `scalars/unit` | — | Only net-new scalar package (mallory-math has no Unit type); non-blocking, can slip | ✅ Shipped (#23) |
| `adapters/adapter-onnx` | `tensor-core` | "Start here" mode: use ONNX Runtime Web directly for imported models — doesn't need autograd | ✅ Shipped (#18), verified against real ONNX models |
| `frame-arrow` | — | **No dependency on tensor-core or tensor-wasm** — can be built fully in parallel with the tensor track (tensors and tables are explicitly separate-but-interoperable domains) | ✅ Shipped (#19), verified against real pyarrow/pandas |
| `adapters/adapter-ml-matrix`, `adapter-mathjs`, `adapter-math` (Matrix↔Tensor slice only) | `tensor-core`, `scalar-types`, `mallory-math` | Cheap, low-surface, validate the adapter-isolation pattern early. `adapter-math`'s bigger deliverables (Symbolic→IR, Graph→CSR) land in v2 | `adapter-math`'s Matrix↔Tensor slice ✅ Shipped (#14); `adapter-ml-matrix`/`adapter-mathjs` not started (no filed issue yet — mallory-math already covers ml-matrix's role, revisit if a concrete need surfaces) |

### v2 — Compilation, GPU, scientific core, dataframe I/O

| Package | Depends on | Why here | Status |
|---|---|---|---|
| `tensor-compile` | `tensor-core` | Expression IR, elementwise fusion, temp-memory planning — must land before WebGPU | ✅ Shipped (#11), extended for the Symbolic bridge below |
| `tensor-webgpu` | `tensor-core` **+** `tensor-compile` | Large matmuls/attention-adjacent/image kernels need the fusion/IR layer for GPU dispatch planning | ✅ Shipped (#12) — GEMM/attention/elementwise-fusion WGSL kernels verified against a real headless GPUAdapter; GEMM stays WASM-routed (measured, no crossover on this hardware+naive-kernel combo, see `docs/spikes/webgpu-baseline.md`) |
| `frame-parquet` | `frame-arrow` | Scans/writes, projection/predicate pushdown | ✅ Shipped (#20, #30, #32) — read/write + genuine pushdown, LIST/STRUCT nested types, and a genuinely-lazy `scanParquetLazy` on frame-arrow's new lazy-source extension point |
| `fft`, `signal`, `image` (resize/normalize), convolution/attention primitives, `data` loaders + `trainer` facade, checkpoint format (`stateDict` + `io.writeCheckpoint`) | `tensor-core`, `tensor-compile`, `mallory-iteration` | "Practical ML/media compute" bundle. `data` is built on async-itertools via a **curated facade** — dataframe-safe names only (`chunk(n)` ← `group(n)`, `fold` ← terminal fold, `count*` NOT re-exported since it means row-count in dataframe-land), plus `batch(n, {collate})` → Tensors, epoch shuffling, `data.fromAsync` | `fft` ✅ Shipped (#40, new `mallory-fft` package: `ComplexTensor` + `fft`/`ifft`/`fftPadded`/`rfft`/`irfft`). `image` ✅ Shipped (#41, new `mallory-image` package: `resize`/`normalize`). `trainer` ✅ Shipped (#43, `trainer.configure`/`fit`, decoupled from `data` via a plain `AsyncIterable<{x,y}>` interface). checkpoint format ✅ Shipped (#42, `Module.stateDict()`/`loadStateDict()` + `io.writeCheckpoint`/`loadCheckpoint` — a mallory-plus-specific container, not real `.npz`). `signal` ✅ Shipped (#44, new `mallory-signal` package: `convolve`/`sosFilter`/`butter`/`stft`/`istft`/`findPeaks`/`resamplePoly`, verified against a `scipy.signal` subprocess oracle — lowpass/highpass `butter` only in v1, bandpass/bandstop need a second frequency transform not implemented yet). `async-itertools` → `mallory-iteration` (moved to the `mallory` monorepo, §0); `data` namespace (#22) still blocked on its npm publish (armed but deliberately held for the combined release, 2026-08-13). |
| Symbolic→IR bridge in `adapter-math`: `compileExpr(expr \| string, spec)` | `tensor-compile`, `mallory-math` | mallory's tagged-union `Expr` AST maps ~1:1 onto the IR (const/var/arith, 41 unary funcs incl. erf/sigmoid/relu, `cmp`→comparisons, `piecewise`→`Tensor.where`); unsupported nodes throw typed errors. Differentiate symbolically in mallory, evaluate vectorized over tensors — nearly free | ✅ Shipped (#15) |
| Dense linalg maturity (QR/SVD/eigen), `optimize`, minimal `sparse` (CSR/COO) | `tensor-core`, `tensor-autograd` | `optimize.minimize` needs autodiff; full sparse solvers deferred to v3 | Reference-speed linalg (LU/solve/QR/Cholesky/eig/SVD/leastSquares/pseudoInverse/norms) ✅ Shipped in `adapter-math` (#26). Native WASM `solve()` kernel ✅ Shipped (#39, LU with partial pivoting — the first native-kernel candidate named above); QR/SVD/eigen/Cholesky stay reference-speed for now, next candidates when that work resumes. `optimize`/`sparse`: not started |
| `adapters/adapter-stdlib`, `adapter-tfjs`, `adapter-danfo`, `adapter-arrow` | respective core packages | Larger-surface adapters, deferred until the APIs they bridge to stabilize | Not started (as planned — deferred) |

### v3 — Full dataframe system, Python interop, scientific breadth

| Package | Depends on | Why here |
|---|---|---|
| `interop-python` | `frame-arrow`, `frame-parquet`, `tensor-core` | Needs stable Arrow IPC/Parquet/.npy surfaces to bridge — explicitly last in the source's release sequence | ✅ Shipped (#21) |
| Dataframe hardening: window ops, full groupby/join maturity | `frame-arrow`, `frame-parquet` | — | Not started |
| `sparse.linalg` iterative solvers, `interpolate`, `special`, `stats.distributions`, full `integrate` | `tensor-core`, `tensor-autograd` | "Build later" tier — large algorithm/test surface. `adapter-math`'s Graph→CSR bridge (`toCSR(g)` → `{rowPointers, columnIndices, values, order}`, iterating the `Map` adjacency directly — never via the dense Infinity-sentinel matrix; explicit `missing: "zero"\|"infinity"` policy on the `toDense` companion) lands alongside the minimal `sparse` package | Not started |
| GPU kernel DSL maturity (typed `kernel({inputs, output, expression})`) | `tensor-webgpu` | The *DSL* matures here; unrestricted JS-to-shader transpilation never enters scope, at any version | Not started |
| `expression` (optional) — safe string-expression compilation into `tensor-compile`'s IR, never into JS | `tensor-compile`, `adapter-math` | **Primary route: reuse mallory-math's `Symbolic.parse` via `adapter-math`'s compile bridge** — string → mallory `Expr` AST → validated IR; a bespoke parser is only built if mallory's grammar proves insufficient. Source's math.js section: useful for a REPL/notebook/calculator/graphing view/LLM-facing constrained compute DSL, explicitly non-core; compatible with non-goal 2 because the IR and WASM/WGSL emitters stay fully controlled | Superseded early: `compileExpr` (#15) shipped directly in `adapter-math` in v2, not as a separate `expression` package — same "reuse Symbolic.parse" approach the source recommended, just landed sooner than planned |
| `DecimalTensor` (if ever) | `scalars/*` | Own fixed-width storage family, never boxed `BigNumber` | Not started |
| Full Danfo/pandas behavioral parity in adapters | `adapters/adapter-danfo` | Explicitly bounded, not full parity even here | Not started |

Non-goals from §2 never enter this roadmap at any version.

---

## 6. Layer Deep-Dives

### 6.1 Tensor Core, WASM Kernels, Autograd & Compilation

#### `tensor-core` v1 scope
- **Dtypes:** `bool`, `u8/i8`, `u16/i16`, `u32/i32`, `f16/bf16/f32/f64`. No complex dtypes in v1.
  - **Inconsistency found in the source on re-read:** its `DType` union stops at 32-bit ints, yet its own ONNX example constructs `Tensor.ones([1, 4], { dtype: "i64" })` from a `BigInt64Array` — transformer models genuinely need int64 `input_ids`. Resolve before freezing the dtype list: either add `i64/u64` (BigInt64Array-backed) to v1, or define an explicit i32→i64 conversion policy at the ONNX adapter boundary.
- **Core object:** `Tensor` with `shape`, `ndim`, `size`, `dtype`, `device` (`"wasm" | "webgpu"`), `requiresGrad`, `grad`.
  - *Note (2026-09-23):* the `device` field was never built. [RFC 0001](./rfcs/0001-device-backends.md) (**Proposed**, awaiting decision) proposes replacing it with device-typed arrays over the `@johnhenry/tensor-backend` contract, with `Tensor` staying the host type; `packages/tensor-mlx` is its experimental prototype.
- **Constructors:** `Tensor.scalar/from/fromTypedArray/zeros/ones/full/arange/linspace/eye/empty/concat/stack/where/loadNpy`. `einsum` is stretch, not core-blocking.
- **Shape/view ops:** `reshape/flatten/squeeze/unsqueeze/transpose/permute/broadcastTo/contiguous/at/slice/select/gather/cast/clone/detach`; `reshape` supports `-1` shape inference (source-explicit). **View vs. contiguous must stay semantically distinct from day one** — `permute()` never copies, `contiguous()` copies iff not already contiguous, enforced by tests.
- **Elementwise + broadcasting:** `add/sub/mul/div/pow/neg/exp/log/sqrt/abs/clip` plus comparison (`eq/ne/lt/lte/gt/gte`) and logical ops, `any/all`.
- **Reductions/masking:** `sum/mean/max/argmax/std/variance(ddof)/cumsum/cumprod/sort/argsort/topK/take/gather/scatter/mask/Tensor.where`.
- **matmul + dense linalg:** `matmul`, `dot` only (full `linalg.solve/cholesky/qr/svd` stays out of `tensor-core` itself, though `linalg.matmul/solve` is Stage-1 for the bundle as a whole).
- **Random:** `random.seed/uniform/normal/randint`.
- **Activations:** `softmax/relu/sigmoid/gelu`.
- **I/O:** `.npy` load/save — "implement early." `.npz` is a fast-follow.
- **Hard architectural rules:** no `Proxy`-based indexing anywhere; no implicit device transfer.

#### `tensor-wasm`: Rust/WASM crate boundary
- **Toolchain:** `wasm-bindgen` + `wasm-pack`, target `wasm32-unknown-unknown` (not WASI — needs to run in-browser and in Node/Deno without syscalls). Two-layer API: friendly TS methods (`a.matmul(b)`) over a low-level BLAS-like ABI (`kernels.gemmF32({outOffset, aOffset, bOffset, m, n, k, strides, alpha, beta})`), hand-written to take flat numeric params on the hot path to avoid `wasm-bindgen` object-marshalling overhead.
- **SIMD:** "where measured," not blanket — scalar + `std::arch::wasm32` SIMD128 paths behind feature detection, gated by an actual benchmark showing speedup before merging.
- **Memory:** single growable `WebAssembly.Memory`, bump/arena allocator for kernel temporaries, `Tensor` storage as offset/length into `memory.buffer` (never copied in/out per op). Strides/offsets are first-class in every kernel signature so non-contiguous views don't force a `contiguous()` copy first.
- **`...Into` functions:** `ops.addInto(out, a, b)` etc. — write directly into `out`'s WASM buffer offset, zero intermediate allocation, return `out` for chaining. This is the tier-3 "no-allocation production kernel" rung of a 3-tier allocation model (1: backend auto-plans/fuses; 2: explicit `scope()`/`keep()`; 3: explicit `...Into`).

#### `tensor-autograd`
- **Tape:** reverse-mode, dynamically built (define-by-run). Support **both** functional (`grad.valueAndGrad`, `grad.of`) and PyTorch-style tape/`.backward()` styles. `Tensor.variable(values)` is the grad-enabled constructor (the source's translation of `requires_grad_(True)`); `grad.checkpoint` exists in the full API but is deferred to v2. `grad.noGrad()`/`grad.enable()` toggle recording. **No in-place ops on tensors with an active tape record in v1** — hard constraint, not a "later" TODO.
- **Ops needing gradients first:** elementwise arithmetic, broadcasting-aware `sum/mean`, `matmul`, `softmax/relu/sigmoid/gelu`. Non-differentiable ops (`argmax`, `sort`, comparisons) should raise on `.backward()`, not silently produce wrong gradients.
- **Minimal `nn`/`optim` slice:** `nn.Parameter`, `nn.Module` (abstract `forward()`), `nn.Linear/Embedding/LayerNorm`, `nn.mseLoss/crossEntropy`, `optim.AdamW`. RMSNorm/Sequential/Dropout and plain SGD are fast-follows, not blocking.
- **v2 training-workflow surface (gap incorporated on re-read):** the source's TensorFlow.js translations also name a `trainer` facade (`trainer.configure(...)`, `await trainer.fit({x, y})`, `trainer.fit(dataLoader)`) and a `data` namespace (`data.fromAsync`, async datasets, batching, transforms), with model persistence as `model.stateDict()` + `io.writeCheckpoint`/`io.loadCheckpoint`. None of this blocks v1 autograd, but §5's v2 "data loaders + checkpoint format" bundle means this specific API surface.

#### `tensor-compile`
- **Expression IR (v1 cut):** a small, closed, typed graph IR — elementwise/broadcast nodes only, each typed by `(dtype, shape)`, built via tracing a `compile(fn)`-wrapped function. **Reuse this same IR as the shared lowering target for both WASM fusion and the future WebGPU kernel DSL** (`tensor-webgpu`) — the practical win of building this package before `tensor-webgpu`.
- **Fusion (v1 cut):** elementwise-chain fusion only (e.g. `(a+b)*c` → `relu` collapsed into one WASM loop). Matmul+bias+activation fusion and reduction fusion are out of scope for v1.
- **Why after autograd, not before:** fusion must be provably transparent to the tape (fused op's backward must match the unfused sequence exactly) — that's only checkable once the unfused eager reference from `tensor-autograd` exists as a differential baseline.

#### Milestone / acceptance criteria (proposed, not in source)
- **M1 (`tensor-core`):** full v1 op list implemented + tested; view/contiguous semantics enforced with pointer-identity-style tests; zero runtime dependency on `tensor-wasm` for basic construction (pure-JS/TypedArray fallback), unblocking parallel autograd API work.
- **M2 (`tensor-wasm`):** every `tensor-core` v1 op has a WASM kernel with the friendly/low-level split; `...Into` variants for add/mul/matmul/softmax demonstrated allocation-free; SIMD path only merges with a measured benchmark attached; differential tests pass against a NumPy oracle.
- **M3 (`tensor-autograd`):** `.backward()` matches finite-difference gradients within tolerance for every differentiable op; a toy training loop (linear regression or XOR-MLP) converges end-to-end; in-place mutation on a tracked tensor is rejected at runtime with a clear error.
- **M4 (`tensor-compile`):** elementwise fusion demonstrated with a measured allocation/wall-clock win; fused output verified within f32 tolerance of the unfused eager path for both forward and backward; `compile()` proven strictly opt-in (untouched eager paths are byte-for-byte unaffected).

#### Testing strategy: differential testing against NumPy
Not specified in the source — proposed here. Run a **Python subprocess** (not Pyodide — too heavy for test infra) as the oracle: write inputs as `.npy` (dogfoods `tensor-core`'s own required I/O feature), invoke a small script running the equivalent NumPy op, compare via `allclose`-style relative+absolute tolerance (explicit, reviewed per-op — matmul/reductions need looser tolerance than elementwise). Extend the same pattern to gradients (compare against `torch.autograd` if available, for a true reverse-mode reference). **Third gradient oracle: mallory-math's `DualNumber` forward-mode autodiff** — `dualGrad`/`dualGradN` test helpers in `adapter-math`, algorithmically independent of both the reverse tape and finite differences, pure JS so it runs where Python isn't available (browser CI, local watch mode). **mallory-math scalar oracles for later namespaces:** its `SpecialFunctions.erf/gamma/beta`, `Distributions`, `Statistics`, `Numerical.rk4/gaussLegendre/levenbergMarquardt`, `MatrixMath` decompositions, and `FFT` serve as pure-JS scalar reference implementations in differential tests for `special`/`stats`/`integrate`/`optimize`/`linalg`/`fft` (elementwise-map the mallory scalar over the tensor input, compare within per-op tolerances) — policy: **oracle-ize, don't merge**; nothing from those mallory namespaces is ever re-exported from the runtime. Fusion correctness is self-referential (fused vs. unfused in this runtime — NumPy has no fusion concept to diff against). Flag the Python+NumPy CI dependency explicitly and scope it to the tensor-layer test suites only.

#### Top risks
1. **View/stride correctness across the WASM ABI boundary** — non-contiguous broadcasted views feeding `...Into` kernels is exactly where off-by-one stride bugs produce *plausible but wrong* output rather than a crash. Non-contiguous-view cases must be first-class in differential tests, not an afterthought.
2. **SIMD-path/scalar-path divergence** — two implementations per hot kernel that must agree bit-for-bit/tolerance-for-tolerance, an ongoing maintenance tax likely underestimated by "SIMD where measured" framing alone.
3. **The in-place-mutation non-goal is a real, unresolved API-usability gap** — PyTorch users expect some in-place ops for training-loop memory efficiency; deferring indefinitely may push users toward low-level `...Into` ops even in ordinary model code, blurring the intended 3-tier allocation model. Needs an explicit follow-up decision on *when*, not just "later."

---

### 6.2 Arrow-Backed DataFrame Layer & Python Interoperability

#### `frame-arrow` v1 scope
Build as a thin, ergonomic layer over the `apache-arrow` npm package (the project's official JS implementation — `Table`/`RecordBatch`/`Vector`/`Schema`), not a reimplementation of columnar storage.
- **`Frame`:** `schema/columns/length`, `select/drop/rename/withColumns`, `filter/sortBy/limit/slice`, `groupBy().aggregate()`, `join`, `concat`, `nullCount/fillNull/dropNull`, `toArrow/toRows/toTensor/toCSV/toIPC` (`toParquet` lives in `frame-parquet` — keep this package free of a Parquet dependency).
- **`Series<T>`:** `name/dtype/length`, `cast/isNull/fillNull/unique/valueCounts/toTensor`.
- **Expression builders:** `col(name)` + comparison/logical combinators, plus the `fn.*` aggregate/scalar helpers the source uses throughout its examples (`fn.count/sum/mean/stddev/month`) and the `.overAll()` whole-column modifier from its Danfo-section normalization example (`col("spend").sub(fn.mean(col("spend")).overAll())...`). Full window functions stay in v3, but `overAll()`-style whole-column aggregates inside `withColumns` are needed for v1 parity with the source's own examples — a gap incorporated on re-read.
- **Hard design constraints:** immutable, expression-oriented by default (every transform builds a new logical-plan node — this is what makes lazy planning/column pruning/predicate pushdown/worker execution/backend substitution possible); no `.loc`/`.iloc`-style overloaded indexing, no in-place mutation, no `Proxy`-based indexing.
- **Lazy planner v1 minimum:** a `.collect()` materialization boundary plus column pruning and predicate pushdown for `select`/`filter` at the logical-plan level, even with single-threaded in-process execution. Worker/WASM query execution, window operations, and pivot/melt are explicitly deferred.

#### `frame-parquet` v1 scope
Scan, write, projection pushdown, predicate pushdown, layered on `frame-arrow`'s expression model (`Frame.readParquet(path, {columns, filter})` pushes down; `scanParquet()` plugs into the lazy planner, including glob/partitioned scans — the source's own example is `Frame.scanParquet("/events/*.parquet")`). Write options per the source: `compression: "zstd"`, `rowGroupSize`.

**The source conversation names no Parquet library — this is an open v1 decision.** Current landscape:

| Library | Model | Tradeoff |
|---|---|---|
| `parquet-wasm` (Rust `parquet`/`arrow` crates → WASM) | Reads/writes directly to/from Arrow buffers | Zero-copy with `apache-arrow`, but ~1.2MB brotli WASM payload + init step |
| `hyparquet` / `hyparquet-writer` | Pure JS, zero-dependency, plain JS objects | Works everywhere, no WASM step, but no Arrow-native zero-copy |
| `parquetjs` | Pure JS, async | Effectively unmaintained (~7 years) — rule out |

**DECIDED 2026-08-10 (spike overturned the earlier parquet-wasm lean — see `docs/spikes/parquet-bakeoff.md`): `hyparquet` + `hyparquet-writer` (+ `hyparquet-compressors` for zstd) for frame-parquet v1**, behind an engine-agnostic interface so parquet-wasm can slot in later as an optional bulk-decode accelerator. Measured evidence: parquet-wasm 0.7.2's column projection is broken on 2 of 3 read paths (sync `readParquet` silently ignores `columns`; `ParquetFile.read({columns})` produces IPC that crashes apache-arrow — upstream #810) and it exposes no min/max statistics to build pruning on (#863/#864); hyparquet has real automatic stats-based row-group skipping + bloom filters + page-index pruning (verified 599 KB fetched vs 8.4 MB full scan), 20× smaller payload (~91 KB vs 1.82 MB gzip), and much stronger maintenance signals (858k dl/wk, corporate-backed vs 129k, solo). Cost accepted: plain-JS-array output means ~3× slower conversion to Arrow (~+108 ms/1M values) — acceptable at frame scale, and the revisit condition (bulk-decode-bound workloads) is documented in the spike. **Footgun to guard in frame-parquet:** hyparquet-writer with `codec:'ZSTD'` and no compressor silently writes a corrupt file — frame-parquet must hard-fail that configuration.

#### `interop-python`
A **PyPI package** (not npm), `pip install`-able alongside `pyarrow`/`pandas`. Concrete v1: Arrow IPC read/write convenience wrappers (`interop_python.read_ipc/write_ipc`), same for Parquet, plus `.npy`/`.npz` helpers/examples. The optional Arrow C Data Interface/PyCapsule bridge (same-process zero-copy Python↔native Node addon) is explicitly deferred — it only pays off once a native Node addon exists downstream of `tensor-wasm` work.

#### Pandas-compatibility matrix (source-verbatim, turned into a checklist)

| Capability | Status |
|---|---|
| Read pandas-written files | **Yes** — Arrow IPC/Parquet/CSV/JSON/.npy |
| Export data for pandas to consume | **Yes** — Arrow IPC/Parquet → PyArrow → `to_pandas()` |
| Import a pandas DataFrame into JS | **Yes**, indirectly — Python exports, JS reads |
| Same-process zero-copy Python↔native Node addon | **Potentially** — Arrow C Data Interface/PyCapsule |
| Same-process zero-copy Python↔JS/WASM browser | **No** — different process/runtime/memory models |
| Implement `pandas.DataFrame` semantics exactly | **Technically possible, strategically poor** — not pursued |
| Run existing pandas Python code unchanged | **No** — use CPython/Pyodide/server-side Python instead |

**Rejected-alternative note, must be preserved in project docs so it isn't "helpfully" re-added later:** do not target Python's `__dataframe__()` DataFrame Interchange Protocol — pandas deprecates it and drops the fallback entirely in pandas 4.0. Arrow IPC is the portable answer for browser/Node/Deno.

#### `.npy`/`.npz`/ONNX/DLPack scope
Only `.npy` is true v1 ("implement early," lives in `tensor-core`, not the dataframe packages). `.npz` follows immediately after, still tensor-level. ONNX is interop-only (see §6.3). DLPack is native/GPU-only, no browser story, no zero-copy across WASM↔CPython — a `tensor-wasm`/native-addon-era concern, not dataframe v1.

#### Testing strategy
Concrete bidirectional conformance suite built around the source's own round-trip example (JS writes Arrow IPC/Parquet → Python `pyarrow` reads → `.to_pandas()`):
- **JS → Python:** write a `Frame` covering the full type matrix (ints, floats, strings, booleans, nulls, nested/list columns) to IPC and Parquet; Python subprocess reads via `pyarrow`, converts to pandas, asserts schema+values against fixtures.
- **Python → JS:** inverse — pandas fixture (ideally via `interop-python` itself, dogfooding it) writes IPC/Parquet; JS reads back through `frame-arrow`/`frame-parquet`, asserts equivalence.
- **CI mechanics:** pinned Python venv (`pyarrow`/`pandas`/`numpy` — pin explicitly, IPC format and pandas 4.0's interchange-protocol removal both make version pinning matter), shell out via `child_process.execFileSync`, compare JSON-serialized schema+checksums rather than diffing binaries. Run for Arrow IPC, Parquet, and `.npy`/`.npz` separately. Wire as an integration stage after `frame-arrow`/`frame-parquet` build, owned by `interop-python`.

#### Top risks
1. ~~JS Parquet library maturity unresolved~~ **RESOLVED 2026-08-10 by spike** (`docs/spikes/parquet-bakeoff.md`): hyparquet chosen with measured evidence; parquet-wasm's projection bugs and missing stats APIs confirmed the "verify, don't README-trust" instinct. Residual risk: hyparquet's null-vs-`undefined` wart (#168) and BigInt i64 policy, tracked in the spike doc.
2. ~~`apache-arrow` JS/PyArrow semantic parity~~ **RESOLVED 2026-08-10 by spike** (`docs/spikes/arrow-parity.md`): full binary parity across basic types+nulls, int64 (bit-exact incl. >2^53), dictionary encoding, chunked batches, nested list/struct, and timestamps — both directions, IPC file and stream. All caveats are JS API-surface (bigint JSON serialization throws; timestamp[us] builder needs `makeData` + `BigInt64Array`; `get()` returns float epoch-ms losing ns precision in the accessor only). frame-arrow v1 can safely claim: primitives, i64-as-bigint, dictionary<utf8>, chunking, ts[ms]/[us], flat list/struct; defer ns timestamps, decimal/date/time, deep nesting, delta dictionaries, large_utf8.
3. **Browser bundle size / WASM cost stacking** — `apache-arrow` + `parquet-wasm`'s ~1.2MB payload + any `tensor-wasm` kernels compound against the stated in-browser interactive-use ambition. No sizing budget or lazy-loading strategy is defined yet — needs to be an explicit non-functional requirement, not a post-hoc discovery.

---

### 6.3 WebGPU Acceleration & Scientific Computing (SciPy-equivalent)

#### `tensor-webgpu` — realism check
Sequenced 7th of 8 in the source's release order, after `tensor-compile` — itself a signal WebGPU is a second-tier backend, WASM is the v1 foundation. **This sequencing is validated by current WebGPU ecosystem state** (independent research, not from the source conversation):
- Browser support only reached "critical mass" around January 2026. Firefox shipped WebGPU default-on for Windows (v141) and macOS/Apple Silicon (v145); Linux remains experimental-flag-only, Android targeted late 2026. Safari shipped default WebGPU starting Safari 26.
- Node has **no native WebGPU** — access requires a third-party Dawn-based native addon (`node-webgpu`) with its own prebuild/platform-matrix risk. Deno has first-class built-in WebGPU (via `wgpu`) — a materially better native story than Node here.

**Recommended v1 cut:** `matmul`/GEMM above a measured size threshold (small matmuls stay on WASM), attention-adjacent primitives (QK^T, softmax, weighted-sum — the minimum to make `nn` training loops GPU-usable), elementwise fusion via `tensor-compile`'s IR. Chromium-family browsers only for v1, documented explicitly; Node support ships as "works with a documented native-addon install step," not a first-class guarantee. Device moves stay explicit/async per the non-goals list.

#### SciPy-equivalent namespaces — scope + sequencing

| Order | Namespace | v1 surface |
|---|---|---|
| 1 | `linalg` (dense) | `solve/lu/qr/svd/eigh/matrixExp/norm` — named result objects, not tuple unpacking |
| 2 | `sparse` | CSR first (`sparse.csr({values, columnIndices, rowPointers, shape})`), COO next, CSC only if justified; `sparse.linalg.cg/solve` |
| 3 | `optimize` | `optimize.minimize({objective, initial, method, gradient: "autodiff", tolerance})` — autodiff-by-default, explicit analytical-gradient override allowed; `root/leastSquares/brent/assignment` |
| 4 | `integrate` | `quad`; `solveIVP({tSpan, initial, derivative, method: "rk45", tEval})` as the **only** ODE entry point — no `odeint` alias, ever |
| 5 | `interpolate` | `linear/cubicSpline/regularGrid/scattered` |
| 6 | `fft`/`signal` | `fft/rfft`; `convolve/butter/sosFilter/stft/istft/findPeaks/resamplePoly` with explicit channel/batch-axis options (video-relevant, not 1-D-audio-only) |
| — | `special`, `stats.distributions` | Optional, separate packages — large algorithm/test surface |

Note: `linalg.matmul/solve` is Stage-1 (core, v1), but the rest of this table is Stage-3 (v3 in §5) — the SciPy layer as a whole is third-priority, not near-term; don't parallelize it with `tensor-webgpu`.

**Complex-number dependency for `fft` (gap found on re-read):** the source defers complex *dtypes* to "later projects" yet schedules `fft` in Stage 2 (v2). Its math.js section supplies the resolution: a dedicated **`ComplexTensor`** with interleaved or split typed storage (`complex.fromParts(real, imag)` — its own example feeds this into `fft.fft`), accelerated, and explicitly distinct from any generic `Tensor<T>`. Plan accordingly: v2 `fft` ships either as `rfft`/`irfft`-only (real-valued API) or alongside a minimal `ComplexTensor`; full `fft.fft` on complex inputs requires the latter. **Boundary contract (decided with the Mallory integration):** `ComplexTensor` storage stays interleaved/split TypedArrays; `at()`/`item()` return a mallory-math `ComplexNumber`, and constructors accept `ComplexNumber[]` at the edges. Boxed at edges, flat in kernels — consistent with non-goals 3 and 9.

**stdlib-derived detail worth keeping:** NaN-aware reductions (`stats.mean(x, { ignoreNaN: true })`) — the source specifically calls out stdlib's specialized typed-array NaN-aware accumulation (including extended-precision single-precision variants) as valuable to match.

#### Kernel-compile DSL (GPU.js-inspired) — v1/v2 plan
- **v1 — explicit WGSL only, no DSL:** `gpuKernel.define({name, inputs: TensorSpec[], output: TensorSpec, wgsl: string})`. Hand-written, vetted WGSL — the WGSL analogue of "no arbitrary user-defined WASM kernels without a validated ABI." Shape/dtype validation happens at the `TensorSpec` boundary (declared, not inferred), sidestepping the hardest part of a DSL.
- **v2 — typed expression DSL:** only after v1 has real usage and `tensor-compile`'s IR exists. A strict pure/side-effect-free subset over `TensorSpec` inputs, lowering to the *same* IR already used for WASM fusion, then emitting WGSL from that IR — never transpiling JS syntax directly.
- **Never in scope, at any version:** arbitrary JS-callback-to-shader transpilation (GPU.js's actual mechanism) — ruled out categorically, not just deferred.

#### ONNX Runtime integration — v1 scope
"Start here" = mode 1 only: thin adapter over ONNX Runtime Web.
```ts
onnx.load(modelSource: ArrayBuffer | URL, options?): Promise<OnnxModel>
model.run(inputs: Record<string, Tensor>): Promise<Record<string, Tensor>>
```
Input/output marshalling is the entire v1 job — no attempt to unify tensor types or share memory beyond ORT's own API. Backend selection (wasm/webgpu/webgl within ORT) delegates to ORT itself; v1 does not route ORT execution through `tensor-webgpu`'s device abstraction. Modes 2 (adapt runtime tensors to ORT storage) and 3 (import/compile ONNX graphs into this runtime) are deferred — mode 3 explicitly risks "reimplementing a huge operator compatibility matrix," exactly the scope-creep trap the source conversation warns about elsewhere.

#### Testing strategy
WebGPU-in-CI is a known pain point (independent research): headless Chrome + Dawn/SwiftShader (software Vulkan) is the standard no-GPU-hardware path, but still needs an X11 display for adapter init — true headless fails. **This machine already has the equivalent pattern solved for WebGL** (Xvfb `:99` + Mesa llvmpipe, verified via `gl-report` — see the `headless-webgl` memory entry); the same Xvfb-fronting approach applies directly to Dawn+SwiftShader.
1. **WASM path = CI baseline of record** for all correctness tests (numerical accuracy, autodiff gradients, SciPy-namespace algorithms) — no GPU or display server needed, runs on every OS/arch matrix cell.
2. **WebGPU correctness** (not performance) tests run against Dawn+SwiftShader under Xvfb, checked as numerical-equivalence against the WASM backend's output.
3. **GPU performance tests gated behind a real hardware-available runner** — scheduled/on-demand, not per-PR, since software rendering can't represent real perf.
4. **Cross-browser WebGPU matrix** (Firefox/Safari) is pre-release/nightly tier, not per-commit — given current fragmentation.
5. Every `tensor-webgpu` kernel gets a WASM-backend counterpart as a differential-test oracle, even ones without a production WASM fallback.

#### Top risks
1. **WebGPU ecosystem maturity/fragmentation** (independent research) — cross-browser support only reached critical mass ~January 2026; Node has no native WebGPU and depends on third-party Dawn bindings with real prebuild/platform risk. A v1 shipped now builds on a foundation still visibly moving across multiple runtimes simultaneously.
2. **SciPy-namespace scope creep** — the source doc itself warns about this ("a partial NumPy + partial PyTorch + partial SciPy... with none of the foundations finished"); each sub-namespace (sparse solvers, ODE integrators, special functions) is a large correctness-critical library in its own right, independent of the sequencing being sound.
3. **Autodiff-by-default gradients in `optimize.minimize` is an unproven engineering commitment** — requires objective functions to be expressible in the traceable autodiff subset, a real constraint SciPy itself doesn't impose (finite-difference gradients work on any Python callable). Needs validation against real optimization use cases before being locked in, or it risks silently falling back to numerical gradients (undermining the stated advantage) or being unusably restrictive for objectives with control flow.

---

## 7. Consolidated Risk Register

Ranked by how early each risk becomes load-bearing:

1. **Scope creep across the full API surface** (project-level). Eleven v1 packages plus a Cargo workspace before v1 even completes. The source conversation names this exact failure mode explicitly. Mitigation: enforce the v1→v2→v3 gate in §5 as a hard rule (no `sparse`/`optimize` work starts before `tensor-compile` ships), not a suggestion.
2. **View/stride correctness across the WASM ABI boundary** (tensor layer). Off-by-one stride bugs on non-contiguous views produce plausible-but-wrong output, not crashes. Mitigation: non-contiguous-view cases are first-class in differential tests from M1 onward.
3. **JS Parquet library immaturity** (dataframe layer). Both viable options are young relative to `pyarrow.parquet`. Mitigation: a dedicated spike before `frame-parquet` work starts, verifying pushdown and nested-type support directly.
4. **WebGPU ecosystem fragmentation** (GPU layer). Still visibly moving across browsers/runtimes as of the plan's writing. Mitigation: Chromium-only v1 target, documented explicitly; Node WebGPU treated as best-effort, not guaranteed.
5. **Sustaining this at solo/small-team scale** (project-level, own judgment). This machine's own multi-package monorepo history (`ai.matey`, `mcp-query`/`agent-query-core`) already shows maintenance strain at roughly a third of this package count, without a Rust/WASM/WebGPU build step on top. No mitigation beyond realistic pacing — this is a multi-quarter-plus effort at minimum, not a sprint.
6. **Rust + WASM + WebGPU + Arrow are each individually hard to ramp contributors on** (project-level, own judgment). The source conversation's own rigor requirements (validated ABIs, differential tests, vetted WGSL) exist precisely because these domains are unusually easy to get wrong — a contributor bottleneck independent of headcount or scope discipline.
7. **SIMD-path/scalar-path divergence** (tensor layer). Two implementations per hot kernel that must stay in sync — an ongoing tax, not a one-time cost.
8. **Autodiff-by-default `optimize.minimize` is unproven** (scientific layer). Validate against real use cases before locking in as a headline differentiator.
9. **In-place-mutation non-goal is an unresolved UX gap** (tensor layer). Needs an explicit "when," not an indefinite "later," or users will route around it via low-level ops in ordinary model code.
10. **Browser bundle-size stacking** (dataframe layer). Arrow + Parquet-WASM + tensor-WASM compound against the in-browser use case with no sizing budget defined yet.

## 8. Immediate Next Steps

All five items below are **DONE** as of 2026-08-11 — kept for historical record, see [`ROADMAP.md`](../ROADMAP.md) for what's actually next.

1. ~~Confirm package naming~~ **RESOLVED 2026-08-10:** math family, unscoped `mallory-*` npm names (see naming note in the header; the `@mallory` scope is owned by another npm user; all needed `mallory-*` names verified available). The earlier Google-Malloy collision concern is closed by the rename. *(Addendum: this repo was later renamed again, to the `@johnhenry` npm scope — see the addendum note in the header above.)*
1b. ~~Ops prerequisite: npm auth~~ **RESOLVED** — npm publish tooling and CI are wired up (`docs/RELEASING.md`); first real publish awaits the `NPM_TOKEN` secret being armed (a deliberate manual gate, not a blocker on any code work).
1c. ~~async-itertools fix-and-publish workstream~~ **RESOLVED 2026-08-10, differently than planned:** rather than fix-in-place, `async-itertools` was consolidated into the `mallory` monorepo as `mallory-iteration` (full git history preserved) — the transduce leak fix and the rest of this checklist happened as part of that move. Its npm publish (which unblocks the v2 `data` namespace, #22) is still pending.
2. ~~Scaffold the monorepo~~ **DONE** — npm workspaces, Cargo workspace, the `build:wasm → build` dependency, and every package/adapter/scalar listed in §5's v1 row are live.
3. ~~Stand up the differential-testing harness~~ **DONE** — see `docs/TESTING.md`; every numeric package's test suite runs a NumPy (or pyarrow/pandas, for frame-arrow) differential oracle.
4. ~~Spike the Parquet library decision and the `apache-arrow`/PyArrow parity check~~ **DONE 2026-08-10** — three spike docs in `docs/spikes/` (arrow-parity, parquet-bakeoff, cellgraph-study); Parquet = hyparquet, arrow parity confirmed, CellGraph = design reference only (its 5 sharp edges reported upstream as johnhenry/mallory#12–#16 — that repo was `mallory-graph` at the time these issues were filed, later renamed to `mallory`; same issue numbers, all fixed and closed).
5. ~~Set up the Xvfb + Dawn/SwiftShader headless WebGPU CI path~~ **DONE 2026-08-12 as part of issue #12** — `tensor-webgpu` shipped with a real headless GPUAdapter verified under Xvfb (Chrome-for-Testing + Xvfb steps in `.github/workflows/ci.yml`; not yet confirmed on an actual GitHub Actions runner, see `docs/spikes/webgpu-baseline.md`).

## 9. Open Questions

Carried directly from the source conversation's own unexplored follow-ups:
1. ~~Which ml-matrix decompositions to prioritize first~~ **RESOLVED 2026-08-11 as issue #26** — ship a reference-speed `linalg` surface now (delegating to mallory-math's decompositions via `adapter-math`), labeled reference-speed; `solve` and matmul-adjacent paths are the first native-kernel candidates when that work starts. Shipped.
2. ~~How to structure Arrow-backed DataFrames with a Danfo.js-like API~~ **RESOLVED 2026-08-13: no new `adapter-danfo` package for now; `frame-arrow` keeps its own expression-oriented API.** `frame-arrow`'s v1 API (#19) already shipped its own design (`Frame`/`Series`, `col()`/`fn.*`, `.overAll()`) rather than mimicking Danfo, and non-goal 11 is explicit that Danfo/TensorFlow.js/math.js/stdlib are "semantic references and optional adapter targets only, never foundations." §5 already lists full Danfo/pandas parity under the lowest-priority "build later" tier. Revisit if a concrete Danfo-migration consumer shows up — not a permanent no, just not speculative work without one.
3. **Implementing Complex and Fraction types in Rust for WASM** — filed as issue #27 to make the decision discoverable; **maintainer decision 2026-08-13 (reaffirming 2026-08-11): keep open as "someday", now with a concrete implementation roadmap on the issue** (what would change in `crates/tensor-wasm-kernels`, the JS wiring, and the consumer, plus the reopen trigger: a measured bottleneck in `mallory-fft`'s JS `transform()` loop). The two halves diverge sharply: split-storage Complex kernels fit the existing flat extern-C ABI cleanly (an `fft_f64_split` kernel is the natural first candidate); Fraction/Rational is bigint-backed and can never cross the flat-numeric ABI — if ever built, it's a separate crate with a different marshalling design, and nothing needs it. `mallory-fft`'s shipped `ComplexTensor` (#40, split real/imag storage, boxed only at edges) empirically confirmed the original flat-storage prediction.

Raised by this plan (not in the source):
4. ~~`i64`/`u64` dtype decision~~ **RESOLVED** during `tensor-core` M1 (§5) — both are `BigInt64Array`/`BigUint64Array`-backed, documented in `packages/tensor-core/src/dtype.ts`.
5. ~~Public naming~~ **RESOLVED 2026-08-10** — renamed to the math family, unscoped `mallory-*` npm packages (§8 item 1). *(Addendum: later renamed again to the `@johnhenry` scoped names — see the header addendum.)*
6. ~~npm-only vs. dual npm+JSR publishing~~ **RESOLVED 2026-08-11 as issue #25: dual, from the start.** Every publishable package has a generated `jsr.json` (`@johnhenry/<name>`) and a `jsr-release` CI job (OIDC Trusted Publishing); see `docs/RELEASING.md`'s JSR section for the one-time manual jsr.io setup that arms it.
7. ~~Browser bundle-size budget~~ **CLOSED 2026-08-13 as issue #24: no numeric budget, final** (not "no budget for now" — the earlier revisit-with-data framing is retired). The lazy-loading POLICY remains in force and is the actual size control: dynamic `import()` at the exact call site (e.g. `frame-arrow`'s `toTensor()`), heavy capabilities in their own opt-in packages. With 17 independently-installable packages, per-package granularity IS the budget; a repo-wide number would mostly measure apache-arrow's external weight. The 150 KB-gzip-core recommendation stays on the issue thread as a reference if anyone later wants a size check. A concrete consumer-app size problem gets a new, data-attached issue, not this one reopened.
