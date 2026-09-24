---
rfc: 0001
title: "Device/backend abstraction: adopt @johnhenry/tensor-backend as the device contract"
status: Accepted with changes (2026-09-24)
issue: https://github.com/johnhenry/math-plus/issues/124
prototype: packages/tensor-mlx (issue #125)
created: 2026-09-23
decider: John Henry (repo owner)
---

# RFC 0001: Device/backend abstraction

**Status: Accepted with changes (2026-09-24).** See §12 for the decision and the
answers to §11. The one change from the recommendation: uploads become async
(Q2), in line with PLAN.md non-goal 5.

## 1. Summary

- math-plus should **depend on** `@johnhenry/tensor-backend` for its device
  contract, not fork it (**Option A**).
- tensor-core's `Tensor` stays the **host type**: the eager, strided,
  full-dtype reference that NumPy checks. It does **not** become a
  backend.
- Each accelerator gets a math-plus device package, the first being
  `@johnhenry/math-plus-tensor-mlx`. The package wraps a `Backend` in a
  math-plus-style array API.
- Data crosses between host and device only through explicit
  `device.fromTensor(t)` and `await array.toTensor()`.
- The contract's shared conformance suite is the gate that every device
  must pass.
- Autograd goes on top later, as one tape over `Backend` handles. It is not
  built into any device.

## 2. Problem

- `docs/PLAN.md` §6.1 specifies
  `Tensor.device: "wasm" | "webgpu"`. It was never built.
- Acceleration exists today only as separate storage types with separate
  APIs:
  - `WasmTensor` (`tensor-wasm`): f32 only, 1-D/2-D `...Into` kernels,
    manual `free()`. Its README states "There is no `setBackend` API
    anywhere in math-plus".
  - `GPUTensor` (`tensor-webgpu`): `await toWebGPU(t)` and
    `gpu.toTensor()`, Chromium-only in v1.
- Code written against one of these cannot run on another, and neither
  shares a test suite with the other. The result has three consequences:
  1. **No portability.** An attention block written for `GPUTensor` has to
     be rewritten for `WasmTensor`, and neither version runs on a native
     GPU under Node.
  2. **No native GPU path on Node or Bun.** The only native code is Rust
     over `Deno.dlopen`, it runs on the CPU only, and PLAN §0 lists Bun as
     untested.
  3. **No shared correctness gate.** Each accelerated package tests itself
     against tensor-core in its own way.

Meanwhile laya-js has shipped:

- a small, explicit-backend contract, `@johnhenry/tensor-backend` 0.1.0;
- a data-driven conformance suite for that contract;
- four implementations: pure-TS CPU, native MLX, WebGPU, and a structural
  bridge to `Tensor.fromTypedArray`.

The MLX implementation, `@johnhenry/backend-mlx` 0.1.0, is published with a
prebuilt runtime for darwin/arm64. It is hand-written against Apple's mlx-c
and measured at Python-MLX speed. All three packages resolve on npm as of
this RFC, verified with `npm view` on 2026-09-23.

## 3. Goals

1. **Code that runs on any backend.** Kernels and model code written once
   against a contract run on every implementation of it.
2. **Native Apple Silicon compute on Node and Bun**, without writing a
   second FFI binding (canonical-implementation rule, `AGENTS.md`).
3. **One conformance gate** that every device implementation passes, in
   f32 and f16. It complements, and does not replace, math-plus's NumPy
   differential tests.
4. **Keep what tensor-core is good at**: strided views, the full dtype set
   (`f64`, `i64`, `u8`, ...), eager semantics, and its role as the
   NumPy-checked reference. None of this should be diluted by device
   concerns.
5. **Adopt without breaking anything.** No existing package's API changes
   because of this RFC.

## 4. Non-goals

These are consistent with `docs/PLAN.md` §2:

- **No global default backend** (non-goal 10). There is no `setBackend`
  and no module-level default device. You create a device and pass it.
  This matches tensor-backend's own first design rule: "Backends are passed
  explicitly; there is no global default".
- **No implicit copies** (non-goal 5 and §6.1's "no implicit device
  transfer"). No op accepts arrays from two devices, and no op uploads a
  host `Tensor` on your behalf. Section 7.4 covers the one place where this
  RFC reads non-goal 5 slightly differently (upload is synchronous).
- **No automatic device placement or cost model.** Nothing picks a device
  for you. Size-threshold routing such as tensor-webgpu's
  `chooseGemmBackend` stays an explicit helper that the caller invokes.
- **No merged storage model in this RFC.** `Tensor` does not gain a
  `device` field, and `WasmTensor`/`GPUTensor` are not deleted or
  re-plumbed here. Section 9 sketches their migration, which is future
  work.
- **No byte-exact cross-device results.** Devices are compared within
  per-op tolerances (PLAN §2's reasoning on cross-platform snapshots).
- **No API dependence on MLX-the-Python-library semantics** beyond what the
  contract fixes, in the spirit of non-goal 11. MLX is an implementation;
  the contract is the API.

## 5. Background: the contract, and how it differs from tensor-core

`@johnhenry/tensor-backend` (`src/index.ts`) defines a `Backend<T>` whose
design rules are:

- backends are passed explicitly;
- about 40 **synchronous** ops return **opaque handles**, and `read()` is
  the only async op;
- evaluation may be lazy (MLX graphs, queued WebGPU passes);
- `scope(fn)` and `dispose()` manage lifetimes;
- masks and indices are built on the host;
- the fused ops `linear`, `layerNorm`, `rope`, `sdpa` and `gelu` are
  required, so each backend can use its best kernel;
- dtype names match tensor-core's.

The conformance subpath runs 49 golden cases over 30 ops, generated from
Python MLX 0.32.2, in f32 and in f16.

Its scope was set by one consumer, a ModernBERT-style encoder. The gaps
relative to tensor-core are what the decision is really about:

| | tensor-core `Tensor` | tensor-backend `Backend` |
|---|---|---|
| dtypes | `bool u8 i8 u16 i16 u32 i32 u64 i64 f16 bf16 f32 f64` | `f32 f16 bf16 i32 bool` |
| storage | typed array + shape + strides + offset; views share storage | opaque handle; layout is the backend's business |
| execution | eager | may be lazy; `read`/`flush` force it |
| elementwise | ~40 unary/binary ops, comparisons, logicals | `add sub mul div maximum where scale exp log relu gelu` |
| reductions | `sum mean min max argmax variance std cumsum cumprod sort argsort topK any all` | `sum max softmax sort` (single axis) |
| indexing | `slice` (steps) `select take gather mask nonzero` | `slice` (unit step) `split concat embedding gatherRows` |
| NN | `softmax relu sigmoid gelu` (tanh approx.) | `linear layerNorm rope sdpa` + exact-erf `gelu` |
| memory | GC | `dispose`/`scope` (+ finalizer safety net) |

The contract is a **device instruction set**. tensor-core is a **host
array library**. They overlap, but neither is a subset of the other.

## 6. Options

### Option A: adopt `@johnhenry/tensor-backend` as-is, as a dependency (recommended)

Under this option:

- math-plus device packages depend on `@johnhenry/tensor-backend@^0.1` and
  implement or consume its `Backend`.
- Each backend implementation is an ordinary dependency, for example
  `@johnhenry/backend-mlx`.
- When math-plus needs ops the contract lacks:
  - short term, the device package composes them from the contract, for
    example `minimum = -maximum(-a, -b)`, `min = -max(-x)` and
    `mean = sum · 1/n` in the prototype;
  - long term, they are added **upstream** as additive, minor-version
    contract changes, each with new conformance cases.

**Pros**

- It follows the canonical-implementation rule: exactly one contract and
  one MLX binding exist anywhere.
- It inherits working, measured implementations (MLX, WebGPU, CPU) and the
  conformance suite on day one.
- It matches non-goal 10 by construction.
- Upstream fixes (ABI detection, allocator stats, compile) reach math-plus
  through a version bump.
- The prototype shows the fit: 89 tests, 0 skipped, on Node and Bun.

**Cons and risks**

- **Cross-repo governance.** A contract change math-plus needs requires a
  laya-js release. Mitigation: same owner; additive changes only; pin
  `^0.1` and treat any 0.x minor as potentially breaking.
- **The contract is inference-shaped.** Training and general numerics need
  more ops, including comparisons, `sqrt`/`pow`, and `argmax`. Each one is
  an upstream PR plus fixtures. Their generator lives in laya-mlx, which
  is Python.
- **License mix.** The contract and backends are Apache-2.0, while
  math-plus is MIT. Depending on Apache-2.0 packages from MIT ones is
  compatible, and no Apache-2.0 source is vendored here.
- **Native distribution weight.** `backend-mlx-darwin-arm64` is a 64 MB
  download on Apple Silicon. It is an optionalDependency: only
  darwin/arm64 installs fetch it, and `--omit=optional` skips it.

### Option B: fork into `math-plus-tensor-device`

Copy the interface, host helpers and conformance suite into this repo as
`@johnhenry/math-plus-tensor-device`, then evolve it freely: add f64, the
remaining ops, and training needs.

**Pros**

- math-plus fully controls its contract and can grow it without cross-repo
  releases.
- The contract could be designed for general numerics from the start, for
  example by adding f64 on backends that support it.

**Cons**

- It **violates the canonical-implementation rule** at the contract level:
  two near-identical contracts would drift, and every backend would need
  to implement both or pick a side.
- The MLX, WebGPU and CPU backends target tensor-backend's `Backend`. A
  fork would need adapters for them or forks of them too, and forking
  backend-mlx means a second hand-bound mlx-c FFI. That second binding is
  exactly what #125 warns against.
- The fixtures come from laya-mlx's Python generator. A fork either
  re-hosts that generator or lets the fixtures go stale.

### Option C: status quo

Keep `WasmTensor` and `GPUTensor` as separate, explicitly imported storage
types, and add MLX, if at all, as another bespoke type.

**Pros**

- No new abstraction and no cross-repo dependency.

**Cons**

- It solves none of §2: no portability, no Node/Bun GPU, no shared gate.
- Each new accelerator adds its own ad-hoc API and its own test story.

### Comparison

| | A: adopt | B: fork | C: status quo |
|---|---|---|---|
| Canonical-implementation rule | kept | broken (2 contracts) | n/a |
| Native MLX on Node/Bun | now (prototype) | after re-binding or adapting | bespoke, if ever |
| Conformance suite | inherited, shared | copied, drifts | none shared |
| Contract evolution speed | upstream release | local | n/a |
| Non-goal 10 (no global default) | by construction | by design | yes |
| Breaking changes to existing packages | none | none | none |

## 7. Recommendation: Option A, layered as follows

### 7.1 tensor-core's `Tensor` is the host type, not a backend

**Do not make `Tensor` implement `Backend`**, for two reasons:

- **Semantics.** A `Backend` handle is opaque and possibly lazy. `Tensor`'s
  value lies in exposed storage (`data`, `strides`, `offset`), views that
  never copy, and a dtype set the contract cannot express. Making `Tensor`
  a backend would either shrink `Tensor` to the contract or bloat the
  contract to `Tensor`.
- **Role.** tensor-core is the NumPy-checked **reference**. It is the thing
  every device is compared against, so it should not also be one of the
  devices under test.

Instead, **`HostTensor` ↔ `Tensor` is the transfer boundary**, and it is
zero-copy on the host side in both directions:

- `hostFromTensor(t)` views the tensor's contiguous storage. For f16 it
  re-views the raw bits as a `Float16Array`.
- `tensorFromHost(h)` wraps a buffer that was just read, through
  tensor-backend's existing `toMathPlusArgs`, which is the canonical copy
  of that mapping.

Each transfer then costs exactly one copy, made by the backend (for MLX,
host memory into unified memory). Non-contiguous tensors are rejected
rather than packed silently.

A separate, optional question: should math-plus also provide a **CPU
reference `Backend` implemented with tensor-core kernels**? That would be
an adapter, not `Tensor` itself. laya-js already ships
`@johnhenry/backend-cpu`, and the canonical-implementation rule says there
should be only one CPU reference backend. See open question 3.

### 7.2 One device package per accelerator, over the contract

`@johnhenry/math-plus-tensor-<device>` wraps one `Backend` in the
math-plus array idiom:

- method chaining;
- `(axis?, { keepDims })` reductions;
- no implicit dtype promotion. This rule is enforced in the wrapper even
  where MLX itself would promote.

The package also exposes the raw `Backend` as `device.backend`, so code
written against the contract and the conformance suite work unchanged.
`packages/tensor-mlx` is the first such package, and §10 lists its API.

Why a wrapper rather than exposing `Backend` directly:

- math-plus users should get math-plus semantics (dtype strictness,
  explicit-transfer errors that name the fix).
- The layer is also where composed ops live until they move upstream.

### 7.3 Amend PLAN §6.1: device is a type, not a field

Replace `Tensor.device: "wasm" | "webgpu"` with **device-typed arrays**:
`MlxArray`, `GPUTensor` and `WasmTensor` are distinct types, and a device
array knows its owning device.

- Mixing devices, or passing a host `Tensor` where a device array is
  expected, is a type error at compile time and a thrown error with a
  remedy at runtime.
- This enforces non-goal 5 more strongly than a runtime `device` string
  could.

The transfer spelling becomes `device.fromTensor(t)` for upload and
`await arr.toTensor()` for download. That mirrors tensor-webgpu's existing
`gpu.toTensor()` and frame-arrow's `Series.toTensor()`.

### 7.4 Laziness vs non-goal 6, and the synchronous upload

**Laziness.** Non-goal 6 says "No magic lazy execution — eager is default,
compilation/fusion is opt-in". MLX is lazy. This RFC proposes reading the
non-goal as being about **observable semantics**. The prototype enforces
that reading:

- shape and dtype errors throw at the call site;
- values are visible only through explicit `eval`, `toTensor` or `toHost`;
- a result never depends on when evaluation happens;
- **`compile` (fusion) stays opt-in**. The prototype does not even wrap
  it.

If the owner reads non-goal 6 strictly instead, the device package can call
`eval` after every op. That would give up most of MLX's value, so this is
**open question 1**.

**Synchronous upload.** Non-goal 5 says transfers are "explicit and
async-visible (`await x.to("webgpu")`)". Under this RFC:

- download is async (`toTensor`);
- upload (`fromTensor`) is **synchronous but explicit**, because the
  contract's `fromHost` is synchronous: MLX's unified-memory copy never
  needs to await.

This RFC reads "async-visible" as "visible at the call site". **Open
question 2** asks whether upload must return a Promise anyway, for
uniformity with a future WebGPU device.

### 7.5 Autograd on top: one tape over backend handles

`tensor-autograd` today builds a define-by-run tape over tensor-core
`Tensor`s. This RFC proposes that device autograd be the **same tape
design, generic over `Backend<T>`**. The proposal has four parts:

1. **One VJP per op**, written in contract ops and generic over the
   backend. For example, the VJP of `matmul` is two `matmul`s plus a
   `transpose`; broadcasting gradients use `sum` over broadcast axes plus
   `reshape`, the device analogue of `sumToShape`. This keeps a single
   copy of each derivative rule across all devices.
2. **Handles, not arrays, on the tape.** The tape records backend handles
   and owns their lifetimes. Saved activations are released when the tape
   is released. That is the `scope`/`dispose` discipline, not garbage
   collection.
3. **Oracles.** Existing `tensor-autograd` on tensor-core is the
   differential reference. Finite differences and adapter-math's
   `DualNumber` gradients (`docs/TESTING.md`) remain the independent
   checks.
4. **Contract additions it needs first.** These would go upstream:
   - a comparison or `sign` op, for `relu'` and `max` gradients;
   - `neg` and `sqrt`/`rsqrt`, for `layerNorm'`, which can be composed
     today but slowly;
   - an `erf` or `geluGrad`, for `gelu'`;
   - `argmax`, for cross-entropy.

   Until those exist, device autograd stays **deferred**. Training on
   tensor-core keeps working unchanged. This matches #123's scope, which is
   transformer readiness on tensor-core.

In-place mutation stays forbidden on recorded tensors (non-goal 7). The
contract has no in-place ops.

### 7.6 Conformance story

The conformance work has three layers, each with its own oracle:

1. **Contract conformance.** Every `Backend` that math-plus depends on or
   ships runs
   `runConformance(() => device.backend, loadOpCases(), …)` in the device
   package's own tests: the MLX GPU and CPU devices in the prototype. The
   runs are skip-don't-fail off-platform. The golden fixtures come from
   Python MLX in laya-mlx. When math-plus needs an op, its cases are added
   upstream with the op.
2. **math-plus-level differential tests.** The wrapper API, including
   composed ops, dtype rules and reductions over all axes, is compared
   against a **NumPy oracle** (`docs/TESTING.md`):
   - f32 with tight tolerances;
   - f16 on f16-rounded inputs, with 2e-2 tolerance;
   - casts bit-exact against `astype`.

   The oracle resolves through `$MATH_PLUS_ORACLE_PYTHON`, else `python3`,
   and follows the same skip-don't-fail contract. The prototype runs every
   case in one Python process.
3. **Host-boundary tests** that run on every platform: zero-copy views,
   contiguity and dtype rejection. These need no native library.

On CI's Linux runners, layers 1 and 2 skip because MLX needs Apple
Silicon. Local runs on Apple Silicon must report 0 skipped, the same
discipline AGENTS.md applies to the oracles. **Open question 5** asks
about adding a macOS arm64 CI leg.

## 8. Consequences for the dependency graph

```
@johnhenry/tensor-backend (Apache-2.0, laya-js)       contract + conformance
        ^                         ^
        |                         |
@johnhenry/backend-mlx  ---->  (implements)            mlx-c FFI, koffi / bun:ffi
        ^
        |
@johnhenry/math-plus-tensor-mlx  ---->  @johnhenry/math-plus-tensor-core (host type)
```

- tensor-core gains **no** dependency, and its API does not change.
- Only device packages depend on the contract.

## 9. Migration of WasmTensor and GPUTensor

Neither type changes in this RFC. This section describes where each would
go.

- **`WasmTensor` (tensor-wasm).** It stays a **kernel-level storage type**:
  f32, 1-D/2-D, zero-allocation `...Into`, manual `free`, trap poisoning.
  - It covers too few ops (`add sub mul div matmul solve`) to implement
    the contract.
  - A future `createWasmBackend()` could implement `Backend` over
    `Kernels`, composing or falling back per op, once #121 (blocked SIMD
    GEMM) makes it worth routing through.
  - Until then, `WasmTensor` stays what its README says: explicitly
    imported, and not a backend.
- **`GPUTensor` (tensor-webgpu).** Two WebGPU implementations now exist:
  tensor-webgpu (GEMM, attention primitives, and IR→WGSL fusion from
  tensor-compile) and laya-js's `@johnhenry/backend-webgpu`, which
  implements the full contract. The canonical-implementation rule says
  they should converge. Two paths:
  - **(a)** Adopt backend-webgpu as the `Backend`, ship a
    `math-plus-tensor-webgpu` device facade like tensor-mlx, and move
    tensor-webgpu's unique value, the tensor-compile IR→WGSL fusion, to
    run on it.
  - **(b)** Make tensor-webgpu implement the contract itself.

  Path (a) keeps one WebGPU runtime, and the runtime improvements from #126
  already flow in that direction. Either way, `toWebGPU(t)` and
  `gpu.toTensor()` keep working during a deprecation window. This choice
  is **open question 6**.
- **Deno native (tensor-wasm `NativeKernels`).** It is unaffected. A Deno
  MLX path would need a `Deno.dlopen` adapter in backend-mlx, upstream
  (about 60 lines per laya-js's binding decision doc). Until then
  tensor-mlx is excluded from JSR. *(Done: backend-mlx 0.3.0 ships the loader, and
  tensor-mlx 0.2 runs under Deno 2 and publishes to JSR; #147.)*

## 10. The prototype (#125): `@johnhenry/math-plus-tensor-mlx`

- **Dependencies.** It depends on the **published**
  `@johnhenry/backend-mlx@^0.1.0` and `@johnhenry/tensor-backend@^0.1.0`.
  It contains no FFI of its own.
- **API.**
  - `createMlxDevice({ device })` returns an `MlxDevice`, which provides
    `fromTensor`, `fromHost`, `eval`, `scope`, `where`, `liveArrays`,
    `memory`, `destroy` and `backend`.
  - `MlxArray` provides:
    - elementwise and broadcast ops: `add`, `sub`, `mul`, `div`,
      `maximum`, `minimum`, `neg`, `exp`, `log`, `relu`, `gelu`;
    - reductions: `sum`, `mean`, `max`, `min`, `softmax`;
    - `matmul` and `layerNorm`;
    - `cast`, including f16 and bf16, plus `reshape` and `transpose`;
    - `toTensor`, `toHost`, `eval` and `dispose`.
- **Guarantees.**
  - Transfers are explicit; ops refuse host tensors and foreign-device
    arrays.
  - There is no implicit dtype promotion.
  - Uploads of non-contiguous tensors are rejected.
  - There is no global device.
- **Tests.**
  - 89 tests on Node and on Bun: NumPy differential, the tensor-backend
    conformance suite on the GPU and CPU devices, and transfer/lifetime
    tests. All pass with 0 skipped on an Apple M2.
  - The suite skips cleanly off-platform.
- **Registration.** The package is not published to JSR
  (`JSR_EXCLUDED_DIRS`, now checked by the manifest-drift test).
- **Status.** Experimental. Its README lists its limitations: no
  autograd, no compile, only the contract's op set, darwin/arm64 only.
- **Since the decision (tensor-mlx 0.2, #147).** It is on
  `backend-mlx@^0.3.0` and `tensor-backend@^0.2.0`. `fromTensor`/`fromHost`
  are async (Q2). `MlxArray` gained the general-numerics ops (Q7) through
  tensor-backend's compose helpers. It runs on Deno 2 and is published to
  JSR (Q4). There are 213 tests on Node, Bun and Deno, with 0 skipped.

## 11. Open questions for the decider

1. **Non-goal 6.** Is "lazy internally, eager-observable, `compile`
   opt-in" acceptable for device packages? Or must every op evaluate
   eagerly?
2. **Non-goal 5.** Must upload return a Promise, for API uniformity with
   WebGPU, even when the backend copies synchronously?
3. **CPU reference backend.** Should math-plus reuse laya's
   `@johnhenry/backend-cpu`? Or should it provide a tensor-core-kernel
   `Backend` and have laya depend on that? Only one of these should exist.
4. **Deno.** Should a `Deno.dlopen` adapter in backend-mlx, upstream, be a
   prerequisite for publishing tensor-mlx to JSR? Or is JSR-exclusion fine
   long-term?
5. **CI.** Should the project add a macOS arm64 GitHub Actions leg, so MLX
   conformance and differential tests run somewhere other than a developer
   machine?
6. **WebGPU convergence.** Path (a) or path (b) in §9?
7. **Contract growth process.** Should math-plus-driven ops (comparisons,
   `sqrt`, `argmax`, `erf`) be added to tensor-backend under a
   "general numerics" section? Or should they go in a second, math-plus-owned
   contract that *extends* `Backend`? The second choice is additive, not a
   fork, but it is still a second place for ops to live.

## 12. Decision

**Accepted with changes**, 2026-09-24, by the owner. The recommendation (Option A,
§7) stands: math-plus adopts `@johnhenry/tensor-backend` as its device
contract, with tensor-core's `Tensor` as the host type and devices passed explicitly.
One change from the recommendation: uploads are async (Q2).

Answers to §11:

1. **Evaluation model: accepted.** Devices may be lazy internally, as long as
   errors surface at the call site and results look eager to the caller.
   `compile()` stays opt-in. PLAN.md non-goal 6 is amended to "eager-observable".
2. **Upload: async.** `fromHost`/`fromTensor` return a Promise, like `read`. This
   follows PLAN.md non-goal 5 ("device transfer is explicit and
   async-visible"), so a transfer is never silently synchronous in either
   direction. It is a breaking change to the tensor-backend contract and all
   of its backends; it ships as a minor version bump while the packages are 0.x.
3. **CPU reference: math-plus owns it.** math-plus ships the CPU `Backend`,
   built on tensor-core's kernels, so GEMM and friends live only in tensor-core.
   laya-js's `@johnhenry/backend-cpu` becomes a thin re-export, then is deprecated.
4. **Deno: build the adapter.** A `Deno.dlopen` loader goes into
   `@johnhenry/backend-mlx`, next to `bun:ffi` and koffi. After that,
   tensor-mlx is published to JSR.
5. **CI: add a macOS arm64 leg.** It starts non-blocking and becomes
   required once stable, modelled on laya-js's `mlx-macos` job.
6. **WebGPU: path (a).** `@johnhenry/backend-webgpu` becomes the single WebGPU
   runtime. math-plus ships a `tensor-webgpu` device facade, like tensor-mlx,
   and moves its IR→WGSL fusion onto it. `toWebGPU()` and `gpu.toTensor()`
   keep working through a deprecation window.
7. **Contract growth: in tensor-backend.** A "general numerics" section
   (comparisons, `sqrt`, `argmax`, `erf`, …) is added to the one contract as
   optional ops with default compositions in `compose.ts`, plus conformance
   cases. There is no second contract.

The implementation work is tracked in the follow-up issues linked from #124.
