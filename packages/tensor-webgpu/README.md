# @johnhenry/math-plus-tensor-webgpu

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fmath-plus-tensor-webgpu.svg)](https://www.npmjs.com/package/@johnhenry/math-plus-tensor-webgpu)
[![license](https://img.shields.io/npm/l/%40johnhenry%2Fmath-plus-tensor-webgpu.svg)](../../LICENSE)

math-plus's WebGPU device. It is a facade over
[`@johnhenry/backend-webgpu`](https://www.npmjs.com/package/@johnhenry/backend-webgpu),
which is the single WebGPU runtime for math-plus and laya-js
([RFC 0001](../../docs/rfcs/0001-device-backends.md) §12 Q6, path (a);
issue [#146](https://github.com/johnhenry/math-plus/issues/146)). The same
shape as [`@johnhenry/math-plus-tensor-mlx`](../tensor-mlx):

- You create a device. There is no global default.
- Data moves only through explicit, async transfers to and from
  tensor-core `Tensor`s.
- The ops are the [`@johnhenry/tensor-backend`](https://www.npmjs.com/package/@johnhenry/tensor-backend)
  contract, implemented by backend-webgpu: GEMM (skinny, subgroup-matrix
  and tiled kernels), fused flash attention, softmax, LayerNorm, RoPE,
  elementwise ops, reductions and the general-numerics section.

What this package adds on top is **elementwise fusion**: a
`@johnhenry/math-plus-tensor-compile` expression becomes one WGSL dispatch
on the same runtime. It also keeps the measured WASM-vs-WebGPU GEMM
threshold, and the pre-0.2 `GPUDevice` + `GPUTensor` API as deprecated
shims.

Browsers use `navigator.gpu`. Deno uses its built-in WebGPU. Node ≥ 24 and
Bun ≥ 1.2 use Dawn, through the `webgpu` package that backend-webgpu
depends on.

## Install

```bash
npm install @johnhenry/math-plus-tensor-webgpu @johnhenry/math-plus-tensor-core
```

## Quick start

```ts
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { createWebGpuDevice, webGpuUnavailableReason } from "@johnhenry/math-plus-tensor-webgpu";

if (await webGpuUnavailableReason()) throw new Error("no WebGPU here");
const gpu = await createWebGpuDevice();          // requests an adapter + device

const x = await gpu.fromTensor(Tensor.from([1, 2, 3, 4, 5, 6]).reshape([2, 3])); // explicit upload
const w = await gpu.fromTensor(Tensor.from([0.5, -1, 2, 0, 1, 1]).reshape([2, 3]));

const b = gpu.backend;                           // the tensor-backend ops
const y = gpu.scope(() => b.softmax(b.linear(x, w), -1)); // intermediates are freed

// Fusion: one dispatch for the whole expression
const f = gpu.compile(2, (p, q) => p.mul(q).add(1).gelu());
const z = f(x, x);

console.log(await gpu.toTensor(y), await gpu.toTensor(z)); // explicit downloads
for (const t of [x, w, y, z]) gpu.dispose(t);
gpu.destroy();
```

## API

**Device.** `createWebGpuDevice(opts?)` returns a `Promise<WebGpuDevice>`
and rejects when no adapter is available. `webGpuUnavailableReason()`
resolves to a string explaining why, or `null`.

Options are backend-webgpu's `createWebGpuBackend` options
(`preferF16`, `powerPreference`, `subgroupMatrix`, `profiling`,
`maxBatch`, `maxPooledBytes`, `gemm`, `gemmTuning`, `sleepWhileWaiting`),
plus:

- `device?: GPUDevice`. Use a device you already have, such as
  `detectWebGPU().device`. If the device already has a backend, the facade
  shares it (see [One runtime per device](#one-runtime-per-device)).
  `destroy()` never destroys a device you passed in.

`WebGpuDevice`:

- `backend`: the `WebGpuBackend`. Its ops are the device's ops, and it is
  the target of the conformance suite. `backend.rt` exposes the runtime
  (`stats`, `trim()`, `startProfiling()`, `stopProfiling()`).
- Transfers in: `fromTensor(t)` and `fromHost(hostTensor)`, both async. The
  tensor must be C-contiguous (call `.contiguous()` first) and have a
  device dtype: f32, f16, bf16, i32 or bool. Other dtypes throw; cast them
  explicitly first.
- Transfers out: `toTensor(x)` returns a new tensor-core `Tensor`.
  `toHost(x)` returns a `HostTensor`. Both are async.
- Fusion: `fuse(expr, inputs)` and `compile(numInputs, fn)`, described
  below.
- Lifetime: `scope(fn)`, `dispose(x)`, `sync()`, `destroy()`.
- Introspection: `device`, `info` (adapter summary), `supports(dtype)`,
  and `name` (`"webgpu"`).

The host conversion (`hostFromTensor`, `tensorFromHost`) is shared with
tensor-mlx and lives in `@johnhenry/math-plus-tensor-cpu`.

### Elementwise fusion

`gpu.fuse(expr, inputs)` evaluates a tensor-compile expression in one
dispatch. The expression is an `IRNode`, or a `Traced` built with
`Traced.input(i)`. There is no intermediate buffer per op.
`gpu.compile(n, fn)` traces `fn` once and returns
`(...tensors) => tensor`.

- The lowering is `compileIRToWGSL`'s (every `UnaryOp`, `BinaryOp` and
  `CmpOp`, `select`, and the canonical f32 erf and exact GELU). It is
  compiled, cached and batched by backend-webgpu's runtime.
  `compileIRToKernel(node, n)` returns that kernel description.
- Inputs must be f32 and all the same shape. Views with an element offset
  are fine. Cast other dtypes first, and broadcast with backend ops first.
- The result is an f32 tensor of that shape, tracked by the enclosing
  `scope`.

### One runtime per device

Each `GPUDevice` gets exactly one `WebGpuBackend`. Dispatches are batched
into a compute pass that is submitted later, so two runtimes on one device
could reorder each other's work. `createWebGpuDevice()` registers the
backend it creates. `backendFor(device)` returns a device's backend,
creating one if needed; the deprecated functions use it. A second backend
for the same device is refused.

### GEMM threshold

`chooseGemmBackend(m, n, k?)` returns `"webgpu"` when the output has at
least `GEMM_ELEMENT_THRESHOLD` = 192² elements **and** the product does at
least `GEMM_WORK_THRESHOLD` = 2²² multiply-adds; otherwise it returns
`"wasm"`. This was measured end to end (upload, compute and readback of
host arrays) against tensor-wasm's SIMD128 GEMM on an Apple M2.

It was re-measured under Dawn on backend-webgpu's GEMM for 0.2.0 (see
[`docs/spikes/webgpu-tiled-gemm.md`](../../docs/spikes/webgpu-tiled-gemm.md)).
The rule still sends no measured shape to WebGPU that runs slower there.
The new path also wins some shapes the rule leaves on WASM (large k on
small outputs). The rule is unchanged because headless Chrome was not
re-measured, and it deliberately follows the more conservative browser
crossover.

This is one machine's number, and it ignores residency. Re-run
`scripts/measure-gemm-threshold.ts` on your own hardware.

## Migrating from 0.1 (deprecated API)

The 0.1 surface still works. It now runs on backend-webgpu's runtime and
is **deprecated**: it will be removed in the first minor release after
laya-js ships the runtime hooks (see the CHANGELOG).

| 0.1 | Use instead |
|---|---|
| `detectWebGPU()` + `GPUDevice` | `createWebGpuDevice()`. `detectWebGPU` stays supported if you want to own the device; then pass `createWebGpuDevice({ device })` |
| `toWebGPU(t, device)`, `GPUTensor.from*` | `await gpu.fromTensor(t)` / `gpu.fromHost(h)` |
| `gpuTensor.toTensor()` / `toFloat32Array()` | `await gpu.toTensor(x)` / `gpu.toHost(x)` |
| `gpuTensor.free()` | `gpu.dispose(x)` or `gpu.scope(...)` |
| `runGemm(device, a, b, { transB })` | `gpu.backend.matmul(a, b)` / `gpu.backend.linear(x, w, bias?)` |
| `runGemmWGSL` / `runGemmF16WGSL` | upload, `matmul`/`linear`, `toHost` |
| `runAttention(device, q, k, v, { mask, scale })` | `gpu.backend.sdpa(q, k, v, mask, scale)` on `[B, H, L, D]` tensors with a **bool** mask |
| `runQKT` / `runSoftmax` / `runWeightedSum` | `matmul` + `transpose` / `softmax` / `matmul` |
| `runElementwiseWGSL(device, node, arrays, n)` | `gpu.fuse(node, tensors)` / `gpu.compile(n, fn)` |
| `startProfiling` / `stopProfiling` / `configureGPURuntime` | `gpu.backend.rt.startProfiling()` / `stopProfiling()` / `rt.sleepWhileWaiting` |
| `requestDawnGPU` (`./dawn`) | nothing: `createWebGpuDevice()` finds Dawn itself (or `getGpu({ unsafe })` from backend-webgpu) |

`GPUTensor.handle` is the backend tensor behind a `GPUTensor`, so you can
migrate one call at a time:
`createWebGpuDevice({ device }).backend.matmul(a.handle, b.handle)`.

Behaviour changes in the shims:

- **GEMM** runs on backend-webgpu's kernels. `kernel: "skinny" |
  "subgroup-matrix" | "tiled"` still forces a kernel family, through the
  backend's per-shape tuning table. The preconditions changed slightly:
  skinny needs M ≤ 64, and subgroup-matrix no longer needs N % 4 for A·B.
  For A·B, when subgroup matrices apply, B is transposed once so the faster
  `linear` path is used.
- **`runAttention`**:
  - `skipMaskedTiles` and `kernel: "generic"` are ignored. The backend's
    fast kernel (head dim 32/64) always skips masked key tiles; its generic
    kernel never does.
  - Fully masked query rows still produce 0.
  - On a device with less than 32 KiB of workgroup memory (the default
    limit is 16 KiB), attention is composed from matmul and softmax,
    because backend-webgpu@0.3's kernels don't check the limit.
    `detectWebGPU()` and `createWebGpuDevice()` raise the limit.
- **Removed** (they were the duplicated kernels and runtime):
  - GEMM: `planGemm`, `selectGemmKernel`, `GEMM_CONFIG`, and the WGSL
    generators `tiledGemmWGSL`, `skinnyGemmWGSL`, `subgroupMatrixGemmWGSL`.
  - Attention: `planAttention`, `fastAttentionWGSL`,
    `genericAttentionWGSL`, `genericAttentionConfig`.
  - Runtime helpers: `acquireBuffer`, `releaseBuffer`, `dispatchKernel`,
    `getKernel`, `parseWGSLBindings`, `writeBytes`, `readBackBytes`, …
  - `gpuRuntimeStats`: use `backend.rt.stats`.
  - `sleepThresholdMs`.
- **Readback sleep is off by default.** backend-webgpu@0.3 sleeps before
  any readback expected to take over 3 ms, instead of letting Dawn
  busy-poll. On this package's 2–6 ms GEMM and attention readbacks that
  measured 15–60% more latency, so backends created here turn it off. Turn
  it on with `configureGPURuntime(device, { sleepWhileWaiting: true })` or
  `createWebGpuDevice({ sleepWhileWaiting: true })`.

## Limitations

- **The device API has no chainable array wrapper yet**, unlike
  `MlxArray`: ops are called on `gpu.backend` with backend tensors. A shared
  array API for the device packages is future work. It waits on tensor-mlx
  moving to the async-upload contract (tensor-backend 0.2), because
  number operands need a synchronous constant op.
- **Fusion does not broadcast**, and it is f32-only.
- Fusion and the shims use members that backend-webgpu@0.3 makes public
  but does not document (`backend.rt.kernel` / `dispatch` / `acquire`,
  and the `WebGpuTensor` and `WebGpuBackend` constructors), all confined
  to `src/bridge.ts`. Documented hooks are proposed in laya-js
  (`feat/runtime-hooks`).
- The deprecated shims are 2-D (GEMM) and 3-D (attention), and f32 except
  for GEMM.
- Subgroup matrices need Dawn's experimental
  `chromium-experimental-subgroup-matrix`. In practice that means Apple
  GPUs, with `allow_unsafe_apis` (which `createWebGpuDevice()` sets under
  Dawn) or Chrome's `--enable-unsafe-webgpu`. A device you create yourself
  only gets them after `detectWebGPU()` or
  `registerGemmAdapter(device, adapter)`, because Dawn doesn't expose the
  needed adapter info on the device.
- The limitations of backend-webgpu itself apply as well: see
  [its README](https://github.com/johnhenry/laya-js/tree/main/packages/backend-webgpu#limitations).

## Tests

`npm test` and `npm run test:bun` run the same 62 tests under Node and
Bun. GPU tests run on a real adapter, either Dawn in-process or headless
Chrome over CDP (`$MATH_PLUS_WEBGPU_HARNESS=dawn|chrome`; the default is
Dawn, falling back to Chrome). They skip, never fail, where neither exists.

- The **tensor-backend conformance suite** runs through the facade, in f32
  and in f16/bf16 where the device supports them.
- **Facade tests**:
  - transfers of every dtype, including views
  - refusal of implicit conversions
  - backend ops against tensor-core on the CPU
  - `fuse`/`compile` against tensor-compile's CPU `forward`
  - scope tracking
  - interop with `GPUTensor`
- **GEMM**: every kernel family × {f32, f16} × both B layouts, against a
  NumPy oracle (`scripts/gemm_oracle.py`).
- **Fused attention**: every mask shape, including fully masked rows and
  the composed fallback, against a NumPy oracle
  (`scripts/attention_oracle.py`).
- **Fusion cross-checks** against the CPU interpreter, including the fuzzer
  (issue #58) and an `Interval` precision oracle (issue #36).
- **Runtime behaviour**:
  - one backend per device
  - cache hits through the shims
  - the fused kernel's per-dispatch uniforms
  - the deprecated runtime knobs
- The Bun **`writeBuffer` byteOffset** repro (`test/bun/`).

## Provenance

Built for issue #12 on the `docs/spikes/webgpu-baseline.md` spike. Its GEMM
and attention kernels were ported from laya-js (#126, #133), and then
replaced by backend-webgpu in #146. Part of the
[math-plus](https://github.com/johnhenry/math-plus) monorepo; family docs
are at <https://opensource.johnhenry.me/math/>.
