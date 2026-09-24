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
- Uploads are **chainable arrays** (`WebGpuArray`): the one `DeviceArray`
  API every math-plus device shares (from
  [`@johnhenry/math-plus-tensor-cpu`](../tensor-cpu#the-shared-device-array-api-arraydevice--devicearray);
  the same class is tensor-mlx's `MlxArray`), so `x.matmul(w).add(1).softmax()`
  reads the same on WebGPU, MLX and the CPU.
- The ops are the [`@johnhenry/tensor-backend`](https://www.npmjs.com/package/@johnhenry/tensor-backend)
  contract, implemented by backend-webgpu: GEMM (skinny, subgroup-matrix
  and tiled kernels), fused flash attention, softmax, LayerNorm, RoPE,
  elementwise ops, reductions and the general-numerics section.

What this package adds on top is **elementwise fusion**: a
`@johnhenry/math-plus-tensor-compile` expression becomes one WGSL dispatch
on the same runtime. It also keeps the measured WASM-vs-WebGPU GEMM
threshold. The pre-0.2 `GPUDevice` + `GPUTensor` API was removed in 0.3.0
(see [Removed in 0.3.0](#removed-in-030)).

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

const x = await gpu.fromTensor(Tensor.from([1, 2, 3, 4, 5, 6]).reshape([2, 3])); // explicit upload: a WebGpuArray
const w = await gpu.fromTensor(Tensor.from([0.5, -1, 2, 0, 1, 1]).reshape([2, 3]));

const y = gpu.scope(() => x.matmul(w.transpose()).add(1).softmax(-1)); // chainable; intermediates are freed

// Ops the arrays do not wrap: gpu.backend with .handle, then wrap() the result
const lin = gpu.wrap(gpu.backend.linear(x.handle, w.handle));

// Fusion: one dispatch for the whole expression
const f = gpu.compile(2, (p, q) => p.mul(q).add(1).gelu());
const z = f(x, x);

console.log(await y.toTensor(), await lin.toTensor(), await z.toTensor()); // explicit downloads
for (const t of [x, w, y, lin, z]) t.dispose();
gpu.destroy();
```

## API

**Device.** `createWebGpuDevice(opts?)` returns a `Promise<WebGpuDevice>`
and rejects when no adapter is available. `webGpuUnavailableReason()`
resolves to a string explaining why, or `null`.

Options are backend-webgpu's `createWebGpuBackend` options
(`preferF16`, `powerPreference`, `subgroupMatrix`, `profiling`,
`maxBatch`, `maxPooledBytes`, `gemm`, `gemmTuning`, `sleepWhileWaiting`,
`sleepThresholdMs`), plus:

- `device?: GPUDevice`. Use a device you already have, such as
  `detectWebGPU().device`. If the device already has a backend, the facade
  shares it (see [One runtime per device](#one-runtime-per-device)).
  `destroy()` never destroys a device you passed in.
- `adapter?: GPUAdapter`. With `device`, the adapter it came from, so
  subgroup-matrix GEMM can be detected. Not needed for a device from
  `detectWebGPU()`, whose adapter is remembered.

**Readback sleep.** Under Dawn, backend-webgpu sleeps before a readback
instead of letting Dawn busy-poll `mapAsync` (a full core under Bun), when
the expected wait is longer than `sleepThresholdMs`. Backends created here
default that threshold to **15 ms** (backend-webgpu's own default is 3 ms):
readbacks of a few milliseconds keep polling at full speed, and long waits
use about 3× less CPU for about 2% more latency (measured in
[`docs/spikes/webgpu-runtime.md`](../../docs/spikes/webgpu-runtime.md#since-146-backend-webgpus-sleep-and-sleepthresholdms-15)).
Browsers don't busy-poll, so there is no sleep for `navigator.gpu`.

`WebGpuDevice` extends tensor-cpu's `ArrayDevice`:

- Transfers in: `fromTensor(t)` and `fromHost(hostTensor)`, both async,
  both resolving to a `WebGpuArray`. The tensor must be C-contiguous (call
  `.contiguous()` first) and have a dtype the device `supports()`: f32,
  bf16, i32, bool, and f16 when the adapter has `shader-f16`. Anything else
  throws **synchronously**, before any Promise; cast it explicitly first.
- `WebGpuArray` is the shared `DeviceArray`: `add` `sub` `mul` `div`
  `maximum` `minimum` `pow`, `neg` `abs` `exp` `log` `sqrt` `rsqrt` `tanh`
  `sigmoid` `erf` `relu` `gelu`, the comparisons and logical ops, `sum`
  `mean` `max` `min` `argmax` `argmin` `cumsum` `softmax`, `matmul`,
  `layerNorm`, `cast`, `reshape`, `transpose`, `toTensor()`/`toHost()`,
  `eval()`, `dispose()`, `handle` — see
  [tensor-cpu's README](../tensor-cpu#the-shared-device-array-api-arraydevice--devicearray)
  for the list and the dtype rules.
- `backend`: the `WebGpuBackend`, for the ops the arrays do not wrap
  (`linear`, `sdpa`, `rope`, `slice`, `concat`, `embedding`, quantized
  weights, …) and the target of the conformance suite. Pass `x.handle` in
  and `gpu.wrap(result)` to get an array back. `backend.rt` exposes the
  runtime (`stats`, `trim()`, `startProfiling()`, `stopProfiling()`).
- Transfers out: `x.toTensor()` / `x.toHost()` on an array, or
  `gpu.toTensor(x)` / `gpu.toHost(x)`, which also take a raw backend
  tensor. All async.
- Fusion: `fuse(expr, inputs)` and `compile(numInputs, fn)`, described
  below.
- Lifetime: `scope(fn)` (keeps returned arrays and raw backend tensors),
  `x.dispose()` or `gpu.dispose(x)` (arrays or raw tensors), `eval()`,
  `sync()`, `destroy()`.
- Also `where(cond, a, b)`, `wrap(handle)`, `supports(dtype)`, `device`
  (the `GPUDevice`), `info` (adapter summary), and `name` (`"webgpu"`).

The host conversion (`hostFromTensor`, `tensorFromHost`) is shared with
tensor-mlx and lives in `@johnhenry/math-plus-tensor-cpu`.

### Elementwise fusion

`gpu.fuse(expr, inputs)` evaluates a tensor-compile expression in one
dispatch. The expression is an `IRNode`, or a `Traced` built with
`Traced.input(i)`. There is no intermediate buffer per op.
`gpu.compile(n, fn)` traces `fn` once and returns
`(...arrays) => array`.

- The lowering is `compileIRToWGSL`'s (every `UnaryOp`, `BinaryOp` and
  `CmpOp`, `select`, and the canonical f32 erf and exact GELU).
  `compileIRToElementwise(node, n)` returns it as the expression and helper
  functions that backend-webgpu's `elementwise` hook runs. The backend
  compiles, caches and batches the kernel on its runtime.
- Inputs broadcast against each other with NumPy's rules, as in
  tensor-compile's CPU `forward`: `[B, N]` with `[N]`, `[B, 1]` or `[1]`,
  both sides at once, up to rank 8. Views with an element offset are fine.
  Inputs must be f32; cast other dtypes first.
- The result is an f32 `WebGpuArray` of the broadcast shape, tracked by
  the enclosing `scope`. Raw backend tensors in (e.g. `gpu.backend.slice`
  views) give a raw backend tensor out.

### One runtime per device

Each `GPUDevice` gets exactly one `WebGpuBackend`. Dispatches are batched
into a compute pass that is submitted later, so two runtimes on one device
could reorder each other's work. `createWebGpuDevice()` registers the
backend it creates, and `createWebGpuDevice({ device })` for a device that
already has one shares it. A second backend for the same device is refused.

### GEMM threshold

`chooseGemmBackend(m, n, k?)` returns `"webgpu"` when the output has at
least `GEMM_ELEMENT_THRESHOLD` = 256² elements **and** the product does at
least `GEMM_WORK_THRESHOLD` = 2²⁴ multiply-adds; otherwise it returns
`"wasm"`. It prices a host-array call end to end (upload, `matmul` or
`linear`, readback) against tensor-wasm's SIMD128 GEMM, on an Apple M2.

For 0.3.0 it was re-measured on 43 shapes in three environments: Dawn,
headless Chrome, and a real, visible Chromium without subgroup matrices
(see [`docs/spikes/webgpu-tiled-gemm.md`](../../docs/spikes/webgpu-tiled-gemm.md#re-measured-in-three-environments-tensor-webgpu-030-2026-09-24)).
Dawn wins from 160³, but both browsers lose at 192³, and the visible
browser also loses a few shapes the previous rule (`m·n >= 192²` and
`m·n·k >= 2²²`) sent to WebGPU. The new rule sends no measured shape to a
slower WebGPU in any of the three. It leaves some wins on WASM, such as
Dawn's from 160³ and large-k products like 192x4096x192.

This is one machine's number, and it ignores residency: operands already
on the GPU make WebGPU cheaper at every size. Re-run
`scripts/measure-gemm-threshold.ts` (Dawn or headless Chrome) and
`scripts/gemm-threshold-page/serve.ts` (any browser) on your own hardware.

## Changed in 0.4.0: uploads are chainable arrays

`gpu.fromTensor()` and `gpu.fromHost()` now resolve to a `WebGpuArray`
(the shared `DeviceArray`) instead of a raw backend `WebGpuTensor`, so the
WebGPU device has the same chainable API as tensor-mlx and the CPU device.
This is the one breaking change; `gpu.backend` is unchanged.

| 0.3 | 0.4 |
|---|---|
| `const x = await gpu.fromTensor(t); gpu.backend.matmul(x, y)` | `x.matmul(y)`, or `gpu.wrap(gpu.backend.matmul(x.handle, y.handle))` |
| raw tensors for `gpu.backend.*` code: `await gpu.fromHost(h)` | `await gpu.backend.fromHost(h)` (unchanged backend API) |
| `await gpu.fromTensor(badTensor)` rejects | throws synchronously (non-contiguous, or a dtype the device does not support — f16 without `shader-f16` is refused, not widened) |
| `gpu.fuse(expr, tensors)` / `compile(n, fn)(...tensors)` | unchanged for raw tensors; arrays in give an array out |
| `gpu.toTensor(x)`, `gpu.toHost(x)`, `gpu.dispose(x)`, `gpu.scope(fn)` | unchanged, and they also take arrays (`x.toTensor()`, `x.dispose()` work too) |

## Removed in 0.3.0

The pre-0.2 `GPUDevice` + `GPUTensor` API, deprecated in 0.2.0, is gone.
Its replacements, all on a `gpu` from `createWebGpuDevice()`:

| Removed | Use instead |
|---|---|
| `toWebGPU(t, device)`, `GPUTensor.fromFloat32Array` / `fromFloat16Bits` | `await gpu.fromTensor(t)` / `await gpu.fromHost(h)` |
| `GPUTensor.fromBuffer(device, buffer, shape)` | `gpu.backend.wrapBuffer(buffer, shape, dtype)` |
| `gpuTensor.toTensor()` / `toFloat32Array()` / `toUint16Array()` | `await gpu.toTensor(x)` / `await gpu.toHost(x)` |
| `gpuTensor.free()` | `gpu.dispose(x)`, or `gpu.scope(fn)` |
| `runGemm(device, a, b, { transB })`, `runGemmWGSL`, `runGemmF16WGSL` | `gpu.backend.matmul(a, b)` / `gpu.backend.linear(x, w, bias?)` (x·Wᵀ). Where subgroup matrices apply (Dawn or `--enable-unsafe-webgpu` on Apple GPUs) and M > 64, `linear(a, transpose(b, [1, 0]))` is faster than `matmul` for A·B: `matmul` only has the tiled kernel |
| `runAttention(device, q, k, v, { mask, scale })` | `gpu.backend.sdpa(q, k, v, mask, scale)` on `[B, H, L, D]` tensors with a **bool** mask. Fully masked query rows are undefined (the shim returned 0) |
| `runQKT` / `runSoftmax` / `runWeightedSum` | `matmul(q, transpose(k, …))` / `softmax(x, -1)` / `matmul(w, v)` |
| `runElementwiseWGSL(device, node, arrays, n)` | `gpu.fuse(node, tensors)` / `gpu.compile(n, fn)` |
| `startProfiling` / `stopProfiling` / `configureGPURuntime` | `gpu.backend.rt.startProfiling()` / `stopProfiling()` / `rt.sleepWhileWaiting`, `rt.sleepThresholdMs` |
| `backendFor(device)` | `(await createWebGpuDevice({ device })).backend` |
| `requestDawnGPU` (the `./dawn` subpath) | nothing: `createWebGpuDevice()` finds Dawn itself; `getGpu({ unsafe })` from backend-webgpu for a raw `GPU` |
| `gemmKernelApplicable`, `GemmOptions.kernel`, `AttentionOptions` | nothing: backend-webgpu picks kernels (its `gemmTuning` table forces a Linear kernel per shape) |

`detectWebGPU`, `registerGemmAdapter`, `gemmCapabilities`,
`subgroupMatrixUsable`, `chooseGemmBackend` and the threshold constants
stay.

## Limitations

- **The arrays wrap the elementwise/reduction/matmul/LayerNorm op set
  only.** The fused transformer ops, slicing, `concat`, `sort` and
  quantized weights stay on `gpu.backend` (use `x.handle` and
  `gpu.wrap()`). Reductions take one axis or all axes. No autograd on
  device arrays.
- **Fusion is f32-only**, like tensor-compile's forward pass. Cast other
  dtypes first. (backend-webgpu's `elementwise` hook could load f16, bf16,
  i32 and bool inputs as f32, but no test covers that yet.) It lowers the
  forward value only, not gradients.
- This package uses only backend-webgpu's documented API
  (`createWebGpuBackend({ device, adapter })` and the `elementwise` hook).
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

`npm test` and `npm run test:bun` run the same 260 tests under Node and
Bun. GPU tests run on a real adapter, either Dawn in-process or headless
Chrome over CDP (`$MATH_PLUS_WEBGPU_HARNESS=dawn|chrome`; the default is
Dawn, falling back to Chrome). They skip, never fail, where neither exists.

- The **tensor-backend conformance suite** runs through the facade, in f32
  and in f16/bf16 where the device supports them.
- The **shared DeviceArray suite** (`test/device-array.test.ts`, the same
  suite tensor-cpu and tensor-mlx run, from tensor-cpu's
  `test/device-array-suite.ts`): every array op against a NumPy oracle in
  f32, f16 (with `shader-f16`), bf16, i32 and bool, casts bit-exact, and
  the transfer, dtype, constant, lifetime and `handle`/`wrap` rules.
- **Facade tests**:
  - transfers of every dtype, including views
  - refusal of implicit conversions
  - arrays and raw backend tensors through `toTensor`/`toHost`/`dispose`/`fuse`
  - backend ops against tensor-core on the CPU
  - `fuse`/`compile` against tensor-compile's CPU `forward`, including
    broadcasting (lower ranks, size-1 axes on both sides, offset views, an
    input the expression ignores)
  - scope tracking
  - sharing a `detectWebGPU()` device, and `destroy`
  - the API removed in 0.3.0 stays removed
- **GEMM** through `gpu.backend.matmul` / `linear`: every kernel family
  (forced through the backend's `gemmTuning` table) × {f32, f16} × both B
  layouts, and GPU-resident chains, against a NumPy oracle
  (`scripts/gemm_oracle.py`).
- **Fused attention** through `gpu.backend.sdpa`: every mask shape, on
  raised and default (16 KiB) workgroup-memory limits, against a NumPy oracle
  (`scripts/attention_oracle.py`). Fully masked query rows are excluded:
  the contract leaves them undefined.
- **Fusion cross-checks** against the CPU interpreter, including the fuzzer
  (issue #58) and an `Interval` precision oracle (issue #36).
- **Runtime behaviour**:
  - one backend per device
  - pipeline, bind-group and buffer-pool hits, and GPU-resident chains
    that create no readback buffer until the caller reads
  - the fused kernel's per-dispatch uniforms
  - the readback-sleep defaults, and the profiler through `backend.rt`
- The Bun **`writeBuffer` byteOffset** repro (`test/bun/`).

## Provenance

Built for issue #12 on the `docs/spikes/webgpu-baseline.md` spike. Its GEMM
and attention kernels were ported from laya-js (#126, #133), and then
replaced by backend-webgpu in #146. Part of the
[math-plus](https://github.com/johnhenry/math-plus) monorepo; family docs
are at <https://opensource.johnhenry.me/math/>.
