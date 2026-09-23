# @johnhenry/math-plus-tensor-webgpu

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fmath-plus-tensor-webgpu.svg)](https://www.npmjs.com/package/@johnhenry/math-plus-tensor-webgpu)
[![license](https://img.shields.io/npm/l/%40johnhenry%2Fmath-plus-tensor-webgpu.svg)](../../LICENSE)

WebGPU-accelerated GEMM (tiled, small-M "skinny", and subgroup-matrix
kernels; f32, or f16 storage with f32 accumulation), attention-adjacent
primitives (QKᵀ / softmax / weighted-sum), and elementwise fusion by
compiling `@johnhenry/math-plus-tensor-compile`'s IR to WGSL. Browsers via
`navigator.gpu`; Node/Bun via Dawn (optional `webgpu` peer, `./dawn`
subpath).

## Install

```bash
npm install @johnhenry/math-plus-tensor-webgpu
```

## Quick start

```js
import {
  detectWebGPU, toWebGPU, GPUTensor, runGemm, runGemmWGSL, runGemmF16WGSL,
  runQKT, runSoftmax, runWeightedSum,
  runElementwiseWGSL, chooseGemmBackend,
} from "@johnhenry/math-plus-tensor-webgpu";

const cap = await detectWebGPU(); // requests a real adapter AND device (+ shader-f16, subgroup matrices when offered)
if (!cap.available) throw new Error(cap.reason);
const { device } = cap;
console.log(cap.gemm); // { f16: true, subgroupMatrix: true } on Apple GPUs with the right flags

// GEMM, GPU-resident: GPUTensor in, GPUTensor out, no host round-trip
const a = GPUTensor.fromFloat32Array(device, aData, [m, k]);
const w = GPUTensor.fromFloat32Array(device, wData, [n, k]);   // a Linear weight
const y = await runGemm(device, a, w, { transB: true });       // [m, n]
const yHost = await y.toTensor();
a.free(); w.free(); y.free();

// GEMM on host arrays (upload + compute + readback per call)
const c = await runGemmWGSL(device, aData, bData, m, k, n);    // Float32Array
const c16 = await runGemmF16WGSL(device, aBits, bBits, m, k, n); // Uint16Array f16 bits in/out, f32 accumulation

// Explicit async device transfer (f32, contiguous tensors only)
const gpuA = await toWebGPU(tensorA, device);

// Attention chain stays GPU-resident — no CPU round-trip between calls
const scores = await runQKT(device, q, k, /* dims */); // NOTE: unscaled — apply 1/sqrt(dim) yourself
const weights = await runSoftmax(device, scores, /* dims */);
const out = await runWeightedSum(device, weights, v, /* dims */);
// You own every GPUTensor you get back, intermediates included: .free() them.
```

## Node and Bun

Node has no built-in WebGPU; this package supports it as "works with a
documented native-addon install step" (docs/PLAN.md §6.3), via Dawn's
official binding, the [`webgpu`](https://www.npmjs.com/package/webgpu) npm
package (prebuilt for darwin universal, linux x64/arm64, win32 x64/arm64).
It's an **optional peer dependency**, loaded only through the separate
`./dawn` subpath so browser bundles never see it:

```bash
npm install webgpu
```

```js
import { detectWebGPU } from "@johnhenry/math-plus-tensor-webgpu";
import { requestDawnGPU } from "@johnhenry/math-plus-tensor-webgpu/dawn";

const gpu = await requestDawnGPU({ unsafe: true }); // null if `webgpu` isn't installed/loadable
const cap = await detectWebGPU({ gpu });
```

`requestDawnGPU` also installs Dawn's `GPUBufferUsage`/`GPUMapMode` globals
when missing. `unsafe: true` creates the Dawn instance with
`allow_unsafe_apis`, which only unlocks experimental features — it's what
exposes subgroup matrices; without it you get the portable kernels. Verified
under Node 24 on macOS (Metal); Bun uses the same addon but isn't part of
this package's test matrix.

## GEMM kernels

| Kernel | Chosen automatically when | Notes |
|---|---|---|
| `tiled` | everything not below | 64x64x16 workgroup-memory tiles; any shape and alignment (vec4 loads when K / N allow) |
| `skinny` | `transB`, K % 4 == 0, M <= 64 | small-M latency path: each weight row read once, split-K |
| `subgroup-matrix` | M > 64, K % 4 == 0 (and N % 4 == 0 unless `transB`), and the device offers f32 8x8x8 subgroup matrices at subgroup size 32 | **experimental, Dawn-only**: `chromium-experimental-subgroup-matrix`, which needs `allow_unsafe_apis` in Node (`requestDawnGPU({ unsafe: true })`) or `--enable-unsafe-webgpu` in Chrome. Apple GPUs only in practice |

All kernels load in f32, accumulate in f32, and round once on store — so
**f16 GEMM is f16 storage with f32 accumulation**, and it needs the
`shader-f16` feature (it throws rather than silently widening without it).
f16 crosses the host boundary as binary16 bits in a `Uint16Array`, the same
representation `@johnhenry/math-plus-tensor-core` uses for `"f16"` tensors.
Pass `kernel: "tiled" | "skinny" | "subgroup-matrix"` to force one (it
throws if the kernel can't handle the shape). Ported from laya-js's WebGPU
backend, where these kernels are verified against MLX; tile sizes are tuned
on an Apple M2.

## The honest threshold

`GEMM_ELEMENT_THRESHOLD` is `128 * 128`: `chooseGemmBackend(m, n)` returns
`"webgpu"` once the output has at least 16,384 elements. That's measured,
end to end (upload + compute + readback), against tensor-wasm's
`matmulInto` on an **Apple M2**: WebGPU wins at every size from n = 128 in
headless Chrome 153 and default-flag Chromium 152, and from n = 96 under
Dawn; below that a ~0.3-0.5 ms per-call floor loses to WASM. Resident
2048³ f32 reaches ≈1.76 TFLOP/s with subgroup matrices. Full numbers:
[`docs/spikes/webgpu-tiled-gemm.md`](../../docs/spikes/webgpu-tiled-gemm.md).

History: v1's naive kernel never crossed over (`Infinity`,
[`docs/spikes/webgpu-baseline.md`](../../docs/spikes/webgpu-baseline.md),
Intel iGPU via ANGLE-GL). **That machine hasn't been re-measured with the
new kernels, nor has any discrete GPU** — the threshold is one machine's
number, ignores k and residency, and weaker/software adapters will cross
later or never. Re-run `scripts/measure-gemm-threshold.ts` before trusting
it on your hardware. A test pins the value so recalibration stays deliberate.

## API surface

| Export | What it is |
|---|---|
| `detectWebGPU({ gpu?, f16?, subgroupMatrix? })` / `toWebGPU` / `GPUTensor` | Capability detection (incl. `gemm` capabilities), explicit device transfer, f32 or f16 GPU-resident tensor (`fromFloat32Array`/`fromFloat16Bits`/`toTensor`/`toFloat32Array`/`toUint16Array`/`free`) |
| `runGemm` | GPU-resident GEMM on `GPUTensor`s (f32/f16, `transB`, `kernel`) |
| `runGemmWGSL` / `runGemmF16WGSL` | GEMM on host `Float32Array` / f16-bits `Uint16Array` |
| `selectGemmKernel` / `gemmKernelApplicable` / `planGemm` / `GEMM_CONFIG` | Kernel choice and generated shader, inspectable |
| `registerGemmAdapter` / `gemmCapabilities` | Enable the subgroup-matrix kernel on a device you created yourself (needs the adapter; `detectWebGPU` does it for you) |
| `tiledGemmWGSL` / `skinnyGemmWGSL` / `subgroupMatrixGemmWGSL` | The WGSL generators |
| `requestDawnGPU` (`/dawn` subpath) | Dawn `GPU` for Node/Bun (optional `webgpu` peer) |
| `runQKT` / `runSoftmax` / `runWeightedSum` | SDPA primitives, `GPUTensor` in/out, chained via queue ordering (no fences needed) |
| `compileIRToWGSL` / `runElementwiseWGSL` | tensor-compile IR → WGSL shader source; upload/dispatch/readback runner |
| `chooseGemmBackend` / `GEMM_ELEMENT_THRESHOLD` | The measured (non-)crossover, see above |
| `gpu-runtime` helpers | Buffer pool (`acquireBuffer`/`releaseBuffer`), pipeline cache, `readBackFloat32`, `workgroupsFor` |

`Tensor` is deliberately **not** monkey-patched with a `.to("webgpu")`
method — `toWebGPU(tensor, device)` is a free function so the dependency
arrow keeps pointing the right way.

## Traps

- **f32/f16 only; contiguous only.** `toWebGPU` rejects other dtypes and
  non-contiguous views (call `.contiguous()` first). Only GEMM computes on
  f16 — attention and elementwise fusion reject f16 `GPUTensor`s.
- **GEMM is 2-D only**: no batched/broadcast matmul, no strided/offset
  operands, no bias/activation epilogue, no mixed dtypes (A, B, C share one).
- **Subgroup matrices are experimental** (Dawn's
  `chromium-experimental-subgroup-matrix`): Dawn/Chromium-only, behind
  `allow_unsafe_apis` / `--enable-unsafe-webgpu`, and its WGSL syntax has
  changed across versions. The package validates the shader on first use,
  tries the older syntax, and falls back to `tiled` if both fail.
- A device you create yourself only gets the subgroup-matrix kernel after
  `registerGemmAdapter(device, adapter)` (Dawn doesn't expose the needed
  adapter info on the device).
- **Manual memory:** `GPUBuffer`s aren't GC'd predictably — `.free()` every
  `GPUTensor`, including chain intermediates you never read back.
- **`runElementwiseWGSL` does not broadcast** — all inputs and the output
  must share `elementCount`; broadcast on the CPU first.
- **`runQKT` is unscaled** — apply `1/sqrt(dim)` yourself.
- `erf` and exact `gelu` lower to an f32 port of tensor-core's canonical
  erf (`src/special.ts`, loop counts from `ERF_F32_PARAMS`): ~1e-7 absolute
  for `erf`, but `erfc`'s *relative* error in the far tail (z → 9) grows
  toward ~1e-5 because WGSL only specifies `exp` to `3 + 2·|x|` ULP.
  IR op `gelu` is exact erf-GELU since #122; `gelu_tanh` is the tanh form.
- WGSL `pow` is NaN for negative bases where JS isn't; comparisons/step
  functions can flip branches within f32 epsilon — exactly the ops the
  GPU-vs-CPU fuzzer (issue #58) deliberately excludes.
- Headless *Chrome* testing needs real infrastructure: an HTTP origin
  (`navigator.gpu` is absent on `about:blank`/`data:` even with flags),
  Xvfb, `$MATH_PLUS_CHROME_PATH` to pin a Chrome binary, and
  `--test-concurrency=1` (concurrent Chrome instances starve
  `requestAdapter()` on one physical GPU).

## Tests

`npm test` — every GPU test runs against a real adapter through one of two
harnesses (`$MATH_PLUS_WEBGPU_HARNESS=dawn|chrome`, default: Dawn in-process,
falling back to headless Chrome with a 3-attempt cold-start retry, issue
#49); GEMM correctness for every kernel x {f32, f16} x both B layouts
against a NumPy oracle (`scripts/gemm_oracle.py`); IR fuzz cross-check against the CPU evaluator (issue #58,
including the "unreferenced input must still be statically used" silent-zeros
guard); GPU-residency and pipeline/buffer-reuse regression tests (issue
#100); an `Interval`-based f32 precision oracle (issue #36).

## Provenance

Built for issue #12 on the `docs/spikes/webgpu-baseline.md` spike. Part of
the [math-plus](https://github.com/johnhenry/math-plus) monorepo; family
docs at <https://opensource.johnhenry.me/math/>.
