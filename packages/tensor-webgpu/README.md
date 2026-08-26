# @johnhenry/math-plus-tensor-webgpu

WebGPU-accelerated GEMM, attention-adjacent primitives (QKᵀ / softmax /
weighted-sum), and elementwise fusion by compiling
`@johnhenry/math-plus-tensor-compile`'s IR to WGSL. **Chromium-family
browsers only in v1.**

## Install

```bash
npm install @johnhenry/math-plus-tensor-webgpu
```

## Quick start

```js
import {
  detectWebGPU, toWebGPU, runGemmWGSL,
  runQKT, runSoftmax, runWeightedSum,
  runElementwiseWGSL, chooseGemmBackend,
} from "@johnhenry/math-plus-tensor-webgpu";

const cap = await detectWebGPU(); // requests a real adapter AND device
if (!cap.available) throw new Error(cap.reason);
const { device } = cap;

// Explicit async device transfer (f32, contiguous tensors only)
const gpuA = await toWebGPU(tensorA, device);

// Attention chain stays GPU-resident — no CPU round-trip between calls
const scores = await runQKT(device, q, k, /* dims */); // NOTE: unscaled — apply 1/sqrt(dim) yourself
const weights = await runSoftmax(device, scores, /* dims */);
const out = await runWeightedSum(device, weights, v, /* dims */);
// You own every GPUTensor you get back, intermediates included: .free() them.
```

## Node support

There is none in v1, and `detectWebGPU()` says so rather than guessing:
`navigator.gpu` requires a Chromium-family browser or a Node WebGPU polyfill
that this package deliberately does not bundle. Tests run against real
headless Chrome (Xvfb + `--enable-unsafe-swiftshader`), not a mock — and
they *skip*, never fail, when no GPU adapter is available.

## The honest threshold

`GEMM_ELEMENT_THRESHOLD` is `Infinity`, so `chooseGemmBackend()` currently
**always returns `"wasm"`**. That is the measured result, not a bug:
tensor-wasm's `matmulInto` beat this package's v1 GEMM at every size from
8×8 to 768×768, by 5–10x — on a machine with no discrete GPU (ANGLE →
Intel iGPU under Xvfb) and with a deliberately naive one-thread-per-output
shader (no shared-memory tiling). A test pins the non-crossover so
recalibrating (`scripts/measure-gemm-threshold.ts`) is a visible, deliberate
change. If you have a real GPU, measure before believing either backend.

## API surface

| Export | What it is |
|---|---|
| `detectWebGPU` / `toWebGPU` / `GPUTensor` | Capability detection, explicit device transfer, f32 GPU-resident tensor (`toTensor`/`toFloat32Array`/`free`) |
| `runGemmWGSL` | Naive WGSL GEMM (one thread per output element) |
| `runQKT` / `runSoftmax` / `runWeightedSum` | SDPA primitives, `GPUTensor` in/out, chained via queue ordering (no fences needed) |
| `compileIRToWGSL` / `runElementwiseWGSL` | tensor-compile IR → WGSL shader source; upload/dispatch/readback runner |
| `chooseGemmBackend` / `GEMM_ELEMENT_THRESHOLD` | The measured (non-)crossover, see above |
| `gpu-runtime` helpers | Buffer pool (`acquireBuffer`/`releaseBuffer`), pipeline cache, `readBackFloat32`, `workgroupsFor` |

`Tensor` is deliberately **not** monkey-patched with a `.to("webgpu")`
method — `toWebGPU(tensor, device)` is a free function so the dependency
arrow keeps pointing the right way.

## Traps

- **f32 only; contiguous only.** `toWebGPU` rejects non-f32 dtypes and
  non-contiguous views (call `.contiguous()` first).
- **Manual memory:** `GPUBuffer`s aren't GC'd predictably — `.free()` every
  `GPUTensor`, including chain intermediates you never read back.
- **`runElementwiseWGSL` does not broadcast** — all inputs and the output
  must share `elementCount`; broadcast on the CPU first.
- **`runQKT` is unscaled** — apply `1/sqrt(dim)` yourself.
- WGSL `pow` is NaN for negative bases where JS isn't; comparisons/step
  functions can flip branches within f32 epsilon — exactly the ops the
  GPU-vs-CPU fuzzer (issue #58) deliberately excludes.
- Headless testing needs real infrastructure: an HTTP origin
  (`navigator.gpu` is absent on `about:blank`/`data:` even with flags),
  Xvfb, `$MATH_PLUS_CHROME_PATH` to pin a Chrome binary, and
  `--test-concurrency=1` (concurrent Chrome instances starve
  `requestAdapter()` on one physical GPU).

## Tests

`npm test` — real-browser harness with a 3-attempt cold-start retry
(issue #49); IR fuzz cross-check against the CPU evaluator (issue #58,
including the "unreferenced input must still be statically used" silent-zeros
guard); GPU-residency and pipeline/buffer-reuse regression tests (issue
#100); an `Interval`-based f32 precision oracle (issue #36).

## Provenance

Built for issue #12 on the `docs/spikes/webgpu-baseline.md` spike. Part of
the [math-plus](https://github.com/johnhenry/math-plus) monorepo; family
docs at <https://opensource.johnhenry.me/math/>.
