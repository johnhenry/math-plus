---
"@johnhenry/math-plus-tensor-webgpu": minor
---

Fast WebGPU GEMM: tiled, small-M "skinny", and (experimental, Dawn/Chromium) subgroup-matrix kernels ported from laya-js, replacing the naive one-thread-per-output shader.

- f32, and f16 storage with f32 accumulation on devices with `shader-f16` (f16 data crosses the host boundary as binary16 bits in a `Uint16Array`, tensor-core's f16 representation).
- New `runGemm` keeps GEMM GPU-resident (`GPUTensor` in/out); `runGemmWGSL` keeps its signature and gains `{ transB, kernel }` options; new `runGemmF16WGSL`.
- `GPUTensor` can hold f16 (`fromFloat16Bits`, `toUint16Array`; `toTensor` returns an `"f16"` tensor); `toWebGPU` accepts f16 tensors. Its `dtype` field widens from `"f32"` to `"f32" | "f16"`.
- `detectWebGPU(options)` accepts `{ gpu, powerPreference, f16, subgroupMatrix }`, requests `shader-f16` / subgroup matrices when offered, and reports `gemm` capabilities.
- Node/Bun support via Dawn: new `@johnhenry/math-plus-tensor-webgpu/dawn` subpath (`requestDawnGPU`), with `webgpu` as an optional peer dependency.
- `GEMM_ELEMENT_THRESHOLD` changes from `Infinity` to `128 * 128`: re-measured on an Apple M2 (docs/spikes/webgpu-tiled-gemm.md), so `chooseGemmBackend` now routes large GEMMs to WebGPU. One machine's number — re-measure on your hardware.
