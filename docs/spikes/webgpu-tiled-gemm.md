# WebGPU tiled GEMM: re-measured crossover (2026-09-23)

Follow-up to [`webgpu-baseline.md`](webgpu-baseline.md), which found that v1's naive
one-thread-per-output GEMM never beat `@johnhenry/math-plus-tensor-wasm` and pinned
`GEMM_ELEMENT_THRESHOLD = Infinity`. `@johnhenry/math-plus-tensor-webgpu` now ships three
kernels ported from laya-js's WebGPU backend (same author; verified there against MLX), plus
f16 storage with f32 accumulation. **Result: on an Apple M2, WebGPU wins end to end at every
square size from n = 128 (n = 96 under Dawn), and `GEMM_ELEMENT_THRESHOLD` is now
`128 * 128`.** That number is from one machine; see [Caveats](#caveats).

## What changed in the kernel

| Kernel | Used when | Shape of the work |
|---|---|---|
| `tiled` | default, any shape/alignment | 64x64x16 workgroup-memory tiles, 4x4 outputs/thread, vec4 loads when K (and N or K for B) % 4 == 0 |
| `skinny` | `transB`, K % 4 == 0, M <= 64 | all M rows in one workgroup: each weight row read once; 4-way split-K reduced in workgroup memory |
| `subgroup-matrix` | device has f32 8x8x8 subgroup matrices at subgroup size 32, M > 64, K % 4 == 0 (and N % 4 == 0 for a `[K,N]` B) | Dawn's experimental `chromium_experimental_subgroup_matrix` (Metal `simdgroup_matrix`); 32x64 tiles, 2 subgroups x 4x4 fragments, next K panel prefetched into registers |

All three convert loads to f32, accumulate in f32, and round once on store. The `[K,N]` B layout
for the subgroup-matrix kernel (row-major right fragments) is new here; laya-js only has the
`transB` form. Correctness: `test/gemm.test.ts` runs every applicable kernel x {f32, f16} x 15
shapes (both B layouts, unaligned K/N, partial M/N tiles, partial last K panel) against a NumPy
float64 oracle with a derived error bound — 44 kernel runs per dtype, all passing under both Dawn
and headless Chrome on this machine. A deliberately broken kernel (dropping the partial last K
panel) fails it immediately.

## Setup

| | |
|---|---|
| Machine | MacBook Air, Apple M2 (10-core GPU), macOS 27.0, fanless (throttles under sustained load — 1.5 s idle between sizes) |
| Dawn | `webgpu@0.6.1` npm package in Node 24.9, `allow_unsafe_apis` (so subgroup matrices are on) |
| Chrome | Google Chrome 153.0.8010.50, headless (`--headless=new --use-angle=metal --enable-unsafe-webgpu`), via `test/helpers.ts`'s CDP harness; subgroup matrices on |
| Chromium (default flags) | Claude desktop's built-in browser pane, Chromium 152.0.7977.130, no WebGPU flags: `shader-f16` yes, **subgroup matrices no** (so this is the portable `tiled`/`skinny` path an ordinary browser gets) |
| WASM | `@johnhenry/math-plus-tensor-wasm` `matmulInto`, run in Node (scalar build — its SIMD128 module only covers add/mul) |
| Method | Exactly v1's: square n x n x n f32, **end to end per call** (allocate, upload, compute, read back, free), median of 5; one untimed warmup per size (new). Script: `packages/tensor-webgpu/scripts/measure-gemm-threshold.ts` |

## Results: square f32, end to end (the threshold's basis)

Times in ms (median of 5). "v1 naive" is the old kernel re-measured on this same machine and
harness (Dawn) for an apples-to-apples comparison; the v1 spike's own numbers were from a
different machine (Intel iGPU via ANGLE-GL).

| n | WASM (Node) | v1 naive, Dawn | **new, Dawn** | kernel (Dawn) | **new, Chrome 153** | new, Chromium 152 default flags (tiled) | WASM / new (Dawn) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 8 | 0.014 | 0.283 | 0.348 | tiled | 0.400 | 0.700 | 0.04x |
| 16 | 0.040 | 0.718 | 0.474 | tiled | 0.400 | — | 0.08x |
| 32 | 0.067 | 0.346 | 0.366 | tiled | 0.500 | 0.500 | 0.18x |
| 48 | 0.065 | 0.690 | 0.478 | tiled | 0.700 | — | 0.14x |
| 64 | 0.146 | 0.751 | 0.341 | tiled | 0.500 | 0.600 | 0.43x |
| **96** | 0.526 | 0.725 | **0.394** | subgroup-matrix | 0.600 | 0.600 | **1.33x** |
| **128** | 1.225 | 0.488 | **0.409** | subgroup-matrix | **0.800** | **0.700** | **3.0x** |
| 192 | 4.483 | 0.666 | 0.495 | subgroup-matrix | 0.700 | — | 9.1x |
| 256 | 12.305 | 1.155 | 0.624 | subgroup-matrix | 1.700 | 1.200 | 20x |
| 384 | 43.651 | 2.838 | 0.888 | subgroup-matrix | 2.300 | — | 49x |
| 512 | 113.603 | 4.118 | 1.944 | subgroup-matrix | 3.800 | 3.800 | 58x |
| 768 | 475.202 | 10.711 | 7.444 | subgroup-matrix | 6.000 | — | 64x |
| 1024 | 1113.966 | 18.040 | 6.155 | subgroup-matrix | 9.000 | 12.500 | 181x |
| 1536 | 3446.254 | 58.463 | 19.627 | subgroup-matrix | 15.600 | — | 176x |
| 2048 | 20351.780 | 129.701 | 29.098 | subgroup-matrix | 27.500 | 35.100 | 699x |

WASM during the Chrome run (same Node code, measured interleaved with Chrome's GPU work) came out
slower at the top end (e.g. 37.6 s at n = 2048, 2.68 ms at n = 128), consistent with the fanless
machine throttling; the Dawn-run WASM column above is the cleaner baseline and is the one used for
the Chromium-152 comparison. Chrome's `performance.now()` is coarsened to 0.1 ms in a
non-cross-origin-isolated page, so its small-n numbers are quantized.

**Crossover** (smallest n from which WebGPU wins at every larger measured size): **n = 96 under
Dawn, n = 128 in Chrome 153 and in default-flag Chromium 152.** Below that, every WebGPU call pays
a ~0.3-0.5 ms floor (submit + `mapAsync` readback), which a scalar WASM matmul of <= 64³ beats.

### Beyond end to end (context, not the threshold's basis)

Resident = `runGemm` on already-uploaded `GPUTensor`s, timed to `onSubmittedWorkDone()` (no
upload, no readback). f16 = `runGemmF16WGSL` end to end (f16 storage, f32 accumulation).

| n | Dawn f16 e2e | Dawn resident | Dawn tiled-only e2e | Chrome resident | Chromium 152 resident (tiled) |
|---:|---:|---:|---:|---:|---:|
| 512 | 2.423 | 0.931 | 2.716 | 1.300 | 1.500 |
| 1024 | 5.179 | 4.049 | 9.397 | 3.200 | 5.500 |
| 2048 | 14.184 | **9.756 (1.76 TFLOP/s)** | 31.437 | 14.900 | 21.400 (0.80 TFLOP/s) |

The resident 2048³ subgroup-matrix figure (1.76 TFLOP/s f32) matches what laya-js measured for
the same kernel (≈1.75-1.97 TFLOP/s). The portable tiled kernel reaches ≈0.8 TFLOP/s resident in
default-flag Chromium 152 (laya-js's separate register-blocked "direct" Linear kernel, not ported
here, gets ≈1.2). For reference, MLX's hand-written Metal GEMM on this chip is ≈2.3 (f32) /
≈3 (f16) TFLOP/s per laya-js's measurements.

### Linear-layer shapes, `x[M,1024] · W[3072,1024]ᵀ` (Dawn; WASM gets W pre-transposed)

| M | kernel | WASM e2e | WebGPU e2e | WebGPU f16 e2e | WebGPU resident |
|---:|---|---:|---:|---:|---:|
| 1 | skinny | 3.459 | 2.396 | 1.293 | 0.513 |
| 33 | skinny | 112.773 | 2.847 | 1.938 | 1.135 |
| 128 | subgroup-matrix | 378.512 | 3.891 | 2.919 | 1.801 |
| 512 | subgroup-matrix | 1645.436 | 9.101 | 8.079 | 4.267 |

Even M = 1 (a single-token Linear, 3.1 M multiply-adds) wins end to end, because k is large; the
m*n-only threshold (3,072 output elements, below 16,384) would route it to WASM. That's a known
limitation of the heuristic's shape, kept for API compatibility (see below).

## Consequence: `GEMM_ELEMENT_THRESHOLD = 128 * 128`

`chooseGemmBackend(m, n)` now returns `"webgpu"` for m*n >= 16,384 — the more conservative of the
measured crossovers (Chrome's), since a browser page is this package's primary target.
`test/device.test.ts` pins it so recalibration stays a deliberate, visible change.

## Caveats

- **One machine.** Apple M2 only. The trycooy dev box (Intel ADL-N iGPU via ANGLE's GL backend
  under Xvfb), where v1 measured no crossover, has **not** been re-measured with the new kernels;
  nor has any discrete GPU. Weaker GPUs and software adapters (SwiftShader in CI) will cross over
  later or never. Re-run `scripts/measure-gemm-threshold.ts` on your hardware.
- **m*n only.** The threshold ignores k (see the M = 1 Linear row) and residency: operands that
  already live on the GPU make WebGPU cheaper at every size. A k-aware or residency-aware
  heuristic is future work.
- **Tile configs are M2-tuned** (from laya-js's sweeps); they're correct everywhere but may be
  suboptimal on other GPUs.
- **Subgroup matrices are experimental.** Dawn-only (`chromium-experimental-subgroup-matrix`),
  needs `allow_unsafe_apis` in Node (`requestDawnGPU({ unsafe: true })`) or
  `--enable-unsafe-webgpu` in Chrome; its WGSL builtin syntax has changed across versions (the
  package tries the current template syntax, then the older bool-argument one, then falls back to
  `tiled`). Default-flag Chromium doesn't expose it at all — the Chromium-152 column is that case.
- The WASM side is `matmulInto`'s scalar kernel; a SIMD/blocked WASM GEMM would move the
  crossover up.
