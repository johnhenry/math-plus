# WebGPU tiled GEMM: measured crossover (2026-09-23, re-measured 2026-09-24)

Follow-up to [`webgpu-baseline.md`](webgpu-baseline.md), which found that v1's naive
one-thread-per-output GEMM never beat `@johnhenry/math-plus-tensor-wasm` and pinned
`GEMM_ELEMENT_THRESHOLD = Infinity`. `@johnhenry/math-plus-tensor-webgpu` now ships three
kernels ported from laya-js's WebGPU backend (same author; verified there against MLX), plus
f16 storage with f32 accumulation. **Result: on an Apple M2, WebGPU wins end to end at every
square size from n = 128 (n = 96 under Dawn), and `GEMM_ELEMENT_THRESHOLD` is now
`128 * 128`.** That number is from one machine; see [Caveats](#caveats).

> **Current threshold (2026-09-24):** `chooseGemmBackend(m, n, k)` picks WebGPU when
> `m·n >= 192²` **and** `m·n·k >= 2²²`, re-measured against tensor-wasm's SIMD128 GEMM (#130)
> with the thermal-aware method — see
> [Re-measured against the SIMD WASM GEMM](#re-measured-against-the-simd-wasm-gemm-2026-09-24).
> The 2026-09-23 sections below it are kept as the record of the `128 * 128` threshold they
> produced against the old scalar WASM kernel.

## Re-measured against the SIMD WASM GEMM (2026-09-24)

The first measurement (below) raced WebGPU against `matmulInto`'s **scalar** kernel (~1.7 GFLOP/s
at 1024³). #130 then gave tensor-wasm a blocked SIMD128 GEMM (~37 GFLOP/s at 1024³,
[`wasm-simd.md`](wasm-simd.md)), so every WASM time below n = 2048 got 5-40x shorter, and the
first measurement also predated [`docs/BENCHMARKING.md`](../BENCHMARKING.md)'s method (fixed
5-iteration loop, 1.5 s pauses, backends not alternated). Both are fixed here.

### Setup

| | |
|---|---|
| Machine | MacBook Air M2 (`Mac14,15`, 8 CPU cores, 10-core GPU, 24 GiB), macOS 27.0 (26A428), **on AC**, fanless. `pmset -g therm`: no thermal or performance warning recorded at the start of either run. Load average 4-9 during the runs (other agents shared the machine; the GPU itself was held under `~/gpu.lock`) |
| Method | `scripts/bench/thermal.ts` `runGrid`: 5 s cooldown before every (shape, backend) cell, 1 untimed warmup, timing window <= 1 s (3-30 samples), **median**; WASM and WebGPU alternated inside each cell (order swapped every other cell), one process per harness |
| What is timed | End to end per call. WASM: `WasmTensor.fromArray` x2 + `matmulInto` (SIMD128 build, `simdAvailable: true`) + `toFloat32Array` + free, in Node 24.9. WebGPU: `runGemmWGSL` on `Float32Array`s (upload, dispatch, `mapAsync` readback, pooled buffers) |
| Dawn | `webgpu@0.6.1` in the same Node process, `allow_unsafe_apis` (subgroup matrices on), the package's functions called directly: no harness in the timed call |
| Chrome | Headless Google Chrome (`--headless=new --use-angle=metal --enable-unsafe-webgpu`) via `test/helpers.ts`'s CDP harness, subgroup matrices on. Each call is timed inside the page (`performance.now()`, returned as `{ selfTimedMs }`), so the CDP round trip is excluded; that clock is coarsened to 0.1 ms, so small Chrome numbers are quantized. WASM still runs in Node |
| Script | `packages/tensor-webgpu/scripts/measure-gemm-threshold.ts` (now on `runGrid`); `OUT=` writes raw rows incl. min/max/n and run order |

### Results (ms, median; ratio = WASM / WebGPU, bold = WebGPU faster)

Each harness was a separate run a few minutes apart, so each has its own interleaved WASM column.
Min/max/sample counts are in the script output; the spread was small except where noted in the
raw rows (e.g. 48³ WASM on the Dawn run, 0.048-1.809 ms).

#### Square n³

| m x k x n | m·n | m·n·k | WASM (Dawn run) | **Dawn** | ratio | WASM (Chrome run) | **Chrome** | ratio | new rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8x8x8 | 64 | 512 | 0.009 | 0.322 | 0.03x | 0.010 | 0.400 | 0.02x | wasm |
| 16x16x16 | 256 | 4096 | 0.073 | 0.376 | 0.19x | 0.028 | 0.400 | 0.07x | wasm |
| 32x32x32 | 1024 | 32768 | 0.054 | 0.438 | 0.12x | 0.054 | 0.600 | 0.09x | wasm |
| 48x48x48 | 2304 | 110592 | 0.364 | 0.520 | 0.70x | 0.116 | 0.600 | 0.19x | wasm |
| 64x64x64 | 4096 | 262144 | 0.115 | 0.574 | 0.20x | 0.038 | 0.400 | 0.10x | wasm |
| 96x96x96 | 9216 | 884736 | 0.298 | 0.425 | 0.70x | 0.093 | 0.500 | 0.19x | wasm |
| 128x128x128 | 16384 | 2097152 | 0.510 | 0.643 | 0.79x | 0.193 | 0.800 | 0.24x | wasm |
| 160x160x160 | 25600 | 4096000 | 0.870 | 0.691 | **1.26x** | 0.871 | 1.000 | 0.87x | wasm |
| 192x192x192 | 36864 | 7077888 | 1.013 | 0.676 | **1.50x** | 0.979 | 0.800 | **1.22x** | webgpu |
| 256x256x256 | 65536 | 16777216 | 1.654 | 0.889 | **1.86x** | 1.737 | 1.300 | **1.34x** | webgpu |
| 384x384x384 | 147456 | 56623104 | 3.622 | 1.165 | **3.11x** | 3.671 | 2.000 | **1.84x** | webgpu |
| 512x512x512 | 262144 | 134217728 | 7.769 | 1.413 | **5.50x** | 7.742 | 3.300 | **2.35x** | webgpu |
| 1024x1024x1024 | 1048576 | 1073741824 | 58.349 | 5.731 | **10.18x** | 85.130 | 8.200 | **10.38x** | webgpu |
| 2048x2048x2048 | 4194304 | 8589934592 | 463.155 | 22.658 | **20.44x** | 495.754 | 26.400 | **18.78x** | webgpu |

#### k sweep at fixed m = n

| m x k x n | m·n | m·n·k | WASM (Dawn run) | **Dawn** | ratio | WASM (Chrome run) | **Chrome** | ratio | new rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 32x16x32 | 1024 | 16384 | 0.008 | 0.275 | 0.03x | 0.024 | 0.400 | 0.06x | wasm |
| 32x64x32 | 1024 | 65536 | 0.011 | 0.324 | 0.03x | 0.026 | 0.500 | 0.05x | wasm |
| 32x256x32 | 1024 | 262144 | 0.025 | 0.451 | 0.05x | 0.086 | 0.900 | 0.10x | wasm |
| 32x1024x32 | 1024 | 1048576 | 0.240 | 0.945 | 0.25x | 0.277 | 1.300 | 0.21x | wasm |
| 32x4096x32 | 1024 | 4194304 | 0.346 | 2.436 | 0.14x | 0.291 | 4.000 | 0.07x | wasm |
| 64x16x64 | 4096 | 65536 | 0.023 | 0.288 | 0.08x | 0.085 | 0.400 | 0.21x | wasm |
| 64x256x64 | 4096 | 1048576 | 0.085 | 0.467 | 0.18x | 0.280 | 0.600 | 0.47x | wasm |
| 64x1024x64 | 4096 | 4194304 | 0.268 | 0.828 | 0.32x | 0.738 | 1.700 | 0.43x | wasm |
| 64x4096x64 | 4096 | 16777216 | 1.065 | 2.722 | 0.39x | 1.206 | 4.900 | 0.25x | wasm |
| 96x16x96 | 9216 | 147456 | 0.048 | 0.363 | 0.13x | 0.158 | 0.600 | 0.26x | wasm |
| 96x64x96 | 9216 | 589824 | 0.077 | 0.615 | 0.13x | 0.192 | 0.800 | 0.24x | wasm |
| 96x256x96 | 9216 | 2359296 | 0.175 | 0.673 | 0.26x | 0.515 | 1.100 | 0.47x | wasm |
| 96x1024x96 | 9216 | 9437184 | 0.594 | 0.720 | 0.82x | 0.978 | 1.800 | 0.54x | wasm |
| 96x4096x96 | 9216 | 37748736 | 2.433 | 1.720 | **1.41x** | 2.571 | 4.500 | 0.57x | wasm |
| 128x16x128 | 16384 | 262144 | 0.086 | 0.368 | 0.23x | 0.211 | 0.600 | 0.35x | wasm |
| 128x64x128 | 16384 | 1048576 | 0.125 | 0.373 | 0.34x | 0.317 | 0.600 | 0.53x | wasm |
| 128x256x128 | 16384 | 4194304 | 0.306 | 0.481 | 0.64x | 0.670 | 1.100 | 0.61x | wasm |
| 128x1024x128 | 16384 | 16777216 | 1.025 | 0.712 | **1.44x** | 1.613 | 1.300 | **1.24x** | wasm |
| 128x4096x128 | 16384 | 67108864 | 4.237 | 2.032 | **2.09x** | 3.749 | 4.600 | 0.81x | wasm |
| 192x16x192 | 36864 | 589824 | 0.186 | 0.394 | 0.47x | 0.189 | 0.700 | 0.27x | wasm |
| 192x64x192 | 36864 | 2359296 | 0.280 | 0.478 | 0.59x | 0.735 | 0.800 | 0.92x | wasm |
| 192x256x192 | 36864 | 9437184 | 0.648 | 0.457 | **1.42x** | 1.234 | 1.100 | **1.12x** | webgpu |
| 192x1024x192 | 36864 | 37748736 | 2.263 | 0.743 | **3.05x** | 2.522 | 2.400 | **1.05x** | webgpu |
| 192x4096x192 | 36864 | 150994944 | 8.474 | 2.265 | **3.74x** | 10.969 | 5.500 | **1.99x** | webgpu |

#### Linear x[M,1024] · W[3072,1024]ᵀ (WASM gets W pre-transposed)

| m x k x n | m·n | m·n·k | WASM (Dawn run) | **Dawn** | ratio | WASM (Chrome run) | **Chrome** | ratio | new rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1x1024x3072ᵀ | 3072 | 3145728 | 2.389 | 2.041 | **1.17x** | 1.883 | 2.700 | 0.70x | wasm |
| 4x1024x3072ᵀ | 12288 | 12582912 | 2.017 | 2.112 | 0.96x | 2.075 | 2.600 | 0.80x | wasm |
| 16x1024x3072ᵀ | 49152 | 50331648 | 4.267 | 2.515 | **1.70x** | 3.668 | 2.800 | **1.31x** | webgpu |
| 64x1024x3072ᵀ | 196608 | 201326592 | 12.106 | 4.346 | **2.79x** | 11.855 | 4.100 | **2.89x** | webgpu |
| 256x1024x3072ᵀ | 786432 | 805306368 | 45.547 | 5.833 | **7.81x** | 48.261 | 7.900 | **6.11x** | webgpu |

### Reading it

- **Square crossover: n = 160 under Dawn, n = 192 in Chrome** (was 96 / 128 against the scalar
  kernel). The ~0.3-0.5 ms (Dawn) / ~0.4-0.6 ms (Chrome) per-call floor now has to beat a WASM
  GEMM that finishes 128³ in 0.2-0.5 ms.
- **m·n alone is the wrong rule.** At m·n = 192² a small k still loses (192x16x192: 0.47x / 0.27x;
  192x64x192: 0.59x / 0.92x), and a large k on a small output still loses in Chrome
  (128x4096x128: 0.81x, 96x4096x96: 0.57x). The GPU needs both enough output tiles to fill it and
  enough work to cover the fixed per-call cost.
- **m·n·k alone is also wrong**, for the same large-k-small-output cells (64x4096x64 is 16.8 M
  multiply-adds and loses 0.39x / 0.25x on the `tiled` kernel).
- **Chosen rule: `m·n >= 192²` and `m·n·k >= 2²²`.** Over all 43 shapes, it sends no shape to
  WebGPU that measured slower there, under Dawn or Chrome. The shapes it leaves on WASM although
  WebGPU won are Dawn-only wins except one: Dawn 160³ (1.26x), 96x4096x96 (1.41x), 128x1024x128
  (1.44x), 128x4096x128 (2.09x), the single-token Linear 1x1024x3072ᵀ (1.17x), and 128x1024x128
  in Chrome (1.24x). Like the previous threshold, it takes the browser's (more conservative)
  crossover, since a browser page is this package's primary target. `2²²` sits between the
  largest losing product at m·n >= 192² (2.4 M, 192x64x192) and the smallest winning one
  (7.1 M, 192³), in both harnesses.
- The Linear rows moved too: the single-token `M = 1` product no longer wins in Chrome (0.70x;
  1.17x under Dawn), and `M = 4` loses in both, so the old write-up's "even M = 1 wins" is now
  Dawn-only and marginal.
- f16, resident and tiled-only columns were not re-measured (they don't affect the threshold);
  their 2026-09-23 numbers below still describe the GPU side.



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

## 2026-09-23 setup

| | |
|---|---|
| Machine | MacBook Air, Apple M2 (10-core GPU), macOS 27.0, fanless (throttles under sustained load — 1.5 s idle between sizes) |
| Dawn | `webgpu@0.6.1` npm package in Node 24.9, `allow_unsafe_apis` (so subgroup matrices are on) |
| Chrome | Google Chrome 153.0.8010.50, headless (`--headless=new --use-angle=metal --enable-unsafe-webgpu`), via `test/helpers.ts`'s CDP harness; subgroup matrices on |
| Chromium (default flags) | Claude desktop's built-in browser pane, Chromium 152.0.7977.130, no WebGPU flags: `shader-f16` yes, **subgroup matrices no** (so this is the portable `tiled`/`skinny` path an ordinary browser gets) |
| WASM | `@johnhenry/math-plus-tensor-wasm` `matmulInto`, run in Node (scalar build — its SIMD128 module only covers add/mul) |
| Method | Exactly v1's: square n x n x n f32, **end to end per call** (allocate, upload, compute, read back, free), median of 5; one untimed warmup per size (new). Script: `packages/tensor-webgpu/scripts/measure-gemm-threshold.ts` |

## 2026-09-23: square f32, end to end, vs the scalar WASM GEMM (superseded threshold basis)

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

## 2026-09-23 consequence (superseded): `GEMM_ELEMENT_THRESHOLD = 128 * 128`

`chooseGemmBackend(m, n)` now returns `"webgpu"` for m*n >= 16,384 — the more conservative of the
measured crossovers (Chrome's), since a browser page is this package's primary target.
`test/device.test.ts` pins it so recalibration stays a deliberate, visible change.

## Caveats

- **One machine.** Apple M2 only. The trycooy dev box (Intel ADL-N iGPU via ANGLE's GL backend
  under Xvfb), where v1 measured no crossover, has **not** been re-measured with the new kernels;
  nor has any discrete GPU. Weaker GPUs and software adapters (SwiftShader in CI) will cross over
  later or never. Re-run `scripts/measure-gemm-threshold.ts` on your hardware.
- **Residency is ignored.** The threshold (now k-aware, see the 2026-09-24 section) prices
  host-array calls; operands that already live on the GPU make WebGPU cheaper at every size. A
  residency-aware heuristic is future work.
- **Tile configs are M2-tuned** (from laya-js's sweeps); they're correct everywhere but may be
  suboptimal on other GPUs.
- **Subgroup matrices are experimental.** Dawn-only (`chromium-experimental-subgroup-matrix`),
  needs `allow_unsafe_apis` in Node (`requestDawnGPU({ unsafe: true })`) or
  `--enable-unsafe-webgpu` in Chrome; its WGSL builtin syntax has changed across versions (the
  package tries the current template syntax, then the older bool-argument one, then falls back to
  `tiled`). Default-flag Chromium doesn't expose it at all — the Chromium-152 column is that case.
- The 2026-09-23 WASM side was `matmulInto`'s scalar kernel; the SIMD/blocked GEMM (#130) did move
  the crossover up, as predicted (2026-09-24 section). A multi-threaded WASM GEMM would move it
  again.
