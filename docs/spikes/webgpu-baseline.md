# WebGPU pipeline baseline (2026-08-12)

> **Superseded for GEMM** by [`webgpu-tiled-gemm.md`](webgpu-tiled-gemm.md) (2026-09-23): the
> naive kernel measured here was replaced by tiled / skinny / subgroup-matrix kernels, and
> `GEMM_ELEMENT_THRESHOLD` is now `128 * 128` (measured on an Apple M2; this machine's Intel iGPU
> has not been re-measured). The harness findings below still apply.

Status check on the headless-WebGPU seam for `@johnhenry/math-plus-tensor-webgpu` (issue #12). **Headless
WebGPU is real and working on this machine — genuinely hardware-accelerated, not
SwiftShader-only as expected going in. The naive v1 GEMM kernel, however, never beats
`@johnhenry/math-plus-tensor-wasm`'s `matmulInto` at any size tested.** Numbers below are the baseline a
future tiled kernel (or a real discrete-GPU runner) must beat.

## What's working: headless WebGPU itself

| Element | State |
|---|---|
| Browser | `google-chrome-stable` 149.0.7827.200 (`/opt/google/chrome/chrome`) |
| Display | `Xvfb` (private, per-test-file, random display number — not the machine's shared `gl-chrome.service`/`:99`) |
| Launch flags | `--use-angle=gl --use-gl=angle --ignore-gpu-blocklist --enable-unsafe-webgpu --enable-unsafe-swiftshader --no-sandbox --disable-dev-shm-usage` |
| Driving protocol | Raw CDP over `Runtime.evaluate`, no Playwright/Puppeteer — mirrors `~/.local/bin/gl-report`'s pattern on this machine |
| Adapter obtained | A **real hardware adapter** via ANGLE's GL backend against this machine's Intel iGPU (`chrome://gpu` reports an `ADL-N` adapter with `[WebGPU Status] Available`, distinct from the `SwiftShader Device (Subzero)` and `llvmpipe` CPU adapters, both of which `chrome://gpu` reports **Blocklisted — crbug.com/40057808: CPU adapters not fully tested or conformant**) |
| Full compute round-trip | Verified manually before writing any package code: upload two buffers, dispatch `a[i]+b[i]`/`a[i]*b[i]+1`, read back, compare — correct on both the real-hardware path and (separately, forcing `--use-angle=swiftshader`) the CPU-fallback path |

### The two non-obvious things that made this work

1. **`navigator.gpu` is not exposed on `about:blank` or `data:` URLs.** Every flag combination
   tried against `about:blank` reported `"gpu" in navigator === false`, regardless of
   `--enable-unsafe-webgpu`/`--ignore-gpu-blocklist`/etc. Serving a trivial page over a real
   `http://127.0.0.1:PORT/` origin and navigating there instead fixed it immediately, with the
   *exact same* Chrome flags. Likely a secure-context/feature-policy quirk specific to
   `about:`/`data:` documents. `test/helpers.ts` runs a tiny local HTTP server for exactly this
   reason.
2. **Chrome's `/json/new?<url>` DevTools endpoint takes a literal URL, not a percent-encoded
   query value.** An early version of the test harness called
   `fetch(cdpBase + "/json/new?" + encodeURIComponent(pageUrl))`, which made Chrome try to
   navigate to the literal string `http%3A%2F%2F127.0.0.1%3A.../` — an invalid URL — and silently
   fall back to an error page with no real HTTP origin (same symptom as #1: `NO_NAVIGATOR_GPU`).
   Passing the URL unencoded fixed it. This one cost real debugging time because the symptom
   (`navigator.gpu` missing) looked identical to a Chrome-flags problem, right down to it
   surviving a full manual re-verification of the flags in isolation before the actual bug (the
   encoding) was found.

A third, lower-stakes finding: `node --test`'s default file-level concurrency runs each test
file's own headless Chrome instance in parallel, and multiple Chrome processes contending for
the same physical GPU render node intermittently starved one file's `requestAdapter()` to
`null`. `@johnhenry/math-plus-tensor-webgpu`'s `test` script now passes `--test-concurrency=1` to serialize
its own test files against each other (see `test/helpers.ts`'s module doc for the full
reasoning, including why a filesystem lock alone wouldn't fix this specific case).

## Measured: naive GEMM never beats WASM

Methodology: both backends timed **end-to-end per call** — allocate, copy CPU data in, compute,
copy result out, free — matching what a single `Tensor.matmul()` call actually pays today (v1 has
no "keep operands resident across calls" API wired up at the `Tensor` level for either backend).
Median of 5 iterations per size, square `n x n x n` matmuls, f32. Full methodology and the exact
benchmark in `scripts/measure-gemm-threshold.ts`.

| n | elements (n²) | WASM (median ms) | WebGPU (median ms) | WebGPU/WASM ratio |
|---|---|---|---|---|
| 8 | 64 | 0.093 | 46.200 | 497× slower |
| 16 | 256 | 0.128 | 32.800 | 256× slower |
| 32 | 1,024 | 0.217 | 37.000 | 171× slower |
| 48 | 2,304 | 0.233 | 38.700 | 166× slower |
| 64 | 4,096 | 0.498 | 38.800 | 78× slower |
| 96 | 9,216 | 1.834 | 58.900 | 32× slower |
| 128 | 16,384 | 16.585 | 92.000 | 5.5× slower |
| 192 | 36,864 | 22.552 | 236.500 | 10.5× slower |
| 256 | 65,536 | 43.825 | 263.900 | 6.0× slower |
| 384 | 147,456 | 150.154 | 1,141.800 | 7.6× slower |
| 512 | 262,144 | 407.988 | 2,812.500 | 6.9× slower |
| 768 | 589,824 | 1,193.371 | 11,306.300 | 9.5× slower |

**No crossover found.** WASM wins at every tested size, and — critically — the ratio does not
narrow toward 1× as `n` grows. It stabilizes in the 5-10× range for the larger sizes rather than
trending toward a crossover just past what got measured. Sizes above 768 were not measured: at
the observed growth rate, 1024³ alone would cost roughly 25-30 seconds per call, and the trend
gives no reason to expect a different conclusion.

### Why, honestly

Two real, specific reasons — not "GPUs are just slow":

1. **No discrete GPU on this machine.** WebGPU here runs through ANGLE's GL backend against an
   Intel integrated GPU (see above) — a real hardware path, but a much weaker one than a discrete
   GPU's, and further translated through ANGLE rather than a native Vulkan/Metal/D3D12 backend.
2. **`runGemmWGSL`'s shader (`src/gemm.ts`) is naive by design for v1** — one thread per output
   element, reading `A`/`B` straight from global storage buffers on every iteration of the inner
   `k` loop, no shared-memory tiling. This is exactly the kernel shape most exposed to
   memory-bandwidth limits on weaker hardware; a tiled kernel (blocking into workgroup-shared
   memory, the standard GEMM optimization) would very plausibly change these numbers
   substantially. It's documented as future work in `gemm.ts`'s own module doc rather than
   attempted here — v1's job was proving the seam and measuring honestly, not hand-tuning a
   kernel against a single machine's integrated GPU.

## Consequence for `chooseGemmBackend`

`GEMM_ELEMENT_THRESHOLD` in `src/threshold.ts` is set to `Infinity` — `chooseGemmBackend` always
returns `"wasm"` on v1's default configuration. A specific finite threshold would misrepresent
this measurement (it would imply a crossover was found and extrapolated beyond the tested range,
which didn't happen). Re-run `scripts/measure-gemm-threshold.ts` after either a tiled kernel
lands or on a runner with a real discrete GPU, and replace the constant with whatever that run
finds.

## What this means for the rest of v1

Attention primitives (`runQKT`/`runSoftmax`/`runWeightedSum`) and the elementwise-fusion IR
lowering (`fusion-wgsl.ts`) are unaffected by this finding — they're correctness features (verified
against CPU oracles in `test/attention.test.ts` and `test/fusion.test.ts`, all passing on this
same real hardware adapter), not performance ones gated by a threshold. GEMM is the one v1
primitive with an explicit "use WASM below a threshold" design, and this doc is that threshold's
honest, measured answer for this machine: never, until the kernel or the hardware improves.
