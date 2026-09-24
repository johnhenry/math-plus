/**
 * GEMM backend selection (issue #12, v1 scope item 1: "GEMM above a measured
 * size threshold — small matmuls stay on WASM since GPU dispatch overhead
 * dominates at small sizes"). The crossover is MEASURED, not guessed, with
 * `scripts/measure-gemm-threshold.ts` (end to end per call: upload, compute,
 * read back — the cost a host-array matmul actually pays).
 *
 * History, because the number changed for real reasons:
 *
 *  1. v1 (docs/spikes/webgpu-baseline.md): `Infinity`. The naive
 *     one-thread-per-output kernel never beat `@johnhenry/math-plus-tensor-wasm`'s
 *     `matmulInto` from 8x8 to 768x768 on the trycooy dev box (Intel ADL-N
 *     iGPU through ANGLE's GL backend under Xvfb).
 *  2. 2026-09-23 (docs/spikes/webgpu-tiled-gemm.md): `m·n >= 128 * 128`.
 *     Tiled / skinny / subgroup-matrix kernels on an Apple M2, against the
 *     OLD scalar WASM GEMM (~1.7 GFLOP/s at 1024³).
 *  3. 2026-09-24 (same doc, "Re-measured against the SIMD WASM GEMM"): `m·n >= 192 * 192`
 *     AND `m·n·k >= 2^22`. tensor-wasm's SIMD128 blocked GEMM (#130,
 *     ~37 GFLOP/s at 1024³) made WASM ~20x faster in the kernel, which moved
 *     the square crossover from n = 128 to n = 160 (Dawn) / n = 192 (headless
 *     Chrome), measured with docs/BENCHMARKING.md's thermal-aware method. It
 *     also showed that m·n alone is the wrong shape of rule: at m·n = 192²
 *     a small k (16, 64) still loses to WASM (0.27-0.92x), and at m·n <= 128²
 *     a large k (4096) still loses in Chrome (0.57-0.81x), because WebGPU's
 *     per-call floor (~0.3-0.5 ms submit + `mapAsync`) has to be paid for by
 *     enough output tiles AND enough work per tile. Over all 43 measured
 *     shapes (square, a k sweep at five m·n, Linear x·Wᵀ), this rule routes
 *     no shape to WebGPU that measured slower there, under either Dawn or
 *     Chrome; it leaves some Dawn wins (e.g. 160³ at 1.26x) on WASM,
 *     deliberately taking the more conservative (browser) crossover.
 *  4. 2026-09-24, issue #146 (same doc, "Re-measured on backend-webgpu"):
 *     unchanged. GEMM now runs on @johnhenry/backend-webgpu's kernels and
 *     runtime; re-measured under Dawn, the rule still routes no measured
 *     shape to a slower WebGPU, while the new path also wins some shapes it
 *     leaves on WASM (large k on small m·n, e.g. 128x4096x128 at 2.9x).
 *     Not loosened: headless Chrome was not re-measured, and the rule
 *     follows the browser crossover.
 *  5. 0.3.0 (same doc, "Re-measured in three environments"): `m·n >= 256²`
 *     AND `m·n·k >= 2^24`. Measured through the facade callers now use
 *     (`gpu.backend.matmul` for A·B, which is backend-webgpu's tiled
 *     kernel everywhere; `linear` for x·Wᵀ) in Dawn, headless Chrome and a
 *     real, visible Chromium (Chrome 152, no `--enable-unsafe-webgpu`, so
 *     no subgroup matrices). Dawn still wins from 160³, but both browsers
 *     lose at 192³ (0.88x headless, 0.89x visible), and the visible
 *     browser also loses 192x256x192 (0.95x), 192x1024x192 (0.96x) and the
 *     16-row Linear 16x1024x3072ᵀ (0.96x), all of which the previous rule
 *     sent to WebGPU. Over the 43 shapes, `m·n >= 256²` is the smallest
 *     element threshold that routes no shape to a slower WebGPU in any of
 *     the three; `2^24` = 256³ is the smallest product measured at that
 *     m·n, a win in all three (1.41-2.78x). It keeps small-k products on
 *     WASM, which lost in both browsers at m·n = 192² (192x16x192 0.69x /
 *     0.55x, 192x64x192 0.64x visible). Left on WASM although all three
 *     measured a win: 128x1024x128 (1.17-1.45x) and 192x4096x192
 *     (1.21-2.31x).
 *
 * Caveats, loudly: this is ONE machine's number (Apple M2). A weaker GPU,
 * or a software adapter (lavapipe/SwiftShader in CI) crosses over later or
 * never; a faster WASM (more threads) crosses later too. It prices
 * host-array calls — operands that already live on the GPU
 * (`gpu.backend.matmul` on tensors from `gpu.fromTensor`) make WebGPU
 * cheaper at every size. No measured shape has m·n >= 256² with k < 256,
 * so the work test is not pinned down there. Re-run the script (and its
 * browser page, `scripts/gemm-threshold-page/`) on your hardware before
 * trusting it.
 */

/** Output elements (m·n) below which WASM GEMM beat WebGPU end to end on the reference machine, whatever k was. See this module's doc. */
export const GEMM_ELEMENT_THRESHOLD = 256 * 256;

/** Multiply-adds (m·n·k) below which WASM GEMM beat WebGPU end to end on the reference machine, even at m·n >= {@link GEMM_ELEMENT_THRESHOLD}. See this module's doc. */
export const GEMM_WORK_THRESHOLD = 2 ** 24;

/**
 * `"webgpu"` when BOTH `m·n >= GEMM_ELEMENT_THRESHOLD` and
 * `m·n·k >= GEMM_WORK_THRESHOLD`, else `"wasm"`. `k` is optional for
 * compatibility with the earlier `(m, n)` signature; omitted, only the m·n
 * test applies (callers that know k should pass it — the data says it
 * matters). Doesn't factor in residency or GPU queue occupancy.
 */
export function chooseGemmBackend(m: number, n: number, k?: number): "wasm" | "webgpu" {
  const elements = m * n;
  if (elements < GEMM_ELEMENT_THRESHOLD) return "wasm";
  if (k !== undefined && elements * k < GEMM_WORK_THRESHOLD) return "wasm";
  return "webgpu";
}
