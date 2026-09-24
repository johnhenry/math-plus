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
 *  3. Now (same doc, "Re-measured against the SIMD WASM GEMM"): `m·n >= 192 * 192`
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
 *
 * Caveats, loudly: this is ONE machine's number (Apple M2, subgroup-matrix
 * kernel available). A browser without subgroup matrices, a weaker GPU, or a
 * software adapter (lavapipe/SwiftShader in CI) crosses over later or never;
 * a faster WASM (more threads) crosses later too. It prices host-array calls —
 * operands that already live on the GPU (`gpu.backend.matmul` on tensors
 * from `gpu.fromTensor`) make
 * WebGPU cheaper at every size. Re-run the script on your hardware before
 * trusting it.
 */

/** Output elements (m·n) below which WASM GEMM beat WebGPU end to end on the reference machine, whatever k was. See this module's doc. */
export const GEMM_ELEMENT_THRESHOLD = 192 * 192;

/** Multiply-adds (m·n·k) below which WASM GEMM beat WebGPU end to end on the reference machine, even at m·n >= {@link GEMM_ELEMENT_THRESHOLD}. See this module's doc. */
export const GEMM_WORK_THRESHOLD = 2 ** 22;

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
