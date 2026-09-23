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
 *     iGPU through ANGLE's GL backend under Xvfb) — 5-10x slower, with no
 *     narrowing trend, so no finite threshold was honest.
 *  2. Now (docs/spikes/webgpu-tiled-gemm.md): `128 * 128`. With the tiled /
 *     skinny / subgroup-matrix kernels on an Apple M2 (Metal), WebGPU wins
 *     end to end at every measured square size from n = 96 under Dawn
 *     (Node) and from n = 128 in headless Chrome 153 (whose per-call
 *     readback overhead is higher), and loses below — the per-call floor is
 *     ~0.3-0.5 ms of submit + `mapAsync` latency, which WASM beats easily
 *     for tiny matrices. The constant takes the more conservative (Chrome)
 *     crossover.
 *
 * Caveats, loudly: this is ONE machine's number. It has not been
 * re-measured on the trycooy Intel iGPU with the new kernels, nor on any
 * discrete GPU; weaker GPUs or software adapters (SwiftShader) will cross
 * over later or never. It is m*n-based only (ignores k), and it prices
 * host-array calls — operands that already live on the GPU (`runGemm` on
 * `GPUTensor`s) make WebGPU cheaper still. Re-run the script on your
 * hardware before trusting it.
 */

/** Output elements (m*n) at or above which WebGPU GEMM beat WASM end to end on the reference machine — see this module's doc for exactly where and how that was measured. */
export const GEMM_ELEMENT_THRESHOLD = 128 * 128;

/** `"wasm"` below the measured crossover, `"webgpu"` at or above it. Pure size-based heuristic — doesn't factor in `k`, residency, or GPU queue occupancy. */
export function chooseGemmBackend(m: number, n: number): "wasm" | "webgpu" {
  return m * n >= GEMM_ELEMENT_THRESHOLD ? "webgpu" : "wasm";
}
