/**
 * GEMM throughput spike (issue #121) — `matmulInto` GFLOP/s on the SIMD128
 * and scalar-only paths, square n x n x n, f32, resident buffers (kernel
 * call only, no copy-in). Results are recorded in docs/spikes/wasm-simd.md.
 *
 *   npm run build:wasm -w @johnhenry/math-plus-tensor-wasm
 *   node packages/tensor-wasm/scripts/gemm-bench.ts [sizes=64,256,512,1024] [wasmDir]
 *
 * `wasmDir` (optional) points at a directory holding a different pair of
 * built artifacts (`tensor_wasm_kernels.wasm` + `tensor_wasm_kernels_simd128.wasm`)
 * — how the pre-#121 naive-kernel baseline was measured on the same machine.
 * Reports the best of N timed calls after one warm-up call (best-of, not
 * mean: the question is what the kernel can do, and contention on a shared
 * dev box only ever adds time).
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { Kernels } from "../src/index.ts";

const sizes = (process.argv[2] ?? "64,256,512,1024").split(",").map(Number);
const wasmDir = process.argv[3];

async function load(simd: boolean): Promise<Kernels> {
  const scalarBytes = wasmDir ? new Uint8Array(await readFile(join(wasmDir, "tensor_wasm_kernels.wasm"))) : undefined;
  if (!simd) return Kernels.load(scalarBytes, new Uint8Array([0])); // invalid SIMD bytes -> scalar-only
  const simdBytes = wasmDir
    ? new Uint8Array(await readFile(join(wasmDir, "tensor_wasm_kernels_simd128.wasm")))
    : undefined;
  return Kernels.load(scalarBytes, simdBytes);
}

function det(len: number, seed: number): Float32Array {
  return Float32Array.from({ length: len }, (_, i) => Math.sin(i * 12.9898 + seed * 78.233));
}

const rows: Array<Record<string, string | number>> = [];
for (const [label, simd] of [["simd128", true], ["scalar", false]] as const) {
  const kernels = await load(simd);
  if (simd && !kernels.simdAvailable) {
    // e.g. a pre-#121 artifact pair, whose SIMD module has no GEMM export.
    console.warn("simd128: SIMD module unavailable (or has no gemm_f32_simd128) -- skipping that path");
    continue;
  }
  for (const n of sizes) {
    const a = kernels.fromArray(det(n * n, 1), [n, n]);
    const b = kernels.fromArray(det(n * n, 2), [n, n]);
    const out = kernels.zeros([n, n]);
    kernels.matmulInto(out, a, b); // warm-up (also tiers up the wasm code)
    const flops = 2 * n ** 3;
    const iters = Math.max(3, Math.min(50, Math.round(2e9 / flops)));
    let best = Infinity;
    for (let i = 0; i < iters; i++) {
      const t0 = performance.now();
      kernels.matmulInto(out, a, b);
      best = Math.min(best, performance.now() - t0);
    }
    rows.push({ path: label, n, ms: +best.toFixed(3), gflops: +(flops / best / 1e6).toFixed(2) });
    a.free();
    b.free();
    out.free();
  }
}
console.table(rows);
