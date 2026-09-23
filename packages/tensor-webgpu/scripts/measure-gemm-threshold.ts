/**
 * Measures the WASM-vs-WebGPU GEMM crossover on THIS machine, the same way
 * docs/spikes/wasm-baseline.md measured the WASM-vs-pure-JS crossover:
 * real timings, not a guess. Not part of `npm test` (per issue #12: "GPU
 * *performance* tests gate behind real hardware, not per-PR" — this is a
 * spike script you run manually and record the results of in
 * docs/spikes/webgpu-tiled-gemm.md; docs/spikes/webgpu-baseline.md holds the
 * v1 naive-kernel numbers it replaced).
 *
 * Headline columns are timed END-TO-END per call exactly like the v1 spike
 * — allocate, copy CPU data in, compute, copy result out, free — because
 * that's what a single host-array matmul costs: `matmulInto` on fresh
 * `WasmTensor`s vs `runGemmWGSL` on `Float32Array`s. One untimed warmup
 * call per size/backend (v1 had none; its median-of-5 already discarded
 * the compile-bearing first call), then the median of `ITERATIONS`.
 *
 * Extra WebGPU columns, for context rather than for the threshold:
 *  - `f16 e2e`: `runGemmF16WGSL` end to end (half the bytes moved).
 *  - `resident`: `runGemm` on already-uploaded `GPUTensor`s, timed to
 *    `queue.onSubmittedWorkDone()` — the cost when operands stay on the GPU
 *    across calls (no upload, no readback).
 *  - `tiled`: end to end with the portable kernel forced (what a browser
 *    without subgroup matrices gets).
 *
 * Inputs are generated on both sides from the same LCG (nothing crosses
 * the harness boundary but timings), so the Chrome path doesn't pay for
 * serializing 16M-element arrays.
 *
 * Run (from packages/tensor-webgpu, after `npm run build:wasm` at the root):
 *   node scripts/measure-gemm-threshold.ts              # Dawn if available, else Chrome
 *   MATH_PLUS_WEBGPU_HARNESS=chrome node scripts/measure-gemm-threshold.ts
 *   SIZES=8,64,512 ITERATIONS=5 COOLDOWN_MS=2000 node scripts/measure-gemm-threshold.ts
 * Follow AGENTS.md's ~/gpu.lock convention around it.
 */
import path from "node:path";
import { Kernels, WasmTensor } from "@johnhenry/math-plus-tensor-wasm";
import { bundleForBrowser, closeHarness, getHarness, SRC } from "../test/helpers.ts";

const SIZES = (process.env.SIZES ?? "8,16,32,48,64,96,128,192,256,384,512,768,1024,1536,2048").split(",").map(Number);
const ITERATIONS = Number(process.env.ITERATIONS ?? 5);
/** Idle between sizes: the reference machine is a fanless MacBook Air whose GPU throttles under sustained load. */
const COOLDOWN_MS = Number(process.env.COOLDOWN_MS ?? 1500);
/** Linear-layer shapes [M, K, N] (ModernBERT-large MLP-in, B stored [N, K]): the skinny and subgroup-matrix kernels' home turf. */
const LINEAR_SHAPES: [number, number, number][] = [
  [1, 1024, 3072],
  [33, 1024, 3072],
  [128, 1024, 3072],
  [512, 1024, 3072],
];

const LCG_SOURCE = `function lcg(size, seed) {
  let s = seed >>> 0;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) { s = (s * 1664525 + 1013904223) >>> 0; out[i] = (s / 0xffffffff) * 2 - 1; }
  return out;
}`;
const lcg = new Function(`${LCG_SOURCE}; return lcg;`)() as (size: number, seed: number) => Float32Array;

const median = (xs: number[]): number => [...xs].sort((x, y) => x - y)[Math.floor(xs.length / 2)] as number;

function measureWasm(kernels: Kernels, m: number, k: number, n: number): number {
  const a = lcg(m * k, m + 1);
  const b = lcg(k * n, n + 2);
  const once = (): number => {
    const t0 = performance.now();
    const ta = WasmTensor.fromArray(kernels, a, [m, k]);
    const tb = WasmTensor.fromArray(kernels, b, [k, n]);
    const out = kernels.zeros([m, n]);
    kernels.matmulInto(out, ta, tb);
    out.toFloat32Array();
    const t1 = performance.now();
    ta.free();
    tb.free();
    out.free();
    return t1 - t0;
  };
  once();
  return median(Array.from({ length: ITERATIONS }, once));
}

interface GpuTimes {
  e2e: number;
  f16: number | null;
  resident: number;
  tiled: number;
  kernel: string;
}

async function main(): Promise<void> {
  const kernels = await Kernels.load();
  const harness = await getHarness();
  if ("unavailable" in harness) {
    console.error(`Cannot measure the WebGPU side: ${harness.reason}`);
    process.exit(1);
  }
  const bundle = bundleForBrowser([path.join(SRC, "gemm.ts"), path.join(SRC, "device.ts")]);
  const info = await harness.run<{ vendor: string; architecture: string; description: string; gemm: unknown }>(
    `const cap = await detectWebGPU({ gpu: navigator.gpu });
     if (!cap.available) throw new Error(cap.reason);
     const i = cap.adapter.info;
     return { vendor: i.vendor, architecture: i.architecture, description: i.description, gemm: cap.gemm };`,
    bundle,
  );
  console.log(`# harness=${harness.kind} adapter=${JSON.stringify(info)} iterations=${ITERATIONS} (median, 1 warmup)`);

  const measureGpu = (m: number, k: number, n: number, transB: boolean): Promise<GpuTimes> =>
    harness.run<GpuTimes>(
      `
      ${LCG_SOURCE}
      const cap = await detectWebGPU({ gpu: navigator.gpu });
      const device = cap.device;
      const [m, k, n, transB, ITER] = [${m}, ${k}, ${n}, ${transB}, ${ITERATIONS}];
      const a = lcg(m * k, m + 1);
      const b = lcg(k * n, n + 2);
      const med = (xs) => [...xs].sort((x, y) => x - y)[Math.floor(xs.length / 2)];
      const time = async (f) => { await f(); const ts = []; for (let i = 0; i < ITER; i++) { const t0 = performance.now(); await f(); ts.push(performance.now() - t0); } return med(ts); };
      const e2e = await time(() => runGemmWGSL(device, a, b, m, k, n, { transB }));
      const tiled = await time(() => runGemmWGSL(device, a, b, m, k, n, { transB, kernel: "tiled" }));
      let f16 = null;
      if (cap.gemm.f16) {
        const toBits = (x) => new Uint16Array(Float16Array.from(x).buffer);
        const a16 = toBits(a), b16 = toBits(b);
        f16 = await time(() => runGemmF16WGSL(device, a16, b16, m, k, n, { transB }));
      }
      const A = GPUTensor.fromFloat32Array(device, a, [m, k]);
      const B = GPUTensor.fromFloat32Array(device, b, transB ? [n, k] : [k, n]);
      const resident = await time(async () => { const C = await runGemm(device, A, B, { transB }); await device.queue.onSubmittedWorkDone(); C.free(); });
      A.free(); B.free();
      return { e2e, f16, resident, tiled, kernel: selectGemmKernel(m, k, n, transB, gemmCapabilities(device)) };
      `,
      bundle,
    );

  const pause = (): Promise<void> => new Promise((r) => setTimeout(r, COOLDOWN_MS));
  const fmt = (x: number | null): string => (x === null ? "n/a" : x.toFixed(3));

  console.log("\n## square n x n x n, f32\n");
  console.log("| n | elements | kernel | WASM e2e ms | WebGPU e2e ms | WASM/WebGPU | WebGPU f16 e2e ms | WebGPU resident ms | WebGPU tiled-only e2e ms |");
  console.log("|---|---|---|---|---|---|---|---|---|");
  const rows: { n: number; wasm: number; gpu: number }[] = [];
  for (const n of SIZES) {
    const wasm = measureWasm(kernels, n, n, n);
    await pause();
    const g = await measureGpu(n, n, n, false);
    rows.push({ n, wasm, gpu: g.e2e });
    console.log(`| ${n} | ${n * n} | ${g.kernel} | ${fmt(wasm)} | ${fmt(g.e2e)} | ${(wasm / g.e2e).toFixed(2)}x | ${fmt(g.f16)} | ${fmt(g.resident)} | ${fmt(g.tiled)} |`);
    await pause();
  }

  console.log("\n## Linear shapes x[M,K] · W[N,K]ᵀ, f32 (WASM gets W pre-transposed to [K,N])\n");
  console.log("| M x K x N | kernel | WASM e2e ms | WebGPU e2e ms | WASM/WebGPU | WebGPU f16 e2e ms | WebGPU resident ms | resident GFLOP/s |");
  console.log("|---|---|---|---|---|---|---|---|");
  for (const [m, k, n] of LINEAR_SHAPES) {
    const wasm = measureWasm(kernels, m, k, n);
    await pause();
    const g = await measureGpu(m, k, n, true);
    const gflops = (2 * m * n * k) / (g.resident * 1e6);
    console.log(`| ${m}x${k}x${n} | ${g.kernel} | ${fmt(wasm)} | ${fmt(g.e2e)} | ${(wasm / g.e2e).toFixed(2)}x | ${fmt(g.f16)} | ${fmt(g.resident)} | ${gflops.toFixed(0)} |`);
    await pause();
  }

  // Crossover = smallest size from which WebGPU wins at EVERY larger measured size too
  // (a single noisy win below a loss doesn't count).
  let crossover: number | undefined;
  for (let i = rows.length - 1; i >= 0; i--) {
    if ((rows[i] as { gpu: number; wasm: number }).gpu < (rows[i] as { wasm: number }).wasm) crossover = (rows[i] as { n: number }).n;
    else break;
  }
  console.log(
    crossover === undefined
      ? `\nNo crossover: WASM faster at the largest measured size (n=${SIZES.at(-1)})`
      : `\nCrossover (end to end, f32): WebGPU faster at every measured n >= ${crossover} (${crossover * crossover} output elements)`,
  );
  await closeHarness();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
