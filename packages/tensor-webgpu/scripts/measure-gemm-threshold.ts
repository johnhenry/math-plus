/**
 * Measures the WASM-vs-WebGPU GEMM crossover on THIS machine: real timings,
 * not a guess. Not part of `npm test` (per issue #12, GPU *performance* is
 * never gated per PR): run it by hand and record the result in
 * docs/spikes/webgpu-tiled-gemm.md, which `src/threshold.ts` cites.
 *
 * Method: docs/BENCHMARKING.md, via `scripts/bench/thermal.ts`'s `runGrid` —
 * a cooldown (default 5 s) before every (cell, backend) measurement, one
 * untimed warmup, a timing window of at most 1 s (3..30 samples, median
 * reported), and WASM/WebGPU alternated inside each cell in one process.
 *
 * Every call is timed END TO END, because that's what a single host-array
 * matmul costs (gemm-threshold-cells.ts has both calls): WASM =
 * `WasmTensor.fromArray` x2 + `matmulInto` + `toFloat32Array` + free
 * (tensor-wasm's SIMD128 GEMM when the runtime has SIMD, #130); WebGPU =
 * the facade on `Float32Array`s: `gpu.fromHost` x2, `gpu.backend.matmul`
 * (A·B) or `gpu.backend.linear` (x·Wᵀ), `gpu.toHost`, dispose — i.e.
 * backend-webgpu's GEMM and runtime. The `kernel` column lists the backend
 * pipelines one call dispatched (`gemm` = tiled, `gemmsg` = subgroup
 * matrices, `gemmskinny`, `gemmdirect`, `splitk`).
 *
 * Harnesses (`$MATH_PLUS_WEBGPU_HARNESS`):
 *  - `dawn` (default here): Dawn in THIS process (backend-webgpu's
 *    `getGpu({ unsafe: true })`), calling the facade directly — no harness
 *    round trip in the timed call.
 *  - `chrome`: headless Chrome via test/helpers.ts's CDP harness. Inputs
 *    live in the page; each call times itself with the page's
 *    `performance.now()` (returned as `{ selfTimedMs }`), so the CDP round
 *    trip is NOT in the sample. Chrome coarsens that clock to 0.1 ms in a
 *    non-cross-origin-isolated page, so small-cell medians are quantized.
 *    WASM runs in this Node process.
 *  - A real, visible browser: `node scripts/gemm-threshold-page/serve.ts`
 *    serves a page that runs the same cells, both sides in the page.
 *
 * Cells: square n³, a k sweep at fixed m·n (does k move the crossover?), and
 * Linear shapes x[M,K]·W[N,K]ᵀ (WASM gets W pre-transposed to [K,N]).
 *
 * Run (from packages/tensor-webgpu, after `npm run build:wasm` at the root),
 * inside the ~/gpu.lock convention (AGENTS.md):
 *   node scripts/measure-gemm-threshold.ts
 *   MATH_PLUS_WEBGPU_HARNESS=chrome node scripts/measure-gemm-threshold.ts
 *   GROUPS=square SIZES=64,128 BENCH_COOL_S=0 node scripts/measure-gemm-threshold.ts   # smoke only
 *   OUT=results.json node scripts/measure-gemm-threshold.ts                             # raw rows + machine info
 */
import { writeFileSync } from "node:fs";
import path from "node:path";
import { getGpu } from "@johnhenry/backend-webgpu";
import { Kernels, WasmTensor } from "@johnhenry/math-plus-tensor-wasm";
import { formatGrid, machineInfo, runGrid, type GridRow } from "../../../scripts/bench/thermal.ts";
import { bundleForBrowser, closeHarness, getHarness, SRC } from "../test/helpers.ts";
import {
  cellInputs,
  cellLabel as label,
  facadeGemm,
  FACADE_GEMM_SOURCE,
  gemmCells,
  kernelsOf,
  KERNEL_PROBE_SOURCE,
  LCG_SOURCE,
  wasmGemm,
  type Cell,
} from "./gemm-threshold-cells.ts";

const list = (key: string): number[] | undefined => process.env[key]?.split(",").map(Number);
const cells = gemmCells({
  sizes: list("SIZES"),
  sweepMN: list("SWEEP_MN"),
  sweepK: list("SWEEP_K"),
  linearM: list("LINEAR_M"),
  groups: process.env.GROUPS ? new Set(process.env.GROUPS.split(",")) : undefined,
});

interface GpuSide {
  kind: "dawn" | "chrome";
  info: unknown;
  /** Make `c` the current cell (inputs placed wherever the GPU call reads them). */
  prepare(c: Cell): Promise<void>;
  /** One end-to-end call on the current cell. */
  call(): Promise<unknown>;
  kernel(c: Cell): Promise<string>;
  close(): Promise<void>;
}

async function dawnSide(): Promise<GpuSide> {
  const { detectWebGPU } = await import("../src/device.ts");
  const { createWebGpuDevice } = await import("../src/facade.ts");
  const entry = await getGpu({ unsafe: true });
  if (!entry) throw new Error("Dawn: the `webgpu` package is not installed or failed to load");
  const cap = await detectWebGPU({ gpu: entry });
  if (!cap.available || !cap.device || !cap.adapter) throw new Error(`Dawn: ${cap.reason ?? "no device"}`);
  const device = cap.device;
  const gpu = await createWebGpuDevice({ device });
  const i = cap.adapter.info;
  let cur: { c: Cell; a: Float32Array; b: Float32Array } | undefined;
  return {
    kind: "dawn",
    info: { vendor: i.vendor, architecture: i.architecture, description: i.description, gemm: cap.gemm },
    prepare: async (c) => {
      const { a, b } = cellInputs(c);
      cur = { c, a, b };
    },
    call: () => {
      const { c, a, b } = cur!;
      return facadeGemm(gpu, c.m, c.k, c.n, c.transB, a, b);
    },
    kernel: (c) => {
      const { a, b } = cellInputs(c);
      return kernelsOf(gpu, () => facadeGemm(gpu, c.m, c.k, c.n, c.transB, a, b));
    },
    close: async () => {
      gpu.destroy();
      device.destroy();
    },
  };
}

async function chromeSide(): Promise<GpuSide> {
  const harness = await getHarness();
  if ("unavailable" in harness) throw new Error(harness.reason);
  if (harness.kind !== "chrome") throw new Error(`expected the chrome harness, got ${harness.kind}`);
  const bundle = bundleForBrowser([path.join(SRC, "facade.ts"), path.join(SRC, "device.ts")]);
  // The bundle's declarations are local to one evaluation, so park what later calls need on globalThis.
  const info = await harness.run(
    `${LCG_SOURCE}
     ${FACADE_GEMM_SOURCE}
     ${KERNEL_PROBE_SOURCE}
     const cap = await detectWebGPU({ gpu: navigator.gpu });
     if (!cap.available) throw new Error(cap.reason);
     const gpu = await createWebGpuDevice({ device: cap.device });
     globalThis.__mp = { lcg, gpu, facadeGemm, kernelsOf };
     const i = cap.adapter.info;
     return { vendor: i.vendor, architecture: i.architecture, description: i.description, gemm: cap.gemm, userAgent: navigator.userAgent, crossOriginIsolated: globalThis.crossOriginIsolated };`,
    bundle,
  );
  let cur: Cell | undefined;
  return {
    kind: "chrome",
    info,
    prepare: async (c) => {
      cur = c;
      await harness.run(`const P = globalThis.__mp;
        P.a = P.lcg(${c.m * c.k}, ${c.m + 1});
        P.b = P.lcg(${c.k * c.n}, ${c.n + 2});`);
    },
    call: () => {
      const c = cur!;
      return harness.run<{ selfTimedMs: number }>(`const P = globalThis.__mp;
        const t0 = performance.now();
        await P.facadeGemm(P.gpu, ${c.m}, ${c.k}, ${c.n}, ${c.transB}, P.a, P.b);
        return { selfTimedMs: performance.now() - t0 };`);
    },
    kernel: (c) =>
      harness.run<string>(
        `const P = globalThis.__mp;
         return P.kernelsOf(P.gpu, () => P.facadeGemm(P.gpu, ${c.m}, ${c.k}, ${c.n}, ${c.transB}, P.lcg(${c.m * c.k}, 1), P.lcg(${c.k * c.n}, 2)));`,
      ),
    close: closeHarness,
  };
}

async function main(): Promise<void> {
  const kernels = await Kernels.load();
  const want = (process.env.MATH_PLUS_WEBGPU_HARNESS ?? "dawn").toLowerCase();
  const gpu = want === "chrome" ? await chromeSide() : await dawnSide();
  const machine = machineInfo();
  console.log(`# harness=${gpu.kind} adapter=${JSON.stringify(gpu.info)} wasmSimd=${kernels.simdAvailable}`);
  console.log(`# machine=${JSON.stringify(machine)}`);

  // runGrid runs both backends of a cell back to back after the cell's
  // cooldown, and the first call of each backend is the untimed warmup — so
  // generating a cell's inputs lazily here never lands in a timed sample.
  let prepared: Cell | undefined;
  let wasmIn: { a: Float32Array; bKN: Float32Array } | undefined;
  const ensure = async (c: Cell): Promise<void> => {
    if (prepared === c) return;
    const { a, bKN } = cellInputs(c);
    wasmIn = { a, bKN };
    await gpu.prepare(c);
    prepared = c;
  };
  const rows = await runGrid<Cell>({
    cells,
    label,
    backends: [
      {
        name: "wasm",
        run: async (c) => {
          await ensure(c);
          wasmGemm(kernels, WasmTensor, c, wasmIn!.a, wasmIn!.bKN);
        },
      },
      {
        name: "webgpu",
        run: async (c) => {
          await ensure(c);
          return gpu.call();
        },
      },
    ],
  });

  const at = (c: Cell, backend: string): GridRow => rows.find((r) => r.cell === label(c) && r.backend === backend)!;
  const kernelOf = new Map<Cell, string>();
  for (const c of cells) kernelOf.set(c, await gpu.kernel(c));

  const fmt = (r: GridRow): string => `${r.medianMs.toFixed(3)} (${r.minMs.toFixed(3)}–${r.maxMs.toFixed(3)}, n=${r.n})`;
  for (const group of ["square", "k-sweep", "linear"] as const) {
    const gc = cells.filter((c) => c.group === group);
    if (gc.length === 0) continue;
    console.log(`\n## ${group} (${gpu.kind}), ms: median (min–max, samples)\n`);
    console.log("| m x k x n | m·n | m·n·k | kernel | WASM e2e | WebGPU e2e | WASM/WebGPU |");
    console.log("|---|---:|---:|---|---:|---:|---:|");
    for (const c of gc) {
      const w = at(c, "wasm");
      const g = at(c, "webgpu");
      console.log(
        `| ${label(c)} | ${c.m * c.n} | ${c.m * c.n * c.k} | ${kernelOf.get(c)} | ${fmt(w)} | ${fmt(g)} | ${(w.medianMs / g.medianMs).toFixed(2)}x |`,
      );
    }
  }

  // Square crossover: smallest n from which WebGPU wins at EVERY larger measured n
  // (a single noisy win below a loss doesn't count).
  const sq = cells.filter((c) => c.group === "square");
  let crossover: number | undefined;
  for (let i = sq.length - 1; i >= 0; i--) {
    const c = sq[i]!;
    if (at(c, "webgpu").medianMs < at(c, "wasm").medianMs) crossover = c.n;
    else break;
  }
  if (sq.length > 0) {
    console.log(
      crossover === undefined
        ? `\nNo square crossover: WASM faster at the largest measured n (${sq.at(-1)!.n})`
        : `\nSquare crossover (end to end, f32): WebGPU faster at every measured n >= ${crossover} (m·n = ${crossover * crossover}, m·n·k = ${crossover ** 3})`,
    );
  }
  console.log(`\n${formatGrid(rows)}`);
  if (process.env.OUT) writeFileSync(process.env.OUT, JSON.stringify({ harness: gpu.kind, adapter: gpu.info, machine, cells, rows }, null, 1));
  await gpu.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
