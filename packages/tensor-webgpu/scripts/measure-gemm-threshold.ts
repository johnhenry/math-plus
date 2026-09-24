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
 * matmul costs: WASM = `WasmTensor.fromArray` x2 + `matmulInto` +
 * `toFloat32Array` + free (tensor-wasm's SIMD128 GEMM when the runtime has
 * SIMD, #130); WebGPU = `runGemmWGSL` on `Float32Array`s (upload, dispatch,
 * `mapAsync` readback, pooled buffers) — since issue #146 running on
 * @johnhenry/backend-webgpu's GEMM and runtime. The `kernel` column lists
 * the backend pipelines one call dispatched (e.g. `gemmsg`, `gemmskinny`,
 * `copy+gemmsg` for A·B through bᵀ, `gemm` = tiled).
 *
 * Harnesses (`$MATH_PLUS_WEBGPU_HARNESS`):
 *  - `dawn` (default here): Dawn in THIS process (`src/dawn.ts`), calling the
 *    package's own functions directly — no harness round trip in the timed
 *    call.
 *  - `chrome`: headless Chrome via test/helpers.ts's CDP harness. Inputs
 *    live in the page; each call times itself with the page's
 *    `performance.now()` (returned as `{ selfTimedMs }`), so the CDP round
 *    trip is NOT in the sample. Chrome coarsens that clock to 0.1 ms in a
 *    non-cross-origin-isolated page, so small-cell medians are quantized.
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
import { Kernels, WasmTensor } from "@johnhenry/math-plus-tensor-wasm";
import { formatGrid, machineInfo, runGrid, type GridRow } from "../../../scripts/bench/thermal.ts";
import { bundleForBrowser, closeHarness, getHarness, SRC } from "../test/helpers.ts";

interface Cell {
  group: "square" | "k-sweep" | "linear";
  m: number;
  k: number;
  n: number;
  transB: boolean;
}

const list = (key: string, dflt: string): number[] => (process.env[key] ?? dflt).split(",").map(Number);
const SIZES = list("SIZES", "8,16,32,48,64,96,128,160,192,256,384,512,1024,2048");
/** k sweep: square m = n outputs, each with these k. */
const SWEEP_MN = list("SWEEP_MN", "32,64,96,128,192");
const SWEEP_K = list("SWEEP_K", "16,64,256,1024,4096");
/** Linear rows M against ModernBERT-large's MLP-in weight [3072, 1024]. */
const LINEAR_M = list("LINEAR_M", "1,4,16,64,256");
const GROUPS = new Set((process.env.GROUPS ?? "square,k-sweep,linear").split(","));

const cells: Cell[] = [];
if (GROUPS.has("square")) for (const n of SIZES) cells.push({ group: "square", m: n, k: n, n, transB: false });
if (GROUPS.has("k-sweep")) {
  for (const mn of SWEEP_MN) for (const k of SWEEP_K) if (k !== mn) cells.push({ group: "k-sweep", m: mn, k, n: mn, transB: false });
}
if (GROUPS.has("linear")) for (const m of LINEAR_M) cells.push({ group: "linear", m, k: 1024, n: 3072, transB: true });

const label = (c: Cell): string => `${c.m}x${c.k}x${c.n}${c.transB ? "ᵀ" : ""}`;

const LCG_SOURCE = `function lcg(size, seed) {
  let s = seed >>> 0;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) { s = (s * 1664525 + 1013904223) >>> 0; out[i] = (s / 0xffffffff) * 2 - 1; }
  return out;
}`;
const lcg = new Function(`${LCG_SOURCE}; return lcg;`)() as (size: number, seed: number) => Float32Array;

/** One cell's inputs. `b` is [K,N], or [N,K] when transB; `bKN` is always [K,N] (what WASM's matmulInto takes). */
function inputs(c: Cell): { a: Float32Array; b: Float32Array; bKN: Float32Array } {
  const a = lcg(c.m * c.k, c.m + 1);
  const b = lcg(c.k * c.n, c.n + 2);
  if (!c.transB) return { a, b, bKN: b };
  const bKN = new Float32Array(c.k * c.n);
  for (let j = 0; j < c.n; j++) for (let p = 0; p < c.k; p++) bKN[p * c.n + j] = b[j * c.k + p] as number;
  return { a, b, bKN };
}

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
  const { requestDawnGPU } = await import("../src/dawn.ts");
  const { detectWebGPU } = await import("../src/device.ts");
  const { runGemmWGSL } = await import("../src/gemm.ts");
  const { backendFor } = await import("../src/bridge.ts");
  const gpu = await requestDawnGPU({ unsafe: true });
  if (!gpu) throw new Error("Dawn: the `webgpu` package is not installed or failed to load");
  const cap = await detectWebGPU({ gpu });
  if (!cap.available || !cap.device || !cap.adapter) throw new Error(`Dawn: ${cap.reason ?? "no device"}`);
  const device = cap.device;
  const i = cap.adapter.info;
  let cur: { c: Cell; a: Float32Array; b: Float32Array } | undefined;
  return {
    kind: "dawn",
    info: { vendor: i.vendor, architecture: i.architecture, description: i.description, gemm: cap.gemm },
    prepare: async (c) => {
      const { a, b } = inputs(c);
      cur = { c, a, b };
    },
    call: () => {
      const { c, a, b } = cur!;
      return runGemmWGSL(device, a, b, c.m, c.k, c.n, { transB: c.transB });
    },
    kernel: async (c) => {
      // Record the backend pipelines one call dispatches.
      const rt = backendFor(device).rt;
      const keys: string[] = [];
      const orig = rt.dispatch;
      rt.dispatch = function (this: typeof rt, k, ...rest) {
        keys.push(k.key.split(":")[0]!);
        return orig.call(this, k, ...rest);
      };
      try {
        const { a, b } = inputs(c);
        await runGemmWGSL(device, a, b, c.m, c.k, c.n, { transB: c.transB });
      } finally {
        rt.dispatch = orig;
      }
      return keys.join("+");
    },
    close: async () => device.destroy(),
  };
}

async function chromeSide(): Promise<GpuSide> {
  const harness = await getHarness();
  if ("unavailable" in harness) throw new Error(harness.reason);
  if (harness.kind !== "chrome") throw new Error(`expected the chrome harness, got ${harness.kind}`);
  const bundle = bundleForBrowser([path.join(SRC, "gemm.ts"), path.join(SRC, "device.ts")]);
  // The bundle's declarations are local to one evaluation, so park what later calls need on globalThis.
  const info = await harness.run(
    `${LCG_SOURCE}
     const cap = await detectWebGPU({ gpu: navigator.gpu });
     if (!cap.available) throw new Error(cap.reason);
     globalThis.__mp = { lcg, device: cap.device, runGemmWGSL, backendFor };
     const i = cap.adapter.info;
     return { vendor: i.vendor, architecture: i.architecture, description: i.description, gemm: cap.gemm, userAgent: navigator.userAgent };`,
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
        await P.runGemmWGSL(P.device, P.a, P.b, ${c.m}, ${c.k}, ${c.n}, { transB: ${c.transB} });
        return { selfTimedMs: performance.now() - t0 };`);
    },
    kernel: (c) =>
      harness.run<string>(
        `const P = globalThis.__mp; const rt = P.backendFor(P.device).rt; const keys = []; const orig = rt.dispatch;
         rt.dispatch = function (k, ...rest) { keys.push(k.key.split(":")[0]); return orig.call(this, k, ...rest); };
         try { await P.runGemmWGSL(P.device, P.lcg(${c.m * c.k}, 1), P.lcg(${c.k * c.n}, 2), ${c.m}, ${c.k}, ${c.n}, { transB: ${c.transB} }); }
         finally { rt.dispatch = orig; }
         return keys.join("+");`,
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
    const { a, bKN } = inputs(c);
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
          const { a, bKN } = wasmIn!;
          const ta = WasmTensor.fromArray(kernels, a, [c.m, c.k]);
          const tb = WasmTensor.fromArray(kernels, bKN, [c.k, c.n]);
          const out = kernels.zeros([c.m, c.n]);
          kernels.matmulInto(out, ta, tb);
          out.toFloat32Array();
          ta.free();
          tb.free();
          out.free();
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
