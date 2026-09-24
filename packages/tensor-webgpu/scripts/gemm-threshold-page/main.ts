/**
 * The GEMM threshold benchmark in a real browser page: the same 43 cells,
 * the same end-to-end calls (../gemm-threshold-cells.ts) and the same
 * thermal-aware `runGrid` (scripts/bench/thermal.ts) as
 * measure-gemm-threshold.ts, with BOTH sides in the page — tensor-wasm's
 * SIMD GEMM in the page's WASM engine, WebGPU through
 * `createWebGpuDevice()` on the page's `navigator.gpu` (no
 * `--enable-unsafe-webgpu`, so no subgroup matrices in a normal Chrome).
 *
 * Served by serve.ts (cross-origin isolated, so `performance.now()` is not
 * coarsened to 0.1 ms). Query parameters: `groups`, `sizes`, `sweepMN`,
 * `sweepK`, `linearM` (comma lists), `cool` (seconds, default 5),
 * `window` (ms, default 1000), `autostart=0` to wait for the button.
 * Progress and results are in `window.__gemm` and rendered as Markdown in
 * `#out` (read them with DevTools, or any page-reading tool).
 */
import { Kernels, WasmTensor } from "@johnhenry/math-plus-tensor-wasm";
import { formatGrid, runGrid, type GridRow } from "../../../../scripts/bench/thermal.ts";
import { detectWebGPU } from "../../src/device.ts";
import { createWebGpuDevice } from "../../src/facade.ts";
import { cellInputs, cellLabel, facadeGemm, gemmCells, kernelsOf, wasmGemm, type Cell } from "../gemm-threshold-cells.ts";

interface State {
  status: "idle" | "running" | "done" | "error";
  done: number;
  total: number;
  info?: unknown;
  rows: GridRow[];
  kernels: Record<string, string>;
  markdown?: string;
  error?: string;
}

const state: State = { status: "idle", done: 0, total: 0, rows: [], kernels: {} };
(globalThis as { __gemm?: State }).__gemm = state;

const params = new URLSearchParams(location.search);
const list = (key: string): number[] | undefined => params.get(key)?.split(",").map(Number);
const out = document.getElementById("out")!;
const show = (): void => {
  out.textContent = `status: ${state.status} ${state.done}/${state.total}\n${state.error ?? ""}\n${state.markdown ?? state.rows.map((r) => `${r.backend.padEnd(7)} ${r.cell.padEnd(14)} ${r.medianMs.toFixed(3)} ms (n=${r.n})`).join("\n")}`;
};

async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

function table(cells: readonly Cell[], at: (c: Cell, backend: string) => GridRow): string {
  const fmt = (r: GridRow): string => `${r.medianMs.toFixed(3)} (${r.minMs.toFixed(3)}–${r.maxMs.toFixed(3)}, n=${r.n})`;
  const lines: string[] = [];
  for (const group of ["square", "k-sweep", "linear"] as const) {
    const gc = cells.filter((c) => c.group === group);
    if (!gc.length) continue;
    lines.push(`\n## ${group} (page), ms: median (min–max, samples)\n`);
    lines.push("| m x k x n | m·n | m·n·k | kernel | WASM e2e | WebGPU e2e | WASM/WebGPU |");
    lines.push("|---|---:|---:|---|---:|---:|---:|");
    for (const c of gc) {
      const w = at(c, "wasm");
      const g = at(c, "webgpu");
      lines.push(`| ${cellLabel(c)} | ${c.m * c.n} | ${c.m * c.n * c.k} | ${state.kernels[cellLabel(c)]} | ${fmt(w)} | ${fmt(g)} | ${(w.medianMs / g.medianMs).toFixed(2)}x |`);
    }
  }
  return lines.join("\n");
}

async function main(): Promise<void> {
  state.status = "running";
  const cells = gemmCells({
    sizes: list("sizes"),
    sweepMN: list("sweepMN"),
    sweepK: list("sweepK"),
    linearM: list("linearM"),
    groups: params.get("groups") ? new Set(params.get("groups")!.split(",")) : undefined,
  });
  state.total = cells.length * 2;
  const base = new URL("./wasm/", location.href).href;
  const kernels = await Kernels.load(await fetchBytes(`${base}tensor_wasm_kernels.wasm`), await fetchBytes(`${base}tensor_wasm_kernels_simd128.wasm`));
  const cap = await detectWebGPU();
  if (!cap.available || !cap.device || !cap.adapter) throw new Error(`WebGPU: ${cap.reason}`);
  const gpu = await createWebGpuDevice({ device: cap.device });
  const i = cap.adapter.info;
  state.info = {
    vendor: i.vendor,
    architecture: i.architecture,
    description: i.description,
    gemm: cap.gemm,
    wasmSimd: kernels.simdAvailable,
    userAgent: navigator.userAgent,
    crossOriginIsolated,
    hardwareConcurrency: navigator.hardwareConcurrency,
  };
  show();

  let prepared: Cell | undefined;
  let inputs: ReturnType<typeof cellInputs> | undefined;
  const ensure = (c: Cell): ReturnType<typeof cellInputs> => {
    if (prepared !== c) {
      inputs = cellInputs(c);
      prepared = c;
    }
    return inputs!;
  };
  state.rows = await runGrid<Cell>({
    cells,
    label: cellLabel,
    cooldownMs: Number(params.get("cool") ?? 5) * 1000,
    windowMs: Number(params.get("window") ?? 1000),
    onResult: (row) => {
      state.rows.push(row);
      state.done++;
      show();
    },
    backends: [
      { name: "wasm", run: (c) => void wasmGemm(kernels, WasmTensor, c, ensure(c).a, ensure(c).bKN) },
      { name: "webgpu", run: (c) => facadeGemm(gpu, c.m, c.k, c.n, c.transB, ensure(c).a, ensure(c).b) },
    ],
  });
  for (const c of cells) {
    const { a, b } = cellInputs(c);
    state.kernels[cellLabel(c)] = await kernelsOf(gpu, () => facadeGemm(gpu, c.m, c.k, c.n, c.transB, a, b));
  }
  const at = (c: Cell, backend: string): GridRow => state.rows.find((r) => r.cell === cellLabel(c) && r.backend === backend)!;
  state.markdown = `# adapter=${JSON.stringify(state.info)}\n${table(cells, at)}\n\n${formatGrid(state.rows)}`;
  state.status = "done";
  show();
}

const start = (): void => {
  if (state.status !== "idle") return;
  main().catch((e: unknown) => {
    state.status = "error";
    state.error = e instanceof Error ? (e.stack ?? e.message) : String(e);
    show();
  });
};
document.getElementById("start")!.addEventListener("click", start);
show();
if (params.get("autostart") !== "0") start();
