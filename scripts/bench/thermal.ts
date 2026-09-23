/**
 * Helper for thermal-aware benchmark spikes. The methodology and the reasons for
 * it are in docs/BENCHMARKING.md; this file is the reusable mechanics:
 *
 * - {@link timeCell}: cooldown, then warmup, then a timing window of at most
 *   `windowMs` (bounded by `minRuns`..`maxRuns`), reporting the median.
 * - {@link runGrid}: runs every cell on every backend, ALTERNATING backends
 *   within each cell and within one process, so a machine that heats up
 *   during the run penalises every backend about equally.
 * - {@link machineInfo}: machine, runtime and thermal state to paste next to
 *   the numbers.
 *
 * Runs under Node (type stripping) and Bun. No dependencies. Example:
 *
 *   import { machineInfo, runGrid, formatGrid } from "../../../scripts/bench/thermal.ts";
 *   const grid = await runGrid({
 *     cells: [64, 256, 1024],
 *     label: (n) => `n=${n}`,
 *     backends: [
 *       { name: "wasm", run: (n) => wasmMatmul(n) },
 *       { name: "webgpu", run: (n) => gpuMatmul(n) },
 *     ],
 *   });
 *   console.log(formatGrid(grid));
 *   console.log(JSON.stringify({ machine: machineInfo(), grid }));
 *
 * Environment overrides (a quick smoke run needs no code change):
 * `BENCH_COOL_S` (default 5), `BENCH_WINDOW_MS` (default 1000).
 */
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import os from "node:os";

export interface CellOptions {
  /** Idle time before the cell, so it starts cold. Default `BENCH_COOL_S`*1000, else 5000. */
  cooldownMs?: number;
  /** Upper bound on the timed window (the warmup is not counted). Default `BENCH_WINDOW_MS`, else 1000. */
  windowMs?: number;
  /** Always take at least this many samples, even if they exceed the window. Default 3. */
  minRuns?: number;
  /** Never take more than this many samples. Default 30. */
  maxRuns?: number;
  /** Untimed calls before the window (JIT, pipeline compile, buffer pools). Default 1. */
  warmup?: number;
  /** Injected for tests; defaults to a real timer. */
  sleep?: (ms: number) => Promise<void>;
  /** Injected for tests; defaults to `performance.now`. */
  now?: () => number;
}

export interface CellResult {
  medianMs: number;
  minMs: number;
  maxMs: number;
  n: number;
  /** Wall time of the timed window, to confirm it stayed at or under `windowMs`. */
  windowMs: number;
}

const envNumber = (key: string): number | undefined => {
  const v = process.env[key];
  return v === undefined || v === "" ? undefined : Number(v);
};

const realSleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** Time one (cell, backend) pair: cooldown, warmup, then a window of at most `windowMs`. */
export async function timeCell(fn: () => unknown, opts: CellOptions = {}): Promise<CellResult> {
  const cooldownMs = opts.cooldownMs ?? (envNumber("BENCH_COOL_S") ?? 5) * 1000;
  const windowMs = opts.windowMs ?? envNumber("BENCH_WINDOW_MS") ?? 1000;
  const minRuns = opts.minRuns ?? 3;
  const maxRuns = opts.maxRuns ?? 30;
  const warmup = opts.warmup ?? 1;
  const sleep = opts.sleep ?? realSleep;
  const now = opts.now ?? (() => performance.now());

  if (cooldownMs > 0) await sleep(cooldownMs);
  for (let i = 0; i < warmup; i++) await fn();
  const samples: number[] = [];
  const start = now();
  while (samples.length < maxRuns && (samples.length < minRuns || now() - start < windowMs)) {
    const a = now();
    await fn();
    samples.push(now() - a);
  }
  const window = now() - start;
  samples.sort((x, y) => x - y);
  return {
    medianMs: samples[Math.floor(samples.length / 2)]!,
    minMs: samples[0]!,
    maxMs: samples[samples.length - 1]!,
    n: samples.length,
    windowMs: window,
  };
}

export interface GridBackend<C> {
  name: string;
  run: (cell: C) => unknown;
}

export interface GridSpec<C> extends CellOptions {
  cells: readonly C[];
  backends: readonly GridBackend<C>[];
  label?: (cell: C) => string;
  /** Called after each measurement; defaults to one line on stderr. `null` silences it. */
  onResult?: ((row: GridRow) => void) | null;
}

export interface GridRow extends CellResult {
  cell: string;
  backend: string;
  /** Position in the run (0-based), so drift over time can be checked afterwards. */
  order: number;
}

/**
 * Every cell on every backend. Backends alternate inside each cell, and their
 * order is reversed on every other cell (A,B then B,A) so neither backend
 * always runs on the warmer machine.
 */
export async function runGrid<C>(spec: GridSpec<C>): Promise<GridRow[]> {
  const label = spec.label ?? ((c: C) => String(c));
  const report = spec.onResult === undefined ? defaultReport : spec.onResult;
  const rows: GridRow[] = [];
  for (let i = 0; i < spec.cells.length; i++) {
    const cell = spec.cells[i]!;
    const order = i % 2 === 0 ? spec.backends : [...spec.backends].reverse();
    for (const backend of order) {
      const res = await timeCell(() => backend.run(cell), spec);
      const row = { cell: label(cell), backend: backend.name, order: rows.length, ...res };
      rows.push(row);
      report?.(row);
    }
  }
  return rows;
}

function defaultReport(r: GridRow): void {
  console.error(
    `${r.backend.padEnd(10)} ${r.cell.padEnd(16)} median ${r.medianMs.toFixed(3)} ms  (min ${r.minMs.toFixed(3)}, max ${r.maxMs.toFixed(3)}, n=${r.n}, window ${r.windowMs.toFixed(0)} ms)`,
  );
}

/** Median table as Markdown: one row per cell, one column per backend. */
export function formatGrid(rows: readonly GridRow[]): string {
  const backends = [...new Set(rows.map((r) => r.backend))];
  const cells = [...new Set(rows.map((r) => r.cell))];
  const at = (c: string, b: string) => rows.find((r) => r.cell === c && r.backend === b);
  const lines = [
    `| cell | ${backends.map((b) => `${b} (ms)`).join(" | ")} |`,
    `|---|${backends.map(() => "---:").join("|")}|`,
    ...cells.map((c) => `| ${c} | ${backends.map((b) => at(c, b)?.medianMs.toFixed(3) ?? "").join(" | ")} |`),
  ];
  return lines.join("\n");
}

/** Best-effort thermal state. Returns `null` where the platform offers nothing without root. */
export function thermalSnapshot(): string | null {
  try {
    if (process.platform === "darwin") {
      // "No thermal warning level has been recorded" = not throttled right now.
      return execFileSync("pmset", ["-g", "therm"], { encoding: "utf8", timeout: 2000 }).trim().replace(/\s*\n\s*/g, "; ");
    }
    if (process.platform === "linux" && existsSync("/sys/class/thermal")) {
      const zones = readdirSync("/sys/class/thermal").filter((z) => z.startsWith("thermal_zone"));
      const temps = zones.map((z) => {
        const read = (f: string) => readFileSync(`/sys/class/thermal/${z}/${f}`, "utf8").trim();
        try {
          return `${read("type")}=${(Number(read("temp")) / 1000).toFixed(1)}C`;
        } catch {
          return null;
        }
      });
      return temps.filter(Boolean).join(", ") || null;
    }
  } catch {
    // fall through
  }
  return null;
}

/** Machine, runtime and thermal state to record next to every published number. */
export function machineInfo(): Record<string, unknown> {
  const bun = (globalThis as { Bun?: { version: string } }).Bun;
  let model: string | null = null;
  try {
    if (process.platform === "darwin") model = execFileSync("sysctl", ["-n", "hw.model"], { encoding: "utf8" }).trim();
  } catch {
    // not fatal
  }
  return {
    date: new Date().toISOString(),
    platform: `${process.platform}-${process.arch}`,
    os: `${os.type()} ${os.release()}`,
    model,
    cpu: os.cpus()[0]?.model ?? null,
    cores: os.cpus().length,
    memGiB: Math.round(os.totalmem() / 2 ** 30),
    runtime: bun ? `bun ${bun.version}` : `node ${process.version}`,
    onBattery: onBattery(),
    thermal: thermalSnapshot(),
    loadavg: os.loadavg().map((x) => Number(x.toFixed(2))),
    cooldownS: envNumber("BENCH_COOL_S") ?? 5,
    windowMs: envNumber("BENCH_WINDOW_MS") ?? 1000,
  };
}

function onBattery(): boolean | null {
  try {
    if (process.platform === "darwin") return /Battery Power/.test(execFileSync("pmset", ["-g", "batt"], { encoding: "utf8", timeout: 2000 }));
  } catch {
    // not fatal
  }
  return null;
}
