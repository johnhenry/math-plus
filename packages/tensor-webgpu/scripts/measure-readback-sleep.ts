/**
 * Measures what backend-webgpu's pre-readback sleep costs on this package's
 * typical readbacks, to pick `SLEEP_THRESHOLD_MS_DEFAULT` (src/bridge.ts).
 * Not part of `npm test`: run it by hand and record the result in
 * docs/spikes/webgpu-runtime.md.
 *
 * Before a readback, backend-webgpu (Dawn) sleeps for ~80% of the wait the
 * same amount of work took last time, when that expected wait exceeds
 * `sleepThresholdMs`, instead of letting Dawn busy-poll `mapAsync`. Three
 * settings of the same runtime, alternated inside each cell:
 *  - `poll`: `sleepWhileWaiting = false` (this package's default in 0.2.0),
 *  - `sleep>15`: sleeping with a 15 ms threshold (the new default),
 *  - `sleep>3`: backend-webgpu's own default threshold, for reference.
 *
 * Method: docs/BENCHMARKING.md via `scripts/bench/thermal.ts`'s `runGrid`
 * (5 s cooldown before every measurement, one warmup, ≤ 1 s windows, the
 * settings alternated and their order reversed every other cell, one
 * process). Each call encodes the op on GPU-resident inputs and awaits
 * `backend.read` of the result: wall time is dispatch + GPU + readback.
 * CPU is the process's user+system time over the same call (median).
 *
 * Run from packages/tensor-webgpu, inside the ~/gpu.lock convention:
 *   node scripts/measure-readback-sleep.ts
 *   BENCH_COOL_S=0 node scripts/measure-readback-sleep.ts   # smoke only
 */
import { createWebGpuDevice } from "../src/index.ts";
import type { WebGpuTensor } from "@johnhenry/backend-webgpu";
import { machineInfo, runGrid } from "../../../scripts/bench/thermal.ts";

const gpu = await createWebGpuDevice();
const b = gpu.backend;
console.error(`adapter: ${gpu.info.vendor} ${gpu.info.architecture} (${gpu.info.source}), subgroup matrices: ${b.hasSubgroupMatrix}`);

function rand(n: number, seed: number): Float32Array {
  let s = seed >>> 0;
  return Float32Array.from({ length: n }, () => ((s = (s * 1664525 + 1013904223) >>> 0) / 0xffffffff) * 2 - 1);
}
const up = (shape: number[], seed: number): Promise<WebGpuTensor> =>
  b.fromHost({ dtype: "f32", shape, data: rand(shape.reduce((a, d) => a * d, 1), seed) });

interface Cell {
  label: string;
  run: () => WebGpuTensor;
}

const [x1024, w1024, x2048, w2048, q, k, v] = await Promise.all([
  up([1024, 1024], 1),
  up([1024, 1024], 2),
  up([2048, 2048], 3),
  up([2048, 2048], 4),
  up([16, 1, 512, 64], 5),
  up([16, 1, 512, 64], 6),
  up([16, 1, 512, 64], 7),
]);
const win = new Uint8Array(512 * 512);
for (let i = 0; i < 512; i++) for (let j = Math.max(0, i - 64); j <= Math.min(511, i + 64); j++) win[i * 512 + j] = 1;
const window64 = await b.fromHost({ dtype: "bool", shape: [512, 512], data: win });

const cells: Cell[] = [
  { label: "linear 1024³", run: () => b.linear(x1024, w1024) },
  { label: "matmul 1024³ (tiled)", run: () => b.matmul(x1024, w1024) },
  { label: "sdpa B16·L512·D64", run: () => b.sdpa(q, k, v, null, 0.125) },
  { label: "sdpa ±64 window", run: () => b.sdpa(q, k, v, window64, 0.125) },
  { label: "linear 2048³", run: () => b.linear(x2048, w2048) },
  { label: "4× linear 2048³", run: () => b.scope(() => { let y = x2048; for (let i = 0; i < 4; i++) y = b.linear(y, w2048); return y; }) },
];

const settings: { name: string; sleep: boolean; threshold: number }[] = [
  { name: "poll", sleep: false, threshold: 15 },
  { name: "sleep>15", sleep: true, threshold: 15 },
  { name: "sleep>3", sleep: true, threshold: 3 },
];

const cpu = new Map<string, number[]>();
const rows = await runGrid({
  cells,
  label: (c) => c.label,
  backends: settings.map((s) => ({
    name: s.name,
    run: async (c: Cell) => {
      b.rt.sleepWhileWaiting = s.sleep;
      b.rt.sleepThresholdMs = s.threshold;
      const c0 = process.cpuUsage();
      const y = c.run();
      await b.read(y);
      const d = process.cpuUsage(c0);
      b.dispose(y);
      const key = `${c.label}|${s.name}`;
      if (!cpu.has(key)) cpu.set(key, []);
      cpu.get(key)!.push((d.user + d.system) / 1000);
    },
  })),
});

const median = (xs: number[]): number => [...xs].sort((a, z) => a - z)[Math.floor(xs.length / 2)]!;
const lines = [
  `| cell | ${settings.map((s) => `${s.name} wall / CPU (ms)`).join(" | ")} |`,
  `|---|${settings.map(() => "---:").join("|")}|`,
];
for (const c of cells) {
  const cols = settings.map((s) => {
    const r = rows.find((row) => row.cell === c.label && row.backend === s.name)!;
    return `${r.medianMs.toFixed(2)} / ${median(cpu.get(`${c.label}|${s.name}`)!).toFixed(2)}`;
  });
  lines.push(`| ${c.label} | ${cols.join(" | ")} |`);
}
console.log(lines.join("\n"));
console.log(JSON.stringify({ machine: machineInfo(), rows }));
gpu.destroy();
