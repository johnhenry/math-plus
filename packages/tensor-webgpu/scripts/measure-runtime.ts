/**
 * Measurements behind docs/spikes/webgpu-runtime.md (issue #126). Manual
 * spike script, not part of `npm test` (GPU performance is never gated per
 * PR — docs/TESTING.md). Runs in-process on Dawn, under Node or Bun.
 *
 *   node scripts/measure-runtime.ts [dispatch|attention|readback|all]
 *   bun  scripts/measure-runtime.ts readback
 *
 * Sections:
 *  - dispatch: host CPU time to encode + submit one dispatch. (a) The old
 *    per-dispatch path (pooled 16-byte uniform buffer + `createBindGroup`
 *    with an auto layout), re-enacted here, vs `dispatchKernel` (uniform
 *    ring + cached bind group) on the same kernel and the same persistent
 *    buffers; (b) op level: `runGemm` / `runQKT` on GPU-resident operands
 *    (fresh output buffer per call, so a bind-group cache miss each time).
 *    Set MEASURE_SRC=<dir> to run (b) against another copy of src/ (the
 *    "before" numbers came from the base branch's src/).
 *  - attention: fused `runAttention` at B=16, L=512, D=64 with no mask,
 *    sliding-window, key-padding and causal masks, masked-key-tile skipping
 *    on vs off; the three-primitive chain for context. GPU time from the
 *    timestamp profiler (median of RUNS forwards).
 *  - readback: process CPU time and wall time per (1 or 4 x 2048^3 GEMM +
 *    a 16-byte readback) with sleep-while-waiting on vs off (threshold
 *    lowered to 3 ms so both sizes sleep when on).
 *
 * The reference machine (Apple M2 MacBook Air) is fanless and throttles
 * under sustained load: every cell is preceded by COOLDOWN_MS of idle.
 * Follow AGENTS.md's ~/gpu.lock convention around it.
 */
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_DIR = process.env.MEASURE_SRC ? path.resolve(process.env.MEASURE_SRC) : path.resolve(HERE, "../src");
const COOLDOWN_MS = Number(process.env.COOLDOWN_MS ?? 4000);
const RUNS = Number(process.env.RUNS ?? 15);
const which = process.argv[2] ?? "all";
const runtime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined" ? "bun" : "node";

const median = (xs: number[]): number => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] as number;
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));
const cool = (): Promise<void> => sleep(COOLDOWN_MS);
const fmt = (x: number, d = 2): string => x.toFixed(d);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Mod = any;
const dawn: Mod = await import(path.join(HERE, "../src/dawn.ts"));
const device_: Mod = await import(path.join(SRC_DIR, "device.ts"));
const gemm: Mod = await import(path.join(SRC_DIR, "gemm.ts"));
const attn: Mod = await import(path.join(SRC_DIR, "attention.ts"));
const rt: Mod = await import(path.join(SRC_DIR, "gpu-runtime.ts"));

const gpu = await dawn.requestDawnGPU({ unsafe: true });
if (!gpu) throw new Error("Dawn unavailable");
const cap = await device_.detectWebGPU({ gpu, timestampQuery: true });
if (!cap.available) throw new Error(cap.reason);
const device: GPUDevice = cap.device;
const { GPUTensor } = device_;

function lcg(size: number, seed: number): Float32Array {
  let s = seed >>> 0;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

console.log(`# runtime=${runtime} src=${path.relative(process.cwd(), SRC_DIR) || "."} adapter=${cap.adapter.info?.description ?? "?"}`);

// ---- dispatch -----------------------------------------------------------------

async function measureDispatch(): Promise<void> {
  const N = 2000;
  const M = 64;
  const A = GPUTensor.fromFloat32Array(device, lcg(M * M, 1), [M, M]);
  const B = GPUTensor.fromFloat32Array(device, lcg(M * M, 2), [M, M]);

  // (a) mechanism A/B on persistent buffers (new src only).
  if (rt.dispatchKernel) {
    const plan = gemm.planGemm("tiled", "f32", M, M, M, false);
    const out = device.createBuffer({ size: M * M * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const legacyPipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: plan.code }), entryPoint: "main" } });
    const legacy = (): void => {
      const dims = rt.acquireBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(dims.buffer, 0, new Uint32Array([M, M, M, 0]));
      const bg = device.createBindGroup({
        layout: legacyPipeline.getBindGroupLayout(0),
        entries: [A.buffer, B.buffer, out, dims.buffer].map((buffer: GPUBuffer, i: number) => ({ binding: i, resource: { buffer } })),
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(legacyPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(plan.groups[0], plan.groups[1]);
      pass.end();
      device.queue.submit([enc.finish()]);
      rt.releaseBuffer(device, dims);
    };
    const current = (): void => {
      rt.dispatchKernel(device, plan.code, [A.buffer, B.buffer, out], [plan.groups[0], plan.groups[1]], { uniform: new Uint32Array([M, M, M, 0]) });
    };
    // Interleaved (legacy, current, legacy, ...) so drift and heat hit both alike.
    const cells: [string, () => void, number[]][] = [
      ["legacy (uniform buffer + createBindGroup)", legacy, []],
      ["dispatchKernel (ring + cached bind group)", current, []],
    ];
    for (let rep = 0; rep < 6; rep++) {
      for (const [, f, samples] of cells) {
        await cool();
        for (let i = 0; i < 50; i++) f();
        await device.queue.onSubmittedWorkDone();
        const t0 = performance.now();
        for (let i = 0; i < N; i++) f();
        samples.push(((performance.now() - t0) * 1000) / N);
        await device.queue.onSubmittedWorkDone();
      }
    }
    for (const [name, , samples] of cells) {
      console.log(`dispatch  ${name.padEnd(44)} ${fmt(median(samples))} µs/dispatch (median of 6 x ${N}; min ${fmt(Math.min(...samples))})`);
    }
    // Encode-only cost (what batching would pay per dispatch): same kernel, one pass, no submit per dispatch.
    const bg = device.createBindGroup({
      layout: legacyPipeline.getBindGroupLayout(0),
      entries: [A.buffer, B.buffer, out, rt.acquireBuffer(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST).buffer].map(
        (buffer: GPUBuffer, i: number) => ({ binding: i, resource: { buffer } }),
      ),
    });
    const batched: number[] = [];
    for (let rep = 0; rep < 3; rep++) {
      await cool();
      const t0 = performance.now();
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let i = 0; i < N; i++) {
        pass.setPipeline(legacyPipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(plan.groups[0], plan.groups[1]);
      }
      pass.end();
      device.queue.submit([enc.finish()]);
      batched.push(((performance.now() - t0) * 1000) / N);
      await device.queue.onSubmittedWorkDone();
    }
    console.log(`dispatch  ${"reference: N dispatches in ONE pass + submit".padEnd(44)} ${fmt(median(batched))} µs/dispatch`);
    out.destroy();
  }

  // (b) op level, GPU-resident operands, fresh output per call.
  const q = GPUTensor.fromFloat32Array(device, lcg(4 * 64 * 32, 3), [4, 64, 32]);
  const ops: [string, () => Promise<{ free(): void }>][] = [
    ["runGemm 64x64x64 (resident)", () => gemm.runGemm(device, A, B)],
    ["runQKT 4x64x64x32 (resident)", () => attn.runQKT(device, q, q, 4, 64, 64, 32)],
  ];
  for (const [name, op] of ops) {
    const samples: number[] = [];
    for (let rep = 0; rep < 5; rep++) {
      await cool();
      const outs: { free(): void }[] = [];
      for (let i = 0; i < 50; i++) outs.push(await op());
      await device.queue.onSubmittedWorkDone();
      const t0 = performance.now();
      for (let i = 0; i < N; i++) outs.push(await op());
      samples.push(((performance.now() - t0) * 1000) / N);
      await device.queue.onSubmittedWorkDone();
      for (const o of outs) o.free();
    }
    console.log(`op        ${name.padEnd(44)} ${fmt(median(samples))} µs/call host time (median of 5 x ${N})`);
  }
  for (const t of [A, B, q]) t.free();
}

// ---- attention ------------------------------------------------------------------

async function measureAttention(): Promise<void> {
  const Bt = 16;
  const L = 512;
  const D = 64;
  const q = GPUTensor.fromFloat32Array(device, lcg(Bt * L * D, 11), [Bt, L, D]);
  const k = GPUTensor.fromFloat32Array(device, lcg(Bt * L * D, 12), [Bt, L, D]);
  const v = GPUTensor.fromFloat32Array(device, lcg(Bt * L * D, 13), [Bt, L, D]);
  const window = new Float32Array(L * L);
  for (let i = 0; i < L; i++) for (let j = Math.max(0, i - 64); j <= Math.min(L - 1, i + 64); j++) window[i * L + j] = 1;
  const pad = new Float32Array(Bt * L);
  const lens: number[] = [];
  for (let b = 0; b < Bt; b++) {
    const len = Math.round(L * (0.25 + (0.75 * b) / (Bt - 1)));
    lens.push(len);
    pad.fill(1, b * L, b * L + len);
  }
  const causal = new Float32Array(L * L);
  for (let i = 0; i < L; i++) causal.fill(1, i * L, i * L + i + 1);
  const masks: [string, Float32Array | null, number[]][] = [
    ["none", null, []],
    ["sliding window ±64", window, [L, L]],
    [`key padding (lens ${lens[0]}..${lens.at(-1)})`, pad, [Bt, 1, L]],
    ["causal", causal, [L, L]],
  ];
  const timeIt = async (f: () => Promise<{ free(): void }[]>): Promise<number> => {
    // Idle first (thermal), then ~150 ms of the same work so the GPU clock
    // has ramped up before anything is timed.
    await cool();
    const w0 = performance.now();
    while (performance.now() - w0 < 150) {
      for (const o of await f()) o.free();
      await device.queue.onSubmittedWorkDone();
    }
    const ms: number[] = [];
    for (let r = 0; r < RUNS; r++) {
      rt.startProfiling(device);
      const outs = await f();
      const timings = (await rt.stopProfiling(device)) as { ms: number }[];
      ms.push(timings.reduce((s, t) => s + t.ms, 0));
      for (const o of outs) o.free();
    }
    return median(ms);
  };
  console.log(`attention B=${Bt} L=${L} D=${D} (GPU ms per forward, median of ${RUNS}, timestamp queries)`);
  for (const [name, data, shape] of masks) {
    const mask = data ? GPUTensor.fromFloat32Array(device, data, shape) : undefined;
    const cells: string[] = [];
    for (const skip of mask ? [false, true] : [true]) {
      const t = await timeIt(async () => [await attn.runAttention(device, q, k, v, { mask, skipMaskedTiles: skip })]);
      cells.push(`${mask ? (skip ? "skip" : "no-skip") : "fused"} ${fmt(t)}`);
    }
    console.log(`  ${name.padEnd(36)} ${cells.join("   ")}`);
    mask?.free();
  }
  const chain = await timeIt(async () => {
    const s = await attn.runQKT(device, q, k, Bt, L, L, D);
    const w = await attn.runSoftmax(device, s, Bt * L, L);
    return [s, w, await attn.runWeightedSum(device, w, v, Bt, L, L, D)];
  });
  console.log(`  ${"none, 3-primitive chain (unscaled)".padEnd(36)} ${fmt(chain)}`);
  for (const t of [q, k, v]) t.free();
}

// ---- readback -------------------------------------------------------------------

async function measureReadback(): Promise<void> {
  const N = 2048;
  const ITER = 30;
  const A = GPUTensor.fromFloat32Array(device, lcg(N * N, 5), [N, N]);
  for (const gemms of [1, 4]) {
    console.log(`readback: ${ITER} x (${gemms} x runGemm ${N}^3 + one 16-byte readback), ${runtime}`);
    const step = async (): Promise<void> => {
      const outs = [];
      for (let g = 0; g < gemms; g++) outs.push(await gemm.runGemm(device, A, A));
      await rt.readBackBytes(device, (outs.at(-1) as { buffer: GPUBuffer }).buffer, 16);
      for (const o of outs) o.free();
    };
    for (const on of [false, true, false, true]) {
      // Threshold lowered from the 15 ms default so the ~9 ms case sleeps too (that is how the default was chosen).
      rt.configureGPURuntime(device, { sleepWhileWaiting: on, sleepThresholdMs: 3 });
      await cool();
      for (let i = 0; i < 5; i++) await step(); // warm the wait estimate
      const c0 = process.cpuUsage();
      const t0 = performance.now();
      const lat: number[] = [];
      for (let i = 0; i < ITER; i++) {
        const t1 = performance.now();
        await step();
        lat.push(performance.now() - t1);
      }
      const wall = performance.now() - t0;
      const cpu = process.cpuUsage(c0);
      const cpuMs = (cpu.user + cpu.system) / 1000;
      console.log(
        `  sleepWhileWaiting=${String(on).padEnd(5)} median latency ${fmt(median(lat))} ms   process CPU ${fmt(cpuMs / ITER)} ms/iter   CPU/wall ${fmt((100 * cpuMs) / wall, 0)}%`,
      );
    }
  }
  A.free();
}

if (which === "dispatch" || which === "all") await measureDispatch();
if (which === "attention" || which === "all") await measureAttention();
if (which === "readback" || which === "all") await measureReadback();
device.destroy();
process.exit(0);
