/**
 * Manually-run CPU benchmark for tensor-core's contiguous fast paths and
 * blocked GEMM (issue #120). NOT part of `npm test` — performance is never
 * gated per-PR (see docs/TESTING.md); numbers are recorded in
 * docs/spikes/tensor-core-fast-paths.md.
 *
 *   node packages/tensor-core/scripts/bench-fast-paths.ts            # all cases
 *   BENCH_ONLY=matmul node packages/tensor-core/scripts/bench-fast-paths.ts
 *   BENCH_IDLE_MS=5000 node ...   # idle between cases (fanless machines throttle)
 *
 * Each case: one warmup call, then `reps` timed calls; reports the median.
 */
import { Tensor, random } from "../src/index.ts";

const IDLE_MS = Number(process.env.BENCH_IDLE_MS ?? 3000);
const ONLY = process.env.BENCH_ONLY;

interface Case {
  name: string;
  reps: number;
  /** FLOPs per call, for a GFLOP/s column (matmul only). */
  flops?: number;
  setup: () => () => unknown;
}

const rng = random.seed(120);
const u = (shape: number[]) => random.uniform(shape, { rng, min: -1, max: 1 });

const cases: Case[] = [
  ...[256, 1024].map((n): Case => ({
    name: `matmul f32 ${n}x${n} @ ${n}x${n}`,
    reps: n === 1024 ? 3 : 10,
    flops: 2 * n * n * n,
    setup: () => {
      const a = u([n, n]);
      const b = u([n, n]);
      return () => a.matmul(b);
    },
  })),
  {
    name: "matmul f32 [8,128,64] @ [8,64,128] (batched)",
    reps: 10,
    flops: 2 * 8 * 128 * 64 * 128,
    setup: () => {
      const a = u([8, 128, 64]);
      const b = u([8, 64, 128]);
      return () => a.matmul(b);
    },
  },
  {
    name: "add f32 [16,128,1024] + same shape",
    reps: 10,
    setup: () => {
      const a = u([16, 128, 1024]);
      const b = u([16, 128, 1024]);
      return () => a.add(b);
    },
  },
  {
    name: "add f32 [16,128,1024] + [1024] (bias broadcast)",
    reps: 10,
    setup: () => {
      const a = u([16, 128, 1024]);
      const b = u([1024]);
      return () => a.add(b);
    },
  },
  {
    name: "mul f32 [16,128,1024] * scalar",
    reps: 10,
    setup: () => {
      const a = u([16, 128, 1024]);
      return () => a.mul(0.5);
    },
  },
  {
    name: "exp f32 [16,128,1024]",
    reps: 10,
    setup: () => {
      const a = u([16, 128, 1024]);
      return () => a.exp();
    },
  },
  {
    name: "add f32 [16,128,1024] transposed view (strided, general path)",
    reps: 5,
    setup: () => {
      const a = u([16, 1024, 128]).transpose([0, 2, 1]);
      const b = u([16, 128, 1024]);
      return () => a.add(b);
    },
  },
  {
    name: "sum f32 [16,128,1024] axis=-1",
    reps: 10,
    setup: () => {
      const a = u([16, 128, 1024]);
      return () => a.sum(-1);
    },
  },
  {
    name: "sum f32 [16,128,1024] axis=1",
    reps: 10,
    setup: () => {
      const a = u([16, 128, 1024]);
      return () => a.sum(1);
    },
  },
  {
    name: "max f32 [16,128,1024] axis=-1",
    reps: 10,
    setup: () => {
      const a = u([16, 128, 1024]);
      return () => a.max(-1);
    },
  },
  {
    name: "softmax f32 [16,128,1024] axis=-1",
    reps: 5,
    setup: () => {
      const a = u([16, 128, 1024]);
      return () => a.softmax(-1);
    },
  },
  {
    name: "variance f32 [16,128,1024] axis=-1",
    reps: 5,
    setup: () => {
      const a = u([16, 128, 1024]);
      return () => a.variance(-1);
    },
  },
];

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

for (const c of cases) {
  if (ONLY && !c.name.includes(ONLY)) continue;
  const run = c.setup();
  run(); // warmup (JIT + first-touch allocation)
  const times: number[] = [];
  for (let r = 0; r < c.reps; r++) {
    const t0 = performance.now();
    run();
    times.push(performance.now() - t0);
  }
  times.sort((x, y) => x - y);
  const median = times[Math.floor(times.length / 2)] as number;
  const gflops = c.flops ? `  ${(c.flops / (median * 1e6)).toFixed(2)} GFLOP/s` : "";
  console.log(`${c.name.padEnd(64)} ${median.toFixed(2).padStart(10)} ms${gflops}`);
  await sleep(IDLE_MS);
}
