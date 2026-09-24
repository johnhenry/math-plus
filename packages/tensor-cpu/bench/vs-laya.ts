/**
 * Benchmark: this package's CPU Backend vs laya-js's `@johnhenry/backend-cpu@0.2.0`
 * (the implementation it replaces; a devDependency for this comparison only)
 * on encoder-shaped work. Follows docs/BENCHMARKING.md via scripts/bench/thermal.ts:
 * 5 s cooldown before every (cell, backend), ≤1 s timing windows, backends
 * alternated in one process, machine + thermal state recorded.
 *
 *   npm run build && node packages/tensor-cpu/bench/vs-laya.ts     # or: bun packages/tensor-cpu/bench/vs-laya.ts
 *
 * `BENCH_CELLS=sdpa,matmul` runs only the cells whose name contains one of the filters.
 *
 * Each timed call is compute only: inputs are uploaded once per cell, the
 * call runs the op(s) inside `scope` and disposes the result (no `read`).
 */
import { host, type Backend, type HostTensor, type Tensor } from "@johnhenry/tensor-backend";
import { createCpuBackend as createLaya } from "@johnhenry/backend-cpu";
import { formatGrid, machineInfo, runGrid } from "../../../scripts/bench/thermal.ts";
import { createCpuBackend } from "../src/index.ts";

type B = Backend<Tensor>;

function rand(shape: number[], seed: number, lo = -1, hi = 1): HostTensor {
  let s = seed >>> 0 || 1;
  const n = shape.reduce((a, b) => a * b, 1);
  const v = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    v[i] = lo + ((s >>> 8) / 2 ** 24) * (hi - lo);
  }
  return { dtype: "f32", shape, data: v };
}

interface Cell {
  name: string;
  inputs: HostTensor[];
  run: (b: B, xs: Tensor[]) => Tensor;
}

// ModernBERT-base-like encoder layer: hidden 768, 12 heads × 64, GEGLU MLP 1152, batch 4 × 128 tokens
// (4, not 16, sequences so one call stays under the 1 s timing window).
const [NB, L, Dm, NH, Dh, Dff] = [4, 128, 768, 12, 64, 1152];
const encoderLayer = (b: B, [x, wqkv, wo, wi, wout, ln1, ln2, mask]: Tensor[]): Tensor => {
  const h = b.layerNorm(x!, ln1!, null, 1e-5);
  const qkv = b.reshape(b.linear(h, wqkv!), [NB, L, 3, NH, Dh]);
  const [q, k, v] = b.split(b.transpose(qkv, [2, 0, 3, 1, 4]), 3, 0).map((t) => b.reshape(t, [NB, NH, L, Dh]));
  const att = b.sdpa(b.rope(q!, 160000), b.rope(k!, 160000), v!, mask!, 1 / Math.sqrt(Dh));
  const merged = b.reshape(b.transpose(att, [0, 2, 1, 3]), [NB, L, Dm]);
  const x1 = b.add(x!, b.linear(merged, wo!));
  const up = b.linear(b.layerNorm(x1, ln2!, null, 1e-5), wi!);
  const [value, gate] = b.split(up, 2, -1);
  return b.add(x1, b.linear(b.mul(b.gelu(value!), gate!), wout!));
};
const padMask = host("bool", [NB, 1, 1, L], Array.from({ length: NB * L }, (_, i) => ((i % L) < L - (i % 7) * 9 ? 1 : 0)));

const CELLS: Cell[] = [
  {
    name: "linear [16·128,1024]×[1024,1024]ᵀ + bias",
    inputs: [rand([16 * 128, 1024], 1), rand([1024, 1024], 2, -0.05, 0.05), rand([1024], 3)],
    run: (b, [x, w, c]) => b.linear(x!, w!, c!),
  },
  {
    name: "sdpa B=1 H=16 L=128 D=64 (bool key mask)",
    inputs: [rand([1, 16, 128, 64], 4), rand([1, 16, 128, 64], 5), rand([1, 16, 128, 64], 6), host("bool", [1, 1, 1, 128], Array.from({ length: 128 }, (_, i) => (i < 100 ? 1 : 0)))],
    run: (b, [q, k, v, m]) => b.sdpa(q!, k!, v!, m!, 0.125),
  },
  {
    name: "matmul [16,128,64]@[16,64,128]",
    inputs: [rand([16, 128, 64], 7), rand([16, 64, 128], 8)],
    run: (b, [x, y]) => b.matmul(x!, y!),
  },
  {
    name: "encoder layer (ModernBERT-base-like, 4×128 tokens)",
    inputs: [
      rand([NB, L, Dm], 9), rand([3 * Dm, Dm], 10, -0.05, 0.05), rand([Dm, Dm], 11, -0.05, 0.05),
      rand([2 * Dff, Dm], 12, -0.05, 0.05), rand([Dm, Dff], 13, -0.05, 0.05), rand([Dm], 14), rand([Dm], 15), padMask,
    ],
    run: encoderLayer,
  },
];

const filters = (process.env.BENCH_CELLS ?? "").split(",").filter(Boolean);
const selected = filters.length ? CELLS.filter((c) => filters.some((f) => c.name.includes(f))) : CELLS;

const backends: Array<{ name: string; b: B }> = [
  { name: "math-plus-tensor-cpu", b: createCpuBackend() as unknown as B },
  { name: "backend-cpu@0.2.0", b: createLaya() as unknown as B },
];
const uploaded = new Map<string, Tensor[]>();
for (const { name, b } of backends) {
  for (const c of selected) uploaded.set(`${name}|${c.name}`, await Promise.all(c.inputs.map((h) => b.fromHost(h))));
}

const rows = await runGrid({
  cells: selected,
  label: (c) => c.name,
  backends: backends.map(({ name, b }) => ({
    name,
    run: (c: Cell) => {
      const out = b.scope(() => c.run(b, uploaded.get(`${name}|${c.name}`)!));
      b.dispose(out);
    },
  })),
});
console.log(formatGrid(rows));
console.log(JSON.stringify({ machine: machineInfo(), rows }));
