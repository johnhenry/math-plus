/**
 * Differential tests: createCpuBackend() vs a NumPy (float64) oracle
 * (docs/TESTING.md), for what the tensor-backend conformance fixtures do not
 * reach: shapes that straddle the GEMM's 4×4 / 64-column blocks and
 * linearNT's 256-row blocks, grouped-query attention, broadcast and additive
 * masks, two-sided broadcasting, middle-axis reductions, i32/bool result
 * dtypes, negative gather indices, and the numerics ops over wide ranges.
 *
 * Skip-don't-fail: without a python3 that imports numpy every case is
 * reported as skipped; a real run must show 0 skipped.
 */
import { host, toF32, type DType, type HostTensor } from "@johnhenry/tensor-backend";
import { createCpuBackend, type CpuBackend, type CpuTensor } from "../src/index.ts";
import { assertClose, oracleSkip, runOracleBatch, seeded, testFns } from "./helpers.ts";
import { makeTest } from "../../../test/harness.ts";

// Each file imports bun:test itself (see test/harness.ts).
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const harness = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
const { describe, itUnless } = testFns(harness);

const EXACT = { atol: 0, rtol: 0 };
const TIGHT = { atol: 1e-6, rtol: 1e-6 };
const ACCUM = { atol: 1e-5, rtol: 1e-5 };

let seedCounter = 1;
const f32 = (shape: number[], lo = -2, hi = 2): HostTensor => host("f32", shape, seeded(shape.reduce((a, b) => a * b, 1), 7919 * seedCounter++, lo, hi));
const i32 = (shape: number[], lo: number, hi: number): HostTensor =>
  host("i32", shape, seeded(shape.reduce((a, b) => a * b, 1), 104729 * seedCounter++, lo, hi).map(Math.floor));
const bool = (shape: number[], pTrue = 0.5): HostTensor =>
  host("bool", shape, seeded(shape.reduce((a, b) => a * b, 1), 15485863 * seedCounter++, 0, 1).map((v) => (v < pTrue ? 1 : 0)));

interface Case {
  name: string;
  op: string;
  args?: Record<string, unknown>;
  inputs: HostTensor[];
  run: (b: CpuBackend, xs: CpuTensor[]) => CpuTensor;
  tol?: { atol: number; rtol: number };
  /** Result dtype the backend must produce. */
  dtype?: DType;
}

const unary = (fn: string, shape: number[], lo: number, hi: number, tol = TIGHT): Case => ({
  name: `${fn} [${lo}, ${hi})`,
  op: "unary",
  args: { fn },
  inputs: [f32(shape, lo, hi)],
  run: (b, [x]) => (b as unknown as Record<string, (t: CpuTensor) => CpuTensor>)[fn]!(x!),
  tol,
});

const binary = (fn: string, a: HostTensor, c: HostTensor, dtype: DType, tol = TIGHT): Case => ({
  name: `${fn} [${a.shape}] ${a.dtype} , [${c.shape}] ${c.dtype}`,
  op: "binary",
  args: { fn },
  inputs: [a, c],
  run: (b, [x, y]) => (b as unknown as Record<string, (p: CpuTensor, q: CpuTensor) => CpuTensor>)[fn]!(x!, y!),
  tol,
  dtype,
});

const reduce = (fn: string, x: HostTensor, axis: number, keepDims: boolean, dtype: DType, tol = ACCUM): Case => ({
  name: `${fn} [${x.shape}] ${x.dtype} axis ${axis}${keepDims ? " keepDims" : ""}`,
  op: "reduce",
  args: { fn, axis, keepDims },
  inputs: [x],
  run: (b, [t]) => (b as unknown as Record<string, (p: CpuTensor, a: number, k: boolean) => CpuTensor>)[fn]!(t!, axis, keepDims),
  tol,
  dtype,
});

const sdpaCase = (name: string, B: number, H: number, Hk: number, Lq: number, Lk: number, D: number, mask: HostTensor | null): Case => {
  const scale = 1 / Math.sqrt(D);
  return {
    name: `sdpa ${name}`,
    op: "sdpa",
    args: { scale },
    inputs: [f32([B, H, Lq, D]), f32([B, Hk, Lk, D]), f32([B, Hk, Lk, D]), ...(mask ? [mask] : [])],
    run: (b, [q, k, v, m]) => b.sdpa(q!, k!, v!, m ?? null, scale),
    tol: ACCUM,
    dtype: "f32",
  };
};

// A key-padding mask [B,1,1,Lk] where every row keeps at least key 0.
const padMask = (B: number, Lk: number): HostTensor => {
  const m = bool([B, 1, 1, Lk], 0.7);
  for (let b = 0; b < B; b++) (m.data as Uint8Array)[b * Lk] = 1;
  return m;
};
// A causal mask [1,1,L,L] (row i attends to keys 0..i).
const causal = (L: number): HostTensor => host("bool", [1, 1, L, L], Array.from({ length: L * L }, (_, k) => (k % L <= Math.floor(k / L) ? 1 : 0)));

const CASES: Case[] = [
  // ---- GEMM-backed ops, sizes straddling the 4x4 register block, 64-column panel and 256-row block
  { name: "linear [2,37,96]·[70,96]ᵀ + bias", op: "linear", inputs: [f32([2, 37, 96]), f32([70, 96]), f32([70])], run: (b, [x, w, c]) => b.linear(x!, w!, c!), tol: ACCUM },
  { name: "linear [300,33]·[67,33]ᵀ (2 row blocks, no bias)", op: "linear", inputs: [f32([300, 33]), f32([67, 33])], run: (b, [x, w]) => b.linear(x!, w!), tol: ACCUM },
  { name: "linear [3,5] i32 input computes in f32", op: "linear", inputs: [i32([3, 5], -4, 4), f32([6, 5])], run: (b, [x, w]) => b.linear(x!, w!), tol: ACCUM, dtype: "f32" },
  { name: "matmul [2,1,5,7]@[3,7,6] (broadcast batches)", op: "matmul", inputs: [f32([2, 1, 5, 7]), f32([3, 7, 6])], run: (b, [x, y]) => b.matmul(x!, y!), tol: ACCUM },
  { name: "matmul [65,9]@[9,69]", op: "matmul", inputs: [f32([65, 9]), f32([9, 69])], run: (b, [x, y]) => b.matmul(x!, y!), tol: ACCUM },
  // ---- attention
  sdpaCase("GQA H=4 Hk=2, no mask", 2, 4, 2, 9, 9, 8, null),
  sdpaCase("GQA H=6 Hk=1, Lq≠Lk, key-padding mask [B,1,1,Lk]", 2, 6, 1, 5, 11, 16, padMask(2, 11)),
  sdpaCase("causal mask [1,1,L,L]", 1, 3, 3, 13, 13, 8, causal(13)),
  sdpaCase("additive f32 mask [B,H,Lq,Lk]", 1, 2, 2, 7, 7, 4, f32([1, 2, 7, 7], -3, 1)),
  sdpaCase("B=1 H=16 L=128 D=64 (encoder shape)", 1, 16, 16, 128, 128, 64, padMask(1, 128)),
  // ---- position / norm / activations
  { name: "rope base 10000 [1,3,17,16]", op: "rope", args: { base: 10000 }, inputs: [f32([1, 3, 17, 16])], run: (b, [x]) => b.rope(x!, 10000), tol: ACCUM },
  { name: "rope base 160000 [2,2,70,32]", op: "rope", args: { base: 160000 }, inputs: [f32([2, 2, 70, 32])], run: (b, [x]) => b.rope(x!, 160000), tol: ACCUM },
  { name: "layerNorm weight only [4,5,24]", op: "layer_norm", args: { eps: 1e-5, has_weight: true }, inputs: [f32([4, 5, 24], -5, 9), f32([24])], run: (b, [x, w]) => b.layerNorm(x!, w!, null, 1e-5), tol: ACCUM },
  { name: "layerNorm bias only [3,40]", op: "layer_norm", args: { eps: 1e-6, has_bias: true }, inputs: [f32([3, 40]), f32([40])], run: (b, [x, c]) => b.layerNorm(x!, null, c!, 1e-6), tol: ACCUM },
  { name: "softmax middle axis [3,6,5]", op: "softmax", args: { axis: 1 }, inputs: [f32([3, 6, 5], -40, 40)], run: (b, [x]) => b.softmax(x!, 1), tol: TIGHT },
  { name: "softmax i32 input -> f32", op: "softmax", args: { axis: -1 }, inputs: [i32([4, 7], -5, 5)], run: (b, [x]) => b.softmax(x!, -1), tol: TIGHT, dtype: "f32" },
  { name: "geglu [3,5,24]", op: "geglu", inputs: [f32([3, 5, 24], -6, 6)], run: (b, [x]) => b.geglu(x!), tol: TIGHT },
  { name: "meanPool [3,7,5], one fully-masked row", op: "mean_pool", inputs: [f32([3, 7, 5]), host("bool", [3, 7], [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])], run: (b, [x, m]) => b.meanPool(x!, m!), tol: ACCUM },
  unary("gelu", [200], -10, 10),
  unary("erf", [200], -6, 6),
  unary("tanh", [100], -20, 20),
  unary("sigmoid", [100], -90, 90),
  unary("sqrt", [50], 0, 100),
  unary("rsqrt", [50], 0.01, 100),
  unary("exp", [50], -30, 30),
  unary("log", [50], 0.001, 1000),
  // ---- broadcasting, both sides, all result dtypes
  binary("add", f32([4, 1, 3]), f32([1, 5, 1]), "f32"),
  binary("maximum", f32([2, 3, 1]), f32([4]), "f32"),
  binary("div", i32([3, 4], -9, 9), i32([4], 1, 5), "f32"),
  binary("mul", i32([2, 3], -9, 9), i32([3, 1, 1], -3, 3), "i32", EXACT),
  binary("pow", f32([3, 4], 0.1, 3), f32([4], -2, 2), "f32"),
  binary("less", f32([3, 1]), f32([1, 4]), "bool", EXACT),
  binary("greaterEqual", i32([2, 5], -3, 3), i32([5], -3, 3), "bool", EXACT),
  binary("logicalAnd", bool([3, 4]), bool([4]), "bool", EXACT),
  binary("logicalOr", bool([2, 1, 3]), bool([4, 1]), "bool", EXACT),
  {
    name: "where cond[2,1,4] a[3,1] b[4]",
    op: "where",
    inputs: [bool([2, 1, 4]), f32([3, 1]), f32([4])],
    run: (b, [c, x, y]) => b.where(c!, x!, y!),
    tol: EXACT,
    dtype: "f32",
  },
  // ---- reductions / scans on middle axes
  reduce("sum", f32([3, 4, 5]), 1, true, "f32"),
  reduce("sum", i32([3, 4, 5], -100, 100), 2, false, "i32", EXACT),
  reduce("mean", f32([3, 4, 5]), 0, false, "f32"),
  reduce("max", f32([3, 4, 5]), 1, false, "f32", EXACT),
  reduce("min", i32([6, 7], -50, 50), 0, true, "i32", EXACT),
  reduce("argmax", f32([3, 4, 5]), 1, false, "i32", EXACT),
  reduce("argmin", f32([3, 9]), -1, true, "i32", EXACT),
  { name: "cumsum f32 axis 1", op: "cumsum", args: { axis: 1 }, inputs: [f32([3, 50, 2])], run: (b, [x]) => b.cumsum(x!, 1), tol: ACCUM, dtype: "f32" },
  { name: "cumsum bool -> i32", op: "cumsum", args: { axis: -1 }, inputs: [bool([4, 9])], run: (b, [x]) => b.cumsum(x!, -1), tol: EXACT, dtype: "i32" },
  { name: "sort axis 0 [5,3]", op: "sort", args: { axis: 0 }, inputs: [f32([5, 3])], run: (b, [x]) => b.sort(x!, 0), tol: EXACT },
  // ---- layout and gathers
  { name: "transpose [2,3,4,5] perm [0,2,1,3]", op: "transpose", args: { perm: [0, 2, 1, 3] }, inputs: [f32([2, 3, 4, 5])], run: (b, [x]) => b.transpose(x!, [0, 2, 1, 3]), tol: EXACT },
  { name: "transpose [4,6] i32", op: "transpose", args: { perm: [1, 0] }, inputs: [i32([4, 6], -9, 9)], run: (b, [x]) => b.transpose(x!, [1, 0]), tol: EXACT, dtype: "i32" },
  { name: "slice [5,6,7] begin [1,-4,0] end [4,6,-2]", op: "slice", args: { begin: [1, 2, 0], end: [4, 6, 5] }, inputs: [f32([5, 6, 7])], run: (b, [x]) => b.slice(x!, [1, -4, 0], [4, 6, -2]), tol: EXACT },
  { name: "concat axis 1 of [2,3,4],[2,1,4],[2,2,4]", op: "concat", args: { axis: 1 }, inputs: [f32([2, 3, 4]), f32([2, 1, 4]), f32([2, 2, 4])], run: (b, xs) => b.concat(xs, 1), tol: EXACT },
  { name: "embedding [11,6] ids [2,3]", op: "embedding", inputs: [f32([11, 6]), i32([2, 3], 0, 11)], run: (b, [t, i]) => b.embedding(t!, i!), tol: EXACT },
  { name: "gatherRows [2,7,3] idx [2,4] (negative indices)", op: "gather_rows", inputs: [f32([2, 7, 3]), i32([2, 4], -7, 7)], run: (b, [x, i]) => b.gatherRows(x!, i!), tol: EXACT },
];

describe("tensor-cpu vs NumPy", () => {
  let want: ReturnType<typeof runOracleBatch> = [];
  let loaded = false;
  const oracle = () => {
    if (!loaded) {
      want = runOracleBatch(CASES.map((c) => ({ op: c.op, inputs: c.inputs, args: c.args })));
      loaded = true;
    }
    return want;
  };
  CASES.forEach((c, i) => {
    itUnless(oracleSkip, c.name, async () => {
      const expected = oracle()[i]!;
      const b = createCpuBackend();
      const xs = await Promise.all(c.inputs.map((h) => b.fromHost(h)));
      const out = b.scope(() => c.run(b, xs));
      const got = await b.read(out);
      if (JSON.stringify(got.shape) !== JSON.stringify([...expected.shape])) throw new Error(`${c.name}: shape [${got.shape}] want [${expected.shape}]`);
      const wantDtype = c.dtype ?? "f32";
      if (got.dtype !== wantDtype) throw new Error(`${c.name}: dtype ${got.dtype} want ${wantDtype}`);
      const tol = c.tol ?? TIGHT;
      const e = expected.contiguous();
      assertClose(toF32(got), Float32Array.from(e.data as ArrayLike<number>), tol.atol, tol.rtol, c.name);
      b.destroy();
    });
  });
});
