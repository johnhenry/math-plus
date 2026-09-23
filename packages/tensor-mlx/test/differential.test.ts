/**
 * Differential tests: MlxArray ops vs a NumPy oracle (docs/TESTING.md), in
 * f32 and — for float ops — f16. Inputs enter through `fromTensor` and
 * results leave through `toHost`, so every case also crosses the explicit
 * transfer boundary.
 *
 * Skip-don't-fail: off darwin/arm64, without libmlxc, or without a python3
 * that imports numpy, every case is reported as skipped. On an Apple Silicon
 * machine with the oracle available a real run must show 0 skipped.
 *
 * f16 cases run MLX in f16 on f16-rounded inputs and compare against NumPy's
 * f32/f64 answer on the same rounded inputs, with a 2e-2 tolerance (the
 * split @johnhenry/tensor-backend's conformance suite uses).
 */
import { toF32 } from "@johnhenry/tensor-backend";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { createMlxDevice, type DeviceDType, type MlxArray, type MlxDevice } from "../src/index.ts";
import {
  assertClose,
  f16RoundedF32,
  f16Tensor,
  mlxSkip,
  oracleSkip,
  runOracleBatch,
  seeded,
  testFns,
  type OracleCase,
} from "./helpers.ts";
import { makeTest } from "../../../test/harness.ts";

// Each file imports bun:test itself (see test/harness.ts).
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const harness = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
const { beforeAll, describe, itUnless } = testFns(harness);

type Float = "f32" | "f16";

interface Case {
  name: string;
  op: string;
  /** Input shapes; values are seeded per case. */
  shapes: number[][];
  args?: Record<string, unknown>;
  /** Value range for inputs (log needs positives). */
  range?: [number, number];
  /** How to compute it on the device. */
  run: (d: MlxDevice, xs: MlxArray[]) => MlxArray;
  tol?: { atol: number; rtol: number };
  dtypes?: Float[];
}

const TIGHT = { atol: 1e-5, rtol: 1e-5 };
const ACCUM = { atol: 1e-4, rtol: 1e-4 };
const F16 = { atol: 2e-2, rtol: 2e-2 };

const CASES: Case[] = [
  // elementwise + broadcasting
  { name: "add [3,4]+[4]", op: "add", shapes: [[3, 4], [4]], run: (_, [a, b]) => a!.add(b!) },
  { name: "sub [2,1,4]-[3,1]", op: "sub", shapes: [[2, 1, 4], [3, 1]], run: (_, [a, b]) => a!.sub(b!) },
  { name: "mul [2,3]*[2,3]", op: "mul", shapes: [[2, 3], [2, 3]], run: (_, [a, b]) => a!.mul(b!) },
  { name: "div [4,5]/[5]", op: "div", shapes: [[4, 5], [5]], range: [0.5, 3], run: (_, [a, b]) => a!.div(b!) },
  { name: "maximum [3,4],[3,1]", op: "maximum", shapes: [[3, 4], [3, 1]], run: (_, [a, b]) => a!.maximum(b!) },
  { name: "minimum [3,4],[4]", op: "minimum", shapes: [[3, 4], [4]], run: (_, [a, b]) => a!.minimum(b!) },
  { name: "add scalar 2.5", op: "add", shapes: [[2, 5]], args: { scalar: 2.5 }, run: (_, [a]) => a!.add(2.5) },
  { name: "mul scalar -0.75", op: "mul", shapes: [[2, 5]], args: { scalar: -0.75 }, run: (_, [a]) => a!.mul(-0.75) },
  { name: "minimum scalar 0.25", op: "minimum", shapes: [[6]], args: { scalar: 0.25 }, run: (_, [a]) => a!.minimum(0.25) },
  { name: "neg", op: "neg", shapes: [[2, 3]], run: (_, [a]) => a!.neg() },
  { name: "exp", op: "exp", shapes: [[3, 4]], run: (_, [a]) => a!.exp() },
  { name: "log", op: "log", shapes: [[3, 4]], range: [0.1, 4], run: (_, [a]) => a!.log() },
  { name: "relu", op: "relu", shapes: [[3, 4]], run: (_, [a]) => a!.relu() },
  { name: "gelu (exact erf)", op: "gelu", shapes: [[4, 6]], range: [-4, 4], run: (_, [a]) => a!.gelu() },
  // reductions
  { name: "sum axis 0", op: "sum", shapes: [[5, 3]], args: { axis: 0 }, run: (_, [a]) => a!.sum(0), tol: ACCUM },
  { name: "sum axis -1 keepDims", op: "sum", shapes: [[2, 3, 4]], args: { axis: -1, keepdims: true }, run: (_, [a]) => a!.sum(-1, { keepDims: true }), tol: ACCUM },
  { name: "sum all", op: "sum", shapes: [[4, 6]], run: (_, [a]) => a!.sum(), tol: ACCUM },
  { name: "sum all keepDims", op: "sum", shapes: [[2, 3]], args: { keepdims: true }, run: (_, [a]) => a!.sum(undefined, { keepDims: true }), tol: ACCUM },
  { name: "mean axis 1", op: "mean", shapes: [[3, 8]], args: { axis: 1 }, run: (_, [a]) => a!.mean(1), tol: ACCUM },
  { name: "mean all", op: "mean", shapes: [[4, 5]], run: (_, [a]) => a!.mean(), tol: ACCUM },
  { name: "max axis 0", op: "max", shapes: [[4, 3]], args: { axis: 0 }, run: (_, [a]) => a!.max(0) },
  { name: "max all", op: "max", shapes: [[3, 7]], run: (_, [a]) => a!.max() },
  { name: "min axis -1 keepDims", op: "min", shapes: [[3, 5]], args: { axis: -1, keepdims: true }, run: (_, [a]) => a!.min(-1, { keepDims: true }) },
  { name: "softmax axis -1", op: "softmax", shapes: [[3, 7]], args: { axis: -1 }, range: [-6, 6], run: (_, [a]) => a!.softmax() },
  { name: "softmax axis 0", op: "softmax", shapes: [[5, 2]], args: { axis: 0 }, range: [-6, 6], run: (_, [a]) => a!.softmax(0) },
  // linear algebra & NN
  { name: "matmul [4,8]@[8,5]", op: "matmul", shapes: [[4, 8], [8, 5]], run: (_, [a, b]) => a!.matmul(b!), tol: ACCUM },
  { name: "matmul batched [2,3,4]@[4,5]", op: "matmul", shapes: [[2, 3, 4], [4, 5]], run: (_, [a, b]) => a!.matmul(b!), tol: ACCUM },
  {
    name: "layerNorm weight+bias",
    op: "layer_norm",
    shapes: [[3, 16], [16], [16]],
    args: { eps: 1e-5, has_weight: true, has_bias: true },
    run: (_, [x, w, b]) => x!.layerNorm(w!, b!, 1e-5),
    tol: ACCUM,
  },
  { name: "layerNorm no affine", op: "layer_norm", shapes: [[2, 2, 8]], args: { eps: 1e-6 }, run: (_, [x]) => x!.layerNorm(null, null, 1e-6), tol: ACCUM },
  { name: "transpose [2,3,4] -> [2,0,1]", op: "transpose", shapes: [[2, 3, 4]], args: { axes: [2, 0, 1] }, run: (_, [a]) => a!.transpose([2, 0, 1]) },
];

function floatInputs(c: Case, dtype: Float): { device: Tensor[]; oracle: Tensor[] } {
  const [lo, hi] = c.range ?? [-2, 2];
  const device: Tensor[] = [];
  const oracle: Tensor[] = [];
  c.shapes.forEach((shape, i) => {
    const n = shape.reduce((p, d) => p * d, 1);
    const vals = seeded(n, hash(c.name) + i * 7919, lo, hi);
    if (dtype === "f16") {
      device.push(f16Tensor(vals, shape));
      oracle.push(f16RoundedF32(vals, shape));
    } else {
      const t = Tensor.fromTypedArray(Float32Array.from(vals), shape, { dtype: "f32" });
      device.push(t);
      oracle.push(t);
    }
  });
  return { device, oracle };
}

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

const skip = mlxSkip ?? oracleSkip;

describe("tensor-mlx vs NumPy", () => {
  let device: MlxDevice;
  const expected = new Map<string, Tensor>();
  const plan: { key: string; c: Case; dtype: Float; inputs: Tensor[]; oracleInputs: Tensor[] }[] = [];
  for (const c of CASES) {
    for (const dtype of c.dtypes ?? (["f32", "f16"] as const)) {
      const { device: inputs, oracle: oracleInputs } = floatInputs(c, dtype);
      plan.push({ key: `${c.name} ${dtype}`, c, dtype, inputs, oracleInputs });
    }
  }

  beforeAll(() => {
    if (skip) return;
    device = createMlxDevice();
    const jobs: OracleCase[] = plan.map((p) => ({ op: p.c.op, inputs: p.oracleInputs, args: p.c.args }));
    const results = runOracleBatch(jobs);
    plan.forEach((p, i) => expected.set(p.key, results[i]!));
  });

  for (const p of plan) {
    itUnless(skip, p.key, async () => {
      const xs = p.inputs.map((t) => device.fromTensor(t));
      const out = device.scope(() => p.c.run(device, xs));
      const want = expected.get(p.key)!;
      const got = await out.toHost();
      if (got.dtype !== p.dtype) throw new Error(`${p.key}: dtype ${got.dtype}, want ${p.dtype} (no implicit promotion)`);
      if (JSON.stringify(got.shape) !== JSON.stringify([...want.shape])) {
        throw new Error(`${p.key}: shape [${got.shape}] want [${want.shape}]`);
      }
      const tol = p.dtype === "f16" ? F16 : (p.c.tol ?? TIGHT);
      assertClose(toF32(got), want.contiguous().data as Float32Array, tol.atol, tol.rtol, p.key);
      out.dispose();
      for (const x of xs) x.dispose();
    });
  }

  // ---- cast: bit-exact against NumPy's astype (RNE for f16/bf16, truncation for i32) ----
  const castVals = [0, 1, -1, 0.1, 1 / 3, 65504, 65520, 1e-8, -2.5, 3.9999, -7.5, 1e5, 6.1e-5];
  const castCases: { to: DeviceDType; from: Tensor }[] = [
    { to: "f16", from: Tensor.fromTypedArray(Float32Array.from(castVals), [castVals.length], { dtype: "f32" }) },
    { to: "bf16", from: Tensor.fromTypedArray(Float32Array.from(castVals), [castVals.length], { dtype: "f32" }) },
    { to: "i32", from: Tensor.fromTypedArray(Float32Array.from(castVals), [castVals.length], { dtype: "f32" }) },
    { to: "bool", from: Tensor.fromTypedArray(Float32Array.from(castVals), [castVals.length], { dtype: "f32" }) },
    { to: "f32", from: Tensor.fromTypedArray(Int32Array.from([0, 1, -5, 2 ** 24 + 1, -(2 ** 30)]), [5], { dtype: "i32" }) },
  ];
  let castWant: Tensor[] = [];
  beforeAll(() => {
    if (skip) return;
    castWant = runOracleBatch(castCases.map((c) => ({ op: "cast", inputs: [c.from], args: { dtype: c.to } })));
  });
  castCases.forEach((c, i) => {
    itUnless(skip, `cast ${c.from.dtype} -> ${c.to} (exact)`, async () => {
      const x = device.fromTensor(c.from);
      const y = x.cast(c.to);
      const got = await y.toTensor(); // f16/bf16 come back as raw bits, like NumPy's .view(uint16)
      const want = castWant[i]!;
      const g = Array.from(got.data as ArrayLike<number>);
      const w = Array.from(want.data as ArrayLike<number>);
      if (JSON.stringify(g) !== JSON.stringify(w)) throw new Error(`cast -> ${c.to}: got ${g} want ${w}`);
      x.dispose();
      y.dispose();
    });
  });

  // ---- where (bool condition, broadcasting) ----
  itUnless(skip, "where(cond[3,1], a[3,4], b[4])", async () => {
    const cond = Tensor.fromTypedArray(Uint8Array.from([1, 0, 1]), [3, 1], { dtype: "bool" });
    const a = Tensor.fromTypedArray(Float32Array.from(seeded(12, 11)), [3, 4], { dtype: "f32" });
    const b = Tensor.fromTypedArray(Float32Array.from(seeded(4, 12)), [4], { dtype: "f32" });
    const [want] = runOracleBatch([{ op: "where", inputs: [cond, a, b] }]);
    const out = device.where(device.fromTensor(cond), device.fromTensor(a), device.fromTensor(b));
    assertClose(toF32(await out.toHost()), want!.data as Float32Array, 0, 0, "where");
  });
});
