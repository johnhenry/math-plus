/**
 * Differential tests: MlxArray ops vs a NumPy oracle (docs/TESTING.md), in
 * f32, f16 and bf16 for float ops, and in i32/bool where an op is defined
 * there. Inputs enter through `fromTensor` and results leave through
 * `toHost`, so every case also crosses the explicit transfer boundary.
 *
 * Skip-don't-fail: off darwin/arm64, without libmlxc, or without a python3
 * that imports numpy, every case is reported as skipped. On an Apple Silicon
 * machine with the oracle available a real run must show 0 skipped.
 *
 * f16/bf16 cases run MLX in that dtype on inputs rounded to it and compare
 * against NumPy's f32/f64 answer on the same rounded inputs, with a 2e-2
 * (f16) / 5e-2 (bf16) tolerance — the split @johnhenry/tensor-backend's
 * conformance suite uses. bool and i32 results (comparisons, arg-reductions,
 * integer ops) must match exactly.
 */
import { bf16BitsToF32, f32ToBf16Bits, toF32 } from "@johnhenry/tensor-backend";
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

type RunDType = "f32" | "f16" | "bf16" | "i32" | "bool";

interface Case {
  name: string;
  op: string;
  /** Input shapes; values are seeded per case. */
  shapes: number[][];
  args?: Record<string, unknown>;
  /** Value range for inputs (log needs positives). */
  range?: [number, number];
  /** Round inputs to integers (ties for comparisons / arg-reductions). */
  integral?: boolean;
  /** How to compute it on the device. */
  run: (d: MlxDevice, xs: MlxArray[]) => MlxArray;
  tol?: { atol: number; rtol: number };
  /** Input dtypes to run (default f32/f16/bf16). */
  dtypes?: RunDType[];
  /** Result dtype when it is not the input's (comparisons → bool, arg-reductions → i32). */
  out?: "bool" | "i32";
}

const TIGHT = { atol: 1e-5, rtol: 1e-5 };
const ACCUM = { atol: 1e-4, rtol: 1e-4 };
const F16 = { atol: 2e-2, rtol: 2e-2 };
const BF16 = { atol: 5e-2, rtol: 5e-2 };
const EXACT = { atol: 0, rtol: 0 };
const FLOATS: RunDType[] = ["f32", "f16", "bf16"];
const NUMERIC: RunDType[] = [...FLOATS, "i32"];
const ANY: RunDType[] = [...NUMERIC, "bool"];

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
  // ---- general numerics (tensor-backend 0.2 optional ops, native in backend-mlx) ----
  { name: "sqrt", op: "sqrt", shapes: [[3, 5]], range: [0.05, 9], run: (_, [a]) => a!.sqrt() },
  { name: "rsqrt", op: "rsqrt", shapes: [[3, 5]], range: [0.1, 9], run: (_, [a]) => a!.rsqrt() },
  { name: "pow [3,4]**[4]", op: "pow", shapes: [[3, 4], [4]], range: [0.25, 2.5], run: (_, [a, b]) => a!.pow(b!), tol: ACCUM },
  { name: "pow scalar 2 (negative base)", op: "pow", shapes: [[2, 5]], args: { scalar: 2 }, run: (_, [a]) => a!.pow(2) },
  { name: "pow scalar -0.5", op: "pow", shapes: [[6]], args: { scalar: -0.5 }, range: [0.2, 5], run: (_, [a]) => a!.pow(-0.5) },
  { name: "abs", op: "abs", shapes: [[3, 4]], run: (_, [a]) => a!.abs(), dtypes: NUMERIC },
  { name: "neg i32", op: "neg", shapes: [[7]], range: [-50, 50], run: (_, [a]) => a!.neg(), dtypes: ["i32"] },
  { name: "tanh", op: "tanh", shapes: [[4, 5]], range: [-4, 4], run: (_, [a]) => a!.tanh() },
  { name: "sigmoid", op: "sigmoid", shapes: [[4, 5]], range: [-8, 8], run: (_, [a]) => a!.sigmoid() },
  { name: "erf", op: "erf", shapes: [[5, 6]], range: [-3.5, 3.5], run: (_, [a]) => a!.erf() },
  { name: "equal [3,4],[4] (ties)", op: "equal", shapes: [[3, 4], [4]], integral: true, range: [-2, 2], run: (_, [a, b]) => a!.equal(b!), dtypes: ANY, out: "bool" },
  { name: "notEqual [3,4],[3,1] (ties)", op: "not_equal", shapes: [[3, 4], [3, 1]], integral: true, range: [-2, 2], run: (_, [a, b]) => a!.notEqual(b!), dtypes: ANY, out: "bool" },
  { name: "less [2,5],[2,5]", op: "less", shapes: [[2, 5], [2, 5]], integral: true, range: [-3, 3], run: (_, [a, b]) => a!.less(b!), dtypes: NUMERIC, out: "bool" },
  { name: "lessEqual [2,5],[5]", op: "less_equal", shapes: [[2, 5], [5]], integral: true, range: [-3, 3], run: (_, [a, b]) => a!.lessEqual(b!), dtypes: NUMERIC, out: "bool" },
  { name: "greater scalar 0.5", op: "greater", shapes: [[3, 4]], args: { scalar: 0.5 }, run: (_, [a]) => a!.greater(0.5), out: "bool" },
  { name: "greaterEqual [4,3],[3] (ties)", op: "greater_equal", shapes: [[4, 3], [3]], integral: true, range: [-3, 3], run: (_, [a, b]) => a!.greaterEqual(b!), dtypes: NUMERIC, out: "bool" },
  { name: "equal i32 scalar 1", op: "equal", shapes: [[8]], args: { scalar: 1 }, range: [-2, 3], run: (_, [a]) => a!.equal(1), dtypes: ["i32"], out: "bool" },
  { name: "logicalAnd [3,4],[4]", op: "logical_and", shapes: [[3, 4], [4]], run: (_, [a, b]) => a!.logicalAnd(b!), dtypes: ["bool"] },
  { name: "logicalOr [3,4],[3,1]", op: "logical_or", shapes: [[3, 4], [3, 1]], run: (_, [a, b]) => a!.logicalOr(b!), dtypes: ["bool"] },
  { name: "logicalNot", op: "logical_not", shapes: [[2, 6]], run: (_, [a]) => a!.logicalNot(), dtypes: ["bool"] },
  { name: "argmax axis 1", op: "argmax", shapes: [[4, 7]], args: { axis: 1 }, run: (_, [a]) => a!.argmax(1), dtypes: NUMERIC, out: "i32" },
  { name: "argmax all (ties: first)", op: "argmax", shapes: [[3, 5]], integral: true, range: [-3, 3], run: (_, [a]) => a!.argmax(), dtypes: NUMERIC, out: "i32" },
  { name: "argmin axis 0 keepDims", op: "argmin", shapes: [[5, 3]], args: { axis: 0, keepdims: true }, run: (_, [a]) => a!.argmin(0, { keepDims: true }), dtypes: NUMERIC, out: "i32" },
  { name: "argmin all keepDims", op: "argmin", shapes: [[2, 3, 4]], args: { keepdims: true }, run: (_, [a]) => a!.argmin(undefined, { keepDims: true }), out: "i32" },
  { name: "cumsum axis 1", op: "cumsum", shapes: [[3, 6]], args: { axis: 1 }, run: (_, [a]) => a!.cumsum(1), dtypes: NUMERIC, tol: ACCUM },
  { name: "cumsum axis 0", op: "cumsum", shapes: [[5, 2]], args: { axis: 0 }, run: (_, [a]) => a!.cumsum(0), tol: ACCUM },
  { name: "cumsum all (flattened)", op: "cumsum", shapes: [[2, 3, 2]], run: (_, [a]) => a!.cumsum(), dtypes: NUMERIC, tol: ACCUM },
  { name: "sum i32 axis 0", op: "sum", shapes: [[4, 3]], args: { axis: 0 }, range: [-100, 100], run: (_, [a]) => a!.sum(0), dtypes: ["i32"] },
  { name: "min i32 all", op: "min", shapes: [[3, 4]], range: [-100, 100], run: (_, [a]) => a!.min(), dtypes: ["i32"] },
  { name: "add i32 scalar -70000 (both halves)", op: "add", shapes: [[5]], args: { scalar: -70000 }, range: [-9, 9], run: (_, [a]) => a!.add(-70000), dtypes: ["i32"] },
  { name: "mul i32 scalar 2^20+3", op: "mul", shapes: [[4]], args: { scalar: 2 ** 20 + 3 }, range: [-9, 9], run: (_, [a]) => a!.mul(2 ** 20 + 3), dtypes: ["i32"] },
  { name: "sub bf16 scalar 1/3", op: "sub", shapes: [[6]], args: { scalar: 1 / 3 }, run: (_, [a]) => a!.sub(1 / 3), dtypes: ["bf16"] },
];

function inputsFor(c: Case, dtype: RunDType): { device: Tensor[]; oracle: Tensor[] } {
  const [lo, hi] = c.range ?? [-2, 2];
  const device: Tensor[] = [];
  const oracle: Tensor[] = [];
  c.shapes.forEach((shape, i) => {
    const n = shape.reduce((p, d) => p * d, 1);
    let vals = seeded(n, hash(c.name) + i * 7919, lo, hi);
    if (c.integral || dtype === "i32") vals = vals.map(Math.round);
    switch (dtype) {
      case "f16":
        device.push(f16Tensor(vals, shape));
        oracle.push(f16RoundedF32(vals, shape));
        break;
      case "bf16": {
        const bits = Uint16Array.from(vals, f32ToBf16Bits);
        device.push(Tensor.fromTypedArray(bits, shape, { dtype: "bf16" }));
        oracle.push(Tensor.fromTypedArray(Float32Array.from(bits, bf16BitsToF32), shape, { dtype: "f32" }));
        break;
      }
      case "i32": {
        const t = Tensor.fromTypedArray(Int32Array.from(vals), shape, { dtype: "i32" });
        device.push(t);
        oracle.push(t);
        break;
      }
      case "bool": {
        const t = Tensor.fromTypedArray(Uint8Array.from(vals, (v) => (v > 0 ? 1 : 0)), shape, { dtype: "bool" });
        device.push(t);
        oracle.push(t);
        break;
      }
      default: {
        const t = Tensor.fromTypedArray(Float32Array.from(vals), shape, { dtype: "f32" });
        device.push(t);
        oracle.push(t);
      }
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
  const plan: { key: string; c: Case; dtype: RunDType; inputs: Tensor[]; oracleInputs: Tensor[] }[] = [];
  for (const c of CASES) {
    for (const dtype of c.dtypes ?? FLOATS) {
      const { device: inputs, oracle: oracleInputs } = inputsFor(c, dtype);
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
      const xs = await Promise.all(p.inputs.map((t) => device.fromTensor(t)));
      const out = device.scope(() => p.c.run(device, xs));
      const want = expected.get(p.key)!;
      const got = await out.toHost();
      const wantDtype = p.c.out ?? p.dtype;
      if (got.dtype !== wantDtype) throw new Error(`${p.key}: dtype ${got.dtype}, want ${wantDtype} (no implicit promotion)`);
      if (JSON.stringify(got.shape) !== JSON.stringify([...want.shape])) {
        throw new Error(`${p.key}: shape [${got.shape}] want [${want.shape}]`);
      }
      const tol = wantDtype === "bool" || wantDtype === "i32" ? EXACT : p.dtype === "f16" ? F16 : p.dtype === "bf16" ? BF16 : (p.c.tol ?? TIGHT);
      const w = want.contiguous().data as ArrayLike<number>;
      assertClose(toF32(got), Float32Array.from(w, Number), tol.atol, tol.rtol, p.key);
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
      const x = await device.fromTensor(c.from);
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
    const [dc, da, db] = await Promise.all([cond, a, b].map((t) => device.fromTensor(t)));
    const out = device.where(dc!, da!, db!);
    assertClose(toF32(await out.toHost()), want!.data as Float32Array, 0, 0, "where");
  });
});
