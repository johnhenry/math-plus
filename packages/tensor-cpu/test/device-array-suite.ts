/**
 * The ONE behavioural suite of the chainable device-array API
 * (src/device-array.ts), run over every device facade: this package's
 * `createCpuDevice()` (test/device-array.test.ts), tensor-mlx's
 * `createMlxDevice()` and tensor-webgpu's `createWebGpuDevice()` (each
 * package's test/device-array.test.ts calls {@link deviceArraySuite}).
 *
 * - Differential cases: every op against a NumPy oracle
 *   (scripts/device_array_oracle.py), in f32, f16 and bf16 for float ops
 *   (the dtypes the device `supports()`), and in i32/bool where an op is
 *   defined there. Inputs enter through `fromTensor` and results leave
 *   through `toHost`, so every case also crosses the explicit transfer
 *   boundary. f16/bf16 cases run the device in that dtype on inputs rounded
 *   to it and compare against NumPy's f32/f64 answer on the same rounded
 *   inputs with a 2e-2 (f16) / 5e-2 (bf16) tolerance — the split
 *   @johnhenry/tensor-backend's conformance suite uses. bool and i32
 *   results must match exactly; casts are bit-exact.
 * - Behavioural cases: transfers (every dtype, 0-d, offset views, sync
 *   validation errors, refusal of unsupported dtypes), no implicit transfers
 *   or promotion, on-device number constants, lazy-inside/eager-observable
 *   errors, `scope`/`dispose` lifetime, `handle`/`wrap`.
 *
 * Skip-don't-fail: without the device (caller's `skip`) or a python3 that
 * imports numpy, the cases are reported as skipped. A real run must show 0
 * skipped. The suite is typed structurally ({@link SuiteDevice}) and imports
 * nothing at runtime from the package under test, so the three callers can
 * pass devices built from `src/` or from the published `dist/`.
 */
import assert from "node:assert/strict";
import { bf16BitsToF32, f32ToBf16Bits, toF32, type DType, type HostTensor } from "@johnhenry/tensor-backend";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { assertClose, oracleSkip, runNpyOracle, seeded, type NpyJob, type TestFns } from "./helpers.ts";

type Operand = SuiteArray | number;
type Axis = number | undefined;
type KeepDims = { keepDims?: boolean };

/** The DeviceArray surface the suite exercises (structural, so any facade's arrays fit). */
export interface SuiteArray {
  readonly shape: readonly number[];
  readonly dtype: DType;
  readonly ndim: number;
  readonly size: number;
  readonly disposed: boolean;
  readonly handle: unknown;
  toTensor(): Promise<Tensor>;
  toHost(): Promise<HostTensor>;
  eval(): SuiteArray;
  dispose(): void;
  add(o: Operand): SuiteArray;
  sub(o: Operand): SuiteArray;
  mul(o: Operand): SuiteArray;
  div(o: Operand): SuiteArray;
  maximum(o: Operand): SuiteArray;
  minimum(o: Operand): SuiteArray;
  pow(o: Operand): SuiteArray;
  neg(): SuiteArray;
  abs(): SuiteArray;
  exp(): SuiteArray;
  log(): SuiteArray;
  sqrt(): SuiteArray;
  rsqrt(): SuiteArray;
  tanh(): SuiteArray;
  sigmoid(): SuiteArray;
  erf(): SuiteArray;
  relu(): SuiteArray;
  gelu(): SuiteArray;
  equal(o: Operand): SuiteArray;
  notEqual(o: Operand): SuiteArray;
  less(o: Operand): SuiteArray;
  lessEqual(o: Operand): SuiteArray;
  greater(o: Operand): SuiteArray;
  greaterEqual(o: Operand): SuiteArray;
  logicalAnd(o: SuiteArray): SuiteArray;
  logicalOr(o: SuiteArray): SuiteArray;
  logicalNot(): SuiteArray;
  sum(axis?: Axis, opts?: KeepDims): SuiteArray;
  mean(axis?: Axis, opts?: KeepDims): SuiteArray;
  max(axis?: Axis, opts?: KeepDims): SuiteArray;
  min(axis?: Axis, opts?: KeepDims): SuiteArray;
  argmax(axis?: Axis, opts?: KeepDims): SuiteArray;
  argmin(axis?: Axis, opts?: KeepDims): SuiteArray;
  cumsum(axis?: number): SuiteArray;
  softmax(axis?: number): SuiteArray;
  matmul(o: SuiteArray): SuiteArray;
  layerNorm(w?: SuiteArray | null, b?: SuiteArray | null, eps?: number): SuiteArray;
  cast(dtype: DType): SuiteArray;
  reshape(shape: readonly number[]): SuiteArray;
  transpose(axes?: readonly number[]): SuiteArray;
}

/** The ArrayDevice surface the suite exercises. */
export interface SuiteDevice {
  readonly name: string;
  supports(dtype: DType): boolean;
  fromTensor(t: Tensor): Promise<SuiteArray>;
  fromHost(h: HostTensor): Promise<SuiteArray>;
  wrap(h: never): SuiteArray;
  eval(...arrays: never[]): void;
  scope<R>(fn: () => R): R;
  where(cond: SuiteArray, a: SuiteArray, b: SuiteArray): SuiteArray;
  readonly backend: { add(a: never, b: never): unknown };
}

export interface SuiteOptions {
  /** Test-name prefix, e.g. "tensor-mlx". */
  label: string;
  /** Why the device is unavailable here (every case skips), or null. */
  skip: string | null;
  /** The device under test (created by the caller, top-level; null when `skip` is set). */
  device: SuiteDevice | null;
  /** A second device the first must refuse arrays from (another device of the same kind, or a different kind). */
  other: () => SuiteDevice | Promise<SuiteDevice>;
}

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
  run: (xs: SuiteArray[]) => SuiteArray;
  /** f32 tolerance (f16/bf16 use F16/BF16; bool/i32 are exact). */
  tol?: { atol: number; rtol: number };
  /** Input dtypes to run (default f32/f16/bf16), filtered by `supports()`. */
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
  { name: "add [3,4]+[4]", op: "add", shapes: [[3, 4], [4]], run: ([a, b]) => a!.add(b!) },
  { name: "sub [2,1,4]-[3,1]", op: "sub", shapes: [[2, 1, 4], [3, 1]], run: ([a, b]) => a!.sub(b!) },
  { name: "mul [2,3]*[2,3]", op: "mul", shapes: [[2, 3], [2, 3]], run: ([a, b]) => a!.mul(b!) },
  { name: "div [4,5]/[5]", op: "div", shapes: [[4, 5], [5]], range: [0.5, 3], run: ([a, b]) => a!.div(b!) },
  { name: "maximum [3,4],[3,1]", op: "maximum", shapes: [[3, 4], [3, 1]], run: ([a, b]) => a!.maximum(b!) },
  { name: "minimum [3,4],[4]", op: "minimum", shapes: [[3, 4], [4]], run: ([a, b]) => a!.minimum(b!) },
  { name: "add scalar 2.5", op: "add", shapes: [[2, 5]], args: { scalar: 2.5 }, run: ([a]) => a!.add(2.5) },
  { name: "mul scalar -0.75", op: "mul", shapes: [[2, 5]], args: { scalar: -0.75 }, run: ([a]) => a!.mul(-0.75) },
  { name: "minimum scalar 0.25", op: "minimum", shapes: [[6]], args: { scalar: 0.25 }, run: ([a]) => a!.minimum(0.25) },
  { name: "neg", op: "neg", shapes: [[2, 3]], run: ([a]) => a!.neg() },
  { name: "exp", op: "exp", shapes: [[3, 4]], run: ([a]) => a!.exp() },
  { name: "log", op: "log", shapes: [[3, 4]], range: [0.1, 4], run: ([a]) => a!.log() },
  { name: "relu", op: "relu", shapes: [[3, 4]], run: ([a]) => a!.relu() },
  { name: "gelu (exact erf)", op: "gelu", shapes: [[4, 6]], range: [-4, 4], run: ([a]) => a!.gelu() },
  // reductions
  { name: "sum axis 0", op: "sum", shapes: [[5, 3]], args: { axis: 0 }, run: ([a]) => a!.sum(0), tol: ACCUM },
  { name: "sum axis -1 keepDims", op: "sum", shapes: [[2, 3, 4]], args: { axis: -1, keepdims: true }, run: ([a]) => a!.sum(-1, { keepDims: true }), tol: ACCUM },
  { name: "sum all", op: "sum", shapes: [[4, 6]], run: ([a]) => a!.sum(), tol: ACCUM },
  { name: "sum all keepDims", op: "sum", shapes: [[2, 3]], args: { keepdims: true }, run: ([a]) => a!.sum(undefined, { keepDims: true }), tol: ACCUM },
  { name: "mean axis 1", op: "mean", shapes: [[3, 8]], args: { axis: 1 }, run: ([a]) => a!.mean(1), tol: ACCUM },
  { name: "mean all", op: "mean", shapes: [[4, 5]], run: ([a]) => a!.mean(), tol: ACCUM },
  { name: "max axis 0", op: "max", shapes: [[4, 3]], args: { axis: 0 }, run: ([a]) => a!.max(0) },
  { name: "max all", op: "max", shapes: [[3, 7]], run: ([a]) => a!.max() },
  { name: "min axis -1 keepDims", op: "min", shapes: [[3, 5]], args: { axis: -1, keepdims: true }, run: ([a]) => a!.min(-1, { keepDims: true }) },
  { name: "softmax axis -1", op: "softmax", shapes: [[3, 7]], args: { axis: -1 }, range: [-6, 6], run: ([a]) => a!.softmax() },
  { name: "softmax axis 0", op: "softmax", shapes: [[5, 2]], args: { axis: 0 }, range: [-6, 6], run: ([a]) => a!.softmax(0) },
  // linear algebra & NN
  { name: "matmul [4,8]@[8,5]", op: "matmul", shapes: [[4, 8], [8, 5]], run: ([a, b]) => a!.matmul(b!), tol: ACCUM },
  { name: "matmul batched [2,3,4]@[4,5]", op: "matmul", shapes: [[2, 3, 4], [4, 5]], run: ([a, b]) => a!.matmul(b!), tol: ACCUM },
  {
    name: "layerNorm weight+bias",
    op: "layer_norm",
    shapes: [[3, 16], [16], [16]],
    args: { eps: 1e-5, has_weight: true, has_bias: true },
    run: ([x, w, b]) => x!.layerNorm(w!, b!, 1e-5),
    tol: ACCUM,
  },
  { name: "layerNorm no affine", op: "layer_norm", shapes: [[2, 2, 8]], args: { eps: 1e-6 }, run: ([x]) => x!.layerNorm(null, null, 1e-6), tol: ACCUM },
  { name: "transpose [2,3,4] -> [2,0,1]", op: "transpose", shapes: [[2, 3, 4]], args: { axes: [2, 0, 1] }, run: ([a]) => a!.transpose([2, 0, 1]) },
  { name: "transpose default (reversed)", op: "transpose", shapes: [[2, 3, 4]], run: ([a]) => a!.transpose(), dtypes: ANY },
  // ---- general numerics (tensor-backend optional ops, through the compose helpers) ----
  { name: "sqrt", op: "sqrt", shapes: [[3, 5]], range: [0.05, 9], run: ([a]) => a!.sqrt() },
  { name: "rsqrt", op: "rsqrt", shapes: [[3, 5]], range: [0.1, 9], run: ([a]) => a!.rsqrt() },
  { name: "pow [3,4]**[4]", op: "pow", shapes: [[3, 4], [4]], range: [0.25, 2.5], run: ([a, b]) => a!.pow(b!), tol: ACCUM },
  { name: "pow scalar 2 (negative base)", op: "pow", shapes: [[2, 5]], args: { scalar: 2 }, run: ([a]) => a!.pow(2) },
  { name: "pow scalar -0.5", op: "pow", shapes: [[6]], args: { scalar: -0.5 }, range: [0.2, 5], run: ([a]) => a!.pow(-0.5) },
  { name: "abs", op: "abs", shapes: [[3, 4]], run: ([a]) => a!.abs(), dtypes: NUMERIC },
  { name: "neg i32", op: "neg", shapes: [[7]], range: [-50, 50], run: ([a]) => a!.neg(), dtypes: ["i32"] },
  { name: "tanh", op: "tanh", shapes: [[4, 5]], range: [-4, 4], run: ([a]) => a!.tanh() },
  { name: "sigmoid", op: "sigmoid", shapes: [[4, 5]], range: [-8, 8], run: ([a]) => a!.sigmoid() },
  { name: "erf", op: "erf", shapes: [[5, 6]], range: [-3.5, 3.5], run: ([a]) => a!.erf() },
  { name: "equal [3,4],[4] (ties)", op: "equal", shapes: [[3, 4], [4]], integral: true, range: [-2, 2], run: ([a, b]) => a!.equal(b!), dtypes: ANY, out: "bool" },
  { name: "notEqual [3,4],[3,1] (ties)", op: "not_equal", shapes: [[3, 4], [3, 1]], integral: true, range: [-2, 2], run: ([a, b]) => a!.notEqual(b!), dtypes: ANY, out: "bool" },
  { name: "less [2,5],[2,5]", op: "less", shapes: [[2, 5], [2, 5]], integral: true, range: [-3, 3], run: ([a, b]) => a!.less(b!), dtypes: NUMERIC, out: "bool" },
  { name: "lessEqual [2,5],[5]", op: "less_equal", shapes: [[2, 5], [5]], integral: true, range: [-3, 3], run: ([a, b]) => a!.lessEqual(b!), dtypes: NUMERIC, out: "bool" },
  { name: "greater scalar 0.5", op: "greater", shapes: [[3, 4]], args: { scalar: 0.5 }, run: ([a]) => a!.greater(0.5), out: "bool" },
  { name: "greaterEqual [4,3],[3] (ties)", op: "greater_equal", shapes: [[4, 3], [3]], integral: true, range: [-3, 3], run: ([a, b]) => a!.greaterEqual(b!), dtypes: NUMERIC, out: "bool" },
  { name: "equal i32 scalar 1", op: "equal", shapes: [[8]], args: { scalar: 1 }, range: [-2, 3], run: ([a]) => a!.equal(1), dtypes: ["i32"], out: "bool" },
  { name: "logicalAnd [3,4],[4]", op: "logical_and", shapes: [[3, 4], [4]], run: ([a, b]) => a!.logicalAnd(b!), dtypes: ["bool"] },
  { name: "logicalOr [3,4],[3,1]", op: "logical_or", shapes: [[3, 4], [3, 1]], run: ([a, b]) => a!.logicalOr(b!), dtypes: ["bool"] },
  { name: "logicalNot", op: "logical_not", shapes: [[2, 6]], run: ([a]) => a!.logicalNot(), dtypes: ["bool"] },
  { name: "argmax axis 1", op: "argmax", shapes: [[4, 7]], args: { axis: 1 }, run: ([a]) => a!.argmax(1), dtypes: NUMERIC, out: "i32" },
  { name: "argmax all (ties: first)", op: "argmax", shapes: [[3, 5]], integral: true, range: [-3, 3], run: ([a]) => a!.argmax(), dtypes: NUMERIC, out: "i32" },
  { name: "argmin axis 0 keepDims", op: "argmin", shapes: [[5, 3]], args: { axis: 0, keepdims: true }, run: ([a]) => a!.argmin(0, { keepDims: true }), dtypes: NUMERIC, out: "i32" },
  { name: "argmin all keepDims", op: "argmin", shapes: [[2, 3, 4]], args: { keepdims: true }, run: ([a]) => a!.argmin(undefined, { keepDims: true }), out: "i32" },
  { name: "cumsum axis 1", op: "cumsum", shapes: [[3, 6]], args: { axis: 1 }, run: ([a]) => a!.cumsum(1), dtypes: NUMERIC, tol: ACCUM },
  { name: "cumsum axis 0", op: "cumsum", shapes: [[5, 2]], args: { axis: 0 }, run: ([a]) => a!.cumsum(0), tol: ACCUM },
  { name: "cumsum all (flattened)", op: "cumsum", shapes: [[2, 3, 2]], run: ([a]) => a!.cumsum(), dtypes: NUMERIC, tol: ACCUM },
  { name: "sum i32 axis 0", op: "sum", shapes: [[4, 3]], args: { axis: 0 }, range: [-100, 100], run: ([a]) => a!.sum(0), dtypes: ["i32"] },
  { name: "min i32 all", op: "min", shapes: [[3, 4]], range: [-100, 100], run: ([a]) => a!.min(), dtypes: ["i32"] },
  { name: "max i32 axis 1 keepDims", op: "max", shapes: [[3, 4]], args: { axis: 1, keepdims: true }, range: [-100, 100], run: ([a]) => a!.max(1, { keepDims: true }), dtypes: ["i32"] },
  { name: "add i32 scalar -70000 (both halves)", op: "add", shapes: [[5]], args: { scalar: -70000 }, range: [-9, 9], run: ([a]) => a!.add(-70000), dtypes: ["i32"] },
  { name: "mul i32 scalar 2^20+3", op: "mul", shapes: [[4]], args: { scalar: 2 ** 20 + 3 }, range: [-9, 9], run: ([a]) => a!.mul(2 ** 20 + 3), dtypes: ["i32"] },
  { name: "sub i32 [2,3]-[3]", op: "sub", shapes: [[2, 3], [3]], range: [-50, 50], run: ([a, b]) => a!.sub(b!), dtypes: ["i32"] },
  { name: "sub bf16 scalar 1/3", op: "sub", shapes: [[6]], args: { scalar: 1 / 3 }, run: ([a]) => a!.sub(1 / 3), dtypes: ["bf16"] },
  { name: "reshape [2,3,4] -> [4,6] then sum axis 1", op: "sum", shapes: [[4, 6]], args: { axis: 1 }, run: ([a]) => a!.reshape([2, 3, 4]).reshape([4, 6]).sum(1), tol: ACCUM },
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
      case "f16": {
        const half = Float16Array.from(vals);
        device.push(Tensor.fromTypedArray(new Uint16Array(half.buffer), shape, { dtype: "f16" }));
        oracle.push(Tensor.fromTypedArray(Float32Array.from(half), shape, { dtype: "f32" }));
        break;
      }
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

const ORACLE = new URL("../scripts/device_array_oracle.py", import.meta.url).pathname;
const oracle = (jobs: NpyJob[]): Tensor[] => runNpyOracle(ORACLE, jobs);
const vals = async (a: SuiteArray): Promise<number[]> => Array.from((await a.toTensor()).data as ArrayLike<number>, Number);

/**
 * Registers the suite. `device` is created by the caller before this call
 * (top-level await), so the dtypes it `supports()` are known when the cases
 * are registered: unsupported dtypes are not registered as skipped cases —
 * their refusal is asserted instead.
 */
export function deviceArraySuite(fns: TestFns, opts: SuiteOptions): void {
  const { beforeAll, describe, itUnless } = fns;
  const dev = opts.device;
  const supported = (d: RunDType): boolean => (dev ? dev.supports(d) : true);
  const skipAll = opts.skip;
  const skip = opts.skip ?? oracleSkip;
  const d = (): SuiteDevice => dev as SuiteDevice;

  describe(`${opts.label} DeviceArray vs NumPy`, () => {
    const expected = new Map<string, Tensor>();
    const plan: { key: string; c: Case; dtype: RunDType; inputs: Tensor[]; oracleInputs: Tensor[] }[] = [];
    for (const c of CASES) {
      for (const dtype of (c.dtypes ?? FLOATS).filter(supported)) {
        const { device: inputs, oracle: oracleInputs } = inputsFor(c, dtype);
        plan.push({ key: `${c.name} ${dtype}`, c, dtype, inputs, oracleInputs });
      }
    }

    beforeAll(() => {
      if (skip) return;
      const results = oracle(plan.map((p) => ({ op: p.c.op, inputs: p.oracleInputs, args: p.c.args })));
      plan.forEach((p, i) => expected.set(p.key, results[i]!));
    });

    for (const p of plan) {
      itUnless(skip, p.key, async () => {
        const xs = await Promise.all(p.inputs.map((t) => d().fromTensor(t)));
        const out = d().scope(() => p.c.run(xs));
        const want = expected.get(p.key)!;
        const got = await out.toHost();
        const wantDtype = p.c.out ?? p.dtype;
        if (got.dtype !== wantDtype) throw new Error(`${p.key}: dtype ${got.dtype}, want ${wantDtype} (no implicit promotion)`);
        if (JSON.stringify(got.shape) !== JSON.stringify([...want.shape])) throw new Error(`${p.key}: shape [${got.shape}] want [${want.shape}]`);
        const tol = wantDtype === "bool" || wantDtype === "i32" ? EXACT : p.dtype === "f16" ? F16 : p.dtype === "bf16" ? BF16 : (p.c.tol ?? TIGHT);
        assertClose(toF32(got), Float32Array.from(want.contiguous().data as ArrayLike<number>, Number), tol.atol, tol.rtol, p.key);
        out.dispose();
        for (const x of xs) x.dispose();
      });
    }

    // ---- cast: bit-exact against NumPy's astype (RNE for f16/bf16, truncation for i32) ----
    const castVals = [0, 1, -1, 0.1, 1 / 3, 65504, 65520, 1e-8, -2.5, 3.9999, -7.5, 1e5, 6.1e-5];
    const f32 = Tensor.fromTypedArray(Float32Array.from(castVals), [castVals.length], { dtype: "f32" });
    const castCases = (
      [
        { to: "f16", from: f32 },
        { to: "bf16", from: f32 },
        { to: "i32", from: f32 },
        { to: "bool", from: f32 },
        { to: "f32", from: Tensor.fromTypedArray(Int32Array.from([0, 1, -5, 2 ** 24 + 1, -(2 ** 30)]), [5], { dtype: "i32" }) },
        // This suite's own scope is RunDType (the original 5 dtypes) --
        // see its definition above; the 8 dtypes added 2026-09-25 need
        // their own oracle coverage before joining this shared suite,
        // tracked separately from the per-backend work.
      ] as { to: RunDType; from: Tensor }[]
    ).filter((c) => supported(c.to));
    let castWant: Tensor[] = [];
    beforeAll(() => {
      if (skip) return;
      castWant = oracle(castCases.map((c) => ({ op: "cast", inputs: [c.from], args: { dtype: c.to } })));
    });
    castCases.forEach((c, i) => {
      itUnless(skip, `cast ${c.from.dtype} -> ${c.to} (exact)`, async () => {
        const x = await d().fromTensor(c.from);
        const y = x.cast(c.to);
        assert.equal(y.dtype, c.to);
        const got = await y.toTensor(); // f16/bf16 come back as raw bits, like NumPy's .view(uint16)
        assert.deepEqual(Array.from(got.data as ArrayLike<number>), Array.from(castWant[i]!.data as ArrayLike<number>), `cast -> ${c.to}`);
        x.dispose();
        y.dispose();
      });
    });

    itUnless(skip, "where(cond[3,1], a[3,4], b[4])", async () => {
      const cond = Tensor.fromTypedArray(Uint8Array.from([1, 0, 1]), [3, 1], { dtype: "bool" });
      const a = Tensor.fromTypedArray(Float32Array.from(seeded(12, 11)), [3, 4], { dtype: "f32" });
      const b = Tensor.fromTypedArray(Float32Array.from(seeded(4, 12)), [4], { dtype: "f32" });
      const [want] = oracle([{ op: "where", inputs: [cond, a, b] }]);
      const [dc, da, db] = await Promise.all([cond, a, b].map((t) => d().fromTensor(t)));
      const out = d().where(dc!, da!, db!);
      assertClose(toF32(await out.toHost()), want!.data as Float32Array, 0, 0, "where");
      assert.throws(() => d().where(da!, da!, db!), /cond must be bool/);
    });
  });

  describe(`${opts.label} DeviceArray behaviour`, () => {
    itUnless(skipAll, "round-trips every supported dtype through fromTensor/toTensor; refuses the others synchronously", async () => {
      const cases: Tensor[] = [
        Tensor.fromTypedArray(Float32Array.from([1.25, -2, 3e-3]), [3], { dtype: "f32" }),
        Tensor.fromTypedArray(new Uint16Array(Float16Array.from([1.5, -2, 0.1]).buffer), [3], { dtype: "f16" }),
        Tensor.fromTypedArray(Uint16Array.from([0x3fc0, 0xc000, 0x3dcd]), [3], { dtype: "bf16" }),
        Tensor.fromTypedArray(Int32Array.from([7, -2, 2 ** 30]), [3], { dtype: "i32" }),
        Tensor.fromTypedArray(Uint8Array.from([1, 0, 1]), [3], { dtype: "bool" }),
      ];
      for (const t of cases) {
        if (!d().supports(t.dtype as DType)) {
          assert.throws(() => d().fromTensor(t), new RegExp(`does not support ${t.dtype}.*explicitly`), `${t.dtype} is refused, never widened`);
          continue;
        }
        const a = await d().fromTensor(t);
        assert.equal(a.dtype, t.dtype);
        assert.deepEqual([...a.shape], [3]);
        const back = await a.toTensor();
        assert.equal(back.dtype, t.dtype);
        assert.deepEqual(Array.from(back.data as ArrayLike<number>), Array.from(t.data as ArrayLike<number>), t.dtype);
        assert.notEqual(back.data.buffer, t.data.buffer, "the round trip is two real copies, not an alias");
        a.dispose();
      }
      for (const dt of ["f32", "i32", "bool"] as const) assert.equal(d().supports(dt), true, dt);
    });

    itUnless(skipAll, "0-d and offset tensors upload correctly; a downloaded Tensor outlives the array", async () => {
      const s = await d().fromTensor(Tensor.full([], 4, { dtype: "f32" }));
      assert.deepEqual([...s.shape], []);
      assert.deepEqual(await vals(s.mul(2)), [8]);
      const base = Tensor.fromTypedArray(Float32Array.from([0, 1, 2, 3, 4, 5]), [3, 2], { dtype: "f32" });
      const row = await d().fromTensor(base.select(0, 1));
      assert.deepEqual(await vals(row), [2, 3]);
      const t = await row.mul(2).toTensor();
      row.dispose();
      assert.deepEqual([...t.data], [4, 6]);
    });

    itUnless(skipAll, "uploads are async; their validation errors throw synchronously, before any Promise", async () => {
      const p = d().fromTensor(Tensor.from([1, 2, 3]));
      assert.ok(p instanceof Promise, "fromTensor returns a Promise (RFC 0001 §12 Q2)");
      assert.deepEqual(await vals(await p), [1, 2, 3]);
      const hp = d().fromHost({ dtype: "i32", shape: [2], data: Int32Array.from([4, 5]) });
      assert.ok(hp instanceof Promise);
      assert.equal((await hp).dtype, "i32");
      const t = Tensor.fromTypedArray(Float32Array.from([1, 2, 3, 4]), [2, 2], { dtype: "f32" }).transpose();
      assert.throws(() => d().fromTensor(t), /contiguous\(\) first/);
      // f64/i64 support varies by backend (full parity on CPU as of
      // 2026-09-25; MLX/WebGPU have their own real constraints -- see
      // tensor-backend's DType docs) -- assert whichever behavior this
      // device actually has, not a blanket "always refused".
      if (dev?.supports("f64")) {
        assert.deepEqual(await vals(await d().fromTensor(Tensor.from([1], { dtype: "f64" }))), [1]);
      } else {
        assert.throws(() => d().fromTensor(Tensor.from([1], { dtype: "f64" })), /no float64|not support/);
      }
      if (dev?.supports("i64")) {
        assert.deepEqual(await vals(await d().fromTensor(Tensor.from([1], { dtype: "i64" }))), [1]);
      } else {
        assert.throws(() => d().fromTensor(Tensor.from([1], { dtype: "i64" })), /cast\("i32"\)|not support/);
      }
    });

    itUnless(skipAll, "ops refuse tensor-core Tensors and arrays of another device (no implicit transfers)", async () => {
      const a = await d().fromTensor(Tensor.from([1, 2, 3]));
      const host = Tensor.from([1, 2, 3]);
      assert.throws(() => a.add(host as unknown as SuiteArray), /tensor-core Tensor.*fromTensor\(\)/);
      assert.throws(() => a.matmul(host as unknown as SuiteArray), /no implicit transfers/);
      assert.throws(() => a.add(a.handle as SuiteArray), /no implicit transfers/, "a raw backend handle is not an array either");
      const other = await opts.other();
      const b = await other.fromTensor(Tensor.from([1, 2, 3]));
      assert.throws(() => a.add(b), /no implicit transfers/);
      assert.throws(() => d().where(b.greater(0), a, a), /no implicit transfers/);
      b.dispose();
    });

    itUnless(skipAll, "no implicit dtype promotion; float-only ops refuse integers; numbers take the array's dtype", async () => {
      const f = await d().fromTensor(Tensor.from([1, 2, 3]));
      const i = await d().fromTensor(Tensor.from([1, 2, 3], { dtype: "i32" }));
      assert.throws(() => f.add(i), /dtype mismatch f32 vs i32/);
      assert.throws(() => f.matmul(i.reshape([3, 1])), /dtype mismatch/);
      assert.throws(() => i.exp(), /needs a float dtype/);
      assert.throws(() => i.mean(), /needs a float dtype/);
      assert.equal(i.add(2).dtype, "i32", "number operands take the array's dtype");
      assert.equal(i.cast("f32").add(f).dtype, "f32");
      for (const h of ["f16", "bf16"] as const) {
        if (d().supports(h)) assert.equal(f.cast(h).mul(0.5).dtype, h, `${h} stays ${h}`);
        else assert.throws(() => f.cast(h), new RegExp(`does not support ${h}`));
      }
    });

    itUnless(skipAll, "number operands become on-device constants: exact over the i32 range, truncating, range-checked", async () => {
      const x = await d().fromTensor(Tensor.from([1, -2, 3]));
      const i = await d().fromTensor(Tensor.from([5, -7, 0], { dtype: "i32" }));
      assert.deepEqual(await vals(x.mul(0.5).add(1)), [1.5, 0, 2.5]);
      assert.deepEqual(await vals(i.add(-(2 ** 31) + 7)), [-(2 ** 31) + 12, -(2 ** 31), -(2 ** 31) + 7], "exact over the i32 range");
      assert.deepEqual(await vals(i.mul(2.9)), [10, -14, 0], "a fractional operand truncates toward zero, like numpy.asarray(v, int32)");
      assert.throws(() => i.add(2 ** 31), /does not fit in i32/);
      const bl = await d().fromTensor(Tensor.fromTypedArray(Uint8Array.from([1, 0, 1]), [3], { dtype: "bool" }));
      assert.deepEqual(await vals(bl.equal(1)), [1, 0, 1], "bool constants are v != 0");
      assert.throws(() => bl.logicalAnd(1 as unknown as SuiteArray), /operand/);
      const e = await d().fromTensor(Tensor.fromTypedArray(new Float32Array(0), [0, 3], { dtype: "f32" }));
      assert.deepEqual([...e.add(1).shape], [0, 3], "empty arrays work too");
    });

    itUnless(skipAll, "numerics ops keep the dtype rules", async () => {
      const f = await d().fromTensor(Tensor.from([1, 4, 9]));
      const i = await d().fromTensor(Tensor.from([1, -4, 9], { dtype: "i32" }));
      const bl = await d().fromTensor(Tensor.fromTypedArray(Uint8Array.from([1, 0, 1]), [3], { dtype: "bool" }));
      for (const op of ["sqrt", "rsqrt", "tanh", "sigmoid", "erf", "gelu", "log", "softmax"] as const) assert.throws(() => i[op](), /needs a float dtype/, op);
      assert.throws(() => i.pow(2), /needs a float dtype/);
      assert.throws(() => i.layerNorm(), /needs a float dtype/);
      assert.throws(() => f.logicalAnd(f), /needs bool operands/);
      assert.throws(() => bl.logicalAnd(f), /dtype mismatch/);
      assert.throws(() => bl.neg(), /not defined for bool/);
      assert.throws(() => bl.cumsum(), /not defined for bool/);
      assert.throws(() => bl.sum(), /not defined for bool/);
      assert.throws(() => f.less(i), /dtype mismatch f32 vs i32/);
      assert.equal(i.abs().dtype, "i32");
      assert.equal(i.neg().dtype, "i32");
      assert.equal(i.cumsum().dtype, "i32");
      assert.equal(f.less(2).dtype, "bool");
      assert.equal(bl.equal(bl).dtype, "bool");
      assert.equal(f.argmax().dtype, "i32");
      assert.equal(i.argmin(0).dtype, "i32");
      assert.deepEqual(await vals(f.sqrt()), [1, 2, 3]);
      assert.deepEqual(await vals(i.abs()), [1, 4, 9]);
      assert.equal((await f.argmax().toTensor()).item(), 2);
      assert.deepEqual(await vals(bl.logicalNot()), [0, 1, 0]);
      assert.throws(() => f.sum(1), /axis 1 out of range for ndim 1/);
      assert.throws(() => f.cumsum(-2), /out of range/);
    });

    itUnless(skipAll, "lazy inside, eager-observable: shape errors throw at the call site, values appear at eval/toTensor", async () => {
      const a = await d().fromTensor(Tensor.from([1, 2, 3]));
      const b = await d().fromTensor(Tensor.from([1, 2]));
      assert.throws(() => a.add(b));
      assert.throws(() => a.reshape([2, 2]));
      assert.throws(() => a.reshape([3, 1]).matmul(a.reshape([3, 1])));
      const y = a.exp().sum();
      assert.equal(y.eval(), y, "eval() returns this");
      d().eval(y as never, a as never);
      d().eval();
      const got = (await y.toTensor()).item() as number;
      assert.ok(Math.abs(got - (Math.E + Math.exp(2) + Math.exp(3))) < 1e-4);
      assert.equal(a.ndim, 1);
      assert.equal(a.size, 3);
    });

    itUnless(skipAll, "scope frees intermediates, keeps returned arrays (and raw handles), nests, and marks freed arrays", async () => {
      const x = await d().fromTensor(Tensor.from([1, 2]));
      let inner: SuiteArray | undefined;
      let nestedTmp: SuiteArray | undefined;
      const { y, z, raw } = d().scope(() => {
        inner = x.exp();
        const nested = d().scope(() => {
          nestedTmp = x.mul(10);
          return nestedTmp.add(1);
        });
        return { y: inner.add(x), z: nested, raw: d().backend.add(x.handle as never, x.handle as never) };
      });
      assert.equal(inner!.disposed, true);
      assert.equal(nestedTmp!.disposed, true);
      assert.equal(y.disposed, false);
      assert.equal(z.disposed, false, "a nested scope's result survives the outer scope when returned from it");
      assert.throws(() => inner!.exp(), /used after dispose/);
      assert.deepEqual(await vals(z), [11, 21]);
      const kept = d().wrap(raw as never);
      assert.deepEqual(await vals(kept), [2, 4], "a raw backend handle returned from scope is kept");
      y.dispose();
      y.dispose(); // idempotent
      assert.equal(y.disposed, true);
      let leaked: SuiteArray | undefined;
      assert.throws(() => d().scope(() => { leaked = x.exp(); throw new Error("boom"); }), /boom/);
      assert.equal(leaked!.disposed, true, "a throwing scope frees everything it created");
      assert.equal(x.disposed, false, "arrays from outside the scope are untouched");
    });

    itUnless(skipAll, "handle and wrap bridge to device.backend for the ops the arrays do not wrap", async () => {
      const x = await d().fromTensor(Tensor.from([1, 2, 3]));
      const sum = d().wrap(d().backend.add(x.handle as never, x.handle as never) as never);
      assert.deepEqual(await vals(sum.mul(0.5)), [1, 2, 3]);
      x.dispose();
      assert.throws(() => x.handle, /used after dispose/);
    });
  });
}
