/**
 * Differential tests: tensor-core vs a NumPy subprocess oracle
 * (docs/PLAN.md §6.1). Exchange format is .npy in BOTH directions, so every
 * run also exercises tensor-core's .npy reader and writer against NumPy's.
 *
 * Python resolution: $MATH_PLUS_ORACLE_PYTHON, else `python3` on PATH. If no
 * interpreter with numpy is found the suite SKIPS (it does not fail) — CI
 * environments that guarantee the oracle should also `grep -c "skipped 0"`.
 * On trycooy: nix-shell -p "python3.withPackages(ps: [ps.numpy])" provides
 * one; pip wheels don't work on NixOS.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor, isBigIntDType, type DType } from "../src/index.ts";

const ORACLE_SCRIPT = new URL("../scripts/numpy_oracle.py", import.meta.url)
  .pathname;

function findOraclePython(): string | undefined {
  const candidates = [
    process.env.MATH_PLUS_ORACLE_PYTHON,
    "python3",
  ].filter((c): c is string => Boolean(c));
  for (const candidate of candidates) {
    try {
      execFileSync(candidate, ["-c", "import numpy"], { stdio: "ignore" });
      return candidate;
    } catch {
      // try next
    }
  }
  return undefined;
}

const PYTHON = findOraclePython();
const skip = PYTHON
  ? false
  : "no python with numpy found (set MATH_PLUS_ORACLE_PYTHON)";

interface OracleJob {
  op: string;
  inputs?: string[];
  axis?: number;
  permutation?: number[];
  scalar?: number;
  arange?: [number, number, number];
  dtype?: string;
  specs?: Array<[number | null, number | null, number | null] | null>;
  ddof?: number;
  k?: number;
  largest?: boolean;
  condition?: string;
  fn?: string; // "unary" op dispatch (issue #64)
  min?: number; // clip
  max?: number; // clip
  padding?: Array<[number, number]>; // pad
  value?: number; // pad fill value
  sections?: number | number[]; // split
  outputs?: string[]; // split (multi-output)
  counts?: number | number[]; // repeat
  shift?: number; // roll
  output: string;
}

function assertBoolEqual(actual: Tensor, expected: Tensor, op: string): void {
  assert.equal(actual.dtype, "bool", `${op}: dtype`);
  assert.deepEqual([...actual.shape], [...expected.shape], `${op}: shape`);
  const a = actual.contiguous();
  const e = expected.contiguous();
  for (let i = 0; i < a.size; i++) {
    assert.equal(a.data[a.offset + i], e.data[e.offset + i], `${op}: element ${i}`);
  }
}

function runOracle(dir: string, job: Omit<OracleJob, "output">): Tensor {
  const output = join(dir, `out-${Math.random().toString(36).slice(2)}.npy`);
  const jobPath = join(dir, "job.json");
  writeFileSync(jobPath, JSON.stringify({ ...job, output }));
  execFileSync(PYTHON as string, [ORACLE_SCRIPT, jobPath], {
    stdio: ["ignore", "ignore", "pipe"],
  });
  return Tensor.fromNpy(new Uint8Array(readFileSync(output)));
}

/** For "split", whose result is N differently-shaped parts — one output path per part. */
function runOracleMulti(dir: string, job: Omit<OracleJob, "output" | "outputs">, count: number): Tensor[] {
  const outputs = Array.from({ length: count }, (_, i) =>
    join(dir, `out-multi-${Math.random().toString(36).slice(2)}-${i}.npy`),
  );
  const jobPath = join(dir, "job.json");
  writeFileSync(jobPath, JSON.stringify({ ...job, outputs }));
  execFileSync(PYTHON as string, [ORACLE_SCRIPT, jobPath], {
    stdio: ["ignore", "ignore", "pipe"],
  });
  return outputs.map((p) => Tensor.fromNpy(new Uint8Array(readFileSync(p))));
}

function saveTensor(dir: string, name: string, t: Tensor): string {
  const path = join(dir, `${name}.npy`);
  writeFileSync(path, t.toNpy());
  return path;
}

/** Per-op absolute/relative tolerances (looser for accumulation-order-sensitive ops). */
const TOLERANCES: Record<string, { rtol: number; atol: number }> = {
  "default:f32": { rtol: 1e-5, atol: 1e-6 },
  "default:f64": { rtol: 1e-12, atol: 1e-12 },
  "sum:f32": { rtol: 1e-4, atol: 1e-5 },
  "matmul:f32": { rtol: 1e-4, atol: 1e-5 },
  "variance:f32": { rtol: 1e-3, atol: 1e-4 },
  "variance:f64": { rtol: 1e-9, atol: 1e-9 },
  "std:f32": { rtol: 1e-3, atol: 1e-4 },
  "std:f64": { rtol: 1e-9, atol: 1e-9 },
  "cumsum:f32": { rtol: 1e-4, atol: 1e-5 },
  "cumprod:f32": { rtol: 1e-3, atol: 1e-4 },
  "mean:f32": { rtol: 1e-4, atol: 1e-5 },
};

function assertClose(actual: Tensor, expected: Tensor, op: string): void {
  assert.deepEqual([...actual.shape], [...expected.shape], `${op}: shape`);
  const tolerance =
    TOLERANCES[`${op}:${actual.dtype}`] ??
    TOLERANCES[`default:${actual.dtype}`] ??
    { rtol: 0, atol: 0 };
  const a = actual.contiguous();
  const e = expected.contiguous();
  for (let i = 0; i < a.size; i++) {
    const av = a.data[a.offset + i];
    const ev = e.data[e.offset + i];
    if (typeof av === "bigint" || typeof ev === "bigint") {
      assert.equal(av, ev, `${op}: element ${i}`);
    } else {
      const bound =
        tolerance.atol + tolerance.rtol * Math.abs(ev as number);
      assert.ok(
        Math.abs((av as number) - (ev as number)) <= bound,
        `${op}: element ${i}: ${av} vs ${ev} (bound ${bound})`,
      );
    }
  }
}

function randomTensor(shape: number[], dtype: DType): Tensor {
  const size = shape.reduce((x, y) => x * y, 1);
  const values = Array.from({ length: size }, () =>
    isBigIntDType(dtype)
      ? Math.floor(Math.random() * 200 - 100)
      : Math.random() * 20 - 10,
  );
  return Tensor.from(values, { dtype }).reshape(shape);
}

test("differential vs NumPy", { skip }, async (t) => {
  const dir = mkdtempSync(join(tmpdir(), "math-plus-diff-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));

  await t.test("binary ops, f32 + f64, incl. broadcasting", () => {
    for (const dtype of ["f32", "f64"] as const) {
      const a = randomTensor([3, 4], dtype);
      const b = randomTensor([4], dtype); // broadcast along rows
      const aPath = saveTensor(dir, `a-${dtype}`, a);
      const bPath = saveTensor(dir, `b-${dtype}`, b);
      for (const op of ["add", "sub", "mul", "div"] as const) {
        const expected = runOracle(dir, { op, inputs: [aPath, bPath] });
        assertClose(a[op](b), expected, op);
      }
    }
  });

  await t.test("binary ops on NON-CONTIGUOUS (permuted) views", () => {
    const a = randomTensor([3, 4], "f64");
    const aT = a.permute([1, 0]);
    // Oracle sees the packed transpose; we compute on the strided view.
    const aTPath = saveTensor(dir, "aT", aT); // toNpy packs it
    const b = randomTensor([3], "f64");
    const bPath = saveTensor(dir, "b-nc", b);
    for (const op of ["add", "mul"] as const) {
      const expected = runOracle(dir, { op, inputs: [aTPath, bPath] });
      assertClose(aT[op](b), expected, op);
    }
  });

  await t.test("scalar operand", () => {
    const a = randomTensor([5], "f64");
    const aPath = saveTensor(dir, "a-scalar", a);
    const expected = runOracle(dir, { op: "mul", inputs: [aPath], scalar: 2.5 });
    assertClose(a.mul(2.5), expected, "mul");
  });

  await t.test("reductions: full, axis, negative axis, non-contiguous", () => {
    const a = randomTensor([3, 4], "f64");
    const aPath = saveTensor(dir, "a-red", a);
    assertClose(a.sum(), runOracle(dir, { op: "sum", inputs: [aPath] }), "sum");
    assertClose(a.mean(), runOracle(dir, { op: "mean", inputs: [aPath] }), "mean");
    assertClose(a.sum(0), runOracle(dir, { op: "sum", inputs: [aPath], axis: 0 }), "sum");
    assertClose(a.sum(1), runOracle(dir, { op: "sum", inputs: [aPath], axis: 1 }), "sum");
    assertClose(a.mean(1), runOracle(dir, { op: "mean", inputs: [aPath], axis: 1 }), "mean");

    const aT = a.permute([1, 0]);
    const aTPath = saveTensor(dir, "aT-red", aT);
    assertClose(
      aT.sum(0),
      runOracle(dir, { op: "sum", inputs: [aTPath], axis: 0 }),
      "sum",
    );
  });

  await t.test("i64 add matches NumPy int64 exactly", () => {
    const a = randomTensor([6], "i64");
    const b = randomTensor([6], "i64");
    const expected = runOracle(dir, {
      op: "add",
      inputs: [saveTensor(dir, "a-i64", a), saveTensor(dir, "b-i64", b)],
    });
    assert.equal(expected.dtype, "i64");
    assertClose(a.add(b), expected, "add");
  });

  await t.test("arange matches NumPy", () => {
    const expected = runOracle(dir, {
      op: "arange",
      arange: [0, 10, 3],
      dtype: "float32",
    });
    assertClose(Tensor.arange(0, 10, 3), expected, "arange");
  });

  await t.test("NumPy reads our .npy; we read NumPy's (permute round-trip)", () => {
    const a = randomTensor([2, 3, 4], "f32");
    const expected = runOracle(dir, {
      op: "permute",
      inputs: [saveTensor(dir, "a-perm", a)],
      permutation: [2, 0, 1],
    });
    assertClose(a.permute([2, 0, 1]).contiguous(), expected, "permute");
  });

  await t.test("slice matches NumPy, incl. negative step and partial specs", () => {
    const a = randomTensor([6, 5], "f64");
    const aPath = saveTensor(dir, "a-slice", a);

    const cases: Array<Array<[number | null, number | null, number | null] | null>> = [
      [[1, 5, 2], null],
      [null, [0, 4, 1]],
      [[null, null, -1], null], // reverse axis 0
      [[-4, -1, 1], [1, null, null]],
    ];
    for (const specs of cases) {
      const expected = runOracle(dir, { op: "slice", inputs: [aPath], specs });
      const jsSpecs = specs.map((s) =>
        s === null
          ? null
          : {
              start: s[0] ?? undefined,
              end: s[1] ?? undefined,
              step: s[2] ?? undefined,
            },
      );
      assertClose(a.slice(...jsSpecs).contiguous(), expected, "slice");
    }
  });

  await t.test("slice on a non-contiguous (permuted) view matches NumPy", () => {
    const a = randomTensor([4, 3], "f64");
    const aT = a.permute([1, 0]); // non-contiguous
    const aTPath = saveTensor(dir, "aT-slice", aT); // toNpy packs it
    const specs: Array<[number | null, number | null, number | null] | null> = [
      [0, 2, 1],
      [null, null, -1],
    ];
    const expected = runOracle(dir, { op: "slice", inputs: [aTPath], specs });
    const sliced = aT.slice(
      { start: 0, end: 2 },
      { step: -1 },
    );
    assertClose(sliced.contiguous(), expected, "slice");
  });

  await t.test("matmul 2-D matches NumPy", () => {
    const a = randomTensor([4, 3], "f64");
    const b = randomTensor([3, 5], "f64");
    const expected = runOracle(dir, {
      op: "matmul",
      inputs: [saveTensor(dir, "mm-a", a), saveTensor(dir, "mm-b", b)],
    });
    assertClose(a.matmul(b), expected, "matmul");
  });

  await t.test("matmul on transposed (non-contiguous) operands matches NumPy", () => {
    const a = randomTensor([3, 4], "f64"); // will use a.T: (4,3)
    const b = randomTensor([3, 5], "f64"); // will use b.T: (5,3), then .T again isn't needed
    const aTPath = saveTensor(dir, "mm-aT", a.permute([1, 0])); // packs transposed
    const bPath = saveTensor(dir, "mm-b2", b);
    const expected = runOracle(dir, { op: "matmul", inputs: [aTPath, bPath] });
    // a.permute([1,0]) is a non-contiguous (4,3) view; matmul must not copy it first.
    assertClose(a.permute([1, 0]).matmul(b), expected, "matmul");
  });

  await t.test("matmul batched (leading-axis broadcast) matches NumPy", () => {
    const a = randomTensor([2, 3, 4], "f64"); // batch=2
    const b = randomTensor([4, 5], "f64"); // broadcasts across the batch
    const expected = runOracle(dir, {
      op: "matmul",
      inputs: [saveTensor(dir, "mm-batch-a", a), saveTensor(dir, "mm-batch-b", b)],
    });
    assertClose(a.matmul(b), expected, "matmul");
  });

  await t.test("matmul with a 1-D operand matches NumPy's squeeze rules", () => {
    const mat = randomTensor([4, 3], "f64");
    const vecK = randomTensor([3], "f64");
    const vecM = randomTensor([4], "f64");

    const mv = runOracle(dir, {
      op: "matmul",
      inputs: [saveTensor(dir, "mv-a", mat), saveTensor(dir, "mv-b", vecK)],
    });
    assertClose(mat.matmul(vecK), mv, "matmul"); // (4,3)@(3,) -> (4,)

    const vm = runOracle(dir, {
      op: "matmul",
      inputs: [saveTensor(dir, "vm-a", vecM), saveTensor(dir, "vm-b", mat)],
    });
    assertClose(vecM.matmul(mat), vm, "matmul"); // (4,)@(4,3) -> (3,)
  });

  await t.test("dot on 1-D operands matches NumPy (0-d result)", () => {
    const a = randomTensor([6], "f64");
    const b = randomTensor([6], "f64");
    const expected = runOracle(dir, {
      op: "dot",
      inputs: [saveTensor(dir, "dot-a", a), saveTensor(dir, "dot-b", b)],
    });
    assertClose(a.dot(b), expected, "dot");
  });

  await t.test("matmul on i64 operands matches NumPy exactly", () => {
    const a = randomTensor([3, 4], "i64");
    const b = randomTensor([4, 2], "i64");
    const expected = runOracle(dir, {
      op: "matmul",
      inputs: [saveTensor(dir, "mm-i64-a", a), saveTensor(dir, "mm-i64-b", b)],
    });
    assert.equal(expected.dtype, "i64");
    assertClose(a.matmul(b), expected, "matmul");
  });

  await t.test("cast matches NumPy astype across dtype pairs", () => {
    const a = randomTensor([5], "f64");
    const aPath = saveTensor(dir, "cast-a", a);
    for (const dtype of ["f32", "i32", "i64", "u8"] as const) {
      const expected = runOracle(dir, { op: "cast", inputs: [aPath], dtype });
      assertClose(a.cast(dtype), expected, "cast");
    }
  });

  await t.test("comparisons match NumPy, incl. broadcasting", () => {
    const a = randomTensor([4, 3], "f64");
    const b = randomTensor([3], "f64");
    const aPath = saveTensor(dir, "cmp-a", a);
    const bPath = saveTensor(dir, "cmp-b", b);
    for (const op of ["eq", "ne", "lt", "lte", "gt", "gte"] as const) {
      const expected = runOracle(dir, { op, inputs: [aPath, bPath] });
      assertBoolEqual(a[op](b), expected, op);
    }
  });

  await t.test("min/max/argmin/argmax match NumPy, full and per-axis", () => {
    const a = randomTensor([4, 5], "f64");
    const aPath = saveTensor(dir, "extrema-a", a);
    for (const op of ["min", "max"] as const) {
      assertClose(a[op](), runOracle(dir, { op, inputs: [aPath] }), op);
      assertClose(a[op](0), runOracle(dir, { op, inputs: [aPath], axis: 0 }), op);
      assertClose(a[op](1), runOracle(dir, { op, inputs: [aPath], axis: 1 }), op);
    }
    for (const op of ["argmin", "argmax"] as const) {
      assertClose(a[op](), runOracle(dir, { op, inputs: [aPath] }), op);
      assertClose(a[op](0), runOracle(dir, { op, inputs: [aPath], axis: 0 }), op);
      assertClose(a[op](1), runOracle(dir, { op, inputs: [aPath], axis: 1 }), op);
    }
  });

  await t.test("sqrt matches NumPy", () => {
    const a = randomTensor([6], "f64").add(20); // keep positive
    const aPath = saveTensor(dir, "sqrt-a", a);
    assertClose(a.sqrt(), runOracle(dir, { op: "sqrt", inputs: [aPath] }), "sqrt");
  });

  await t.test("variance/std match NumPy, full + per-axis + ddof=1", () => {
    const a = randomTensor([4, 5], "f64");
    const aPath = saveTensor(dir, "var-a", a);
    for (const op of ["variance", "std"] as const) {
      assertClose(a[op](), runOracle(dir, { op, inputs: [aPath] }), op);
      assertClose(a[op](0), runOracle(dir, { op, inputs: [aPath], axis: 0 }), op);
      assertClose(
        a[op](1, { ddof: 1 }),
        runOracle(dir, { op, inputs: [aPath], axis: 1, ddof: 1 }),
        op,
      );
    }
  });

  await t.test("cumsum/cumprod match NumPy, flattened and per-axis", () => {
    const a = randomTensor([3, 4], "f64");
    const aPath = saveTensor(dir, "cum-a", a);
    for (const op of ["cumsum", "cumprod"] as const) {
      assertClose(a[op](), runOracle(dir, { op, inputs: [aPath] }), op);
      assertClose(a[op](1), runOracle(dir, { op, inputs: [aPath], axis: 1 }), op);
    }
  });

  await t.test("sort/argsort match NumPy along the last axis by default", () => {
    const a = randomTensor([4, 5], "f64");
    const aPath = saveTensor(dir, "sort-a", a);
    assertClose(a.sort(), runOracle(dir, { op: "sort", inputs: [aPath] }), "sort");
    assertClose(
      a.argsort(),
      runOracle(dir, { op: "argsort", inputs: [aPath] }),
      "argsort",
    );
    assertClose(
      a.sort(0),
      runOracle(dir, { op: "sort", inputs: [aPath], axis: 0 }),
      "sort",
    );
  });

  await t.test("topK matches NumPy (values + indices, largest and smallest)", () => {
    const a = randomTensor([6], "f64");
    const aPath = saveTensor(dir, "topk-a", a);
    for (const largest of [true, false]) {
      const { values, indices } = a.topK(3, { largest });
      assertClose(
        values,
        runOracle(dir, { op: "topk_values", inputs: [aPath], k: 3, largest }),
        "topk_values",
      );
      assertClose(
        indices,
        runOracle(dir, { op: "topk_indices", inputs: [aPath], k: 3, largest }),
        "topk_indices",
      );
    }
  });

  await t.test("concat/stack match NumPy", () => {
    const a = randomTensor([2, 3], "f64");
    const b = randomTensor([2, 3], "f64");
    const aPath = saveTensor(dir, "cat-a", a);
    const bPath = saveTensor(dir, "cat-b", b);
    assertClose(
      Tensor.concat([a, b], { axis: 0 }),
      runOracle(dir, { op: "concat", inputs: [aPath, bPath], axis: 0 }),
      "concat",
    );
    assertClose(
      Tensor.concat([a, b], { axis: 1 }),
      runOracle(dir, { op: "concat", inputs: [aPath, bPath], axis: 1 }),
      "concat",
    );
    assertClose(
      Tensor.stack([a, b], { axis: 0 }),
      runOracle(dir, { op: "stack", inputs: [aPath, bPath], axis: 0 }),
      "stack",
    );
  });

  await t.test("where matches NumPy", () => {
    const cond = Tensor.from([1, 0, 1, 0, 1, 0], { dtype: "bool" }).reshape([2, 3]);
    const a = randomTensor([2, 3], "f64");
    const b = randomTensor([2, 3], "f64");
    const condPath = saveTensor(dir, "where-cond", cond);
    const aPath = saveTensor(dir, "where-a", a);
    const bPath = saveTensor(dir, "where-b", b);
    const expected = runOracle(dir, {
      op: "where",
      inputs: [aPath, bPath],
      condition: condPath,
    });
    assertClose(Tensor.where(cond, a, b), expected, "where");
  });

  await t.test("log matches NumPy", () => {
    const a = randomTensor([6], "f64").add(20); // keep positive, log's domain
    const aPath = saveTensor(dir, "log-a", a);
    assertClose(a.log(), runOracle(dir, { op: "log", inputs: [aPath] }), "log");
  });

  // ---- unary op-table parity with the compiled IR (issue #64) -------------
  await t.test("exp/pow/abs/neg/sign/trig/hyperbolic/cbrt/log10/log2/expm1/log1p/floor/ceil/round/trunc match NumPy", () => {
    // Per-fn: how to build a SAFE-DOMAIN input tensor from a raw random
    // f64 tensor `r` (values roughly in [-10, 10)), and the Tensor method
    // under test. `roundSafe` avoids exact .5 boundaries, where JS
    // Math.round (half-up) and NumPy's np.round (half-to-even) genuinely
    // disagree by convention -- not a bug on either side, so the test
    // sidesteps it rather than papering over it with a loose tolerance.
    const cases: Array<{
      name: string;
      fn: string;
      scalar?: number;
      domain: (r: Tensor) => Tensor;
      call: (t: Tensor) => Tensor;
    }> = [
      { name: "exp", fn: "exp", domain: (r) => r.mul(0.2), call: (t) => t.exp() }, // keep exp() from overflowing
      { name: "pow", fn: "pow", scalar: 3, domain: (r) => r, call: (t) => t.pow(3) },
      { name: "abs", fn: "abs", domain: (r) => r, call: (t) => t.abs() },
      { name: "neg", fn: "neg", domain: (r) => r, call: (t) => t.neg() },
      { name: "sign", fn: "sign", domain: (r) => r, call: (t) => t.sign() },
      { name: "sin", fn: "sin", domain: (r) => r, call: (t) => t.sin() },
      { name: "cos", fn: "cos", domain: (r) => r, call: (t) => t.cos() },
      { name: "tan", fn: "tan", domain: (r) => r.mul(0.1), call: (t) => t.tan() }, // stay away from tan's poles
      { name: "asin", fn: "arcsin", domain: (r) => r.mul(0.09), call: (t) => t.asin() }, // domain [-1,1]
      { name: "acos", fn: "arccos", domain: (r) => r.mul(0.09), call: (t) => t.acos() },
      { name: "atan", fn: "arctan", domain: (r) => r, call: (t) => t.atan() },
      { name: "sinh", fn: "sinh", domain: (r) => r.mul(0.2), call: (t) => t.sinh() },
      { name: "cosh", fn: "cosh", domain: (r) => r.mul(0.2), call: (t) => t.cosh() },
      { name: "tanh", fn: "tanh", domain: (r) => r, call: (t) => t.tanh() },
      { name: "asinh", fn: "arcsinh", domain: (r) => r, call: (t) => t.asinh() },
      { name: "acosh", fn: "arccosh", domain: (r) => r.abs().add(1.5), call: (t) => t.acosh() }, // domain x >= 1
      { name: "atanh", fn: "arctanh", domain: (r) => r.mul(0.09), call: (t) => t.atanh() }, // domain (-1,1)
      { name: "cbrt", fn: "cbrt", domain: (r) => r, call: (t) => t.cbrt() },
      { name: "log10", fn: "log10", domain: (r) => r.abs().add(0.5), call: (t) => t.log10() }, // domain x > 0
      { name: "log2", fn: "log2", domain: (r) => r.abs().add(0.5), call: (t) => t.log2() },
      { name: "expm1", fn: "expm1", domain: (r) => r.mul(0.2), call: (t) => t.expm1() },
      { name: "log1p", fn: "log1p", domain: (r) => r.abs().add(0.5), call: (t) => t.log1p() }, // domain x > -1
      { name: "floor", fn: "floor", domain: (r) => r, call: (t) => t.floor() },
      { name: "ceil", fn: "ceil", domain: (r) => r, call: (t) => t.ceil() },
      {
        name: "round",
        fn: "round",
        // Push every value at least 0.1 away from the nearest .5 boundary
        // (half-up vs half-to-even only disagree exactly AT .5).
        domain: (r) => {
          const rounded = r.round();
          const frac = r.sub(rounded);
          const nearBoundary = frac.abs().lt(0.1);
          const pushedUp = Tensor.where(frac.gte(0), r.add(0.15), r.sub(0.15));
          return Tensor.where(nearBoundary, pushedUp, r);
        },
        call: (t) => t.round(),
      },
      { name: "trunc", fn: "trunc", domain: (r) => r, call: (t) => t.trunc() },
    ];
    for (const { name, fn, scalar, domain, call } of cases) {
      const raw = randomTensor([12], "f64");
      const a = domain(raw);
      const aPath = saveTensor(dir, `unary-${name}-a`, a);
      assertClose(call(a), runOracle(dir, { op: "unary", fn, inputs: [aPath], scalar }), name);
    }
  });

  // ---- structural/manipulation ops (issue #65) -----------------------------

  await t.test("clip matches NumPy (min only, max only, both)", () => {
    const a = randomTensor([12], "f64").mul(10); // spread wide enough to hit both bounds
    const aPath = saveTensor(dir, "clip-a", a);
    assertClose(a.clip(-3, 3), runOracle(dir, { op: "clip", inputs: [aPath], min: -3, max: 3 }), "clip-both");
    assertClose(a.clip(-3, undefined), runOracle(dir, { op: "clip", inputs: [aPath], min: -3 }), "clip-min");
    assertClose(a.clip(undefined, 3), runOracle(dir, { op: "clip", inputs: [aPath], max: 3 }), "clip-max");
  });

  await t.test("prod matches NumPy, full + per-axis (built on cumprod's own tested loop)", () => {
    const a = randomTensor([2, 3], "f64").mul(0.3).add(1); // keep products from over/underflowing
    const aPath = saveTensor(dir, "prod-a", a);
    assertClose(a.prod(), runOracle(dir, { op: "prod", inputs: [aPath] }), "prod-all");
    assertClose(a.prod(0), runOracle(dir, { op: "prod", inputs: [aPath], axis: 0 }), "prod-axis0");
    assertClose(a.prod(1), runOracle(dir, { op: "prod", inputs: [aPath], axis: 1 }), "prod-axis1");
  });

  await t.test("pad matches NumPy constant-mode padding, last-N-axes convention", () => {
    const a = randomTensor([2, 3], "f64");
    const aPath = saveTensor(dir, "pad-a", a);
    const padding: Array<[number, number]> = [[1, 2]]; // last axis only, axis 0 untouched
    assertClose(
      a.pad(padding, { value: -7 }),
      runOracle(dir, { op: "pad", inputs: [aPath], padding, value: -7 }),
      "pad",
    );
  });

  await t.test("split matches NumPy, equal-sections form and cut-point form", () => {
    const a = randomTensor([8], "f64");
    const aPath = saveTensor(dir, "split-a", a);
    const equal = a.split(4);
    const equalExpected = runOracleMulti(dir, { op: "split", inputs: [aPath], sections: 4 }, 4);
    assert.equal(equal.length, 4);
    equal.forEach((part, i) => assertClose(part, equalExpected[i] as Tensor, `split-equal-${i}`));

    const cut = a.split([2, 5]);
    const cutExpected = runOracleMulti(dir, { op: "split", inputs: [aPath], sections: [2, 5] }, 3);
    assert.equal(cut.length, 3);
    cut.forEach((part, i) => assertClose(part, cutExpected[i] as Tensor, `split-cut-${i}`));
  });

  await t.test("repeat matches NumPy, uniform count and per-index counts", () => {
    const a = randomTensor([4], "f64");
    const aPath = saveTensor(dir, "repeat-a", a);
    assertClose(a.repeat(3), runOracle(dir, { op: "repeat", inputs: [aPath], counts: 3 }), "repeat-uniform");
    const counts = [1, 0, 2, 3];
    assertClose(a.repeat(counts), runOracle(dir, { op: "repeat", inputs: [aPath], counts }), "repeat-perindex");
  });

  await t.test("flip matches NumPy, single axis and all axes", () => {
    const a = randomTensor([2, 3], "f64");
    const aPath = saveTensor(dir, "flip-a", a);
    assertClose(a.flip(1), runOracle(dir, { op: "flip", inputs: [aPath], axis: 1 }), "flip-axis1");
    assertClose(a.flip(), runOracle(dir, { op: "flip", inputs: [aPath] }), "flip-all");
  });

  await t.test("roll matches NumPy, per-axis and flattened (no-axis)", () => {
    const a = randomTensor([2, 3], "f64");
    const aPath = saveTensor(dir, "roll-a", a);
    assertClose(a.roll(2, { axis: 1 }), runOracle(dir, { op: "roll", inputs: [aPath], shift: 2, axis: 1 }), "roll-axis");
    assertClose(a.roll(2), runOracle(dir, { op: "roll", inputs: [aPath], shift: 2 }), "roll-flat");
  });

  await t.test("nonzero matches NumPy argwhere's [count, ndim] coordinate shape", () => {
    // Deliberately sparse fixed pattern (not random) so the expected
    // coordinate set is hand-verifiable, not just "whatever NumPy says".
    const a = Tensor.from([0, 1, 0, 2, 0, 0, 3, 0], { dtype: "f64" }).reshape([2, 4]);
    const aPath = saveTensor(dir, "nonzero-a", a);
    const got = a.nonzero();
    const expected = runOracle(dir, { op: "nonzero", inputs: [aPath] });
    assert.equal(got.dtype, "i64");
    assertClose(got, expected, "nonzero");
    // Row 0 = [0,1,0,2] -> nonzero at col 1 (val 1), col 3 (val 2).
    // Row 1 = [0,0,3,0] -> nonzero at col 2 (val 3).
    assert.deepEqual([...got.contiguous().data], [0n, 1n, 0n, 3n, 1n, 2n]);
  });

  await t.test("relu/sigmoid/gelu match NumPy", () => {
    const a = randomTensor([6], "f64");
    const aPath = saveTensor(dir, "act-a", a);
    for (const op of ["relu", "sigmoid", "gelu"] as const) {
      assertClose(a[op](), runOracle(dir, { op, inputs: [aPath] }), op);
    }
  });

  await t.test("softmax matches NumPy, default and explicit axis, sums to 1", () => {
    const a = randomTensor([4, 5], "f64");
    const aPath = saveTensor(dir, "softmax-a", a);
    assertClose(a.softmax(), runOracle(dir, { op: "softmax", inputs: [aPath] }), "softmax");
    assertClose(
      a.softmax(0),
      runOracle(dir, { op: "softmax", inputs: [aPath], axis: 0 }),
      "softmax",
    );
    const rowSums = a.softmax().sum(1);
    for (const v of rowSums.toArray() as number[]) {
      assert.ok(Math.abs(v - 1) < 1e-9, `row sum ${v} != 1`);
    }
  });
});
