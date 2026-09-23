/**
 * issue #128: `withCompute` (explicit opt-in half-precision arithmetic) and
 * bf16 `.npy` through the ml_dtypes convention, both checked against
 * NumPy (+ ml_dtypes for bf16) via scripts/half_oracle.py.
 *
 * Oracle resolution (skip-don't-fail, per AGENTS.md):
 * - f16 jobs need NumPy: $MATH_PLUS_ORACLE_PYTHON, else `python3`.
 * - bf16 jobs also need ml_dtypes: the first of those two that can import it,
 *   else `uv run --no-project --with numpy --with ml_dtypes python` when `uv`
 *   is on PATH.
 * With no suitable interpreter the affected tests skip with a reason; a real
 * run should show 0 skipped.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { Tensor, withCompute } from "../src/index.ts";

const ORACLE_SCRIPT = new URL("../scripts/half_oracle.py", import.meta.url).pathname;

function works(cmd: string[], imports: string): boolean {
  try {
    execFileSync(cmd[0] as string, [...cmd.slice(1), "-c", `import ${imports}`], { stdio: "ignore", timeout: 120_000 });
    return true;
  } catch {
    return false;
  }
}

function findPython(imports: string): string[] | undefined {
  const plain = [process.env.MATH_PLUS_ORACLE_PYTHON, "python3"].filter((c): c is string => Boolean(c)).map((c) => [c]);
  for (const cmd of plain) if (works(cmd, imports)) return cmd;
  if (imports.includes("ml_dtypes")) {
    const uv = ["uv", "run", "--quiet", "--no-project", "--with", "numpy", "--with", "ml_dtypes", "python"];
    if (works(uv, imports)) return uv;
  }
  return undefined;
}

const NUMPY = findPython("numpy");
const ML_DTYPES = findPython("numpy, ml_dtypes");
const skipF16 = NUMPY ? false : "no python with numpy found (set MATH_PLUS_ORACLE_PYTHON)";
const skipBf16 = ML_DTYPES
  ? false
  : "no python with numpy + ml_dtypes found (set MATH_PLUS_ORACLE_PYTHON, or install uv)";

const dir = mkdtempSync(join(tmpdir(), "tensor-core-half-"));
test.after(() => rmSync(dir, { recursive: true, force: true }));
let jobCounter = 0;

function oracle(cmd: string[], job: Record<string, unknown>): Uint8Array {
  const n = jobCounter++;
  const output = join(dir, `out-${n}.npy`);
  const jobPath = join(dir, `job-${n}.json`);
  writeFileSync(jobPath, JSON.stringify({ ...job, output }));
  execFileSync(cmd[0] as string, [...cmd.slice(1), ORACLE_SCRIPT, jobPath], { stdio: ["ignore", "ignore", "inherit"] });
  return new Uint8Array(readFileSync(output));
}

function save(name: string, t: Tensor): string {
  const p = join(dir, `${name}-${jobCounter++}.npy`);
  writeFileSync(p, t.toNpy());
  return p;
}

/** Deterministic values spread over several binades, including ties-prone ones. */
function values(n: number, seed: number, scale = 4): number[] {
  let s = seed >>> 0;
  return Array.from({ length: n }, () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return ((s / 2 ** 32) * 2 - 1) * scale;
  });
}

function bitsOf(t: Tensor): number[] {
  return [...(t.contiguous().data as Uint16Array)];
}

/** Bit patterns equal, or (for accumulation-order-sensitive ops) at most `ulps` apart. */
function assertHalfBits(actual: Tensor, expected: Tensor, label: string, ulps = 0): void {
  assert.equal(actual.dtype, expected.dtype, `${label}: dtype`);
  assert.deepEqual(actual.shape, expected.shape, `${label}: shape`);
  const a = bitsOf(actual);
  const e = bitsOf(expected);
  for (let i = 0; i < a.length; i++) {
    const ai = a[i] as number;
    const ei = e[i] as number;
    if (ulps === 0) {
      assert.equal(ai, ei, `${label}[${i}]: 0x${ai.toString(16)} vs 0x${ei.toString(16)}`);
    } else {
      assert.ok((ai & 0x8000) === (ei & 0x8000) && Math.abs(ai - ei) <= ulps, `${label}[${i}]: 0x${ai.toString(16)} vs 0x${ei.toString(16)}`);
    }
  }
}

// ---- withCompute: contract (no oracle needed) ---------------------------------

test("withCompute: decodes, computes, re-encodes to the input half dtype; non-compute results pass through", () => {
  const a = Tensor.from([1.5, -2, 0.1], { dtype: "f16" });
  const b = Tensor.from([2, 0.5, 3], { dtype: "f16" });
  const sum = withCompute("f32", [a, b], (x, y) => x.add(y));
  assert.equal(sum.dtype, "f16");
  // f16(0.1) = 0.0999755859375; + 3 = 3.0999755859375 in f32, which rounds to the f16 value 3.099609375
  assert.deepEqual(sum.toArray(), [3.5, -1.5, 3.099609375]);
  const [prod, mask] = withCompute("f32", [a, b], (x, y) => [x.mul(y), x.gt(y)]);
  assert.equal(prod!.dtype, "f16");
  assert.equal(mask!.dtype, "bool", "a bool result is not re-encoded");
  assert.equal(withCompute("f32", [a], (x) => x.argmax()).dtype, "i32", "an index result is not re-encoded");
});

test("withCompute: non-half inputs are passed through untouched (no hidden promotion of them)", () => {
  const h = Tensor.from([1, 2, 3], { dtype: "bf16" });
  const bias = Tensor.from([0.5, 0.5, 0.5], { dtype: "f32" });
  const seen: string[] = [];
  const out = withCompute("f32", [h, bias], (x, y) => {
    seen.push(x!.dtype, y!.dtype);
    return x!.add(y!);
  });
  assert.deepEqual(seen, ["f32", "f32"]);
  assert.equal(out.dtype, "bf16");
  assert.deepEqual(out.toArray(), [1.5, 2.5, 3.5]);
  // An f64 input under compute "f32" stays f64, so mixing still needs an explicit cast.
  const f64 = Tensor.from([1, 1, 1], { dtype: "f64" });
  assert.throws(() => withCompute("f32", [h, f64], (x, y) => x!.add(y!)), /no implicit promotion|dtype mismatch/);
});

test("withCompute: rejects no half input, mixed f16/bf16, a bad compute dtype, and non-Tensor results", () => {
  const f = Tensor.from([1], { dtype: "f16" });
  const b = Tensor.from([1], { dtype: "bf16" });
  const x = Tensor.from([1], { dtype: "f32" });
  assert.throws(() => withCompute("f32", [x], (t) => t), /no f16\/bf16 input/);
  assert.throws(() => withCompute("f32", [f, b], (t) => t), /mix f16 and bf16/);
  assert.throws(() => withCompute("f16" as "f32", [f], (t) => t), /must be "f32" or "f64"/);
  assert.throws(() => withCompute("f32", [f], () => 1 as unknown as Tensor), /must return a Tensor/);
});

test("withCompute: works on non-contiguous half views and leaves the inputs unchanged", () => {
  const m = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f16" }).reshape([2, 3]);
  const before = bitsOf(m);
  const t = withCompute("f32", [m.transpose()], (x) => x.mul(2));
  assert.deepEqual(t.toArray(), [[2, 8], [4, 10], [6, 12]]);
  assert.deepEqual(bitsOf(m), before);
});

test("the error on half arithmetic points at withCompute as well as cast()", () => {
  assert.throws(() => Tensor.from([1], { dtype: "f16" }).add(1), /cast\("f32"\).*withCompute\("f32"/);
});

// ---- withCompute vs NumPy (f16) -------------------------------------------------

test("withCompute f16 vs NumPy: single ops equal NumPy's own float16 ufuncs bit-for-bit", { skip: skipF16 }, () => {
  const a = Tensor.from(values(256, 1, 300), { dtype: "f16" });
  const b = Tensor.from(values(256, 2, 3), { dtype: "f16" });
  const [pa, pb] = [save("a", a), save("b", b)];
  for (const expr of ["add", "mul"] as const) {
    const expected = Tensor.fromNpy(oracle(NUMPY!, { op: "compute", expr, inputs: [pa, pb], half: "f16", compute: "native" }));
    const actual = withCompute("f32", [a, b], (x, y) => (expr === "add" ? x!.add(y!) : x!.mul(y!)));
    assertHalfBits(actual, expected, `f16 ${expr} vs NumPy native float16`);
  }
});

test("withCompute f16 vs NumPy: a chain rounds once, matching f(a.astype(f32), ...).astype(float16)", { skip: skipF16 }, () => {
  const a = Tensor.from(values(300, 3, 8), { dtype: "f16" });
  const b = Tensor.from(values(300, 4, 8), { dtype: "f16" });
  const [pa, pb] = [save("a", a), save("b", b)];
  for (const compute of ["f32", "f64"] as const) {
    const expected = Tensor.fromNpy(
      oracle(NUMPY!, { op: "compute", expr: "chain", inputs: [pa, pb], half: "f16", compute: compute === "f32" ? "float32" : "float64" }),
    );
    const actual = withCompute(compute, [a, b], (x, y) => x!.mul(y!).add(x!).relu());
    assertHalfBits(actual, expected, `f16 chain in ${compute}`);
  }
});

test("withCompute f16 vs NumPy: matmul, softmax and a reduction (within 1 f16 ulp: accumulation order)", { skip: skipF16 }, () => {
  const a = Tensor.from(values(8 * 16, 5, 2), { dtype: "f16" }).reshape([8, 16]);
  const b = Tensor.from(values(16 * 4, 6, 2), { dtype: "f16" }).reshape([16, 4]);
  const [pa, pb] = [save("a", a), save("b", b)];
  const run = (expr: string, inputs: string[]) =>
    Tensor.fromNpy(oracle(NUMPY!, { op: "compute", expr, inputs, half: "f16", compute: "float32" }));
  assertHalfBits(withCompute("f32", [a, b], (x, y) => x!.matmul(y!)), run("matmul", [pa, pb]), "matmul", 1);
  assertHalfBits(withCompute("f32", [a], (x) => x!.softmax(-1)), run("softmax", [pa]), "softmax", 1);
  assertHalfBits(withCompute("f32", [a], (x) => x!.sum(0)), run("sum0", [pa]), "sum(axis 0)", 1);
});

// ---- bf16: withCompute and .npy vs NumPy + ml_dtypes -----------------------------

test("withCompute bf16 vs ml_dtypes: add and a chain in f32 match bit-for-bit", { skip: skipBf16 }, () => {
  const a = Tensor.from(values(300, 7, 50), { dtype: "bf16" });
  const b = Tensor.from(values(300, 8, 5), { dtype: "bf16" });
  const [pa, pb] = [save("a", a), save("b", b)];
  const run = (expr: string) =>
    Tensor.fromNpy(oracle(ML_DTYPES!, { op: "compute", expr, inputs: [pa, pb], half: "bf16", compute: "float32" }), { voidAs: "bf16" });
  assertHalfBits(withCompute("f32", [a, b], (x, y) => x!.add(y!)), run("add"), "bf16 add");
  assertHalfBits(withCompute("f32", [a, b], (x, y) => x!.mul(y!).add(x!).relu()), run("chain"), "bf16 chain");
});

test(".npy bf16: a NumPy + ml_dtypes file (descr '<V2') reads with { voidAs: 'bf16' }, and only with it", { skip: skipBf16 }, () => {
  const src = Tensor.from([1.5, -2.25, 0.1, 3.4e38, 1e-40, 65504, -0, Infinity, ...values(64, 9, 1e4)], { dtype: "f32" });
  const bytes = oracle(ML_DTYPES!, { op: "save_bf16", inputs: [save("src", src)] });
  assert.match(new TextDecoder().decode(bytes.subarray(0, 80)), /'descr': '<V2'/);
  assert.throws(() => Tensor.fromNpy(bytes), /untyped 2-byte void.*voidAs: "bf16"/);
  const t = Tensor.fromNpy(bytes, { voidAs: "bf16" });
  assert.equal(t.dtype, "bf16");
  assertHalfBits(t, src.cast("bf16"), "ml_dtypes astype(bfloat16) vs cast('bf16')");
});

test(".npy bf16: JS-written file is byte-identical to np.save, and reads back via .view(bfloat16)", { skip: skipBf16 }, () => {
  const src = Tensor.from(values(40, 10, 100), { dtype: "f32" });
  const js = src.cast("bf16").toNpy();
  const py = oracle(ML_DTYPES!, { op: "save_bf16", inputs: [save("src", src)] });
  assert.deepEqual([...js], [...py], "JS toNpy() bytes equal np.save(x.astype(ml_dtypes.bfloat16))");
  const back = Tensor.fromNpy(oracle(ML_DTYPES!, { op: "load_bf16", inputs: [save("js-bf16", src.cast("bf16"))] }));
  assert.equal(back.dtype, "f32");
  assert.deepEqual(back.toArray(), src.cast("bf16").cast("f32").toArray());
});
