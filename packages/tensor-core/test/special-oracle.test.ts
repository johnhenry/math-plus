/**
 * PyTorch parity for `Tensor.gelu()` in both `approximate` modes, f64 and f32
 * (issue #122), via `scripts/gelu_torch_oracle.py`. The scalar erf/erfc/GELU
 * themselves are checked against SciPy in @johnhenry/math-plus-special.
 *
 * Same skip-don't-fail contract as the other Python oracles (docs/TESTING.md):
 * PyTorch resolves via $MATH_PLUS_TORCH_ORACLE_PYTHON, else
 * $MATH_PLUS_ORACLE_PYTHON, else `python3`. A real verification run must show
 * 0 skipped.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "../src/index.ts";

const ORACLE_SCRIPT = new URL("../scripts/gelu_torch_oracle.py", import.meta.url).pathname;

function findPython(envVars: readonly (string | undefined)[], probe: string): string | undefined {
  const candidates = [...envVars, "python3"].filter((c): c is string => Boolean(c));
  for (const candidate of candidates) {
    try {
      execFileSync(candidate, ["-c", probe], { stdio: "ignore" });
      return candidate;
    } catch {
      // try next
    }
  }
  return undefined;
}

const TORCH_PYTHON = findPython(
  [process.env.MATH_PLUS_TORCH_ORACLE_PYTHON, process.env.MATH_PLUS_ORACLE_PYTHON],
  "import torch",
);
const TORCH_SKIP = TORCH_PYTHON ? false : "no python with torch found (set MATH_PLUS_TORCH_ORACLE_PYTHON)";

function runOracle(python: string, job: Record<string, unknown>): number[] {
  const dir = mkdtempSync(join(tmpdir(), "math-plus-special-oracle-"));
  try {
    const jobPath = join(dir, "job.json");
    writeFileSync(jobPath, JSON.stringify(job));
    const stdout = execFileSync(python, [ORACLE_SCRIPT, jobPath], {
      stdio: ["ignore", "pipe", "pipe"],
      maxBuffer: 64 * 1024 * 1024,
    });
    return JSON.parse(stdout.toString("utf8")) as number[];
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/** Deterministic PRNG so failures reproduce. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Asserts `|ours - oracle| <= atol + rtol·|oracle|` elementwise and returns the
 * worst RELATIVE error over points with `|oracle| >= relFloor` (for reporting).
 */
function checkClose(
  actual: readonly number[],
  expected: readonly number[],
  xs: readonly number[],
  label: string,
  { rtol, atol = 0, relFloor = Number.MIN_VALUE }: { rtol: number; atol?: number; relFloor?: number },
): number {
  assert.equal(actual.length, expected.length, `${label}: length`);
  let worst = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i]!;
    const e = expected[i]!;
    const diff = Math.abs(a - e);
    assert.ok(diff <= atol + rtol * Math.abs(e), `${label}(${xs[i]}): ours ${a} vs oracle ${e} (diff ${diff})`);
    if (Math.abs(e) >= relFloor) worst = Math.max(worst, diff / Math.abs(e));
  }
  return worst;
}

test("Tensor.gelu() matches torch.nn.functional.gelu in both approximate modes (f64)", { skip: TORCH_SKIP }, (t) => {
  const xs: number[] = [];
  for (let i = -12 * 64; i <= 12 * 64; i++) xs.push(i / 64);
  const rng = mulberry32(7);
  for (let i = 0; i < 1000; i++) xs.push(rng() * 20 - 10);
  const input = Tensor.from(xs, { dtype: "f64" });
  for (const approximate of ["none", "tanh"] as const) {
    const expected = runOracle(TORCH_PYTHON as string, { op: "torch_gelu", xs, approximate, dtype: "float64" });
    const ours = Array.from(input.gelu({ approximate }).data as Float64Array);
    // torch's f64 exact GELU is 0.5·x·(1 + erf(x/√2)), which cancels in the
    // left tail (absolute error ~1e-16·|x| there), so the bound is mixed
    // abs/rel; our own tail accuracy is pinned by @johnhenry/math-plus-special's ndtr test.
    const worst = checkClose(ours, expected, xs, `gelu(${approximate})`, { rtol: 1e-13, atol: 1e-15, relFloor: 1e-3 });
    t.diagnostic(`gelu approximate=${approximate} vs torch f64: max relative error ${worst.toExponential(2)} where |gelu| >= 1e-3`);
  }
  // Default is exact ("none"), like torch.
  assert.deepEqual(Array.from(input.gelu().data as Float64Array), Array.from(input.gelu({ approximate: "none" }).data as Float64Array));
});

test("Tensor.gelu() matches torch.nn.functional.gelu in both approximate modes (f32)", { skip: TORCH_SKIP }, (t) => {
  const xs: number[] = [];
  for (let i = -10 * 32; i <= 10 * 32; i++) xs.push(i / 32);
  const input = Tensor.from(xs, { dtype: "f32" });
  for (const approximate of ["none", "tanh"] as const) {
    const expected = runOracle(TORCH_PYTHON as string, { op: "torch_gelu", xs, approximate, dtype: "float32" });
    const ours = Array.from(input.gelu({ approximate }).data as Float32Array);
    // torch's f32 exact GELU also cancels in the left tail (absolute error
    // ~|x|·2^-24 there), hence the 1e-6 absolute term (|x| <= 10 here).
    const worst = checkClose(ours, expected, xs, `gelu f32(${approximate})`, { rtol: 1e-6, atol: 1e-6, relFloor: 1e-2 });
    t.diagnostic(`gelu approximate=${approximate} vs torch f32: max relative error ${worst.toExponential(2)} where |gelu| >= 1e-2`);
  }
});
