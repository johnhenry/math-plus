/**
 * Differential oracles for the canonical erf/erfc/GELU (src/special.ts, issue #122):
 *
 * - SciPy `special.erf` / `special.erfc` over [-6, 6] (dense) plus both tails
 *   out to erfc's f64 underflow, and `x·special.ndtr(x)` as a tail-accurate
 *   exact-GELU reference.
 * - PyTorch `torch.nn.functional.gelu` in BOTH `approximate` modes, f64 and f32.
 *
 * Same skip-don't-fail contract as the other Python oracles (docs/TESTING.md):
 * SciPy resolves via $MATH_PLUS_SCIPY_ORACLE_PYTHON, else
 * $MATH_PLUS_ORACLE_PYTHON, else `python3`; PyTorch via
 * $MATH_PLUS_TORCH_ORACLE_PYTHON, else $MATH_PLUS_ORACLE_PYTHON, else
 * `python3`. A real verification run must show 0 skipped.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { erf, erfc, geluErf, Tensor } from "../src/index.ts";

const ORACLE_SCRIPT = new URL("../scripts/special_oracle.py", import.meta.url).pathname;

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

const SCIPY_PYTHON = findPython(
  [process.env.MATH_PLUS_SCIPY_ORACLE_PYTHON, process.env.MATH_PLUS_ORACLE_PYTHON],
  "import scipy.special, numpy",
);
const TORCH_PYTHON = findPython(
  [process.env.MATH_PLUS_TORCH_ORACLE_PYTHON, process.env.MATH_PLUS_ORACLE_PYTHON],
  "import torch",
);
const SCIPY_SKIP = SCIPY_PYTHON ? false : "no python with scipy found (set MATH_PLUS_SCIPY_ORACLE_PYTHON)";
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

const SMALLEST_NORMAL = 2 ** -1022;

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

/** [-6, 6] every 1/256 (exact binary fractions), the |x| = 1 seam, tiny |x|, and 2000 random points in [-6, 6]. */
function coreGrid(): number[] {
  const xs: number[] = [];
  for (let i = -6 * 256; i <= 6 * 256; i++) xs.push(i / 256);
  const eps = Number.EPSILON;
  xs.push(1 - eps / 2, 1 + eps, -1 + eps / 2, -1 - eps, 1e-300, -1e-300, 1e-20, 5e-9, -3e-5);
  const rng = mulberry32(122);
  for (let i = 0; i < 2000; i++) xs.push(rng() * 12 - 6);
  return xs;
}

/** Both tails: |x| in [6, 27.25] every 1/16 (erfc underflows to 0 just past 27.2). */
function tailGrid(): number[] {
  const xs: number[] = [];
  for (let i = 6 * 16; i <= 27.25 * 16; i++) xs.push(i / 16, -i / 16);
  return xs;
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

test("erf matches scipy.special.erf over [-6, 6] and both tails", { skip: SCIPY_SKIP }, (t) => {
  const xs = [...coreGrid(), ...tailGrid()];
  const expected = runOracle(SCIPY_PYTHON as string, { op: "erf", xs });
  const worst = checkClose(xs.map(erf), expected, xs, "erf", { rtol: 2e-15 });
  t.diagnostic(`erf vs scipy: max relative error ${worst.toExponential(2)} over ${xs.length} points`);
  // Tensor.erf() is the same function elementwise.
  const viaTensor = Array.from(Tensor.from(xs, { dtype: "f64" }).erf().data as Float64Array);
  assert.deepEqual(viaTensor, xs.map(erf));
});

test("erfc matches scipy.special.erfc over [-6, 6] and both tails (relative, down to underflow)", { skip: SCIPY_SKIP }, (t) => {
  const xs = [...coreGrid(), ...tailGrid()];
  const expected = runOracle(SCIPY_PYTHON as string, { op: "erfc", xs });
  // Relative bound everywhere erfc is a normal f64. Below the smallest normal
  // (x > ~26.55) SciPy flushes erfc to 0 while ours underflows gradually
  // through the subnormals, so those points get an absolute bound of one
  // smallest-normal instead.
  const worst = checkClose(xs.map(erfc), expected, xs, "erfc", { rtol: 5e-15, atol: SMALLEST_NORMAL, relFloor: SMALLEST_NORMAL });
  t.diagnostic(`erfc vs scipy: max relative error ${worst.toExponential(2)} (normal-range results) over ${xs.length} points`);
  const viaTensor = Array.from(Tensor.from(xs, { dtype: "f64" }).erfc().data as Float64Array);
  assert.deepEqual(viaTensor, xs.map(erfc));
});

test("exact GELU matches x·scipy.special.ndtr(x) to full relative precision, including the far-left tail", { skip: SCIPY_SKIP }, (t) => {
  const xs: number[] = [];
  for (let i = -38 * 16; i <= 12 * 16; i++) xs.push(i / 16); // GELU underflows to -0 just past x ≈ -38.5
  const expected = runOracle(SCIPY_PYTHON as string, { op: "gelu_ndtr", xs });
  const ours = xs.map(geluErf);
  // Same subnormal caveat as erfc above (ndtr flushes to 0 first). In the far
  // left tail SciPy's ndtr is itself the less accurate side: it rounds x/√2
  // before erfc, costing ~x²·2^-53 relative (checked against 50-digit mpmath
  // during development: at x = -36.6875 ours is within 1e-16, ndtr is off by
  // 2.3e-13). So the bound grows with x² exactly as the ORACLE's error does.
  let worst = 0;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i]!;
    const rtol = 1e-14 + 4 * x * x * 2 ** -53;
    worst = Math.max(worst, checkClose([ours[i]!], [expected[i]!], [x], "gelu", { rtol, atol: SMALLEST_NORMAL, relFloor: 1e-300 }));
  }
  t.diagnostic(`exact gelu vs x*ndtr(x) over [-38, 12]: max relative difference ${worst.toExponential(2)} (dominated by ndtr's own x²·ε tail error)`);
  // Where ndtr is accurate (|x| <= 6) the agreement is at full precision.
  let worstCore = 0;
  for (let i = 0; i < xs.length; i++) {
    if (Math.abs(xs[i]!) <= 6) worstCore = Math.max(worstCore, Math.abs(ours[i]! - expected[i]!) / Math.max(Math.abs(expected[i]!), 1e-300));
  }
  assert.ok(worstCore < 1e-14, `exact gelu vs x*ndtr(x) on [-6, 6]: ${worstCore}`);
  t.diagnostic(`exact gelu vs x*ndtr(x) on [-6, 6]: max relative error ${worstCore.toExponential(2)}`);
});

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
    // abs/rel; our own tail accuracy is pinned by the ndtr test above.
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
