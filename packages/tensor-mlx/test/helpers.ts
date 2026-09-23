/**
 * Shared test plumbing: runtime-neutral describe/it (Node's node:test or
 * Bun's bun:test), the skip-don't-fail MLX gate, and the NumPy oracle.
 */
import * as nodeTest from "node:test";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { mlxUnavailableReason } from "../src/index.ts";

type ItFn = (name: string, fn: () => void | Promise<void>) => unknown;

export interface TestFns {
  describe: (name: string, fn: () => void) => unknown;
  it: ItFn;
  beforeAll: (fn: () => void | Promise<void>) => unknown;
  /** `it`, or a skipped test carrying `reason` when `reason` is set (skip, never fail). */
  itUnless: (reason: string | null, name: string, fn: () => void | Promise<void>) => void;
}

/**
 * node:test, or bun:test when the CALLING FILE passes its own `bun:test`
 * module: Bun 1.2's node:test shim only registers tests from the first file
 * of a run, and bun:test binds describe/it per importing file (issue #127),
 * so each test file must do the `import("bun:test")` itself.
 */
export function testFns(bunTest: any): TestFns {
  const rawIt: ItFn & { skip: ItFn } = bunTest?.it ?? nodeTest.it;
  return {
    describe: bunTest?.describe ?? nodeTest.describe,
    it: rawIt,
    beforeAll: bunTest?.beforeAll ?? nodeTest.before,
    itUnless(reason, name, fn) {
      if (reason) rawIt.skip(`${name} (skipped: ${reason})`, () => {});
      else rawIt(name, fn);
    },
  };
}

/** Why MLX tests cannot run here (non-darwin-arm64 or no libmlxc), or null. */
export const mlxSkip: string | null = mlxUnavailableReason();

// ---- NumPy oracle (docs/TESTING.md): $MATH_PLUS_ORACLE_PYTHON, else python3 ----

function findOraclePython(): string | undefined {
  for (const candidate of [process.env.MATH_PLUS_ORACLE_PYTHON, "python3"]) {
    if (!candidate) continue;
    try {
      execFileSync(candidate, ["-c", "import numpy"], { stdio: "ignore" });
      return candidate;
    } catch {
      // try the next one
    }
  }
  return undefined;
}

export const PYTHON = findOraclePython();
export const oracleSkip: string | null = PYTHON ? null : "no python3 with numpy (set MATH_PLUS_ORACLE_PYTHON)";

const ORACLE_SCRIPT = new URL("../scripts/numpy_oracle.py", import.meta.url).pathname;

export interface OracleCase {
  op: string;
  inputs: Tensor[];
  args?: Record<string, unknown>;
}

/** Runs every case in ONE Python process; returns NumPy's results in order. */
export function runOracleBatch(cases: OracleCase[]): Tensor[] {
  const dir = mkdtempSync(join(tmpdir(), "tensor-mlx-oracle-"));
  try {
    const jobs = cases.map((c, i) => {
      const inputs = c.inputs.map((t, j) => {
        const p = join(dir, `in-${i}-${j}.npy`);
        writeFileSync(p, t.toNpy());
        return p;
      });
      return { op: c.op, inputs, args: c.args ?? {}, output: join(dir, `out-${i}.npy`) };
    });
    const jobPath = join(dir, "jobs.json");
    writeFileSync(jobPath, JSON.stringify(jobs));
    execFileSync(PYTHON as string, [ORACLE_SCRIPT, jobPath], { stdio: ["ignore", "ignore", "pipe"] });
    return jobs.map((j) => Tensor.fromNpy(new Uint8Array(readFileSync(j.output))));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/** Deterministic pseudo-random floats in [lo, hi) (LCG; no dependency on test ordering). */
export function seeded(n: number, seed: number, lo = -2, hi = 2): number[] {
  let s = seed >>> 0 || 1;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out.push(lo + ((s >>> 8) / 2 ** 24) * (hi - lo));
  }
  return out;
}

/** |got - want| <= atol + rtol·|want| elementwise, NaN-aware; throws with the worst index. */
export function assertClose(got: ArrayLike<number>, want: ArrayLike<number>, atol: number, rtol: number, label: string): void {
  if (got.length !== want.length) throw new Error(`${label}: length ${got.length} != ${want.length}`);
  let worst = 0;
  let worstI = -1;
  for (let i = 0; i < got.length; i++) {
    const g = got[i]!;
    const w = want[i]!;
    if (Number.isNaN(g) || Number.isNaN(w)) {
      if (Number.isNaN(g) !== Number.isNaN(w)) throw new Error(`${label}: NaN mismatch at ${i} (got ${g}, want ${w})`);
      continue;
    }
    if (g === w) continue; // covers matching infinities
    const excess = Math.abs(g - w) - (atol + rtol * Math.abs(w));
    if (excess > worst) {
      worst = excess;
      worstI = i;
    }
  }
  if (worstI >= 0) throw new Error(`${label}: [${worstI}] got ${got[worstI]} want ${want[worstI]} (atol ${atol}, rtol ${rtol})`);
}

/** A tensor-core f16 Tensor (raw IEEE bits in a Uint16Array) holding `values` rounded to f16. */
export function f16Tensor(values: readonly number[], shape: readonly number[]): Tensor {
  const half = Float16Array.from(values);
  return Tensor.fromTypedArray(new Uint16Array(half.buffer), shape, { dtype: "f16" });
}

/** The f32 tensor of `values` after rounding each to f16 (the oracle's view of an f16 input). */
export function f16RoundedF32(values: readonly number[], shape: readonly number[]): Tensor {
  return Tensor.fromTypedArray(Float32Array.from(Float16Array.from(values)), shape, { dtype: "f32" });
}
