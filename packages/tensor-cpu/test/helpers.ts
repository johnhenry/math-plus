/**
 * Shared test plumbing: describe/it over the repo's runtime-neutral harness
 * (test/harness.ts; each test file builds its own Harness with its own
 * `import("bun:test")`, since Bun 1.2 binds tests per importing file) and
 * the batch NumPy oracle (skip-don't-fail).
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import type { HostTensor } from "@johnhenry/tensor-backend";
import type { Harness } from "../../../test/harness.ts";

type Body = () => void | Promise<void>;

export interface TestFns {
  /** Groups tests by name prefix ("outer > inner > test"); `fn` must register synchronously. */
  describe: (name: string, fn: () => void) => void;
  it: (name: string, fn: Body) => void;
  /** Runs once before the file's tests (the harness's file-level `before`). */
  beforeAll: (fn: Body) => void;
  /** `it`, or a skipped test carrying `reason` when `reason` is set (skip, never fail). */
  itUnless: (reason: string | null, name: string, fn: Body) => void;
}

export function testFns(h: Harness): TestFns {
  const prefix: string[] = [];
  const full = (name: string): string => [...prefix, name].join(" > ");
  return {
    describe(name, fn) {
      prefix.push(name);
      try {
        fn();
      } finally {
        prefix.pop();
      }
    },
    it: (name, fn) => h.test(full(name), () => fn()),
    beforeAll: (fn) => h.before(fn),
    itUnless: (reason, name, fn) => h.test(full(name), reason ? { skip: reason } : {}, () => fn()),
  };
}

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

const PYTHON = findOraclePython();
/** Why the NumPy differential tests cannot run here, or null (skip-don't-fail). */
export const oracleSkip: string | null = PYTHON ? null : "no python3 with numpy (set MATH_PLUS_ORACLE_PYTHON)";

const ORACLE_SCRIPT = new URL("../scripts/numpy_oracle.py", import.meta.url).pathname;

export interface OracleJob {
  op: string;
  inputs: HostTensor[];
  args?: Record<string, unknown>;
}

function toTensor(h: HostTensor): Tensor {
  if (h.dtype === "f16" || h.dtype === "bf16") throw new Error("oracle inputs are f32/i32/bool");
  return Tensor.fromTypedArray(h.data as Float32Array | Int32Array | Uint8Array, h.shape, { dtype: h.dtype });
}

/** Runs every backend job through scripts/numpy_oracle.py in ONE Python process; returns NumPy's results in order. */
export function runOracleBatch(jobs: OracleJob[]): Tensor[] {
  return runNpyOracle(ORACLE_SCRIPT, jobs.map((j) => ({ op: j.op, inputs: j.inputs.map(toTensor), args: j.args })));
}

export interface NpyJob {
  op: string;
  inputs: Tensor[];
  args?: Record<string, unknown>;
}

/**
 * Runs every job through the batch oracle `script` in ONE Python process
 * (inputs and outputs exchanged as .npy); returns NumPy's results in order.
 */
export function runNpyOracle(script: string, jobs: NpyJob[]): Tensor[] {
  const dir = mkdtempSync(join(tmpdir(), "tensor-cpu-oracle-"));
  try {
    const spec = jobs.map((j, i) => ({
      op: j.op,
      args: j.args ?? {},
      inputs: j.inputs.map((t, k) => {
        const p = join(dir, `in-${i}-${k}.npy`);
        writeFileSync(p, t.toNpy());
        return p;
      }),
      output: join(dir, `out-${i}.npy`),
    }));
    writeFileSync(join(dir, "jobs.json"), JSON.stringify(spec));
    execFileSync(PYTHON as string, [script, join(dir, "jobs.json")], { stdio: ["ignore", "ignore", "pipe"] });
    return spec.map((j) => Tensor.fromNpy(new Uint8Array(readFileSync(j.output))));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/** Deterministic pseudo-random floats in [lo, hi) (LCG). */
export function seeded(n: number, seed: number, lo = -2, hi = 2): number[] {
  let s = seed >>> 0 || 1;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out.push(lo + ((s >>> 8) / 2 ** 24) * (hi - lo));
  }
  return out;
}

/** |got - want| <= atol + rtol·|want| elementwise, NaN/inf-aware; throws with the worst index. */
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
    if (g === w) continue;
    const excess = Math.abs(g - w) - (atol + rtol * Math.abs(w));
    if (excess > worst) {
      worst = excess;
      worstI = i;
    }
  }
  if (worstI >= 0) throw new Error(`${label}: [${worstI}] got ${got[worstI]} want ${want[worstI]} (atol ${atol}, rtol ${rtol})`);
}
