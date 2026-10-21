/**
 * scipy.signal oracle helper (issue #44). Same skip-don't-fail convention
 * as tensor-core's numpy_oracle.py: if no Python with scipy is found, tests
 * using this SKIP rather than fail. On trycooy:
 * `nix-shell -p "python3.withPackages(ps: [ps.scipy ps.numpy])"` provides
 * one (pip wheels don't work on NixOS) -- see docs/TESTING.md.
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const ORACLE_SCRIPT = new URL("../scripts/scipy_oracle.py", import.meta.url).pathname;

function findOraclePython(): string | undefined {
  const candidates = [process.env.MATH_PLUS_SCIPY_ORACLE_PYTHON, process.env.MATH_PLUS_ORACLE_PYTHON, "python3"].filter(
    (c): c is string => Boolean(c),
  );
  for (const candidate of candidates) {
    try {
      execFileSync(candidate, ["-c", "import scipy.signal, numpy"], { stdio: "ignore" });
      return candidate;
    } catch {
      // try next
    }
  }
  return undefined;
}

export const SCIPY_PYTHON = findOraclePython();
export const SCIPY_SKIP_REASON = SCIPY_PYTHON ? undefined : "no python with scipy found (set MATH_PLUS_SCIPY_ORACLE_PYTHON)";

export function runScipyOracle<T>(job: Record<string, unknown>): T {
  if (!SCIPY_PYTHON) throw new Error("runScipyOracle called without a resolved python (check SCIPY_SKIP_REASON first)");
  const dir = mkdtempSync(join(tmpdir(), "math-plus-signal-oracle-"));
  try {
    const jobPath = join(dir, "job.json");
    writeFileSync(jobPath, JSON.stringify(job));
    const stdout = execFileSync(SCIPY_PYTHON, [ORACLE_SCRIPT, jobPath], { stdio: ["ignore", "pipe", "pipe"], maxBuffer: 256 * 1024 * 1024 });
    return JSON.parse(stdout.toString("utf8")) as T;
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
