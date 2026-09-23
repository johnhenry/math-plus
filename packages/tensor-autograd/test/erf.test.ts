/**
 * The internal erf/erfc behind exact GELU (src/erf.ts, pending tensor-core's
 * canonical erf in #122) against Python's `math.erf`/`math.erfc` (C libm).
 * Skip-don't-fail: $MATH_PLUS_ORACLE_PYTHON, else `python3` on PATH.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { test } from "node:test";
import { erf, erfc, geluExact, geluExactDerivative } from "../src/erf.ts";

function findPython(): string | undefined {
  for (const c of [process.env.MATH_PLUS_ORACLE_PYTHON, "python3"].filter((x): x is string => Boolean(x))) {
    try {
      execFileSync(c, ["-c", "import math"], { stdio: "ignore" });
      return c;
    } catch {
      // next
    }
  }
  return undefined;
}
const PY = findPython();
const skip = PY ? false : "no python3 found (set MATH_PLUS_ORACLE_PYTHON)";

const xs = [
  ...Array.from({ length: 801 }, (_, i) => (i - 400) / 37),
  0, 1e-8, -1e-8, 2.4999999, 2.5, 2.5000001, 5.9, 12, 26, -26,
];

test("erf/erfc/exact-GELU match Python's math module", { skip }, () => {
  const script = `
import json, math, sys
xs = json.load(sys.stdin)
print(json.dumps([[math.erf(x), math.erfc(x), 0.5 * x * math.erfc(-x / math.sqrt(2)),
                   0.5 * math.erfc(-x / math.sqrt(2)) + x * math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)] for x in xs]))`;
  const rows = JSON.parse(execFileSync(PY as string, ["-c", script], { input: JSON.stringify(xs), encoding: "utf8" })) as number[][];
  xs.forEach((x, i) => {
    const [e, c, g, dg] = rows[i] as number[];
    assert.ok(Math.abs(erf(x) - (e as number)) <= 1e-15 * Math.max(1, Math.abs(e as number)), `erf(${x}) = ${erf(x)} vs ${e}`);
    assert.ok(Math.abs(erfc(x) - (c as number)) <= 2e-16 + 1e-12 * (c as number), `erfc(${x}) = ${erfc(x)} vs ${c}`);
    assert.ok(Math.abs(geluExact(x) - (g as number)) <= 1e-15 * Math.max(1, Math.abs(x)), `gelu(${x}) = ${geluExact(x)} vs ${g}`);
    assert.ok(Math.abs(geluExactDerivative(x) - (dg as number)) <= 1e-14, `gelu'(${x}) = ${geluExactDerivative(x)} vs ${dg}`);
  });
});

test("erf/erfc: NaN in, NaN out; symmetric; saturates", () => {
  assert.ok(Number.isNaN(erf(Number.NaN)) && Number.isNaN(erfc(Number.NaN)));
  assert.equal(erf(-1.3), -erf(1.3));
  assert.equal(erf(40), 1);
  assert.equal(erfc(-40), 2);
  assert.equal(erfc(40), 0);
});
