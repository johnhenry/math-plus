/**
 * NumPy differential oracle for the blocked GEMM (issue #121): `matmulInto`
 * on both the SIMD128 and scalar-only paths vs `np.matmul`, on shapes that
 * straddle every block/tile edge of the kernel (MR=4, NR=8, MC=64, KC=256,
 * NC=256) and on transposed (strided) operands.
 *
 * Reuses @johnhenry/math-plus-tensor-core's oracle script and .npy I/O
 * rather than a second copy of either (the canonical-implementation rule).
 * Same resolution and skip-don't-fail contract as tensor-core's suite:
 * $MATH_PLUS_ORACLE_PYTHON, else `python3` on PATH; no numpy → skip. A real
 * verification run must show `skipped 0` (docs/TESTING.md).
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { Kernels } from "../src/index.ts";

// Monorepo-relative on purpose: the oracle script is a test asset of the
// sibling package, not part of its published surface.
const ORACLE_SCRIPT = new URL("../../tensor-core/scripts/numpy_oracle.py", import.meta.url).pathname;

function findOraclePython(): string | undefined {
  for (const candidate of [process.env.MATH_PLUS_ORACLE_PYTHON, "python3"]) {
    if (!candidate) continue;
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
const skip = PYTHON ? false : "no python with numpy found (set MATH_PLUS_ORACLE_PYTHON)";

function det(len: number, seed: number): Float32Array {
  return Float32Array.from({ length: len }, (_, i) => Math.sin(i * 12.9898 + seed * 78.233));
}

function numpyMatmul(dir: string, a: Float32Array, aShape: [number, number], b: Float32Array, bShape: [number, number]): Float32Array {
  const aPath = join(dir, "a.npy");
  const bPath = join(dir, "b.npy");
  const outPath = join(dir, "out.npy");
  writeFileSync(aPath, Tensor.fromTypedArray(a, aShape, { dtype: "f32" }).toNpy());
  writeFileSync(bPath, Tensor.fromTypedArray(b, bShape, { dtype: "f32" }).toNpy());
  const jobPath = join(dir, "job.json");
  writeFileSync(jobPath, JSON.stringify({ op: "matmul", inputs: [aPath, bPath], output: outPath }));
  execFileSync(PYTHON as string, [ORACLE_SCRIPT, jobPath], { stdio: ["ignore", "ignore", "pipe"] });
  const result = Tensor.fromNpy(new Uint8Array(readFileSync(outPath)));
  return Float32Array.from(result.toArray().flat(Infinity) as number[]);
}

// Same tolerance as tensor-core's `matmul:f32` registry entry.
const RTOL = 1e-4;
const ATOL = 1e-5;

test("matmulInto (SIMD128 and scalar paths) matches np.matmul across block edges and transposed operands", { skip }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "tensor-wasm-gemm-"));
  try {
    const withSimd = await Kernels.load();
    const scalarOnly = await Kernels.load(undefined, new Uint8Array([0, 1, 2, 3]));
    assert.equal(withSimd.simdAvailable, true, "the SIMD path must actually be exercised");
    for (const [m, k, n] of [[1, 1, 1], [3, 5, 7], [67, 300, 261], [130, 513, 70]] as const) {
      const aData = det(m * k, m);
      const bData = det(k * n, n);
      const expected = numpyMatmul(dir, aData, [m, k], bData, [k, n]);
      // Physically transposed copies, read back through .transposed().
      const aT = new Float32Array(m * k);
      for (let i = 0; i < m; i++) for (let p = 0; p < k; p++) aT[p * m + i] = aData[i * k + p]!;
      const bT = new Float32Array(k * n);
      for (let p = 0; p < k; p++) for (let j = 0; j < n; j++) bT[j * k + p] = bData[p * n + j]!;

      for (const kernels of [withSimd, scalarOnly]) {
        for (const [ta, tb] of [[false, false], [true, true]] as const) {
          const a = ta ? kernels.fromArray(aT, [k, m]).transposed() : kernels.fromArray(aData, [m, k]);
          const b = tb ? kernels.fromArray(bT, [n, k]).transposed() : kernels.fromArray(bData, [k, n]);
          const out = kernels.zeros([m, n]);
          kernels.matmulInto(out, a, b);
          const got = out.toFloat32Array();
          for (let i = 0; i < expected.length; i++) {
            const e = expected[i]!;
            if (!(Math.abs(got[i]! - e) <= ATOL + RTOL * Math.abs(e))) {
              assert.fail(
                `${kernels.simdAvailable ? "simd" : "scalar"} ${m}x${k}@${k}x${n} tA=${ta} tB=${tb} [${i}]: ${got[i]} vs numpy ${e}`,
              );
            }
          }
          a.free();
          b.free();
          out.free();
        }
      }
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
