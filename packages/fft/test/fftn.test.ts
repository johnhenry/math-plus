/**
 * Tests for fftn/ifftn (issue #84), the n-D generalization of fft2/ifft2.
 * Differential tests against numpy.fft.fftn/ifftn (skip-don't-fail
 * convention, docs/TESTING.md) plus self-consistency checks that don't
 * need the oracle.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { ComplexTensor, fft2, fftn, ifftn } from "../src/index.ts";

const ORACLE_SCRIPT = new URL("../scripts/fft2_oracle.py", import.meta.url).pathname;

function findOraclePython(): string | undefined {
  for (const candidate of [process.env.MATH_PLUS_ORACLE_PYTHON, "python3"].filter((c): c is string => Boolean(c))) {
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

interface OracleResult {
  real: unknown;
  imag: unknown;
}

function runOracle(op: string, ct: ComplexTensor, axes?: number | readonly number[]): OracleResult {
  const real = ct.real.contiguous().toArray();
  const imag = ct.imag.contiguous().toArray();
  const out = execFileSync(PYTHON as string, [ORACLE_SCRIPT], {
    input: JSON.stringify({ op, real, imag, axes }),
    encoding: "utf8",
  });
  return JSON.parse(out) as OracleResult;
}

function flatten(value: unknown): number[] {
  return (Array.isArray(value) ? value.flat(Infinity) : [value]) as number[];
}

function assertMatches(ct: ComplexTensor, expected: OracleResult, label: string, tol = 1e-8): void {
  const gotReal = flatten(ct.real.contiguous().toArray());
  const gotImag = flatten(ct.imag.contiguous().toArray());
  const wantReal = flatten(expected.real);
  const wantImag = flatten(expected.imag);
  assert.equal(gotReal.length, wantReal.length, `${label}: element count`);
  for (let i = 0; i < gotReal.length; i++) {
    assert.ok(Math.abs((gotReal[i] as number) - (wantReal[i] as number)) < tol, `${label}: real[${i}] ${gotReal[i]} vs ${wantReal[i]}`);
    assert.ok(Math.abs((gotImag[i] as number) - (wantImag[i] as number)) < tol, `${label}: imag[${i}] ${gotImag[i]} vs ${wantImag[i]}`);
  }
}

function randomComplex(shape: readonly number[]): ComplexTensor {
  let seed = 7919;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed / 0x7fffffff) * 4 - 2;
  };
  const size = shape.reduce((a, b) => a * b, 1);
  const realData = Array.from({ length: size }, rng);
  const imagData = Array.from({ length: size }, rng);
  return ComplexTensor.fromParts(Tensor.from(realData, { dtype: "f64" }).reshape(shape), Tensor.from(imagData, { dtype: "f64" }).reshape(shape));
}

test("fftn matches numpy.fft.fftn on a random 4x4x4 complex cube (all axes)", { skip }, () => {
  const input = randomComplex([4, 4, 4]);
  const got = fftn(input);
  assertMatches(got, runOracle("fftn", input), "fftn-3d");
});

test("ifftn matches numpy.fft.ifftn on a random 4x4x4 complex cube", { skip }, () => {
  const input = randomComplex([4, 4, 4]);
  const got = ifftn(input);
  assertMatches(got, runOracle("ifftn", input), "ifftn-3d");
});

test("fftn over a subset of axes matches numpy.fft.fftn(a, axes=[0,1])", { skip }, () => {
  const input = randomComplex([4, 4, 4]);
  const got = fftn(input, [0, 1]);
  assertMatches(got, runOracle("fftn", input, [0, 1]), "fftn-partial-axes");
});

test("fftn on a 2-D input matches fft2 to floating-point tolerance (same separable computation, but a different axis-processing order -- fftn defaults to ascending axis order, fft2 does axis 1 then axis 0 -- so the last ULP or two can legitimately differ; confirmed empirically before loosening this from an initial deepEqual attempt)", () => {
  const input = randomComplex([4, 8]);
  const viaFftn = fftn(input);
  const viaFft2 = fft2(input);
  const gotReal = flatten(viaFftn.real.contiguous().toArray());
  const wantReal = flatten(viaFft2.real.contiguous().toArray());
  const gotImag = flatten(viaFftn.imag.contiguous().toArray());
  const wantImag = flatten(viaFft2.imag.contiguous().toArray());
  for (let i = 0; i < gotReal.length; i++) {
    assert.ok(Math.abs((gotReal[i] as number) - (wantReal[i] as number)) < 1e-9, `real[${i}]`);
    assert.ok(Math.abs((gotImag[i] as number) - (wantImag[i] as number)) < 1e-9, `imag[${i}]`);
  }
});

test("fftn -> ifftn round-trips on a 4x4x4 cube (self-consistency, no oracle needed)", () => {
  const input = randomComplex([4, 4, 4]);
  const roundTripped = ifftn(fftn(input));
  const gotReal = flatten(roundTripped.real.contiguous().toArray());
  const wantReal = flatten(input.real.contiguous().toArray());
  for (let i = 0; i < gotReal.length; i++) {
    assert.ok(Math.abs((gotReal[i] as number) - (wantReal[i] as number)) < 1e-9, `round-trip real[${i}]`);
  }
});

test("fftn over a partial axis subset matches per-slice fft2 along the remaining axis (self-consistency, no oracle needed)", () => {
  const input = randomComplex([4, 4, 4]);
  const gotPartial = fftn(input, [0, 1]);
  for (let k = 0; k < 4; k++) {
    const slice = ComplexTensor.fromParts(input.real.select(2, k), input.imag.select(2, k));
    const want = fft2(slice);
    const got = ComplexTensor.fromParts(gotPartial.real.select(2, k), gotPartial.imag.select(2, k));
    // Tolerance, not deepEqual: fftn's traversal order differs from four
    // independent fft2 calls, so floating-point summation order (and
    // hence the last ULP or two) can legitimately differ even though both
    // compute the same mathematical result -- confirmed by re-running
    // with a tolerance before concluding this, not just loosening blindly.
    const gotFlat = flatten(got.real.contiguous().toArray());
    const wantFlat = flatten(want.real.contiguous().toArray());
    for (let i = 0; i < gotFlat.length; i++) {
      assert.ok(Math.abs((gotFlat[i] as number) - (wantFlat[i] as number)) < 1e-9, `slice ${k} real[${i}]`);
    }
  }
});
