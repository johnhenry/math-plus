/**
 * Differential tests for fft2/ifft2/fftshift/ifftshift (issue #69) against
 * NumPy's numpy.fft.fft2/ifft2/fftshift/ifftshift -- skip-don't-fail
 * convention (docs/TESTING.md).
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { ComplexNumber } from "@johnhenry/math-plus-scalar-types";
import { ComplexTensor, fft2, fftshift, ifft2, ifftshift } from "../src/index.ts";

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
  real: number[][];
  imag: number[][];
}

function runOracle(op: string, ct: ComplexTensor, axes?: number | readonly number[]): OracleResult {
  const real = ct.real.contiguous().toArray() as number[][];
  const imag = ct.imag.contiguous().toArray() as number[][];
  const out = execFileSync(PYTHON as string, [ORACLE_SCRIPT], {
    input: JSON.stringify({ op, real, imag, axes }),
    encoding: "utf8",
  });
  return JSON.parse(out) as OracleResult;
}

function assertMatches(ct: ComplexTensor, expected: OracleResult, label: string, tol = 1e-9): void {
  const gotReal = ct.real.contiguous().toArray() as number[][];
  const gotImag = ct.imag.contiguous().toArray() as number[][];
  assert.equal(gotReal.length, expected.real.length, `${label}: row count`);
  for (let i = 0; i < gotReal.length; i++) {
    for (let j = 0; j < (gotReal[i] as number[]).length; j++) {
      const gr = (gotReal[i] as number[])[j] as number;
      const er = (expected.real[i] as number[])[j] as number;
      const gi = (gotImag[i] as number[])[j] as number;
      const ei = (expected.imag[i] as number[])[j] as number;
      assert.ok(Math.abs(gr - er) < tol, `${label}: real[${i}][${j}] ${gr} vs ${er}`);
      assert.ok(Math.abs(gi - ei) < tol, `${label}: imag[${i}][${j}] ${gi} vs ${ei}`);
    }
  }
}

function randomComplex(rows: number, cols: number): ComplexTensor {
  const values: ComplexNumber[] = [];
  let seed = 7919;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed / 0x7fffffff) * 4 - 2;
  };
  for (let i = 0; i < rows * cols; i++) values.push(new ComplexNumber(rng(), rng()));
  const flat = ComplexTensor.fromComplexArray(values);
  return ComplexTensor.fromParts(flat.real.reshape([rows, cols]), flat.imag.reshape([rows, cols]));
}

test("fft2 matches numpy.fft.fft2 on a random 4x8 complex input", { skip }, () => {
  const input = randomComplex(4, 8);
  const got = fft2(input);
  assertMatches(got, runOracle("fft2", input), "fft2", 1e-8);
});

test("ifft2 matches numpy.fft.ifft2 on a random 8x4 complex input", { skip }, () => {
  const input = randomComplex(8, 4);
  const got = ifft2(input);
  assertMatches(got, runOracle("ifft2", input), "ifft2", 1e-8);
});

test("fft2 -> ifft2 round-trips (self-consistency, no oracle needed)", () => {
  const input = randomComplex(4, 4);
  const roundTripped = ifft2(fft2(input));
  const gotReal = roundTripped.real.contiguous().toArray() as number[][];
  const wantReal = input.real.contiguous().toArray() as number[][];
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      assert.ok(
        Math.abs((gotReal[i] as number[])[j]! - (wantReal[i] as number[])[j]!) < 1e-9,
        `round-trip real[${i}][${j}]`,
      );
    }
  }
});

test("fft2 rejects non-2-D input", () => {
  const oneD = ComplexTensor.fromComplexArray([new ComplexNumber(1, 0), new ComplexNumber(2, 0)]);
  assert.throws(() => fft2(oneD), RangeError);
});

test("fftshift matches numpy.fft.fftshift, default (all axes) and explicit axis, even AND odd lengths", { skip }, () => {
  for (const [rows, cols] of [
    [4, 8],
    [5, 7],
  ] as const) {
    const input = randomComplex(rows, cols);
    assertMatches(fftshift(input), runOracle("fftshift", input), `fftshift-all-${rows}x${cols}`);
    assertMatches(fftshift(input, 1), runOracle("fftshift", input, [1]), `fftshift-axis1-${rows}x${cols}`);
  }
});

test("ifftshift matches numpy.fft.ifftshift, default and explicit axis, even AND odd lengths", { skip }, () => {
  for (const [rows, cols] of [
    [4, 8],
    [5, 7],
  ] as const) {
    const input = randomComplex(rows, cols);
    assertMatches(ifftshift(input), runOracle("ifftshift", input), `ifftshift-all-${rows}x${cols}`);
    assertMatches(ifftshift(input, 0), runOracle("ifftshift", input, [0]), `ifftshift-axis0-${rows}x${cols}`);
  }
});

test("fftshift and ifftshift are exact inverses for EVEN-length axes (agree for odd too, per NumPy's own convention, at the whole-array level)", () => {
  const input = randomComplex(4, 6);
  const roundTripped = ifftshift(fftshift(input));
  const gotReal = roundTripped.real.contiguous().toArray() as number[][];
  const wantReal = input.real.contiguous().toArray() as number[][];
  assert.deepEqual(gotReal, wantReal);
});

test("fftshift on a hand-verifiable 1-row case: moves the zero-frequency bin to the center", () => {
  // [0,1,2,3,4,5,6,7] fftshift -> [4,5,6,7,0,1,2,3] (shift by floor(8/2)=4)
  const values = Array.from({ length: 8 }, (_, i) => new ComplexNumber(i, 0));
  const flat = ComplexTensor.fromComplexArray(values);
  const asRow = ComplexTensor.fromParts(flat.real.reshape([1, 8]), flat.imag.reshape([1, 8]));
  const shifted = fftshift(asRow, 1);
  assert.deepEqual(shifted.real.contiguous().toArray(), [[4, 5, 6, 7, 0, 1, 2, 3]]);
});
