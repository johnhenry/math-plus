/**
 * resamplePoly (issue #44). Per resample.ts's own doc comment, this uses a
 * Hamming-windowed FIR (not scipy's default Kaiser(beta=5)), so per-sample
 * output isn't expected to bit-match scipy.signal.resample_poly -- these
 * tests verify RECONSTRUCTION QUALITY instead (does a resampled sinusoid
 * still represent the same underlying signal at the new rate, within a
 * real tolerance), which is the property that actually matters.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { resamplePoly } from "../src/index.ts";
import { SCIPY_SKIP_REASON, runScipyOracle } from "./helpers.ts";

test("resamplePoly: output length matches ceil(len(x) * up / down)", () => {
  const x = Tensor.from(Array.from({ length: 100 }, (_, i) => i), { dtype: "f64" });
  assert.equal(resamplePoly(x, 3, 2).shape[0], Math.ceil((100 * 3) / 2));
  assert.equal(resamplePoly(x, 1, 3).shape[0], Math.ceil(100 / 3));
  assert.equal(resamplePoly(x, 5, 1).shape[0], 500);
});

test("resamplePoly: up=1, down=1 is a no-op (identity)", () => {
  const x = [1, 2, 3, 4, 5];
  const out = resamplePoly(Tensor.from(x, { dtype: "f64" }), 1, 1).toArray() as number[];
  assert.deepEqual(out, x);
});

test("resamplePoly: upsampling a low-frequency sinusoid preserves its frequency and amplitude", () => {
  const n = 400;
  const freq = 0.05; // cycles/sample at the original rate, well within the passband after 2x upsampling
  const original = Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * freq * i));
  const up = 2;
  const down = 1;
  const resampled = resamplePoly(Tensor.from(original, { dtype: "f64" }), up, down).toArray() as number[];

  assert.equal(resampled.length, n * up);

  // The resampled signal should still be sin(2*pi*freq*i/up) at the new sample rate
  // (same underlying continuous sinusoid, now sampled `up` times as densely).
  // Compare over the middle region, away from filter-transient edges.
  const start = 100;
  const end = resampled.length - 100;
  let maxErr = 0;
  for (let i = start; i < end; i++) {
    const expected = Math.sin(2 * Math.PI * freq * (i / up));
    maxErr = Math.max(maxErr, Math.abs((resampled[i] as number) - expected));
  }
  assert.ok(maxErr < 0.05, `max reconstruction error too large: ${maxErr}`);
});

test("resamplePoly: downsampling a low-frequency sinusoid (well below the new Nyquist) preserves it", () => {
  const n = 400;
  const freq = 0.05; // at the ORIGINAL rate
  const original = Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * freq * i));
  const up = 1;
  const down = 2;
  // After downsampling by 2, the new Nyquist (in original-rate cycles/sample) is 0.25 -- freq=0.05 is safely within it.
  const resampled = resamplePoly(Tensor.from(original, { dtype: "f64" }), up, down).toArray() as number[];

  assert.equal(resampled.length, Math.ceil(n / down));

  const start = 50;
  const end = resampled.length - 50;
  let maxErr = 0;
  for (let i = start; i < end; i++) {
    const expected = Math.sin(2 * Math.PI * freq * (i * down));
    maxErr = Math.max(maxErr, Math.abs((resampled[i] as number) - expected));
  }
  assert.ok(maxErr < 0.05, `max reconstruction error too large: ${maxErr}`);
});

test("resamplePoly: a rational (non-integer-ratio) up/down still preserves a well-within-band sinusoid", () => {
  const n = 300;
  const freq = 0.03;
  const original = Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * freq * i));
  const up = 3;
  const down = 2;
  const resampled = resamplePoly(Tensor.from(original, { dtype: "f64" }), up, down).toArray() as number[];

  const start = 80;
  const end = resampled.length - 80;
  let maxErr = 0;
  for (let i = start; i < end; i++) {
    const expected = Math.sin(2 * Math.PI * freq * ((i * down) / up));
    maxErr = Math.max(maxErr, Math.abs((resampled[i] as number) - expected));
  }
  assert.ok(maxErr < 0.05, `max reconstruction error too large: ${maxErr}`);
});

test("resamplePoly: up/down are reduced by their GCD (6/4 behaves like 3/2)", () => {
  const x = Tensor.from(Array.from({ length: 50 }, (_, i) => Math.sin(i * 0.1)), { dtype: "f64" });
  const a = resamplePoly(x, 6, 4);
  const b = resamplePoly(x, 3, 2);
  assert.equal(a.shape[0], b.shape[0]);
  const aArr = a.toArray() as number[];
  const bArr = b.toArray() as number[];
  for (let i = 0; i < aArr.length; i++) {
    assert.ok(Math.abs((aArr[i] as number) - (bArr[i] as number)) < 1e-9, `index ${i}`);
  }
});

test("resamplePoly: rejects a non-1-D Tensor and non-positive-integer up/down", () => {
  const bad2d = Tensor.zeros([2, 2], { dtype: "f64" });
  assert.throws(() => resamplePoly(bad2d, 2, 1), RangeError);
  const x = Tensor.from([1, 2, 3], { dtype: "f64" });
  assert.throws(() => resamplePoly(x, 0, 1), RangeError);
  assert.throws(() => resamplePoly(x, 1, -1), RangeError);
  assert.throws(() => resamplePoly(x, 1.5, 1), RangeError);
});

// Issue #113: the identity path (up === down after GCD reduction) returned the
// input's raw storage labeled f64 -- for an f32 input, a Float32Array under a
// "f64" dtype, which downstream dtype-branching ops misread.
test("resamplePoly: identity path on an f32 input returns a genuine f64 tensor with the input's values (#113)", () => {
  const values = [0.1, -2.5, 3.25, 1e-3, 7];
  const f32 = Tensor.from(values, { dtype: "f32" });
  const expected = Array.from(Float32Array.from(values)); // the f32-rounded inputs, widened exactly
  for (const [up, down] of [
    [1, 1],
    [3, 3],
  ] as const) {
    const out = resamplePoly(f32, up, down);
    assert.equal(out.dtype, "f64", `${up}/${down}: dtype label`);
    assert.ok(out.data instanceof Float64Array, `${up}/${down}: storage must be a Float64Array, got ${out.data.constructor.name}`);
    assert.deepEqual(out.toArray(), expected, `${up}/${down}: values`);
    // A downstream op that trusts the label (the #113 failure mode) sees the right values.
    assert.deepEqual(out.add(Tensor.zeros([values.length], { dtype: "f64" })).toArray(), expected);
  }
  // The identity result is a copy: mutating it never touches the input.
  const out = resamplePoly(f32, 1, 1);
  (out.data as Float64Array)[0] = 999;
  assert.equal((f32.toArray() as number[])[0], Math.fround(0.1));
});

test("resamplePoly: identity path on f16 and i32 inputs decodes values rather than reinterpreting storage (#113)", () => {
  const f16 = Tensor.from([1.5, -0.25, 2], { dtype: "f16" });
  assert.deepEqual(resamplePoly(f16, 2, 2).toArray(), [1.5, -0.25, 2]);
  const i32 = Tensor.from([3, -4, 5], { dtype: "i32" });
  const out = resamplePoly(i32, 1, 1);
  assert.equal(out.dtype, "f64");
  assert.deepEqual(out.toArray(), [3, -4, 5]);
});

test(
  "resamplePoly: identity path matches scipy.signal.resample_poly(x, n, n) on f32 input (#113)",
  { skip: SCIPY_SKIP_REASON },
  () => {
    const values = Array.from({ length: 16 }, (_, i) => Math.sin(i * 0.7) * 3);
    const f32 = Tensor.from(values, { dtype: "f32" });
    for (const n of [1, 4]) {
      const scipy = runScipyOracle<{ y: number[]; dtype: string }>({ op: "resample_poly", x: values, dtype: "float32", up: n, down: n });
      assert.equal(scipy.dtype, "float32"); // scipy keeps f32; this package documents f64-everywhere output
      const out = resamplePoly(f32, n, n);
      assert.equal(out.dtype, "f64");
      assert.deepEqual(out.toArray(), scipy.y, `up=down=${n}`);
    }
  },
);
