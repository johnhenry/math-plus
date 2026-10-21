import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { dft } from "../src/dft.ts";

function naive(re: number[], im: number[], inverse: boolean): { re: number[]; im: number[] } {
  const n = re.length;
  const sign = inverse ? 1 : -1;
  const outRe: number[] = [];
  const outIm: number[] = [];
  for (let k = 0; k < n; k++) {
    let sr = 0;
    let si = 0;
    for (let j = 0; j < n; j++) {
      const a = (sign * 2 * Math.PI * ((j * k) % n)) / n;
      sr += (re[j] as number) * Math.cos(a) - (im[j] as number) * Math.sin(a);
      si += (re[j] as number) * Math.sin(a) + (im[j] as number) * Math.cos(a);
    }
    outRe.push(inverse ? sr / n : sr);
    outIm.push(inverse ? si / n : si);
  }
  return { re: outRe, im: outIm };
}

test("dft: matches the O(n^2) definition for power-of-two, prime, odd and composite lengths, forward and inverse", () => {
  for (const n of [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 17, 37, 64, 97, 100, 127, 250]) {
    const re = Array.from({ length: n }, (_, i) => Math.sin(i * 1.3) + 0.2 * i);
    const im = Array.from({ length: n }, (_, i) => Math.cos(i * 0.7));
    for (const inverse of [false, true]) {
      const want = naive(re, im, inverse);
      const got = dft(Float64Array.from(re), Float64Array.from(im), inverse);
      let scale = 0;
      for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(want.re[i] as number), Math.abs(want.im[i] as number));
      for (let i = 0; i < n; i++) {
        assert.ok(Math.abs((got.re[i] as number) - (want.re[i] as number)) <= 1e-12 * scale, `n=${n} inverse=${inverse} re[${i}]`);
        assert.ok(Math.abs((got.im[i] as number) - (want.im[i] as number)) <= 1e-12 * scale, `n=${n} inverse=${inverse} im[${i}]`);
      }
    }
  }
});

test("dft: does not mutate its input", () => {
  const re = Float64Array.from([1, 2, 3, 4, 5]);
  const im = new Float64Array(5);
  dft(re, im, false);
  assert.deepEqual(Array.from(re), [1, 2, 3, 4, 5]);
});

test("dft: rejects mismatched or empty input", () => {
  assert.throws(() => dft(new Float64Array(3), new Float64Array(4), false), RangeError);
  assert.throws(() => dft(new Float64Array(0), new Float64Array(0), false), RangeError);
});
