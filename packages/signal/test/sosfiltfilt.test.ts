import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { butter, sosFilter, sosfiltfilt } from "../src/index.ts";

function sine(n: number, cycles: number): number[] {
  return Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * cycles * i) / n));
}

test("sosfiltfilt: output has the input length and is 1-D f64", () => {
  const sos = butter(4, 0.2, { btype: "lowpass" });
  const out = sosfiltfilt(sos, Tensor.from(sine(200, 5), { dtype: "f64" }));
  assert.deepEqual([...out.shape], [200]);
});

test("sosfiltfilt: a constant signal passes through unchanged (initial conditions remove the edge transient)", () => {
  const sos = butter(4, 0.2, { btype: "lowpass" });
  const out = sosfiltfilt(sos, Tensor.from(new Array(100).fill(3), { dtype: "f64" })).toArray() as number[];
  for (const v of out) assert.ok(Math.abs(v - 3) < 1e-9, `got ${v}`);
});

test("sosfiltfilt: zero phase -- an in-band sine keeps its phase, whereas one-pass sosFilter lags it", () => {
  const sos = butter(2, 0.3, { btype: "lowpass" });
  const x = sine(400, 8); // 8 cycles / 400 samples = 0.02 of Nyquist-normalised band: well inside
  const t = Tensor.from(x, { dtype: "f64" });
  const zp = sosfiltfilt(sos, t).toArray() as number[];
  const one = sosFilter(sos, t).toArray() as number[];
  let errZp = 0;
  let errOne = 0;
  for (let i = 100; i < 300; i++) {
    errZp = Math.max(errZp, Math.abs(zp[i] - x[i]));
    errOne = Math.max(errOne, Math.abs(one[i] - x[i]));
  }
  assert.ok(errZp < 0.01, `zero-phase error ${errZp}`);
  assert.ok(errOne > errZp * 3, `one-pass error ${errOne} should exceed zero-phase ${errZp}`);
});

test("sosfiltfilt: attenuates out-of-band content with squared magnitude response", () => {
  const sos = butter(4, 0.1, { btype: "lowpass" });
  const hi = sine(400, 100); // 0.5 of Nyquist, far above cutoff
  const out = sosfiltfilt(sos, Tensor.from(hi, { dtype: "f64" })).toArray() as number[];
  const peak = Math.max(...out.slice(100, 300).map(Math.abs));
  assert.ok(peak < 1e-3, `peak ${peak}`);
});

test("sosfiltfilt: matches an explicit odd-padded forward-backward reference (scipy defaults)", () => {
  const sos = butter(2, 0.25, { btype: "lowpass" });
  const x = Array.from({ length: 60 }, (_, i) => Math.sin(i / 3) + 0.1 * i);
  const got = sosfiltfilt(sos, Tensor.from(x, { dtype: "f64" })).toArray() as number[];
  // padlen for 1 section with no zero-valued trailing coeffs: 3 * (2*1+1) = 9
  const pad = 9;
  const n = x.length;
  const ext = [
    ...Array.from({ length: pad }, (_, k) => 2 * x[0] - x[pad - k]),
    ...x,
    ...Array.from({ length: pad }, (_, k) => 2 * x[n - 1] - x[n - 2 - k]),
  ];
  // steady-state initial conditions for a unit step, scaled by the first sample
  const [b0, b1, b2, a0, a1, a2] = sos[0];
  const g = (b0 + b1 + b2) / (a0 + a1 + a2);
  const z2 = b2 - a2 * g;
  const z1 = b1 - a1 * g + z2;
  const run = (sig: number[]): number[] => {
    let s1 = z1 * sig[0];
    let s2 = z2 * sig[0];
    return sig.map((v) => {
      const y = b0 * v + s1;
      s1 = b1 * v - a1 * y + s2;
      s2 = b2 * v - a2 * y;
      return y;
    });
  };
  const fwd = run(ext);
  const back = run([...fwd].reverse()).reverse();
  const want = back.slice(pad, pad + n);
  for (let i = 0; i < n; i++) assert.ok(Math.abs(got[i] - want[i]) < 1e-9, `index ${i}: ${got[i]} vs ${want[i]}`);
});

test("sosfiltfilt: rejects signals not longer than padlen, non-1-D input and empty sos", () => {
  const sos = butter(4, 0.2, { btype: "lowpass" });
  assert.throws(() => sosfiltfilt(sos, Tensor.from([1, 2, 3], { dtype: "f64" })), RangeError);
  assert.throws(() => sosfiltfilt(sos, Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2])), RangeError);
  assert.throws(() => sosfiltfilt([], Tensor.from(sine(50, 2), { dtype: "f64" })), RangeError);
});

test("sosfiltfilt: padlen option can be 0 (no padding) and must be < length", () => {
  const sos = butter(2, 0.25, { btype: "lowpass" });
  const t = Tensor.from(sine(50, 2), { dtype: "f64" });
  assert.equal(sosfiltfilt(sos, t, { padlen: 0 }).shape[0], 50);
  assert.throws(() => sosfiltfilt(sos, t, { padlen: 50 }), RangeError);
});
