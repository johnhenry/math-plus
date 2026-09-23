/**
 * butter (issue #44). Section-by-section SOS coefficients are NOT expected
 * to match scipy.signal.butter's exactly (scipy's own zero/pole-to-section
 * grouping isn't fixed either -- verified empirically: for some orders it
 * pairs 2 zeros with the real pole and 1 with each complex pair, for
 * others the opposite) -- so these tests verify END-TO-END FILTERING
 * BEHAVIOR (apply our own sosFilter to our own butter() output, compare
 * against scipy's sosfilt applied to scipy's own butter() output, on the
 * SAME input signal), which is invariant to section grouping and is the
 * property that actually matters.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { butter, sosFilter, type FilterType } from "../src/index.ts";
import { runScipyOracle, SCIPY_SKIP_REASON } from "./helpers.ts";

function close(a: ArrayLike<number>, b: ArrayLike<number>, tol: number): void {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs((a[i] as number) - (b[i] as number)) < tol, `index ${i}: ${a[i]} vs ${b[i]}`);
}

function testSignal(n: number): number[] {
  return Array.from({ length: n }, (_, i) => Math.sin(i * 0.3) + 0.3 * Math.sin(i * 1.7) + 0.1 * Math.sin(i * 2.9));
}

/** `butter`'s real signature is overloaded (`wn`'s shape depends statically on `btype`) so a caller gets compile-time enforcement -- these generic test loops exercise it with heterogeneous runtime data instead, which needs the loose, un-overloaded shape. */
const butterAny = butter as (order: number, wn: number | readonly [number, number], options: { btype: FilterType }) => ReturnType<typeof butter>;

const CASES: Array<{ order: number; wn: number | readonly [number, number]; btype: FilterType }> = [
  { order: 1, wn: 0.1, btype: "lowpass" },
  { order: 2, wn: 0.3, btype: "lowpass" },
  { order: 3, wn: 0.25, btype: "lowpass" },
  { order: 4, wn: 0.6, btype: "highpass" },
  { order: 5, wn: 0.4, btype: "highpass" },
  { order: 6, wn: 0.5, btype: "lowpass" },
  // Bandpass/bandstop (issue #90) -- both even and odd orders, since the
  // prototype's single real pole (odd order only) can transform into
  // either a complex-conjugate pair OR two real poles depending on
  // bandwidth/center-frequency, a data-dependent shape only the general
  // zpk2sos (not the old lowpass/highpass-only specialization) handles.
  { order: 2, wn: [0.2, 0.5], btype: "bandpass" },
  { order: 3, wn: [0.15, 0.35], btype: "bandpass" },
  { order: 4, wn: [0.3, 0.6], btype: "bandpass" },
  { order: 2, wn: [0.2, 0.5], btype: "bandstop" },
  { order: 3, wn: [0.15, 0.35], btype: "bandstop" },
  { order: 4, wn: [0.3, 0.6], btype: "bandstop" },
];

for (const { order, wn, btype } of CASES) {
  test(`butter(${order}, ${JSON.stringify(wn)}, ${btype}): filtered output matches scipy end-to-end`, { skip: SCIPY_SKIP_REASON }, () => {
    const sos = butterAny(order, wn, { btype });
    const x = testSignal(30);
    const mine = sosFilter(sos, Tensor.from(x, { dtype: "f64" })).toArray() as number[];

    const oracleSos = runScipyOracle<{ sos: number[][] }>({ op: "butter_sos", order, wn, btype }).sos;
    const oracleFiltered = runScipyOracle<{ y: number[] }>({ op: "sosfilt", sos: oracleSos, x }).y;

    close(mine, oracleFiltered, 1e-6);
  });
}

test("butter: an even order produces order/2 sections, an odd order produces (order+1)/2 sections", () => {
  assert.equal(butter(2, 0.3, { btype: "lowpass" }).length, 1);
  assert.equal(butter(4, 0.3, { btype: "lowpass" }).length, 2);
  assert.equal(butter(3, 0.3, { btype: "lowpass" }).length, 2);
  assert.equal(butter(5, 0.3, { btype: "lowpass" }).length, 3);
});

test("butter: lowpass attenuates a high-frequency sinusoid far more than a low-frequency one", () => {
  const sos = butter(4, 0.2, { btype: "lowpass" });
  const n = 200;
  const lowFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 0.05)); // well within passband
  const highFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 2.5)); // well into stopband

  const lowOut = sosFilter(sos, Tensor.from(lowFreq, { dtype: "f64" })).toArray() as number[];
  const highOut = sosFilter(sos, Tensor.from(highFreq, { dtype: "f64" })).toArray() as number[];

  // Compare steady-state (post-transient) RMS amplitude vs the input's own.
  const rms = (arr: number[]) => Math.sqrt(arr.reduce((s, v) => s + v * v, 0) / arr.length);
  const lowRatio = rms(lowOut.slice(100)) / rms(lowFreq.slice(100));
  const highRatio = rms(highOut.slice(100)) / rms(highFreq.slice(100));

  assert.ok(lowRatio > 0.9, `passband signal should pass through near-unchanged, ratio=${lowRatio}`);
  assert.ok(highRatio < 0.1, `stopband signal should be strongly attenuated, ratio=${highRatio}`);
});

test("butter: highpass attenuates a low-frequency sinusoid far more than a high-frequency one", () => {
  const sos = butter(4, 0.3, { btype: "highpass" });
  const n = 200;
  const lowFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 0.02));
  const highFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 2.8));

  const lowOut = sosFilter(sos, Tensor.from(lowFreq, { dtype: "f64" })).toArray() as number[];
  const highOut = sosFilter(sos, Tensor.from(highFreq, { dtype: "f64" })).toArray() as number[];

  const rms = (arr: number[]) => Math.sqrt(arr.reduce((s, v) => s + v * v, 0) / arr.length);
  const lowRatio = rms(lowOut.slice(100)) / rms(lowFreq.slice(100));
  const highRatio = rms(highOut.slice(100)) / rms(highFreq.slice(100));

  assert.ok(lowRatio < 0.1, `stopband signal should be strongly attenuated, ratio=${lowRatio}`);
  assert.ok(highRatio > 0.9, `passband signal should pass through near-unchanged, ratio=${highRatio}`);
});

// ---- bandpass/bandstop (issue #90) -------------------------------------------

test("butter: bandpass/bandstop of order N always produces exactly N sections (2N poles, whether the odd-order leftover real prototype pole becomes a complex pair or two real poles)", () => {
  for (const btype of ["bandpass", "bandstop"] as const) {
    for (const order of [2, 3, 4, 5, 6]) {
      assert.equal(butterAny(order, [0.2, 0.5], { btype }).length, order, `${btype} order ${order}`);
    }
  }
});

test("butter: bandpass passes a mid-band frequency through near-unchanged, strongly attenuates both a low and a high frequency", () => {
  const sos = butterAny(4, [0.2, 0.5], { btype: "bandpass" });
  const n = 300;
  const lowFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 0.1)); // below the passband
  const midFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 1.0)); // inside [0.2*pi, 0.5*pi] ~ [0.63, 1.57]
  const highFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 2.8)); // above the passband

  const rms = (arr: number[]) => Math.sqrt(arr.reduce((s, v) => s + v * v, 0) / arr.length);
  const ratio = (input: number[]) => rms((sosFilter(sos, Tensor.from(input, { dtype: "f64" })).toArray() as number[]).slice(150)) / rms(input.slice(150));

  const lowRatio = ratio(lowFreq);
  const midRatio = ratio(midFreq);
  const highRatio = ratio(highFreq);

  assert.ok(midRatio > 0.9, `passband signal should pass through near-unchanged, ratio=${midRatio}`);
  assert.ok(lowRatio < 0.1, `below-passband signal should be strongly attenuated, ratio=${lowRatio}`);
  assert.ok(highRatio < 0.1, `above-passband signal should be strongly attenuated, ratio=${highRatio}`);
});

test("butter: bandstop attenuates a mid-band frequency, passes both a low and a high frequency near-unchanged", () => {
  const sos = butterAny(4, [0.2, 0.5], { btype: "bandstop" });
  const n = 300;
  const lowFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 0.1));
  const midFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 1.0));
  const highFreq = Array.from({ length: n }, (_, i) => Math.sin(i * 2.8));

  const rms = (arr: number[]) => Math.sqrt(arr.reduce((s, v) => s + v * v, 0) / arr.length);
  const ratio = (input: number[]) => rms((sosFilter(sos, Tensor.from(input, { dtype: "f64" })).toArray() as number[]).slice(150)) / rms(input.slice(150));

  const lowRatio = ratio(lowFreq);
  const midRatio = ratio(midFreq);
  const highRatio = ratio(highFreq);

  assert.ok(midRatio < 0.1, `stopband signal should be strongly attenuated, ratio=${midRatio}`);
  assert.ok(lowRatio > 0.9, `below-stopband signal should pass through near-unchanged, ratio=${lowRatio}`);
  assert.ok(highRatio > 0.9, `above-stopband signal should pass through near-unchanged, ratio=${highRatio}`);
});

test("butter: rejects malformed wn for bandpass/bandstop (not a 2-tuple, wn[0] >= wn[1], or either endpoint outside (0,1))", () => {
  assert.throws(() => butterAny(2, 0.5, { btype: "bandpass" }), RangeError); // plain number, not a tuple
  assert.throws(() => butterAny(2, [0.5, 0.2], { btype: "bandpass" }), RangeError); // wn[0] > wn[1]
  assert.throws(() => butterAny(2, [0.3, 0.3], { btype: "bandpass" }), RangeError); // wn[0] === wn[1]
  assert.throws(() => butterAny(2, [0, 0.5], { btype: "bandstop" }), RangeError); // wn[0] <= 0
  assert.throws(() => butterAny(2, [0.2, 1], { btype: "bandstop" }), RangeError); // wn[1] >= 1
});

test("butter: rejects a plain number wn for lowpass/highpass only via the type system, but a stray tuple at runtime still throws", () => {
  assert.throws(() => butterAny(2, [0.2, 0.5], { btype: "lowpass" }), RangeError);
});

test("butter: rejects a non-positive-integer order and a wn outside (0, 1)", () => {
  assert.throws(() => butter(0, 0.5, { btype: "lowpass" }), RangeError);
  assert.throws(() => butter(1.5, 0.5, { btype: "lowpass" }), RangeError);
  assert.throws(() => butter(2, 0, { btype: "lowpass" }), RangeError);
  assert.throws(() => butter(2, 1, { btype: "lowpass" }), RangeError);
  assert.throws(() => butter(2, 1.5, { btype: "lowpass" }), RangeError);
});
