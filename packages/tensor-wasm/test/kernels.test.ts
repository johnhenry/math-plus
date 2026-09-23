import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { Kernels } from "../src/index.ts";

test("addInto writes the sum directly into a pre-allocated resident buffer", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array([1, 2, 3, 4]), [4]);
  const b = kernels.fromArray(new Float32Array([10, 20, 30, 40]), [4]);
  const out = kernels.zeros([4]);
  kernels.addInto(out, a, b);
  assert.deepEqual([...out.toFloat32Array()], [11, 22, 33, 44]);
  a.free();
  b.free();
  out.free();
});

test("mulInto writes the product directly into a pre-allocated resident buffer", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array([2, 3, 4]), [3]);
  const b = kernels.fromArray(new Float32Array([5, 6, 7]), [3]);
  const out = kernels.zeros([3]);
  kernels.mulInto(out, a, b);
  assert.deepEqual([...out.toFloat32Array()], [10, 18, 28]);
});

test("matmulInto matches a hand-computed 2x3 @ 3x2 product", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
  const b = kernels.fromArray(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]);
  const out = kernels.zeros([2, 2]);
  kernels.matmulInto(out, a, b);
  assert.deepEqual([...out.toFloat32Array()], [58, 64, 139, 154]);
});

test("matmulInto on a transposed view reads via strides — no copy needed", async () => {
  const kernels = await Kernels.load();
  // Store A^T directly (3x2); .transposed() reads it as (2x3) via strides.
  const aT = kernels.fromArray(new Float32Array([1, 4, 2, 5, 3, 6]), [3, 2]);
  const b = kernels.fromArray(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]);
  const out = kernels.zeros([2, 2]);
  kernels.matmulInto(out, aT.transposed(), b);
  assert.deepEqual([...out.toFloat32Array()], [58, 64, 139, 154]);
});

test("matmulInto rejects inner-dimension mismatch and non-2-D operands", async () => {
  const kernels = await Kernels.load();
  const a = kernels.zeros([2, 3]);
  const bad = kernels.zeros([4, 2]);
  const out = kernels.zeros([2, 2]);
  assert.throws(() => kernels.matmulInto(out, a, bad), RangeError);
  const oneD = kernels.zeros([3]);
  assert.throws(() => kernels.matmulInto(out, oneD, bad), RangeError);
});

// ---- the M2 acceptance criteria (issue #3) ---------------------------------

test("addInto allocates ZERO times across repeated calls on resident buffers", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array(1000).fill(1.5), [1000]);
  const b = kernels.fromArray(new Float32Array(1000).fill(2.5), [1000]);
  const out = kernels.zeros([1000]);
  const before = kernels.allocCallCount;
  for (let i = 0; i < 500; i++) {
    kernels.addInto(out, a, b);
  }
  assert.equal(
    kernels.allocCallCount,
    before,
    "addInto must not call alloc — the whole point of the ...Into interface",
  );
  assert.equal(out.toFloat32Array()[0], 4);
});

// The two benchmark-threshold tests (addInto-vs-pure-JS and SIMD-vs-scalar)
// live in benchmark.bench-test.ts, run in a separate serial `node --test`
// pass after this suite -- see that file's header and issue #49.

// ---- telemetry hook (issue #10) --------------------------------------------

test("allocator emits wasm/alloc.bytes and wasm/alloc.calls metrics when a sink is installed", async () => {
  const { setSink } = await import("@johnhenry/math-plus-telemetry");
  const events: unknown[] = [];
  setSink((e) => events.push(e));
  try {
    const kernels = await Kernels.load();
    kernels.zeros([10]); // one alloc call
    assert.equal(events.length, 2); // bytes + calls, per alloc
    const names = events.map((e) => (e as { name: string }).name);
    assert.deepEqual(names, ["wasm/alloc.bytes", "wasm/alloc.calls"]);
  } finally {
    setSink(null);
  }
});

test("allocator emits NOTHING when no sink is installed (default, zero-cost)", async () => {
  const kernels = await Kernels.load();
  kernels.zeros([10]);
  kernels.zeros([20]);
  // No assertion possible on "no events" without a sink to observe them --
  // this test's job is just to prove it doesn't throw/misbehave by default.
  assert.equal(kernels.allocCallCount, 2);
});

// ---- SIMD128 fast path (issue #13) -----------------------------------------

test("simdAvailable is true on this runtime (Node supports WASM SIMD) — the fast path is actually exercised, not silently skipped", async () => {
  const kernels = await Kernels.load();
  assert.equal(kernels.simdAvailable, true);
});

test("Kernels.load() degrades gracefully when the SIMD artifact is unusable — simdAvailable false, addInto still correct via the scalar fallback", async () => {
  const kernels = await Kernels.load(undefined, new Uint8Array([0, 1, 2, 3])); // not a valid WASM module
  assert.equal(kernels.simdAvailable, false);
  const a = kernels.fromArray(new Float32Array([1, 2, 3, 4]), [4]);
  const b = kernels.fromArray(new Float32Array([10, 20, 30, 40]), [4]);
  const out = kernels.zeros([4]);
  kernels.addInto(out, a, b);
  assert.deepEqual([...out.toFloat32Array()], [11, 22, 33, 44]);
});

test("addInto/mulInto: SIMD-accelerated result is bit-for-bit identical to the scalar fallback, contiguous case", async () => {
  const withSimd = await Kernels.load();
  const scalarOnly = await Kernels.load(undefined, new Uint8Array([0, 1, 2, 3]));
  assert.equal(withSimd.simdAvailable, true);
  assert.equal(scalarOnly.simdAvailable, false);

  const N = 4001; // deliberately not a multiple of 4 -- exercises the SIMD kernel's scalar tail loop
  const aData = Float32Array.from({ length: N }, (_, i) => Math.sin(i) * 100);
  const bData = Float32Array.from({ length: N }, (_, i) => Math.cos(i) * 50);

  for (const [name, op] of [
    ["addInto", (k: Kernels, out: ReturnType<Kernels["zeros"]>, a: ReturnType<Kernels["zeros"]>, b: ReturnType<Kernels["zeros"]>) => k.addInto(out, a, b)],
    ["mulInto", (k: Kernels, out: ReturnType<Kernels["zeros"]>, a: ReturnType<Kernels["zeros"]>, b: ReturnType<Kernels["zeros"]>) => k.mulInto(out, a, b)],
  ] as const) {
    const aSimd = withSimd.fromArray(aData, [N]);
    const bSimd = withSimd.fromArray(bData, [N]);
    const outSimd = withSimd.zeros([N]);
    op(withSimd, outSimd, aSimd, bSimd);

    const aScalar = scalarOnly.fromArray(aData, [N]);
    const bScalar = scalarOnly.fromArray(bData, [N]);
    const outScalar = scalarOnly.zeros([N]);
    op(scalarOnly, outScalar, aScalar, bScalar);

    assert.deepEqual([...outSimd.toFloat32Array()], [...outScalar.toFloat32Array()], `${name}: SIMD vs scalar mismatch`);
  }
});

test("addInto: a non-contiguous (strided) view is NOT eligible for the SIMD path but still computes correctly via the fallback", async () => {
  const kernels = await Kernels.load();
  assert.equal(kernels.simdAvailable, true);
  // Every other element of a 6-element buffer -- view1D reports stride=2,
  // disqualifying it from the contiguous-only SIMD kernel (flatSpec's
  // stride === 1 check in addInto/mulInto).
  const full = kernels.fromArray(new Float32Array([1, 10, 2, 20, 3, 30]), [6]);
  const strided = full.view1D(0, 3, 2); // offset=0, length=3, stride=2 -> [1, 2, 3]
  const b = kernels.fromArray(new Float32Array([100, 200, 300]), [3]);
  const out = kernels.zeros([3]);
  kernels.addInto(out, strided, b);
  assert.deepEqual([...out.toFloat32Array()], [101, 202, 303]);
});

// The SIMD-vs-scalar benchmark test moved to benchmark.bench-test.ts (its
// own serial `node --test` pass) -- see that file's header and issue #49.

// ---- subInto/divInto kernel parity with add/mul (issue #66) ---------------

test("subInto writes the difference directly into a pre-allocated resident buffer", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array([10, 20, 30, 40]), [4]);
  const b = kernels.fromArray(new Float32Array([1, 2, 3, 4]), [4]);
  const out = kernels.zeros([4]);
  kernels.subInto(out, a, b);
  assert.deepEqual([...out.toFloat32Array()], [9, 18, 27, 36]);
});

test("divInto writes the quotient directly into a pre-allocated resident buffer", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array([10, 20, 30]), [3]);
  const b = kernels.fromArray(new Float32Array([2, 4, 5]), [3]);
  const out = kernels.zeros([3]);
  kernels.divInto(out, a, b);
  assert.deepEqual([...out.toFloat32Array()], [5, 5, 6]);
});

test("subInto/divInto: a non-contiguous (strided) view computes correctly (no SIMD path exists yet for these, so this is the only code path)", async () => {
  const kernels = await Kernels.load();
  const full = kernels.fromArray(new Float32Array([100, 10, 200, 20, 300, 30]), [6]);
  const strided = full.view1D(0, 3, 2); // [100, 200, 300]
  const b = kernels.fromArray(new Float32Array([1, 2, 3]), [3]);
  const outSub = kernels.zeros([3]);
  kernels.subInto(outSub, strided, b);
  assert.deepEqual([...outSub.toFloat32Array()], [99, 198, 297]);
  const outDiv = kernels.zeros([3]);
  kernels.divInto(outDiv, strided, b);
  assert.deepEqual([...outDiv.toFloat32Array()], [100, 100, 100]);
});

test("divInto matches IEEE 754 division-by-zero semantics (±Infinity/NaN), same as JS/Tensor.div — not a trap", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array([1, -1, 0]), [3]);
  const b = kernels.fromArray(new Float32Array([0, 0, 0]), [3]);
  const out = kernels.zeros([3]);
  kernels.divInto(out, a, b);
  const [pos, neg, nan] = out.toFloat32Array();
  assert.equal(pos, Infinity);
  assert.equal(neg, -Infinity);
  assert.ok(Number.isNaN(nan));
  assert.equal(kernels.poisoned, false, "division by zero must not poison the instance");
});

test("subInto/divInto allocate ZERO times across repeated calls on resident buffers", async () => {
  const kernels = await Kernels.load();
  const a = kernels.fromArray(new Float32Array(1000).fill(9), [1000]);
  const b = kernels.fromArray(new Float32Array(1000).fill(3), [1000]);
  const out = kernels.zeros([1000]);
  const before = kernels.allocCallCount;
  for (let i = 0; i < 500; i++) {
    kernels.subInto(out, a, b);
    kernels.divInto(out, a, b);
  }
  assert.equal(kernels.allocCallCount, before);
  assert.equal(out.toFloat32Array()[0], 3);
});

test("subInto/divInto agree with @johnhenry/math-plus-tensor-core's Tensor.sub/Tensor.div on random data (differential leg)", async () => {
  const kernels = await Kernels.load();
  const N = 257; // deliberately not a round SIMD-friendly size
  const aData = Float32Array.from({ length: N }, (_, i) => Math.sin(i) * 37 + 50); // stays away from 0
  const bData = Float32Array.from({ length: N }, (_, i) => Math.cos(i) * 11 + 20); // stays away from 0

  const a = kernels.fromArray(aData, [N]);
  const b = kernels.fromArray(bData, [N]);
  const outSub = kernels.zeros([N]);
  const outDiv = kernels.zeros([N]);
  kernels.subInto(outSub, a, b);
  kernels.divInto(outDiv, a, b);

  const ta = Tensor.fromTypedArray(aData, [N], { dtype: "f32" });
  const tb = Tensor.fromTypedArray(bData, [N], { dtype: "f32" });
  const expectedSub = ta.sub(tb).toArray() as number[];
  const expectedDiv = ta.div(tb).toArray() as number[];

  const gotSub = [...outSub.toFloat32Array()];
  const gotDiv = [...outDiv.toFloat32Array()];
  for (let i = 0; i < N; i++) {
    assert.ok(Math.abs(gotSub[i]! - expectedSub[i]!) < 1e-4, `sub[${i}]: ${gotSub[i]} vs ${expectedSub[i]}`);
    assert.ok(Math.abs(gotDiv[i]! - expectedDiv[i]!) < 1e-4, `div[${i}]: ${gotDiv[i]} vs ${expectedDiv[i]}`);
  }
});
