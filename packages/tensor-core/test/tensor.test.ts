import assert from "node:assert/strict";
import { makeTest, spyMethod } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import {
  BYTES_PER_ELEMENT,
  Tensor,
  broadcastShapes,
  isBigIntDType,
  random,
} from "../src/index.ts";

// ---- constructors ----------------------------------------------------------

test("zeros allocates the right storage", () => {
  const t = Tensor.zeros([2, 3]);
  assert.equal(t.size, 6);
  assert.equal(t.ndim, 2);
  assert.equal(t.dtype, "f32");
  assert.deepEqual([...t.shape], [2, 3]);
  assert.deepEqual([...t.strides], [3, 1]);
  assert.equal(t.at(1, 2), 0);
});

test("ones / full / arange / fromTypedArray", () => {
  assert.equal(Tensor.ones([2, 2]).at(1, 1), 1);
  assert.equal(Tensor.full([3], 7, { dtype: "i32" }).at(2), 7);
  assert.deepEqual(Tensor.arange(0, 10, 2).toArray(), [0, 2, 4, 6, 8]);
  assert.deepEqual(Tensor.arange(3).toArray(), [0, 1, 2]);
  const backing = new Float64Array([1, 2, 3, 4]);
  const t = Tensor.fromTypedArray(backing, [2, 2], { dtype: "f64" });
  assert.equal(t.data, backing); // no copy
  assert.equal(t.at(1, 0), 3);
});

test("from builds a 1-D tensor with negative indexing", () => {
  const t = Tensor.from([1, 2, 3], { dtype: "f64" });
  assert.equal(t.at(0), 1);
  assert.equal(t.at(-1), 3);
  assert.throws(() => t.at(3), RangeError);
});

test("i64 dtype is BigInt-backed (ONNX input_ids case)", () => {
  const t = Tensor.from([101, 2023, 102], { dtype: "i64" });
  assert.equal(t.at(0), 101n);
  assert.equal(typeof t.at(1), "bigint");
  assert.ok(isBigIntDType(t.dtype));
  assert.equal(BYTES_PER_ELEMENT.i64, 8);
});

test("shape is frozen — no mutation of tensor metadata", () => {
  const t = Tensor.zeros([2, 2]);
  assert.throws(() => {
    (t.shape as number[])[0] = 99;
  }, TypeError);
});

// ---- views: identity semantics ---------------------------------------------

test("permute is a view: shares storage, never copies", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  const p = t.permute([1, 0]);
  assert.equal(p.data, t.data); // pointer identity — same backing buffer
  assert.deepEqual([...p.shape], [3, 2]);
  assert.deepEqual([...p.strides], [1, 3]);
  assert.equal(p.at(2, 1), t.at(1, 2));
  assert.equal(p.isContiguous, false);
});

test("transpose defaults to reversing axes", () => {
  const t = Tensor.zeros([2, 3, 4]);
  const tt = t.transpose();
  assert.deepEqual([...tt.shape], [4, 3, 2]);
  assert.equal(tt.data, t.data);
});

test("contiguous() returns `this` when already contiguous", () => {
  const t = Tensor.from([1, 2, 3, 4]).reshape([2, 2]);
  assert.equal(t.contiguous(), t); // identical object, not a copy
});

test("contiguous() packs a non-contiguous view into new storage", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  const p = t.permute([1, 0]);
  const c = p.contiguous();
  assert.notEqual(c.data, t.data); // new buffer
  assert.equal(c.isContiguous, true);
  assert.deepEqual(c.toArray(), [
    [1, 4],
    [2, 5],
    [3, 6],
  ]);
});

// Issue #99: flatten()/roll()/#cumulative() used to route through
// contiguous() unconditionally, which copies even when the source is
// already contiguous but merely doesn't own its whole buffer (nonzero
// offset, or size < data.length) — a case reshape() already handled
// correctly by checking `isContiguous` alone. `select()` on a leading axis
// is a convenient way to build exactly that kind of view: contiguous, but
// offset by a full "page" into a larger backing buffer.
test("flatten()/roll()/cumsum() reuse an already-contiguous view instead of copying, even with a non-zero offset", () => {
  const t = Tensor.arange(24).reshape([2, 2, 2, 3]);
  const sub = t.select(0, 1); // shape [2,2,3]; contiguous; offset=12; doesn't span the whole buffer
  assert.equal(sub.isContiguous, true);
  assert.notEqual(sub.offset, 0);
  assert.notEqual(sub.size, sub.data.length);

  // Baseline established by the audit: reshape() already avoids copying here.
  assert.equal(sub.reshape([sub.size]).data, sub.data);

  // Regression: flatten() previously called contiguous() unconditionally,
  // which (unlike reshape()) additionally requires offset === 0 and a
  // fully-occupied buffer, so it copied `sub` even though it was already
  // contiguous.
  const flattenSpy = spyMethod(sub, "contiguous");
  assert.equal(sub.flatten().data, sub.data);
  assert.equal(flattenSpy.callCount(), 0, "flatten() should not call contiguous() on an already-contiguous view");
  flattenSpy.restore();

  // roll() (no axis) flattens first; that intermediate step should reuse
  // `sub`'s storage rather than pre-copying (the final take()-based gather
  // still allocates fresh output — that part is unavoidable and untested
  // here; differential.test.ts covers roll()'s correctness).
  const rollSpy = spyMethod(sub, "contiguous");
  sub.roll(1);
  assert.equal(rollSpy.callCount(), 0, "roll() should not call contiguous() on an already-contiguous view");
  rollSpy.restore();

  // cumsum()/cumprod() (axis omitted) flatten via the same fast path
  // (cumsum()'s own output is always a fresh, freshly-allocated tensor —
  // it's the redundant pre-copy of `sub` that this test targets).
  const cumsumSpy = spyMethod(sub, "contiguous");
  sub.cumsum();
  assert.equal(cumsumSpy.callCount(), 0, "cumsum() should not call contiguous() on an already-contiguous view");
  cumsumSpy.restore();
});

test("flatten()/roll()/cumsum() still copy when the source is genuinely non-contiguous", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const p = t.permute([1, 0]); // non-contiguous
  assert.equal(p.isContiguous, false);

  const flattenSpy = spyMethod(p, "contiguous");
  const flattened = p.flatten();
  assert.equal(flattenSpy.callCount(), 1);
  assert.deepEqual(flattened.toArray(), [1, 4, 2, 5, 3, 6]);
  flattenSpy.restore();

  const rollSpy = spyMethod(p, "contiguous");
  p.roll(1);
  assert.equal(rollSpy.callCount(), 1);
  rollSpy.restore();

  const cumsumSpy = spyMethod(p, "contiguous");
  assert.deepEqual(p.cumsum().toArray(), [1, 5, 7, 12, 15, 21]);
  assert.equal(cumsumSpy.callCount(), 1);
  cumsumSpy.restore();
});

test("reshape supports -1 inference and rejects non-contiguous input", () => {
  const t = Tensor.arange(12);
  const r = t.reshape([-1, 4]);
  assert.deepEqual([...r.shape], [3, 4]);
  assert.equal(r.data, t.data); // view
  assert.throws(() => t.reshape([-1, -1]), RangeError);
  assert.throws(() => t.reshape([5, 2]), RangeError);
  const nonContig = t.reshape([3, 4]).permute([1, 0]);
  assert.throws(() => nonContig.reshape([12]), TypeError); // no implicit copy
});

// ---- broadcasting elementwise ops -------------------------------------------

test("broadcastShapes follows NumPy trailing-axis rules", () => {
  assert.deepEqual(broadcastShapes([2, 3], [3]), [2, 3]);
  assert.deepEqual(broadcastShapes([2, 1], [1, 3]), [2, 3]);
  assert.deepEqual(broadcastShapes([], [4]), [4]);
  assert.throws(() => broadcastShapes([2, 3], [4]), RangeError);
});

test("add/sub/mul/div with broadcasting", () => {
  const a = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const row = Tensor.from([10, 20, 30], { dtype: "f64" });
  assert.deepEqual(a.add(row).toArray(), [
    [11, 22, 33],
    [14, 25, 36],
  ]);
  assert.deepEqual(a.sub(1).toArray(), [
    [0, 1, 2],
    [3, 4, 5],
  ]);
  assert.deepEqual(a.mul(a).toArray(), [
    [1, 4, 9],
    [16, 25, 36],
  ]);
  assert.deepEqual(a.div(2).toArray(), [
    [0.5, 1, 1.5],
    [2, 2.5, 3],
  ]);
});

test("elementwise ops on NON-CONTIGUOUS views are correct", () => {
  // Risk register #2: stride bugs produce plausible-but-wrong values.
  const a = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const at = a.permute([1, 0]); // [[1,4],[2,5],[3,6]], strides [1,3]
  const b = Tensor.from([10, 100], { dtype: "f64" }); // broadcasts along rows
  assert.deepEqual(at.add(b).toArray(), [
    [11, 104],
    [12, 105],
    [13, 106],
  ]);
  const col = Tensor.from([1, 2, 3], { dtype: "f64" }).reshape([3, 1]);
  assert.deepEqual(at.mul(col).toArray(), [
    [1, 4],
    [4, 10],
    [9, 18],
  ]);
});

test("dtype mismatch throws (no implicit promotion in M1)", () => {
  const a = Tensor.from([1], { dtype: "f32" });
  const b = Tensor.from([1], { dtype: "f64" });
  assert.throws(() => a.add(b), TypeError);
});

test("i64 elementwise: add/sub/mul work, div directs to cast", () => {
  const a = Tensor.from([10, 20], { dtype: "i64" });
  const b = Tensor.from([3, 4], { dtype: "i64" });
  assert.deepEqual(a.add(b).toArray(), [13n, 24n]);
  assert.deepEqual(a.mul(b).toArray(), [30n, 80n]);
  assert.throws(() => a.div(b), TypeError);
});

// ---- reductions --------------------------------------------------------------

test("sum/mean over all elements", () => {
  const t = Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2]);
  assert.equal(t.sum().item(), 10);
  assert.equal(t.mean().item(), 2.5);
});

test("sum/mean along an axis (incl. negative axis)", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  assert.deepEqual(t.sum(0).toArray(), [5, 7, 9]);
  assert.deepEqual(t.sum(1).toArray(), [6, 15]);
  assert.deepEqual(t.sum(-1).toArray(), [6, 15]);
  assert.deepEqual(t.mean(0).toArray(), [2.5, 3.5, 4.5]);
});

test("reductions on non-contiguous views", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const p = t.permute([1, 0]); // [[1,4],[2,5],[3,6]]
  assert.deepEqual(p.sum(1).toArray(), [5, 7, 9]);
  assert.equal(p.sum().item(), 21);
});

test("mean of integer dtypes returns f64 (NumPy semantics)", () => {
  const t = Tensor.from([1, 2], { dtype: "i32" });
  const m = t.mean();
  assert.equal(m.dtype, "f64");
  assert.equal(m.item(), 1.5);
  const big = Tensor.from([3, 4], { dtype: "i64" });
  assert.equal(big.mean().dtype, "f64");
  assert.equal(big.mean().item(), 3.5);
  assert.equal(big.sum().dtype, "i64");
  assert.equal(big.sum().item(), 7n);
});

// ---- .npy I/O -----------------------------------------------------------------

test(".npy round-trip across dtypes", () => {
  for (const dtype of ["f32", "f64", "i32", "i64", "u8", "bool"] as const) {
    const t = Tensor.from([1, 0, 1, 1, 0, 1], { dtype }).reshape([2, 3]);
    const back = Tensor.fromNpy(t.toNpy());
    assert.equal(back.dtype, dtype, dtype);
    assert.deepEqual([...back.shape], [2, 3], dtype);
    assert.deepEqual(back.toArray(), t.toArray(), dtype);
  }
});

test(".npy of a non-contiguous view serializes packed values", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const back = Tensor.fromNpy(t.permute([1, 0]).toNpy());
  assert.deepEqual(back.toArray(), [
    [1, 4],
    [2, 5],
    [3, 6],
  ]);
});

test(".npy header is 64-byte aligned and v1.0", () => {
  const bytes = Tensor.from([1, 2, 3]).toNpy();
  assert.equal(bytes[6], 1);
  const headerLength = (bytes[8] as number) | ((bytes[9] as number) << 8);
  assert.equal((10 + headerLength) % 64, 0);
});

// ---- indexing & slicing (issue #1) --------------------------------------

test("slice is a view: shares storage, matches basic NumPy slicing", () => {
  const t = Tensor.arange(12).reshape([3, 4]); // [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
  const s = t.slice({ start: 1, end: 3 }, { start: 1, end: 3 });
  assert.equal(s.data, t.data); // pointer identity — a view, not a copy
  assert.deepEqual([...s.shape], [2, 2]);
  assert.deepEqual(s.toArray(), [
    [5, 6],
    [9, 10],
  ]);
});

test("slice: omitted trailing axes are taken whole", () => {
  const t = Tensor.arange(12).reshape([3, 4]);
  const s = t.slice({ start: 1, end: 2 }); // second axis unspecified
  assert.deepEqual([...s.shape], [1, 4]);
  assert.deepEqual(s.toArray(), [[4, 5, 6, 7]]);
});

test("slice: negative start/end count from the end (Python slice semantics)", () => {
  const t = Tensor.arange(10);
  assert.deepEqual(t.slice({ start: -3 }).toArray(), [7, 8, 9]);
  assert.deepEqual(t.slice({ end: -3 }).toArray(), [0, 1, 2, 3, 4, 5, 6]);
  assert.deepEqual(t.slice({ start: -5, end: -2 }).toArray(), [5, 6, 7]);
});

test("slice: negative step reverses via negative stride, still a view", () => {
  const t = Tensor.arange(5); // [0,1,2,3,4]
  const r = t.slice({ step: -1 });
  assert.equal(r.data, t.data);
  assert.equal(r.strides[0], -1);
  assert.deepEqual(r.toArray(), [4, 3, 2, 1, 0]);

  const partial = t.slice({ start: 3, end: 0, step: -1 });
  assert.deepEqual(partial.toArray(), [3, 2, 1]);
});

test("slice composes with permute: slice-of-a-view stays a view", () => {
  const t = Tensor.arange(12).reshape([3, 4]);
  const p = t.permute([1, 0]); // 4x3, non-contiguous
  const s = p.slice({ start: 1, end: 3 });
  assert.equal(s.data, t.data); // still the original buffer, two views deep
  assert.deepEqual([...s.shape], [2, 3]);
  assert.deepEqual(s.toArray(), p.toArray().slice(1, 3));
});

test("slice: zero-length and out-of-range specs clamp rather than throw", () => {
  const t = Tensor.arange(5);
  assert.deepEqual(t.slice({ start: 3, end: 3 }).toArray(), []);
  assert.deepEqual(t.slice({ start: 10 }).toArray(), []);
  assert.deepEqual(t.slice({ end: 100 }).toArray(), [0, 1, 2, 3, 4]);
});

test("slice: step of zero throws", () => {
  const t = Tensor.arange(5);
  assert.throws(() => t.slice({ step: 0 }), RangeError);
});

test("select drops an axis and is a view", () => {
  const t = Tensor.arange(12).reshape([3, 4]);
  const row = t.select(0, 1);
  assert.equal(row.data, t.data);
  assert.deepEqual([...row.shape], [4]);
  assert.deepEqual(row.toArray(), [4, 5, 6, 7]);
  // negative index
  assert.deepEqual(t.select(0, -1).toArray(), [8, 9, 10, 11]);
  assert.throws(() => t.select(0, 5), RangeError);
});

test("take gathers arbitrary indices along an axis (copies)", () => {
  const t = Tensor.arange(12).reshape([3, 4]);
  const picked = t.take([2, 0], { axis: 0 });
  assert.notEqual(picked.data, t.data); // copy, not a view
  assert.deepEqual(picked.toArray(), [
    [8, 9, 10, 11],
    [0, 1, 2, 3],
  ]);
  // negative indices and default axis 0
  assert.deepEqual(
    t.take([-1]).toArray(),
    [[8, 9, 10, 11]],
  );
  assert.throws(() => t.take([99]), RangeError);
});

test("gather is take with (axis, indices) argument order", () => {
  const t = Tensor.arange(12).reshape([3, 4]);
  assert.deepEqual(t.gather(1, [3, 0]).toArray(), t.take([3, 0], { axis: 1 }).toArray());
});

test("mask selects elements where the boolean tensor is true, flattening to 1-D", () => {
  const t = Tensor.from([10, 20, 30, 40], { dtype: "f64" });
  const cond = Tensor.from([1, 0, 1, 0], { dtype: "bool" });
  const selected = t.mask(cond);
  assert.deepEqual([...selected.shape], [2]);
  assert.deepEqual(selected.toArray(), [10, 30]);
});

test("mask rejects non-bool conditions and shape mismatches", () => {
  const t = Tensor.from([1, 2, 3], { dtype: "f64" });
  assert.throws(() => t.mask(Tensor.from([1, 0, 1], { dtype: "f64" })), TypeError);
  assert.throws(
    () => t.mask(Tensor.from([1, 0], { dtype: "bool" })),
    RangeError,
  );
});

test("take/select/mask on non-contiguous views read the correct (view-relative) values", () => {
  const t = Tensor.arange(6).reshape([2, 3]); // [[0,1,2],[3,4,5]]
  const p = t.permute([1, 0]); // [[0,3],[1,4],[2,5]]
  assert.deepEqual(p.select(0, 1).toArray(), [1, 4]);
  assert.deepEqual(p.take([2, 0], { axis: 0 }).toArray(), [
    [2, 5],
    [0, 3],
  ]);
});

// ---- matmul (issue #2) ---------------------------------------------------

test("matmul: 2-D x 2-D, deterministic", () => {
  const a = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const b = Tensor.from([7, 8, 9, 10, 11, 12], { dtype: "f64" }).reshape([3, 2]);
  const c = a.matmul(b);
  assert.deepEqual([...c.shape], [2, 2]);
  // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]]
  assert.deepEqual(c.toArray(), [
    [58, 64],
    [139, 154],
  ]);
});

test("matmul: 1-D @ 1-D collapses to a 0-d scalar", () => {
  const a = Tensor.from([1, 2, 3], { dtype: "f64" });
  const b = Tensor.from([4, 5, 6], { dtype: "f64" });
  const r = a.matmul(b);
  assert.equal(r.ndim, 0);
  assert.equal(r.item(), 32); // 1*4+2*5+3*6
});

test("matmul: 2-D @ 1-D and 1-D @ 2-D squeeze the right axis", () => {
  const mat = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const v3 = Tensor.from([1, 1, 1], { dtype: "f64" });
  const v2 = Tensor.from([1, 1], { dtype: "f64" });
  const mv = mat.matmul(v3); // (2,3)@(3,) -> (2,)
  assert.deepEqual([...mv.shape], [2]);
  assert.deepEqual(mv.toArray(), [6, 15]);
  const vm = v2.matmul(mat); // (2,)@(2,3) -> (3,)
  assert.deepEqual([...vm.shape], [3]);
  assert.deepEqual(vm.toArray(), [5, 7, 9]);
});

test("matmul: batched leading axes broadcast", () => {
  // batch of 2 identity-ish 2x2 matrices times a shared 2x2
  const batch = Tensor.from([1, 0, 0, 1, 2, 0, 0, 2], { dtype: "f64" }).reshape([
    2, 2, 2,
  ]);
  const shared = Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2]);
  const r = batch.matmul(shared);
  assert.deepEqual([...r.shape], [2, 2, 2]);
  assert.deepEqual(r.toArray(), [
    [
      [1, 2],
      [3, 4],
    ],
    [
      [2, 4],
      [6, 8],
    ],
  ]);
});

test("matmul: transposed (non-contiguous) lhs does not copy before computing", () => {
  const a = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const aT = a.permute([1, 0]); // (3,2), non-contiguous
  assert.equal(aT.isContiguous, false);
  const b = Tensor.from([1, 0, 0, 1], { dtype: "f64" }).reshape([2, 2]);
  const r = aT.matmul(b); // (3,2)@(2,2) -> (3,2), should equal aT itself (identity)
  assert.deepEqual(r.toArray(), aT.toArray());
});

test("matmul: inner-dimension mismatch throws", () => {
  const a = Tensor.zeros([2, 3], { dtype: "f64" });
  const b = Tensor.zeros([4, 2], { dtype: "f64" });
  assert.throws(() => a.matmul(b), RangeError);
});

test("matmul: dtype mismatch throws", () => {
  const a = Tensor.zeros([2, 2], { dtype: "f32" });
  const b = Tensor.zeros([2, 2], { dtype: "f64" });
  assert.throws(() => a.matmul(b), TypeError);
});

test("matmul: rejects 0-d (scalar) operands", () => {
  const scalar = Tensor.from([5], { dtype: "f64" }).reshape([]);
  const v = Tensor.from([1, 2], { dtype: "f64" });
  assert.throws(() => scalar.matmul(v), RangeError);
});

test("dot: rejects non-1-D operands", () => {
  const mat = Tensor.zeros([2, 2], { dtype: "f64" });
  const v = Tensor.from([1, 2], { dtype: "f64" });
  assert.throws(() => mat.dot(v), RangeError);
});

test("matmul: i64 exact accumulation, no overflow for modest values", () => {
  const a = Tensor.from([1, 2, 3, 4], { dtype: "i64" }).reshape([2, 2]);
  const b = Tensor.from([5, 6, 7, 8], { dtype: "i64" }).reshape([2, 2]);
  assert.deepEqual(a.matmul(b).toArray(), [
    [19n, 22n],
    [43n, 50n],
  ]);
});

// ---- cast (issue #4) ------------------------------------------------------

test("cast converts values and always copies", () => {
  const f = Tensor.from([1.7, -2.3, 3.5], { dtype: "f64" });
  const i = f.cast("i32");
  assert.notEqual(i.data, f.data);
  assert.equal(i.dtype, "i32");
  assert.deepEqual(i.toArray(), [1, -2, 3]); // truncation toward zero, not rounding
});

test("cast to/from i64 crosses the number<->bigint boundary", () => {
  const nums = Tensor.from([1, 2, 3], { dtype: "i32" });
  const big = nums.cast("i64");
  assert.deepEqual(big.toArray(), [1n, 2n, 3n]);
  const back = big.cast("f64");
  assert.deepEqual(back.toArray(), [1, 2, 3]);
});

test("cast to bool maps non-zero to 1", () => {
  const t = Tensor.from([0, 5, -3, 0], { dtype: "f64" });
  assert.deepEqual(t.cast("bool").toArray(), [0, 1, 1, 0]);
});

test("cast preserves shape and works on non-contiguous views", () => {
  const t = Tensor.arange(6).reshape([2, 3]).permute([1, 0]);
  const casted = t.cast("i32");
  assert.deepEqual([...casted.shape], [3, 2]);
  assert.deepEqual(casted.toArray(), t.toArray());
});

// ---- comparisons (issue #4) ------------------------------------------------

test("eq/ne/lt/lte/gt/gte produce bool tensors with broadcasting", () => {
  const a = Tensor.from([1, 2, 3, 4], { dtype: "f64" });
  const b = Tensor.from([1, 5, 2, 4], { dtype: "f64" });
  assert.equal(a.eq(b).dtype, "bool");
  assert.deepEqual(a.eq(b).toArray(), [1, 0, 0, 1]);
  assert.deepEqual(a.ne(b).toArray(), [0, 1, 1, 0]);
  assert.deepEqual(a.lt(b).toArray(), [0, 1, 0, 0]);
  assert.deepEqual(a.lte(b).toArray(), [1, 1, 0, 1]);
  assert.deepEqual(a.gt(b).toArray(), [0, 0, 1, 0]);
  assert.deepEqual(a.gte(b).toArray(), [1, 0, 1, 1]);
  // broadcasting against a scalar
  assert.deepEqual(a.gt(2).toArray(), [0, 0, 1, 1]);
});

test("comparisons on i64 tensors work via BigInt comparison", () => {
  const a = Tensor.from([1, 5, 3], { dtype: "i64" });
  const b = Tensor.from([2, 5, 1], { dtype: "i64" });
  assert.deepEqual(a.lt(b).toArray(), [1, 0, 0]);
  assert.deepEqual(a.eq(b).toArray(), [0, 1, 0]);
});

test("comparisons require matching dtypes", () => {
  const a = Tensor.from([1], { dtype: "f32" });
  const b = Tensor.from([1], { dtype: "f64" });
  assert.throws(() => a.eq(b), TypeError);
});

// ---- logicals & any/all (issue #4) -----------------------------------------

test("logicalAnd/logicalOr/logicalNot on bool tensors", () => {
  const a = Tensor.from([1, 1, 0, 0], { dtype: "bool" });
  const b = Tensor.from([1, 0, 1, 0], { dtype: "bool" });
  assert.deepEqual(a.logicalAnd(b).toArray(), [1, 0, 0, 0]);
  assert.deepEqual(a.logicalOr(b).toArray(), [1, 1, 1, 0]);
  assert.deepEqual(a.logicalNot().toArray(), [0, 0, 1, 1]);
});

test("logical ops reject non-bool tensors", () => {
  const a = Tensor.from([1, 0], { dtype: "f64" });
  assert.throws(() => a.logicalNot(), TypeError);
});

test("any/all over all elements and per-axis", () => {
  const t = Tensor.from([1, 0, 1, 1, 1, 1], { dtype: "bool" }).reshape([2, 3]);
  assert.equal(t.any().item(), 1);
  assert.equal(t.all().item(), 0);
  assert.deepEqual(t.any(0).toArray(), [1, 1, 1]);
  assert.deepEqual(t.all(0).toArray(), [1, 0, 1]);
  assert.deepEqual(t.all(1).toArray(), [0, 1]);
});

test("any/all work directly on non-bool tensors via truthiness", () => {
  const t = Tensor.from([0, 0, 3], { dtype: "f64" });
  assert.equal(t.any().item(), 1);
  assert.equal(t.all().item(), 0);
});

// ---- min/max & argmin/argmax (issue #4) ------------------------------------

test("min/max over all elements and per-axis", () => {
  const t = Tensor.from([3, 1, 4, 1, 5, 9], { dtype: "f64" }).reshape([2, 3]);
  assert.equal(t.min().item(), 1);
  assert.equal(t.max().item(), 9);
  assert.deepEqual(t.min(0).toArray(), [1, 1, 4]); // per-column min: [3,1]->1, [1,5]->1, [4,9]->4
  assert.deepEqual(t.max(1).toArray(), [4, 9]);
});

test("min/max preserve dtype (incl. i64)", () => {
  const t = Tensor.from([5, 2, 8], { dtype: "i64" });
  const mn = t.min();
  assert.equal(mn.dtype, "i64");
  assert.equal(mn.item(), 2n);
});

test("min/max throw on empty reductions", () => {
  const t = Tensor.zeros([0], { dtype: "f64" });
  assert.throws(() => t.min(), RangeError);
});

test("argmin/argmax: flattened index when axis omitted, first-occurrence tie-break", () => {
  const t = Tensor.from([3, 1, 4, 1, 5, 9], { dtype: "f64" }).reshape([2, 3]);
  assert.equal(t.argmin().dtype, "i32");
  assert.equal(t.argmin().item(), 1); // first "1" at flat index 1, not 3
  assert.equal(t.argmax().item(), 5); // the "9" at flat index 5
});

test("argmin/argmax per axis", () => {
  const t = Tensor.from([3, 1, 4, 1, 5, 9], { dtype: "f64" }).reshape([2, 3]);
  assert.deepEqual(t.argmin(0).toArray(), [1, 0, 0]); // per-column argmin: [3,1]->1, [1,5]->0, [4,9]->0
  assert.deepEqual(t.argmax(1).toArray(), [2, 2]); // index within each row
});

test("min/max/argmin/argmax on non-contiguous views", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  const p = t.permute([1, 0]); // [[1,4],[2,5],[3,6]]
  assert.deepEqual(p.max(1).toArray(), [4, 5, 6]);
  assert.deepEqual(p.argmax(1).toArray(), [1, 1, 1]);
});

// ---- variance/std (issue #4) -----------------------------------------------

test("variance/std over all elements, population (ddof=0) by default", () => {
  const t = Tensor.from([2, 4, 4, 4, 5, 5, 7, 9], { dtype: "f64" });
  // known example: population variance = 4, population std = 2
  assert.ok(Math.abs(t.variance().item() as number - 4) < 1e-9);
  assert.ok(Math.abs(t.std().item() as number - 2) < 1e-9);
});

test("variance: ddof >= count throws", () => {
  const t = Tensor.from([1, 2], { dtype: "f64" });
  assert.throws(() => t.variance(undefined, { ddof: 2 }), RangeError);
});

test("variance/std work on integer dtypes (upcast to f64)", () => {
  const t = Tensor.from([2, 4, 4, 4, 5, 5, 7, 9], { dtype: "i32" });
  const v = t.variance();
  assert.equal(v.dtype, "f64");
  assert.ok(Math.abs(v.item() as number - 4) < 1e-9);
});

// ---- cumsum/cumprod (issue #4) ---------------------------------------------

test("cumsum/cumprod flatten when axis is omitted", () => {
  const t = Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2]);
  assert.deepEqual(t.cumsum().toArray(), [1, 3, 6, 10]);
  assert.deepEqual(t.cumprod().toArray(), [1, 2, 6, 24]);
});

test("cumsum/cumprod along a specific axis preserve shape", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
  assert.deepEqual(t.cumsum(1).toArray(), [
    [1, 3, 6],
    [4, 9, 15],
  ]);
  assert.deepEqual(t.cumsum(0).toArray(), [
    [1, 2, 3],
    [5, 7, 9],
  ]);
});

// ---- sort/argsort/topK (issue #4) ------------------------------------------

test("sort/argsort default to the last axis", () => {
  const t = Tensor.from([3, 1, 2, 6, 4, 5], { dtype: "f64" }).reshape([2, 3]);
  assert.deepEqual(t.sort().toArray(), [
    [1, 2, 3],
    [4, 5, 6],
  ]);
  assert.deepEqual(t.argsort().toArray(), [
    [1, 2, 0],
    [1, 2, 0],
  ]);
});

test("topK largest and smallest, values + indices", () => {
  const t = Tensor.from([3, 1, 4, 1, 5, 9, 2, 6]);
  const top3 = t.topK(3);
  assert.deepEqual(top3.values.toArray(), [9, 6, 5]); // descending
  assert.deepEqual(top3.indices.toArray(), [5, 7, 4]);
  const bottom3 = t.topK(3, { largest: false });
  assert.deepEqual(bottom3.values.toArray(), [1, 1, 2]); // ascending
});

test("topK rejects out-of-range k", () => {
  const t = Tensor.from([1, 2, 3]);
  assert.throws(() => t.topK(0), RangeError);
  assert.throws(() => t.topK(4), RangeError);
});

// ---- concat/stack/where (issue #4) -----------------------------------------

test("concat joins along an existing axis", () => {
  const a = Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2]);
  const b = Tensor.from([5, 6], { dtype: "f64" }).reshape([1, 2]);
  const c = Tensor.concat([a, b], { axis: 0 });
  assert.deepEqual([...c.shape], [3, 2]);
  assert.deepEqual(c.toArray(), [
    [1, 2],
    [3, 4],
    [5, 6],
  ]);
});

test("concat rejects dtype/shape mismatches", () => {
  const a = Tensor.from([1], { dtype: "f32" });
  const b = Tensor.from([1], { dtype: "f64" });
  assert.throws(() => Tensor.concat([a, b]), TypeError);
  const c = Tensor.zeros([2, 3], { dtype: "f64" });
  const d = Tensor.zeros([2, 4], { dtype: "f64" });
  assert.throws(() => Tensor.concat([c, d], { axis: 0 }), RangeError); // mismatched non-concat axis
});

test("stack introduces a new axis", () => {
  const a = Tensor.from([1, 2], { dtype: "f64" });
  const b = Tensor.from([3, 4], { dtype: "f64" });
  const s = Tensor.stack([a, b]);
  assert.deepEqual([...s.shape], [2, 2]);
  assert.deepEqual(s.toArray(), [
    [1, 2],
    [3, 4],
  ]);
});

test("where selects elementwise with broadcasting", () => {
  const cond = Tensor.from([1, 0, 1], { dtype: "bool" });
  const a = Tensor.from([10, 20, 30], { dtype: "f64" });
  const b = Tensor.from([1, 2, 3], { dtype: "f64" });
  assert.deepEqual(Tensor.where(cond, a, b).toArray(), [10, 2, 30]);
  // broadcasting: scalar-shaped b
  assert.deepEqual(Tensor.where(cond, a, Tensor.full([], 0, { dtype: "f64" })).toArray(), [
    10, 0, 30,
  ]);
});

// ---- squeeze/unsqueeze/flatten/broadcastTo (issue #4) ----------------------

test("squeeze drops size-1 axes and is a view", () => {
  const t = Tensor.zeros([1, 3, 1, 2], { dtype: "f64" });
  const s = t.squeeze();
  assert.equal(s.data, t.data);
  assert.deepEqual([...s.shape], [3, 2]);
  const s2 = t.squeeze(0);
  assert.deepEqual([...s2.shape], [3, 1, 2]);
  assert.throws(() => t.squeeze(1), RangeError); // axis 1 has size 3, not 1
});

test("unsqueeze inserts a size-1 axis and is a view", () => {
  const t = Tensor.from([1, 2, 3], { dtype: "f64" });
  const u = t.unsqueeze(0);
  assert.equal(u.data, t.data);
  assert.deepEqual([...u.shape], [1, 3]);
  const u2 = t.unsqueeze(-1);
  assert.deepEqual([...u2.shape], [3, 1]);
  const u3 = t.unsqueeze(1); // appended-position edge case (axis === ndim)
  assert.deepEqual([...u3.shape], [3, 1]);
});

test("flatten collapses a range of axes", () => {
  const t = Tensor.arange(24).reshape([2, 3, 4]);
  const f = t.flatten(1, 2);
  assert.deepEqual([...f.shape], [2, 12]);
  assert.deepEqual(f.toArray(), t.reshape([2, 12]).toArray());
});

test("broadcastTo expands size-1 axes via stride 0", () => {
  const t = Tensor.from([1, 2, 3], { dtype: "f64" }).reshape([1, 3]);
  const b = t.broadcastTo([4, 3]);
  assert.equal(b.data, t.data);
  assert.deepEqual([...b.shape], [4, 3]);
  assert.deepEqual(b.toArray(), [
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
  ]);
  assert.throws(() => t.broadcastTo([4, 5]), RangeError);
});

// ---- unfold (issue #84, sliding-window / patch view) -----------------------

test("unfold: 1D sliding windows of size 3 over 5 elements produce 3 overlapping patches", () => {
  const t = Tensor.from([0, 1, 2, 3, 4], { dtype: "f64" });
  const w = t.unfold([3]);
  assert.deepEqual([...w.shape], [3, 3]);
  assert.deepEqual(w.toArray(), [
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
  ]);
  assert.equal(w.data, t.data, "a pure view, no copy");
});

test("unfold: 2D 2x2 windows over a 4x4 grid match hand-computed patches (verified via a standalone node -e script before writing this)", () => {
  const t = Tensor.arange(16).reshape([4, 4]);
  const w = t.unfold([2, 2]);
  assert.deepEqual([...w.shape], [3, 3, 2, 2]);
  const arr = w.toArray() as number[][][][];
  assert.deepEqual(arr[0]?.[0], [
    [0, 1],
    [4, 5],
  ]);
  assert.deepEqual(arr[1]?.[2], [
    [6, 7],
    [10, 11],
  ]);
  assert.deepEqual(arr[2]?.[2], [
    [10, 11],
    [14, 15],
  ]);
});

test("unfold: an explicit axes subset windows only those axes, leaving the rest untouched", () => {
  const t = Tensor.arange(24).reshape([2, 4, 3]);
  const w = t.unfold([2], [1]);
  assert.deepEqual([...w.shape], [2, 3, 3, 2]);
  const arr = w.toArray() as number[][][][];
  // w[b][i][c][k] === t[b][i+k][c]
  assert.equal(arr[0]?.[1]?.[2]?.[1], (t.toArray() as number[][][])[0]?.[2]?.[2]);
  assert.equal(arr[1]?.[0]?.[0]?.[0], (t.toArray() as number[][][])[1]?.[0]?.[0]);
});

test("unfold: rejects a mismatched windowShape/axes length, a too-large window, and a duplicate axis", () => {
  const t = Tensor.arange(16).reshape([4, 4]);
  assert.throws(() => t.unfold([2, 2, 2]), RangeError);
  assert.throws(() => t.unfold([5]), RangeError);
  assert.throws(() => t.unfold([2, 2], [0, 0]), RangeError);
});

test("sqrt rejects bigint dtypes", () => {
  const t = Tensor.from([4, 9], { dtype: "i64" });
  assert.throws(() => t.sqrt(), TypeError);
});

// ---- random (issue #5) -----------------------------------------------------

test("random: same seed produces the same sequence, different seeds diverge", () => {
  const a = random.uniform([5], { rng: random.seed(42) });
  const b = random.uniform([5], { rng: random.seed(42) });
  assert.deepEqual(a.toArray(), b.toArray());

  const c = random.uniform([5], { rng: random.seed(7) });
  assert.notDeepEqual(a.toArray(), c.toArray());
});

test("random: an Rng's sequence is NOT reset between calls (state advances)", () => {
  const rng = random.seed(1);
  const first = random.uniform([3], { rng });
  const second = random.uniform([3], { rng }); // same rng object, further along
  assert.notDeepEqual(first.toArray(), second.toArray());
});

test("random.uniform: values fall in [min, max) and default to [0, 1)", () => {
  const t = random.uniform([200], { rng: random.seed(1) });
  for (const v of t.toArray() as number[]) {
    assert.ok(v >= 0 && v < 1, `${v} not in [0,1)`);
  }
  const scaled = random.uniform([200], { min: 10, max: 20, rng: random.seed(2) });
  for (const v of scaled.toArray() as number[]) {
    assert.ok(v >= 10 && v < 20, `${v} not in [10,20)`);
  }
});

test("random.uniform: sample mean converges toward the midpoint (moment check)", () => {
  const t = random.uniform([20000], { min: 0, max: 10, rng: random.seed(3) });
  const mean = t.mean().item() as number;
  assert.ok(Math.abs(mean - 5) < 0.2, `sample mean ${mean} too far from 5`);
});

test("random.normal: sample mean/std converge to the requested moments", () => {
  const t = random.normal([20000], { mean: 5, std: 2, rng: random.seed(4) });
  const mean = t.mean().item() as number;
  const std = t.std().item() as number;
  assert.ok(Math.abs(mean - 5) < 0.1, `sample mean ${mean} too far from 5`);
  assert.ok(Math.abs(std - 2) < 0.1, `sample std ${std} too far from 2`);
});

test("random.randint: integers within [low, high), reproducible, correct dtype", () => {
  const t = random.randint(0, 10, [500], { rng: random.seed(5) });
  assert.equal(t.dtype, "i32");
  for (const v of t.toArray() as number[]) {
    assert.ok(Number.isInteger(v) && v >= 0 && v < 10, `${v} not in [0,10)`);
  }
  const again = random.randint(0, 10, [500], { rng: random.seed(5) });
  assert.deepEqual(t.toArray(), again.toArray());
});

test("random.randint: covers the full range given enough samples", () => {
  const t = random.randint(0, 5, [2000], { rng: random.seed(6) });
  const seen = new Set(t.toArray() as number[]);
  assert.deepEqual([...seen].sort(), [0, 1, 2, 3, 4]);
});

test("random.randint: rejects an empty or inverted range", () => {
  assert.throws(() => random.randint(5, 5, [1], { rng: random.seed(1) }), RangeError);
  assert.throws(() => random.randint(5, 2, [1], { rng: random.seed(1) }), RangeError);
});

test("random respects an explicit dtype", () => {
  const t = random.uniform([10], { dtype: "f64", rng: random.seed(1) });
  assert.equal(t.dtype, "f64");
});

test("Rng.nextBelow rejects a non-positive-integer bound", () => {
  const rng = random.seed(1);
  assert.throws(() => rng.nextBelow(0), RangeError);
  assert.throws(() => rng.nextBelow(-3), RangeError);
});

test("log rejects bigint dtypes", () => {
  const t = Tensor.from([4, 9], { dtype: "i64" });
  assert.throws(() => t.log(), TypeError);
});
