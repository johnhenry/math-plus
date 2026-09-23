/**
 * Fast-path / strided-path equivalence (issue #120).
 *
 * Contiguous inputs now take flat typed-array kernels (src/kernels.ts), a
 * blocked GEMM, and fused softmax/variance; strided views (transposes,
 * stepped slices, stride-0 broadcasts) keep the general path. The kernels
 * are designed to be BIT-IDENTICAL to the general path, so these tests
 * compare with `Object.is` per element — not a tolerance — between:
 *   - a contiguous tensor (fast path), and
 *   - a non-contiguous "strided twin" holding the same logical values
 *     (general path).
 * matmul additionally checks against an independent naive triple loop.
 * NumPy agreement itself is covered by differential.test.ts.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor, random, type DType } from "../src/index.ts";

const rng = random.seed(120);

function rand(shape: number[], dtype: DType = "f32"): Tensor {
  if (dtype === "f32" || dtype === "f64") {
    return random.uniform(shape, { rng, min: -3, max: 3, dtype });
  }
  const lo = dtype === "u8" || dtype === "bool" ? 0 : -50;
  return random.randint(lo, 50, shape, { rng, dtype });
}

/** Same logical values, non-contiguous layout (every other element of a doubled buffer). */
function stridedTwin(t: Tensor): Tensor {
  const twin = Tensor.stack([t, t], { axis: t.ndim }).select(t.ndim, 0);
  if (t.size > 1) assert.equal(twin.isContiguous, false, "twin must be strided");
  return twin;
}

/** Contiguous view at a non-zero storage offset holding `t`'s values. */
function offsetTwin(t: Tensor): Tensor {
  const padded = Tensor.concat([Tensor.zeros([1, ...t.shape.slice(1)], { dtype: t.dtype }), t.ndim ? t : t.unsqueeze(0)]);
  const view = padded.slice({ start: 1 });
  assert.ok(view.isContiguous && view.offset > 0);
  return view;
}

function assertIdentical(actual: Tensor, expected: Tensor, label: string): void {
  assert.equal(actual.dtype, expected.dtype, `${label}: dtype`);
  assert.deepEqual([...actual.shape], [...expected.shape], `${label}: shape`);
  const a = actual.contiguous();
  const e = expected.contiguous();
  for (let i = 0; i < a.size; i++) {
    const av = a.data[a.offset + i];
    const ev = e.data[e.offset + i];
    if (!Object.is(av, ev)) {
      assert.fail(`${label}: element ${i} differs: ${String(av)} vs ${String(ev)}`);
    }
  }
}

const NUM_DTYPES: DType[] = ["f32", "f64", "i32", "u8"];
const SHAPE = [3, 5, 7];

test("binary ops: same-shape contiguous fast path matches the strided path (all Number dtypes, incl. offset views)", () => {
  for (const dtype of NUM_DTYPES) {
    const a = rand(SHAPE, dtype);
    const b = rand(SHAPE, dtype); // integer /0 included on purpose: both paths store the same wrapped value
    for (const op of ["add", "sub", "mul", "div"] as const) {
      const fast = a[op](b);
      assertIdentical(fast, stridedTwin(a)[op](stridedTwin(b)), `${dtype} ${op}`);
      assertIdentical(offsetTwin(a)[op](offsetTwin(b)), fast, `${dtype} ${op} offset`);
    }
  }
});

test("binary ops: scalar, bias (trailing-block) and row broadcasts match the strided path, both operand orders", () => {
  for (const dtype of ["f32", "f64", "i32"] as DType[]) {
    const x = rand(SHAPE, dtype);
    const cases: Array<[string, Tensor]> = [
      ["scalar 0-d", rand([], dtype)],
      ["scalar [1,1,1]", rand([1, 1, 1], dtype)],
      ["bias [7]", rand([7], dtype)],
      ["bias [5,7]", rand([5, 7], dtype)],
      ["bias [1,5,7]", rand([1, 5, 7], dtype)],
      ["row [3,5,1]", rand([3, 5, 1], dtype)],
      ["row [3,1,1]", rand([3, 1, 1], dtype)],
      ["mixed [1,5,1] (general path)", rand([1, 5, 1], dtype)],
    ];
    for (const [label, y] of cases) {
      for (const op of ["add", "sub", "mul", "div"] as const) {
        assertIdentical(x[op](y), stridedTwin(x)[op](y), `${dtype} x ${op} ${label}`);
        assertIdentical(y[op](x), y[op](stridedTwin(x)), `${dtype} ${label} ${op} x`);
      }
    }
    // Number scalars go through Tensor.full (dtype-rounded) on both paths.
    assertIdentical(x.mul(0.1), stridedTwin(x).mul(0.1), `${dtype} mul 0.1`);
    assertIdentical(x.sub(2.5), stridedTwin(x).sub(2.5), `${dtype} sub 2.5`);
  }
});

test("binary ops: broadcast views and transposes still take the general path correctly", () => {
  const x = rand([4, 6]);
  const col = rand([4, 1]);
  const bcast = col.broadcastTo([4, 6]); // stride-0, not contiguous
  assert.equal(bcast.isContiguous, false);
  assertIdentical(x.add(bcast), x.add(col), "stride-0 broadcast view");
  const t = rand([6, 4]).transpose();
  assertIdentical(x.mul(t), x.mul(t.contiguous()), "transposed operand");
  const rev = x.slice(null, { step: -1 });
  assertIdentical(rev.sub(x), rev.contiguous().sub(x), "negative-step slice");
});

test("empty tensors through the binary fast path", () => {
  const e = Tensor.zeros([0, 4]);
  assert.deepEqual([...e.add(rand([4])).shape], [0, 4]);
  assert.deepEqual([...e.mul(2).shape], [0, 4]);
});

test("unary ops and cast: contiguous fast path matches the strided path", () => {
  for (const dtype of ["f32", "f64"] as DType[]) {
    const x = rand(SHAPE, dtype);
    const pos = x.abs().add(0.5);
    for (const op of ["exp", "tanh", "sigmoid", "gelu", "relu", "sin", "abs", "neg", "floor"] as const) {
      assertIdentical(x[op](), stridedTwin(x)[op](), `${dtype} ${op}`);
    }
    assertIdentical(pos.sqrt(), stridedTwin(pos).sqrt(), `${dtype} sqrt`);
    assertIdentical(pos.log(), stridedTwin(pos).log(), `${dtype} log`);
    assertIdentical(x.pow(3), stridedTwin(x).pow(3), `${dtype} pow`);
  }
  const i = rand(SHAPE, "i32");
  assertIdentical(i.abs(), stridedTwin(i).abs(), "i32 abs");
  const f = random.uniform([50], { rng, min: -300, max: 300, dtype: "f64" });
  for (const target of ["i32", "u8", "i8", "u16", "bool", "f32", "f64", "i64"] as DType[]) {
    assertIdentical(f.cast(target), stridedTwin(f).cast(target), `cast f64 -> ${target}`);
    assertIdentical(offsetTwin(f).cast(target), f.cast(target), `cast offset f64 -> ${target}`);
  }
});

test("contiguous() of an offset contiguous view is an exact packed copy", () => {
  const x = rand([5, 3]);
  const view = x.slice({ start: 2 });
  const packed = view.contiguous();
  assert.equal(packed.offset, 0);
  assert.equal(packed.data.length, packed.size);
  assert.notEqual(packed.data, x.data);
  assertIdentical(packed, stridedTwin(view), "offset view packed");
});

test("comparisons: same-shape and scalar fast paths match the strided path", () => {
  const x = rand(SHAPE, "f64");
  const y = rand(SHAPE, "f64");
  for (const op of ["eq", "ne", "lt", "lte", "gt", "gte"] as const) {
    assertIdentical(x[op](y), stridedTwin(x)[op](stridedTwin(y)), `${op} tensor`);
    assertIdentical(x[op](0), stridedTwin(x)[op](0), `${op} scalar`);
    assertIdentical(x[op](x), stridedTwin(x)[op](x), `${op} self`);
  }
  assertIdentical(x.clip(-1, 1), stridedTwin(x).clip(-1, 1), "clip");
});

test("reductions (sum/mean/min/max, every axis + full) match the strided path", () => {
  for (const dtype of NUM_DTYPES) {
    const x = rand(SHAPE, dtype);
    for (const axis of [undefined, 0, 1, 2, -1]) {
      for (const op of ["sum", "mean", "min", "max"] as const) {
        assertIdentical(x[op](axis), stridedTwin(x)[op](axis), `${dtype} ${op}(${axis})`);
      }
    }
  }
});

test("min/max keep the strided path's NaN semantics on the fast path", () => {
  const vals = [1, NaN, 3, NaN, 0, 2, 5, -1, NaN, 4, 4, 4];
  const x = Tensor.from(vals, { dtype: "f64" }).reshape([3, 4]);
  for (const axis of [undefined, 0, 1]) {
    assertIdentical(x.max(axis), stridedTwin(x).max(axis), `max(${axis})`);
    assertIdentical(x.min(axis), stridedTwin(x).min(axis), `min(${axis})`);
  }
  assert.throws(() => Tensor.zeros([0]).max(), /max of an empty tensor/);
  assert.throws(() => Tensor.zeros([2, 0]).min(1), /min of an empty axis/);
});

test("softmax: fused kernel matches the composed (strided) path bit-for-bit on every axis", () => {
  for (const dtype of ["f32", "f64"] as DType[]) {
    const x = rand([4, 6, 9], dtype).mul(4);
    for (const axis of [-1, 0, 1, 2]) {
      assertIdentical(x.softmax(axis), stridedTwin(x).softmax(axis), `${dtype} softmax(${axis})`);
    }
    assertIdentical(offsetTwin(x).softmax(), x.softmax(), `${dtype} offset softmax`);
    // Rows sum to ~1.
    const s = x.softmax().sum(-1);
    for (let i = 0; i < s.size; i++) assert.ok(Math.abs((s.data[i] as number) - 1) < 1e-5);
  }
  // Large magnitudes: max-subtraction keeps it finite.
  const big = Tensor.from([1000, 1001, 1002], { dtype: "f32" });
  assertIdentical(big.softmax(), stridedTwin(big).softmax(), "large logits");
  assert.ok([...(big.softmax().data as Float32Array)].every(Number.isFinite));
});

test("variance/std: fused kernel matches the composed (strided) path bit-for-bit", () => {
  for (const dtype of ["f32", "f64", "i32"] as DType[]) {
    const x = rand([4, 6, 9], dtype);
    for (const axis of [undefined, 0, 1, -1]) {
      for (const ddof of [0, 1]) {
        assertIdentical(
          x.variance(axis, { ddof }),
          stridedTwin(x).variance(axis, { ddof }),
          `${dtype} variance(${axis}, ddof=${ddof})`,
        );
        assertIdentical(x.std(axis, { ddof }), stridedTwin(x).std(axis, { ddof }), `${dtype} std(${axis})`);
      }
    }
  }
  assert.throws(() => Tensor.from([1, 2]).variance(0, { ddof: 2 }), /ddof=2 >= count=2/);
});

/** Independent naive reference: f64 accumulation in p-order, one rounding on store. */
function naiveMatmul(a: Tensor, b: Tensor): Tensor {
  const [m, k] = a.shape as [number, number];
  const n = b.shape[1] as number;
  const out = Tensor.zeros([m, n], { dtype: a.dtype });
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let p = 0; p < k; p++) acc += (a.at(i, p) as number) * (b.at(p, j) as number);
      out.data[i * n + j] = acc as never;
    }
  }
  return out;
}

test("matmul: blocked GEMM is bit-identical to a naive triple loop (odd sizes straddling the 4x4 / 64-column blocks)", () => {
  for (const dtype of ["f32", "f64", "i32"] as DType[]) {
    for (const [m, k, n] of [[1, 1, 1], [3, 5, 2], [4, 4, 4], [7, 9, 13], [67, 31, 131], [5, 0, 3]] as const) {
      const a = rand([m, k], dtype);
      const b = rand([k, n], dtype);
      assertIdentical(a.matmul(b), naiveMatmul(a, b), `${dtype} ${m}x${k}@${k}x${n}`);
    }
  }
});

test("matmul: strided/transposed/offset operands give the same bits as contiguous ones", () => {
  const a = rand([9, 11]);
  const b = rand([11, 6]);
  const ref = a.matmul(b);
  assertIdentical(stridedTwin(a).matmul(b), ref, "strided lhs");
  assertIdentical(a.matmul(stridedTwin(b)), ref, "strided rhs");
  assertIdentical(a.matmul(b.transpose().contiguous().transpose()), ref, "transposed-view rhs");
  assertIdentical(offsetTwin(a).matmul(offsetTwin(b)), ref, "offset views");
  assertIdentical(a.matmul(b.slice(null, { step: -1 })), a.matmul(b.slice(null, { step: -1 }).contiguous()), "negative stride rhs");
});

test("matmul: batched + broadcast batches and 1-D operands match per-batch 2-D results", () => {
  const a = rand([2, 3, 5, 4]);
  const w = rand([4, 6]); // broadcast across both batch dims
  const out = a.matmul(w);
  assert.deepEqual([...out.shape], [2, 3, 5, 6]);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 3; j++) {
      assertIdentical(out.select(0, i).select(0, j), naiveMatmul(a.select(0, i).select(0, j).contiguous(), w), `batch ${i},${j}`);
    }
  }
  const lhsB = rand([1, 5, 4]);
  const rhsB = rand([3, 4, 2]);
  const bo = lhsB.matmul(rhsB);
  for (let j = 0; j < 3; j++) {
    assertIdentical(bo.select(0, j), naiveMatmul(lhsB.select(0, 0), rhsB.select(0, j).contiguous()), `lhs-broadcast batch ${j}`);
  }
  const v = rand([4]);
  const m = rand([4, 3]);
  assertIdentical(v.matmul(m), naiveMatmul(v.unsqueeze(0), m).reshape([3]), "1-D @ 2-D");
  assertIdentical(m.transpose().matmul(v), naiveMatmul(m.transpose().contiguous(), v.unsqueeze(1)).reshape([3]), "2-D @ 1-D");
  assertIdentical(v.dot(v), naiveMatmul(v.unsqueeze(0), v.unsqueeze(1)).reshape([]), "dot");
});

test("matmul: i64 keeps the BigInt path", () => {
  const a = Tensor.from([1, 2, 3, 4], { dtype: "i64" }).reshape([2, 2]);
  const out = a.matmul(a);
  assert.deepEqual(out.toArray(), [[7n, 10n], [15n, 22n]]);
});
