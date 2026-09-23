/**
 * f16/bf16 storage dtypes: codec correctness, value-converting cast(), and
 * the "numeric kernels reject half dtypes" contract. NumPy-oracle coverage
 * (bit-for-bit astype(float16) and the bf16 RNE reference) lives in
 * differential.test.ts; here the platform's own `Math.f16round` /
 * `Float16Array` (Node >= 24, Bun, current browsers) is the independent
 * reference for f16, and float32 bit-reinterpretation for bf16.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor, decodeHalf, encodeHalf, isHalfDType, random } from "../src/index.ts";

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
const hasFloat16Array = typeof (globalThis as { Float16Array?: unknown }).Float16Array === "function";

function sameNumber(a: number, b: number): boolean {
  return Object.is(a, b) || (Number.isNaN(a) && Number.isNaN(b));
}

test("regression: cast() to f16 converts values instead of truncating to integer bits", () => {
  // The old cast() fell through to Math.trunc for every non-f32/f64 target,
  // so 1.5 became the bit pattern 0x0001 (a subnormal ~6e-8), and reading
  // f16 back gave the raw integer bits (1.5 -> 15872).
  const t = Tensor.from([1.5, -2.25, 0.1, 65504], { dtype: "f32" }).cast("f16");
  assert.deepEqual([...(t.data as Uint16Array)], [0x3e00, 0xc080, 0x2e66, 0x7bff]);
  assert.deepEqual(t.toArray(), [1.5, -2.25, 0.0999755859375, 65504]);
  const back = t.cast("f32");
  assert.deepEqual([...(back.data as Float32Array)], [1.5, -2.25, 0.0999755859375, 65504]);
});

test("regression: cast() from f16/bf16 to integers decodes first", () => {
  const h = Tensor.from([3.75, -2.5, 100], { dtype: "f16" });
  assert.deepEqual(h.cast("i32").toArray(), [3, -2, 100]);
  assert.deepEqual(h.cast("i64").toArray(), [3n, -2n, 100n]);
  const b = Tensor.from([3.75, -2.5, 100], { dtype: "bf16" });
  assert.deepEqual(b.cast("i32").toArray(), [3, -2, 100]);
  assert.deepEqual(Tensor.from([0, -0, 0.5], { dtype: "f16" }).cast("bool").toArray(), [0, 0, 1]);
});

test("f16 decode agrees with Float16Array for all 65536 bit patterns", { skip: !hasFloat16Array && "no Float16Array" }, () => {
  const F16 = (globalThis as unknown as { Float16Array: new (b: ArrayBuffer) => ArrayLike<number> }).Float16Array;
  const bits = new Uint16Array(65536);
  for (let i = 0; i < bits.length; i++) bits[i] = i;
  const native = new F16(bits.buffer);
  for (let i = 0; i < bits.length; i++) {
    assert.ok(sameNumber(decodeHalf("f16", i), native[i] as number), `0x${i.toString(16)}`);
  }
});

test("f16 encode is round-to-nearest-even, matching Math.f16round", { skip: typeof (Math as { f16round?: unknown }).f16round !== "function" && "no Math.f16round" }, () => {
  const f16round = (Math as unknown as { f16round: (x: number) => number }).f16round;
  const probes: number[] = [0, -0, Infinity, -Infinity, Number.NaN, 65504, 65519.999, 65520, 1e-8, 2 ** -25, 3 * 2 ** -26, 2 ** -24, 2 ** -14];
  // every halfway point between adjacent f16 values (ties) plus neighbours
  for (let b = 0; b < 0x7bff; b += 7) {
    const lo = decodeHalf("f16", b);
    const hi = decodeHalf("f16", b + 1);
    const mid = (lo + hi) / 2;
    probes.push(mid, -mid, mid + Number.EPSILON * mid, mid - Number.EPSILON * mid);
  }
  for (let i = 0; i < 5000; i++) probes.push((Math.random() - 0.5) * 10 ** (Math.random() * 14 - 9));
  for (const x of probes) {
    assert.ok(sameNumber(decodeHalf("f16", encodeHalf("f16", x)), f16round(x)), `x=${x}`);
  }
});

test("bf16 decode is float32 with the low 16 bits zeroed (all 65536 patterns)", () => {
  for (let i = 0; i < 65536; i++) {
    u32[0] = i << 16;
    assert.ok(sameNumber(decodeHalf("bf16", i), f32[0] as number), `0x${i.toString(16)}`);
  }
});

test("bf16 encode: exact round-trip of representable values, RNE on ties, overflow to Inf", () => {
  for (let i = 0; i < 65536; i++) {
    const v = decodeHalf("bf16", i);
    if (Number.isNaN(v)) continue;
    assert.equal(encodeHalf("bf16", v), i, `0x${i.toString(16)}`);
  }
  assert.equal(encodeHalf("bf16", 1 + 2 ** -8), 0x3f80); // tie -> even (1.0)
  assert.equal(encodeHalf("bf16", 1 + 3 * 2 ** -8), 0x3f82); // tie -> even (up)
  assert.equal(encodeHalf("bf16", 3.4e38 * 1.01), 0x7f80);
  assert.equal(encodeHalf("bf16", -0), 0x8000);
  assert.equal(encodeHalf("bf16", Number.NaN) & 0x7fc0, 0x7fc0);
});

test("constructors, element access and random encode/decode half dtypes", () => {
  for (const dtype of ["f16", "bf16"] as const) {
    assert.ok(isHalfDType(dtype));
    assert.deepEqual(Tensor.full([2], 2.5, { dtype }).toArray(), [2.5, 2.5]);
    assert.deepEqual(Tensor.ones([2], { dtype }).toArray(), [1, 1]);
    assert.deepEqual(Tensor.arange(0, 2, 0.5, { dtype }).toArray(), [0, 0.5, 1, 1.5]);
    const t = Tensor.from([1, 2, 3, 4], { dtype }).reshape([2, 2]);
    assert.equal(t.at(1, 0), 3);
    assert.equal(t.select(0, 1).select(0, 1).item(), 4);
    // structural ops move bits and stay correct
    assert.deepEqual(t.transpose().contiguous().toArray(), [[1, 3], [2, 4]]);
    assert.deepEqual(Tensor.concat([t, t]).flip(0).toArray(), [[3, 4], [1, 2], [3, 4], [1, 2]]);
    assert.deepEqual(t.pad([[1, 0]], { value: 0.5 }).toArray(), [[0.5, 1, 2], [0.5, 3, 4]]);
    assert.deepEqual(Tensor.where(Tensor.from([1, 0], { dtype: "bool" }), t.select(0, 0), t.select(0, 1)).toArray(), [1, 4]);
    const r = random.uniform([64], { dtype, rng: random.seed(1), min: -1, max: 1 });
    for (const v of r.toArray() as number[]) assert.ok(v >= -1 && v <= 1, `${v}`);
  }
});

test("numeric kernels reject f16/bf16 with a cast() hint instead of computing on bits", () => {
  const t = Tensor.from([1, 2], { dtype: "f16" });
  const u = Tensor.from([1, 2], { dtype: "bf16" });
  const hint = /not supported on (f16|bf16).*cast\("f32"\)/;
  for (const x of [t, u]) {
    assert.throws(() => x.add(x), hint);
    assert.throws(() => x.mul(2), hint);
    assert.throws(() => x.matmul(x), hint);
    assert.throws(() => x.sum(), hint);
    assert.throws(() => x.mean(), hint);
    assert.throws(() => x.max(), hint);
    assert.throws(() => x.argmax(), hint);
    assert.throws(() => x.exp(), hint);
    assert.throws(() => x.sqrt(), hint);
    assert.throws(() => x.abs(), hint);
    assert.throws(() => x.softmax(), hint);
    assert.throws(() => x.gt(1), hint);
    assert.throws(() => x.sort(), hint);
    assert.throws(() => x.cumsum(), hint);
    assert.throws(() => x.variance(), hint);
    assert.throws(() => x.clip(0, 1), hint);
    assert.throws(() => x.nonzero(), hint);
    assert.throws(() => x.any(), hint);
  }
  // the documented route works: cast to f32, compute, cast back
  assert.deepEqual(t.cast("f32").add(t.cast("f32")).cast("f16").toArray(), [2, 4]);
});

test("toNpy/fromNpy round-trip f16 (<f2); bf16 has no .npy dtype", () => {
  const t = Tensor.from([1.5, -0.25, 65504], { dtype: "f16" });
  const back = Tensor.fromNpy(t.toNpy());
  assert.equal(back.dtype, "f16");
  assert.deepEqual(back.toArray(), [1.5, -0.25, 65504]);
  assert.match(new TextDecoder().decode(t.toNpy().subarray(0, 64)), /'descr': '<f2'/);
  assert.throws(() => Tensor.from([1], { dtype: "bf16" }).toNpy(), /no \.npy representation/);
});
