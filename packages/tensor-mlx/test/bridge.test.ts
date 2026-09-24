/**
 * The explicit-transfer boundary and lifetime rules.
 *
 * The first block is pure TypeScript (tensor-core <-> HostTensor views) and
 * runs on every platform. The second needs MLX and skips (never fails) off
 * darwin/arm64 or without libmlxc.
 */
import assert from "node:assert/strict";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { createMlxDevice, hostFromTensor, MlxArray, tensorFromHost, type MlxDevice } from "../src/index.ts";
import { f16Tensor, mlxSkip, testFns } from "./helpers.ts";
import { makeTest } from "../../../test/harness.ts";

// Each file imports bun:test itself (see test/harness.ts).
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const harness = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
const { describe, it, itUnless } = testFns(harness);

describe("host views (no MLX needed)", () => {
  it("hostFromTensor views the tensor's storage without copying", () => {
    const t = Tensor.fromTypedArray(Float32Array.from([1, 2, 3, 4, 5, 6]), [2, 3], { dtype: "f32" });
    const h = hostFromTensor(t);
    assert.equal(h.dtype, "f32");
    assert.deepEqual(h.shape, [2, 3]);
    assert.equal(h.data.buffer, t.data.buffer, "same ArrayBuffer — no copy");
  });

  it("an offset (contiguous) view maps to a subarray of the same buffer", () => {
    const base = Tensor.fromTypedArray(Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7]), [4, 2], { dtype: "f32" });
    const row = base.select(0, 2); // [4, 5], offset 4
    const h = hostFromTensor(row);
    assert.equal(h.data.buffer, base.data.buffer);
    assert.deepEqual([...h.data], [4, 5]);
  });

  it("f16 bits are re-viewed as Float16Array (same bytes), and come back as Uint16Array bits", () => {
    const t = f16Tensor([1.5, -2, 0.1], [3]);
    const h = hostFromTensor(t);
    assert.ok(h.data instanceof Float16Array);
    assert.equal(h.data.buffer, t.data.buffer);
    assert.deepEqual([...h.data], [...Float16Array.from([1.5, -2, 0.1])]);
    const back = tensorFromHost(h);
    assert.equal(back.dtype, "f16");
    assert.ok(back.data instanceof Uint16Array);
    assert.equal(back.data.buffer, t.data.buffer, "tensorFromHost wraps, never copies");
  });

  it("rejects non-contiguous tensors instead of packing them silently", () => {
    const t = Tensor.fromTypedArray(Float32Array.from([1, 2, 3, 4, 5, 6]), [2, 3], { dtype: "f32" }).transpose();
    assert.throws(() => hostFromTensor(t), /contiguous\(\) first/);
    assert.doesNotThrow(() => hostFromTensor(t.contiguous()));
  });

  it("rejects dtypes the device cannot hold, naming the explicit cast", () => {
    assert.throws(() => hostFromTensor(Tensor.from([1, 2], { dtype: "f64" })), /no float64.*cast\("f32"\)/);
    assert.throws(() => hostFromTensor(Tensor.from([1, 2], { dtype: "i64" })), /cast\("i32"\)/);
    assert.throws(() => hostFromTensor(Tensor.from([1, 2], { dtype: "u8" })), /not supported on the device/);
  });
});

describe("MlxDevice transfers & lifetime", () => {
  let dev: MlxDevice;
  const d = () => (dev ??= createMlxDevice());

  itUnless(mlxSkip, "round-trips every device dtype through fromTensor/toTensor", async () => {
    const cases: Tensor[] = [
      Tensor.fromTypedArray(Float32Array.from([1.25, -2, 3e-3]), [3], { dtype: "f32" }),
      f16Tensor([1.5, -2, 0.1], [3]),
      Tensor.fromTypedArray(Uint16Array.from([0x3fc0, 0xc000, 0x3dcd]), [3], { dtype: "bf16" }),
      Tensor.fromTypedArray(Int32Array.from([7, -2, 2 ** 30]), [3], { dtype: "i32" }),
      Tensor.fromTypedArray(Uint8Array.from([1, 0, 1]), [3], { dtype: "bool" }),
    ];
    for (const t of cases) {
      const a = await d().fromTensor(t);
      assert.equal(a.dtype, t.dtype);
      assert.deepEqual(a.shape, [3]);
      const back = await a.toTensor();
      assert.equal(back.dtype, t.dtype);
      assert.deepEqual(Array.from(back.data as ArrayLike<number>), Array.from(t.data as ArrayLike<number>), t.dtype);
      assert.notEqual(back.data.buffer, t.data.buffer, "the round trip is two real copies, not an alias");
      a.dispose();
    }
  });

  itUnless(mlxSkip, "0-d and offset tensors upload correctly", async () => {
    const s = await d().fromTensor(Tensor.full([], 4, { dtype: "f32" }));
    assert.deepEqual(s.shape, []);
    assert.deepEqual([...(await s.mul(2).toTensor()).data], [8]);
    const base = Tensor.fromTypedArray(Float32Array.from([0, 1, 2, 3, 4, 5]), [3, 2], { dtype: "f32" });
    assert.deepEqual([...(await (await d().fromTensor(base.select(0, 1))).toTensor()).data], [2, 3]);
  });

  itUnless(mlxSkip, "ops refuse tensor-core Tensors and foreign-device arrays (no implicit transfers)", async () => {
    const a = await d().fromTensor(Tensor.from([1, 2, 3]));
    const host = Tensor.from([1, 2, 3]);
    assert.throws(() => a.add(host as unknown as MlxArray), /tensor-core Tensor.*fromTensor\(\)/);
    assert.throws(() => a.matmul(host as unknown as MlxArray), /no implicit transfers/);
    const other = createMlxDevice();
    const b = await other.fromTensor(Tensor.from([1, 2, 3]));
    assert.throws(() => a.add(b), /different MlxDevice/);
    b.dispose();
    other.destroy();
  });

  itUnless(mlxSkip, "no implicit dtype promotion; float-only ops refuse integers", async () => {
    const f = await d().fromTensor(Tensor.from([1, 2, 3]));
    const i = await d().fromTensor(Tensor.from([1, 2, 3], { dtype: "i32" }));
    assert.throws(() => f.add(i), /dtype mismatch f32 vs i32/);
    assert.throws(() => i.exp(), /needs a float dtype/);
    assert.throws(() => i.mean(), /needs a float dtype/);
    assert.equal(i.add(2).dtype, "i32", "number operands take the array's dtype");
    assert.equal(f.cast("f16").mul(0.5).dtype, "f16", "f16 stays f16");
    assert.equal(i.cast("f32").add(f).dtype, "f32");
  });

  itUnless(mlxSkip, "uploads are async; their validation errors throw synchronously, before any Promise", async () => {
    const p = d().fromTensor(Tensor.from([1, 2, 3]));
    assert.ok(p instanceof Promise, "fromTensor returns a Promise (RFC 0001 §12 Q2)");
    const a = await p;
    assert.deepEqual([...(await a.toTensor()).data], [1, 2, 3]);
    const hp = d().fromHost({ dtype: "i32", shape: [2], data: Int32Array.from([4, 5]) });
    assert.ok(hp instanceof Promise);
    assert.equal((await hp).dtype, "i32");
    const t = Tensor.fromTypedArray(Float32Array.from([1, 2, 3, 4]), [2, 2], { dtype: "f32" }).transpose();
    assert.throws(() => d().fromTensor(t), /contiguous\(\) first/);
    assert.throws(() => d().fromTensor(Tensor.from([1], { dtype: "f64" })), /no float64/);
  });

  itUnless(mlxSkip, "number operands become on-device constants (no upload, nothing left alive)", async () => {
    const x = await d().fromTensor(Tensor.from([1, -2, 3]));
    const i = await d().fromTensor(Tensor.from([5, -7, 0], { dtype: "i32" }));
    const before = d().liveArrays();
    const y = x.mul(0.5).add(1);
    assert.equal(d().liveArrays(), before + 2, "only the two results are alive");
    assert.deepEqual([...(await y.toTensor()).data], [1.5, 0, 2.5]);
    assert.deepEqual([...(await i.add(-(2 ** 31) + 7).toTensor()).data], [-(2 ** 31) + 12, -(2 ** 31), -(2 ** 31) + 7], "exact over the i32 range");
    assert.deepEqual([...(await i.mul(2.9).toTensor()).data], [10, -14, 0], "a fractional operand truncates toward zero, like numpy.asarray(v, int32)");
    assert.throws(() => i.add(2 ** 31), /does not fit in i32/);
    const e = await d().fromTensor(Tensor.fromTypedArray(new Float32Array(0), [0, 3], { dtype: "f32" }));
    assert.deepEqual(e.add(1).shape, [0, 3], "empty arrays work too");
  });

  itUnless(mlxSkip, "numerics ops keep the dtype rules", async () => {
    const f = await d().fromTensor(Tensor.from([1, 4, 9]));
    const i = await d().fromTensor(Tensor.from([1, -4, 9], { dtype: "i32" }));
    const bl = await d().fromTensor(Tensor.fromTypedArray(Uint8Array.from([1, 0, 1]), [3], { dtype: "bool" }));
    for (const op of ["sqrt", "rsqrt", "tanh", "sigmoid", "erf"] as const) assert.throws(() => i[op](), /needs a float dtype/, op);
    assert.throws(() => i.pow(2), /needs a float dtype/);
    assert.throws(() => f.logicalAnd(f), /needs bool operands/);
    assert.throws(() => bl.logicalAnd(f), /dtype mismatch/);
    assert.throws(() => bl.neg(), /not defined for bool/);
    assert.throws(() => bl.cumsum(), /not defined for bool/);
    assert.throws(() => f.less(i), /dtype mismatch f32 vs i32/);
    assert.equal(i.abs().dtype, "i32");
    assert.equal(i.neg().dtype, "i32");
    assert.equal(i.cumsum().dtype, "i32");
    assert.equal(f.less(2).dtype, "bool");
    assert.equal(bl.equal(bl).dtype, "bool");
    assert.equal(f.argmax().dtype, "i32");
    assert.equal(i.argmin(0).dtype, "i32");
    assert.equal(f.cast("f16").sqrt().dtype, "f16");
    assert.equal(f.cast("bf16").mean().dtype, "bf16");
    assert.deepEqual([...(await f.sqrt().toTensor()).data], [1, 2, 3]);
    assert.deepEqual([...(await i.abs().toTensor()).data], [1, 4, 9]);
    assert.equal((await f.argmax().toTensor()).item(), 2);
    assert.deepEqual([...(await bl.logicalNot().toTensor()).data], [0, 1, 0]);
  });

  itUnless(mlxSkip, "lazy graph: shape errors throw at the call site, values appear at eval/toTensor", async () => {
    const a = await d().fromTensor(Tensor.from([1, 2, 3]));
    const b = await d().fromTensor(Tensor.from([1, 2]));
    assert.throws(() => a.add(b), /broadcast/);
    const y = a.exp().sum();
    assert.equal(y.eval(), y, "eval() returns this");
    const got = (await y.toTensor()).item() as number;
    assert.ok(Math.abs(got - (Math.E + Math.exp(2) + Math.exp(3))) < 1e-4);
  });

  itUnless(mlxSkip, "scope frees intermediates, keeps returned arrays, and marks freed wrappers", async () => {
    const x = await d().fromTensor(Tensor.from([1, 2]));
    const before = d().liveArrays();
    let inner: MlxArray | undefined;
    const { y, z } = d().scope(() => {
      inner = x.exp();
      return { y: inner.add(x), z: x.mul(3) };
    });
    assert.equal(d().liveArrays(), before + 2, "only the two returned arrays survive");
    assert.equal(inner!.disposed, true);
    assert.throws(() => inner!.exp(), /used after dispose/);
    assert.deepEqual([...(await z.toTensor()).data], [3, 6]);
    y.dispose();
    y.dispose(); // idempotent
    z.dispose();
    assert.equal(d().liveArrays(), before);
    assert.throws(() => d().scope(() => { x.exp(); throw new Error("boom"); }), /boom/);
    assert.equal(d().liveArrays(), before, "a throwing scope frees everything it created");
  });

  itUnless(mlxSkip, "a downloaded Tensor outlives the device array", async () => {
    const a = await d().fromTensor(Tensor.from([1, 2, 3]));
    const t = await a.mul(2).toTensor();
    a.dispose();
    assert.deepEqual([...t.data], [2, 4, 6]);
  });

  itUnless(mlxSkip, "memory stays flat across repeated scoped work", async () => {
    const x = await d().fromTensor(Tensor.fromTypedArray(new Float32Array(64 * 64).fill(0.5), [64, 64], { dtype: "f32" }));
    const run = async () => {
      const y = d().scope(() => x.matmul(x).softmax().layerNorm());
      await y.toHost();
      y.dispose();
    };
    await run();
    const live = d().liveArrays();
    for (let i = 0; i < 50; i++) await run();
    assert.equal(d().liveArrays(), live);
  });

  itUnless(mlxSkip, "reports the loaded library and runtime", () => {
    assert.equal(d().name, "mlx");
    assert.equal(d().kind, "gpu");
    assert.match(d().info.libPath, /libmlxc\.dylib$/);
    assert.ok(["node", "bun", "deno"].includes(d().info.runtime));
    if ((globalThis as { Deno?: unknown }).Deno) assert.equal(d().info.runtime, "deno");
  });

  itUnless(mlxSkip, "the CPU device computes the same results", async () => {
    const cpu = createMlxDevice({ device: "cpu" });
    const t = Tensor.fromTypedArray(Float32Array.from([1, 2, 3, 4]), [2, 2], { dtype: "f32" });
    const [gt, ct] = await Promise.all([d().fromTensor(t), cpu.fromTensor(t)]);
    const g = await gt!.matmul(gt!).toTensor();
    const c = await ct!.matmul(ct!).toTensor();
    assert.equal(cpu.kind, "cpu");
    assert.deepEqual([...c.data], [...g.data]);
    assert.deepEqual([...c.data], [7, 10, 15, 22]);
  });
});
