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

// Each file imports bun:test itself (see testFns in helpers.ts).
// @ts-ignore -- bun types are not installed
const bunTest: unknown = (globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null;
const { describe, it, itUnless } = testFns(bunTest);

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
      const a = d().fromTensor(t);
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
    const s = d().fromTensor(Tensor.full([], 4, { dtype: "f32" }));
    assert.deepEqual(s.shape, []);
    assert.deepEqual([...(await s.mul(2).toTensor()).data], [8]);
    const base = Tensor.fromTypedArray(Float32Array.from([0, 1, 2, 3, 4, 5]), [3, 2], { dtype: "f32" });
    assert.deepEqual([...(await d().fromTensor(base.select(0, 1)).toTensor()).data], [2, 3]);
  });

  itUnless(mlxSkip, "ops refuse tensor-core Tensors and foreign-device arrays (no implicit transfers)", () => {
    const a = d().fromTensor(Tensor.from([1, 2, 3]));
    const host = Tensor.from([1, 2, 3]);
    assert.throws(() => a.add(host as unknown as MlxArray), /tensor-core Tensor.*fromTensor\(\)/);
    assert.throws(() => a.matmul(host as unknown as MlxArray), /no implicit transfers/);
    const other = createMlxDevice();
    const b = other.fromTensor(Tensor.from([1, 2, 3]));
    assert.throws(() => a.add(b), /different MlxDevice/);
    b.dispose();
    other.destroy();
  });

  itUnless(mlxSkip, "no implicit dtype promotion; float-only ops refuse integers", () => {
    const f = d().fromTensor(Tensor.from([1, 2, 3]));
    const i = d().fromTensor(Tensor.from([1, 2, 3], { dtype: "i32" }));
    assert.throws(() => f.add(i), /dtype mismatch f32 vs i32/);
    assert.throws(() => i.exp(), /needs a float dtype/);
    assert.throws(() => i.mean(), /needs a float dtype/);
    assert.equal(i.add(2).dtype, "i32", "number operands take the array's dtype");
    assert.equal(f.cast("f16").mul(0.5).dtype, "f16", "f16 stays f16");
    assert.equal(i.cast("f32").add(f).dtype, "f32");
  });

  itUnless(mlxSkip, "lazy graph: shape errors throw at the call site, values appear at eval/toTensor", async () => {
    const a = d().fromTensor(Tensor.from([1, 2, 3]));
    const b = d().fromTensor(Tensor.from([1, 2]));
    assert.throws(() => a.add(b), /broadcast/);
    const y = a.exp().sum();
    assert.equal(y.eval(), y, "eval() returns this");
    const got = (await y.toTensor()).item() as number;
    assert.ok(Math.abs(got - (Math.E + Math.exp(2) + Math.exp(3))) < 1e-4);
  });

  itUnless(mlxSkip, "scope frees intermediates, keeps returned arrays, and marks freed wrappers", async () => {
    const x = d().fromTensor(Tensor.from([1, 2]));
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
    const a = d().fromTensor(Tensor.from([1, 2, 3]));
    const t = await a.mul(2).toTensor();
    a.dispose();
    assert.deepEqual([...t.data], [2, 4, 6]);
  });

  itUnless(mlxSkip, "memory stays flat across repeated scoped work", async () => {
    const x = d().fromTensor(Tensor.fromTypedArray(new Float32Array(64 * 64).fill(0.5), [64, 64], { dtype: "f32" }));
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
    assert.ok(d().info.runtime === "node" || d().info.runtime === "bun");
  });

  itUnless(mlxSkip, "the CPU device computes the same results", async () => {
    const cpu = createMlxDevice({ device: "cpu" });
    const t = Tensor.fromTypedArray(Float32Array.from([1, 2, 3, 4]), [2, 2], { dtype: "f32" });
    const g = await d().fromTensor(t).matmul(d().fromTensor(t)).toTensor();
    const c = await cpu.fromTensor(t).matmul(cpu.fromTensor(t)).toTensor();
    assert.equal(cpu.kind, "cpu");
    assert.deepEqual([...c.data], [...g.data]);
    assert.deepEqual([...c.data], [7, 10, 15, 22]);
  });
});
