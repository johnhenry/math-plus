import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { SafetensorsError, readSafetensors, writeSafetensors } from "../src/index.ts";

test("round-trips every dtype, empty and scalar tensors, and metadata", () => {
  const bytes = writeSafetensors(
    {
      f32: { dtype: "F32", shape: [2, 2], data: new Float32Array([1, 2, 3, 4]) },
      f16: { dtype: "F16", shape: [2], data: new Float16Array([1.5, -0.25]) },
      bf16: { dtype: "BF16", shape: [1], data: new Uint16Array([0x3fc0]) },
      f64: { dtype: "F64", shape: [1], data: new Float64Array([Math.PI]) },
      i64: { dtype: "I64", shape: [2], data: new BigInt64Array([-(2n ** 60n), 7n]) },
      u64: { dtype: "U64", shape: [1], data: new BigUint64Array([2n ** 63n]) },
      i32: { dtype: "I32", shape: [1], data: new Int32Array([-1]) },
      u32: { dtype: "U32", shape: [1], data: new Uint32Array([4e9]) },
      i16: { dtype: "I16", shape: [1], data: new Int16Array([-2]) },
      u16: { dtype: "U16", shape: [1], data: new Uint16Array([65535]) },
      i8: { dtype: "I8", shape: [1], data: new Int8Array([-3]) },
      u8: { dtype: "U8", shape: [3], data: new Uint8Array([1, 2, 3]) },
      bool: { dtype: "BOOL", shape: [2], data: new Uint8Array([1, 0]) },
      empty: { dtype: "F32", shape: [0, 5], data: new Float32Array(0) },
      scalar: { dtype: "F32", shape: [], data: new Float32Array([9]) },
    },
    { format: "pt", note: "é ünïcode" },
  );
  const file = readSafetensors(bytes);
  assert.deepEqual(file.metadata, { format: "pt", note: "é ünïcode" });
  assert.equal(file.header.dataStart % 8, 0, "data section is 8-byte aligned");
  assert.deepEqual([...(file.view("f32") as Float32Array)], [1, 2, 3, 4]);
  assert.deepEqual([...(file.view("f16") as Float16Array)], [1.5, -0.25]);
  assert.deepEqual([...file.toF32("bf16")], [1.5]);
  assert.deepEqual([...(file.view("i64") as BigInt64Array)], [-(2n ** 60n), 7n]);
  assert.deepEqual([...(file.view("u64") as BigUint64Array)], [2n ** 63n]);
  assert.deepEqual([...(file.view("u32") as Uint32Array)], [4e9]);
  assert.deepEqual([...(file.view("bool") as Uint8Array)], [1, 0]);
  assert.deepEqual(file.info("empty").shape, [0, 5]);
  assert.deepEqual(file.info("scalar").shape, []);
  assert.equal(file.toF32("scalar")[0], 9);
  // every tensor starts at a multiple of its element size (largest dtypes first)
  for (const name of file.names()) {
    const info = file.info(name);
    assert.equal((file.header.dataStart + info.dataOffsets[0]) % file.view(name).BYTES_PER_ELEMENT, 0, name);
  }
});

test("tensor order: dtype alignment descending, then name — integer-like names included", () => {
  const bytes = writeSafetensors({
    b: { dtype: "U8", shape: [1], data: new Uint8Array([1]) },
    "10": { dtype: "F32", shape: [1], data: new Float32Array([1]) },
    a: { dtype: "U8", shape: [1], data: new Uint8Array([2]) },
    "2": { dtype: "F32", shape: [1], data: new Float32Array([2]) },
    z: { dtype: "I64", shape: [1], data: new BigInt64Array([3n]) },
  });
  const file = readSafetensors(bytes);
  assert.deepEqual(file.names(), ["z", "10", "2", "a", "b"]);
});

test("accepts a Map and subarray views with non-zero byteOffset", () => {
  const backing = new Float32Array([0, 1, 2, 3]);
  const bytes = writeSafetensors(new Map([["x", { dtype: "F32" as const, shape: [2], data: backing.subarray(2) }]]));
  assert.deepEqual([...(readSafetensors(bytes).view("x") as Float32Array)], [2, 3]);
});

test("rejects bad inputs with SafetensorsError", () => {
  const bad = (fn: () => unknown, code: string) =>
    assert.throws(fn, (e: unknown) => e instanceof SafetensorsError && e.code === code);
  bad(() => writeSafetensors({ x: { dtype: "F32", shape: [3], data: new Float32Array(2) } }), "TensorInvalidInfo");
  bad(() => writeSafetensors({ x: { dtype: "F16", shape: [2], data: new Float32Array(2) } }), "TensorInvalidInfo");
  bad(() => writeSafetensors({ x: { dtype: "Q4" as never, shape: [1], data: new Uint8Array(1) } }), "InvalidDtype");
  bad(() => writeSafetensors({ x: { dtype: "U8", shape: [-1], data: new Uint8Array(0) } }), "InvalidShape");
  bad(() => writeSafetensors({ __metadata__: { dtype: "U8", shape: [1], data: new Uint8Array(1) } }), "InvalidHeaderDeserialization");
  bad(() => writeSafetensors({}, { k: 1 as unknown as string }), "InvalidHeaderDeserialization");
});
