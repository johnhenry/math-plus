import assert from "node:assert/strict";
import { test } from "node:test";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { openSafetensors, readSafetensors, writeSafetensors } from "../src/index.ts";
import { bytesToTensor, fromTensor, readTensor, toTensor } from "../src/tensor.ts";

test("toTensor maps dtypes 1:1; f16/bf16 stay bit patterns that tensor-core decodes", () => {
  const file = readSafetensors(
    writeSafetensors({
      w: { dtype: "F32", shape: [2, 2], data: new Float32Array([1, 2, 3, 4]) },
      h: { dtype: "F16", shape: [2], data: new Float16Array([1.5, -0.25]) },
      b: { dtype: "BF16", shape: [2], data: new Uint16Array([0x3fc0, 0xc040]) },
      i: { dtype: "I64", shape: [1], data: new BigInt64Array([-9n]) },
      m: { dtype: "BOOL", shape: [2], data: new Uint8Array([1, 0]) },
    }),
  );
  const w = toTensor(file, "w");
  assert.equal(w.dtype, "f32");
  assert.deepEqual(w.toArray(), [[1, 2], [3, 4]]);
  assert.equal(w.data.buffer, file.bytes("w").buffer, "aligned data is aliased, not copied");
  const h = toTensor(file, "h");
  assert.equal(h.dtype, "f16");
  assert.ok(h.data instanceof Uint16Array);
  assert.deepEqual(h.toArray(), [1.5, -0.25]);
  assert.deepEqual(h.cast("f32").toArray(), [1.5, -0.25]);
  assert.deepEqual(toTensor(file, "b").toArray(), [1.5, -3]);
  assert.deepEqual(toTensor(file, "i").toArray(), [-9n]);
  assert.equal(toTensor(file, "m").dtype, "bool");
});

test("fromTensor feeds the writer, packing non-contiguous views", () => {
  const t = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f32" }).reshape([2, 3]).transpose();
  const h = Tensor.from([0.5, 2], { dtype: "f16" });
  const file = readSafetensors(writeSafetensors({ t: fromTensor(t), h: fromTensor(h), s: fromTensor(Tensor.arange(4, undefined, 1, { dtype: "i64" }).slice({ start: 1 })) }));
  assert.deepEqual(toTensor(file, "t").toArray(), [[1, 4], [2, 5], [3, 6]]);
  assert.deepEqual(toTensor(file, "h").toArray(), [0.5, 2]);
  assert.deepEqual(toTensor(file, "s").toArray(), [1n, 2n, 3n]);
});

test("readTensor reads lazily; bytesToTensor copies misaligned input", async () => {
  const bytes = writeSafetensors({ x: { dtype: "F16", shape: [3], data: new Float16Array([1, 2, 3]) } });
  const lazy = await openSafetensors(new Blob([bytes]));
  assert.deepEqual((await readTensor(lazy, "x")).toArray(), [1, 2, 3]);
  const odd = new Uint8Array(7);
  odd.set(new Uint8Array(new Float16Array([4, 5, 6]).buffer), 1);
  assert.deepEqual(bytesToTensor("F16", [3], odd.subarray(1)).toArray(), [4, 5, 6]);
});
