import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { load, onnx, UnsupportedDTypeError } from "../src/index.ts";

const FIXTURES = fileURLToPath(new URL("./fixtures/", import.meta.url));

test("load + run: real ONNX model (Add) over float32 tensors", async () => {
  const model = await load(new Uint8Array(readFileSync(`${FIXTURES}tiny_add.onnx`)));
  assert.deepEqual([...model.inputNames], ["a", "b"]);
  assert.deepEqual([...model.outputNames], ["sum"]);

  const a = Tensor.from([1, 2, 3], { dtype: "f32" });
  const b = Tensor.from([10, 20, 30], { dtype: "f32" });
  const outputs = await model.run({ a, b });

  assert.ok(outputs.sum);
  const sum = outputs.sum as Tensor;
  assert.equal(sum.dtype, "f32");
  assert.deepEqual([...sum.shape], [3]);
  assert.deepEqual(sum.toArray(), [11, 22, 33]);

  await model.release();
});

test("load accepts a file path string, not just bytes", async () => {
  const model = await load(`${FIXTURES}tiny_add.onnx`);
  const outputs = await model.run({
    a: Tensor.from([5], { dtype: "f32" }),
    b: Tensor.from([7], { dtype: "f32" }),
  });
  assert.deepEqual((outputs.sum as Tensor).toArray(), [12]);
  await model.release();
});

test("onnx.load namespace form matches the issue's stated API", async () => {
  const model = await onnx.load(new Uint8Array(readFileSync(`${FIXTURES}tiny_add.onnx`)));
  const outputs = await model.run({
    a: Tensor.from([1], { dtype: "f32" }),
    b: Tensor.from([1], { dtype: "f32" }),
  });
  assert.deepEqual((outputs.sum as Tensor).toArray(), [2]);
  await model.release();
});

test("i64 input marshalling: the exact dtype need cited in the issue (input_ids-style int64 tensors)", async () => {
  const model = await load(new Uint8Array(readFileSync(`${FIXTURES}tiny_i64_scale.onnx`)));
  assert.deepEqual([...model.inputNames], ["ids", "scale"]);

  const ids = Tensor.fromTypedArray(new BigInt64Array([1n, 2n, 3n]), [3], { dtype: "i64" });
  const scale = Tensor.from([10, 20, 30], { dtype: "f32" });
  const outputs = await model.run({ ids, scale });

  const scaled = outputs.scaled as Tensor;
  assert.equal(scaled.dtype, "f32");
  assert.deepEqual(scaled.toArray(), [10, 40, 90]);

  await model.release();
});

test("marshals a non-contiguous (transposed) input Tensor correctly — proves .contiguous() is applied before handing data to ORT, not the raw (wrongly-ordered) storage buffer", async () => {
  const model = await load(new Uint8Array(readFileSync(`${FIXTURES}tiny_add_2d.onnx`)));
  const base = Tensor.from([1, 2, 3, 4], { dtype: "f32" }).reshape([2, 2]);
  const transposed = base.transpose(); // a VIEW, logically [[1,3],[2,4]]; underlying storage is still [1,2,3,4]
  const zeros = Tensor.zeros([2, 2], { dtype: "f32" });

  const outputs = await model.run({ a: transposed, b: zeros });
  const sum = outputs.sum as Tensor;
  // Row-major flatten of the LOGICAL (transposed) values, not the raw underlying buffer order.
  assert.deepEqual(sum.toArray(), [
    [1, 3],
    [2, 4],
  ]);
  await model.release();
});

test("bf16 has no ONNX Runtime Web equivalent — throws UnsupportedDTypeError rather than silently mis-mapping to float16", async () => {
  const model = await load(new Uint8Array(readFileSync(`${FIXTURES}tiny_add.onnx`)));
  const bf16Tensor = Tensor.zeros([3], { dtype: "bf16" });
  await assert.rejects(() => model.run({ a: bf16Tensor, b: bf16Tensor }), UnsupportedDTypeError);
  await model.release();
});

test("inputNames/outputNames are exposed before any run() call", async () => {
  const model = await load(new Uint8Array(readFileSync(`${FIXTURES}tiny_i64_scale.onnx`)));
  assert.deepEqual([...model.inputNames], ["ids", "scale"]);
  assert.deepEqual([...model.outputNames], ["scaled"]);
  await model.release();
});
