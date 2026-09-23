import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { MAX_HEADER, SafetensorsError, headerLength, parseHeader, readSafetensors, toFloat32, viewAs } from "../src/index.ts";
import { craft } from "./helpers.ts";

function code(fn: () => unknown, expected: string): void {
  assert.throws(fn, (e: unknown) => {
    assert.ok(e instanceof SafetensorsError, `expected SafetensorsError, got ${String(e)}`);
    assert.equal(e.code, expected, e.message);
    return true;
  });
}

const f32 = (...v: number[]) => new Uint8Array(new Float32Array(v).buffer);

test("parses a minimal valid file; metadata, names, info, views", () => {
  const file = readSafetensors(craft({ __metadata__: { a: "b" }, x: { dtype: "F32", shape: [2], data_offsets: [0, 8] } }, f32(1, 2), true));
  assert.deepEqual(file.metadata, { a: "b" });
  assert.deepEqual(file.names(), ["x"]);
  assert.equal(file.has("x"), true);
  assert.equal(file.has("y"), false);
  assert.deepEqual(file.info("x"), { name: "x", dtype: "F32", shape: [2], dataOffsets: [0, 8] });
  assert.deepEqual([...(file.view("x") as Float32Array)], [1, 2]);
  assert.equal(file.header.dataLength, 8);
  assert.equal(file.header.dataStart % 8, 0);
  code(() => file.info("y"), "TensorNotFound");
});

test("empty header and `__metadata__: null` (as MLX writes it) are accepted", () => {
  assert.equal(readSafetensors(craft("{}")).names().length, 0);
  const f = readSafetensors(craft({ __metadata__: null, x: { dtype: "U8", shape: [1], data_offsets: [0, 1] } }, new Uint8Array([7])));
  assert.deepEqual(f.metadata, {});
  assert.deepEqual([...(f.view("x") as Uint8Array)], [7]);
});

test("header length: too small, too large, truncated, exceeds file", () => {
  code(() => headerLength(new Uint8Array(4)), "HeaderTooSmall");
  const big = new Uint8Array(8);
  new DataView(big.buffer).setBigUint64(0, BigInt(MAX_HEADER + 1), true);
  code(() => parseHeader(big), "HeaderTooLarge");
  const full = craft({ x: { dtype: "U8", shape: [1], data_offsets: [0, 1] } }, new Uint8Array([1]));
  code(() => parseHeader(full.subarray(0, 20)), "InvalidHeaderLength");
  const lying = craft("{}");
  new DataView(lying.buffer).setBigUint64(0, 1000n, true);
  code(() => readSafetensors(lying), "InvalidHeaderLength");
});

test("header text: bad UTF-8, must start with '{', JSON errors, non-object JSON", () => {
  const bad = craft("{}");
  bad[8] = 0xff;
  code(() => readSafetensors(bad), "InvalidHeader");
  code(() => readSafetensors(craft(' {"a":1}')), "InvalidHeaderStart");
  code(() => readSafetensors(craft("[]")), "InvalidHeaderStart");
  code(() => readSafetensors(craft('{"x": {"dtype": "F32",')), "InvalidHeaderDeserialization");
  code(() => readSafetensors(craft('{"x": 5}')), "InvalidHeaderDeserialization");
  code(() => readSafetensors(craft('{"__metadata__": {"k": 1}}')), "InvalidHeaderDeserialization");
  code(() => readSafetensors(craft('{"__metadata__": [1]}')), "InvalidHeaderDeserialization");
});

test("duplicate tensor names are rejected (JSON.parse would silently keep the last)", () => {
  const text = '{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}';
  code(() => readSafetensors(craft(text, new Uint8Array([1]))), "DuplicateTensor");
  // a nested key equal to a tensor name, and escaped quotes in names, are not duplicates
  const ok = '{"__metadata__":{"x":"dtype \\"x\\""},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x\\"":{"dtype":"U8","shape":[0],"data_offsets":[1,1]}}';
  assert.deepEqual(readSafetensors(craft(ok, new Uint8Array([1]))).names(), ["x", 'x"']);
});

test("per-tensor validation: dtype, shape, offsets, size mismatch, overflow", () => {
  const one = new Uint8Array(4);
  code(() => readSafetensors(craft({ x: { dtype: "F8_E4M3", shape: [4], data_offsets: [0, 4] } }, one)), "InvalidDtype");
  code(() => readSafetensors(craft({ x: { dtype: "toString", shape: [4], data_offsets: [0, 4] } }, one)), "InvalidDtype");
  code(() => readSafetensors(craft({ x: { dtype: "U8", shape: [-4], data_offsets: [0, 4] } }, one)), "InvalidShape");
  code(() => readSafetensors(craft({ x: { dtype: "U8", shape: [2.5], data_offsets: [0, 4] } }, one)), "InvalidShape");
  code(() => readSafetensors(craft({ x: { dtype: "U8", shape: [4], data_offsets: [0] } }, one)), "InvalidOffset");
  code(() => readSafetensors(craft({ x: { dtype: "U8", shape: [4], data_offsets: [4, 0] } }, one)), "InvalidOffset");
  code(() => readSafetensors(craft({ x: { dtype: "F32", shape: [2], data_offsets: [0, 4] } }, one)), "TensorInvalidInfo");
  code(() => readSafetensors(craft({ x: { dtype: "F64", shape: [2 ** 30, 2 ** 30], data_offsets: [0, 4] } }, one)), "ValidationOverflow");
});

test("offsets must tile the data section: gaps, overlaps, trailing/missing bytes", () => {
  const eight = new Uint8Array(8);
  const u8 = (b: number, e: number) => ({ dtype: "U8", shape: [e - b], data_offsets: [b, e] });
  // gap at the start / in the middle
  code(() => readSafetensors(craft({ x: u8(2, 8) }, eight)), "InvalidOffset");
  code(() => readSafetensors(craft({ x: u8(0, 3), y: u8(4, 8) }, eight)), "InvalidOffset");
  // overlap
  code(() => readSafetensors(craft({ x: u8(0, 5), y: u8(4, 8) }, eight)), "InvalidOffset");
  // out of range: offsets past the end of the buffer
  code(() => readSafetensors(craft({ x: u8(0, 16) }, eight)), "MetadataIncompleteBuffer");
  // trailing bytes after the last tensor
  code(() => readSafetensors(craft({ x: u8(0, 4) }, eight)), "MetadataIncompleteBuffer");
  // header parsing alone (no file size) still catches gaps/overlaps but not trailing bytes
  assert.equal(parseHeader(craft({ x: u8(0, 4) }, eight)).dataLength, 4);
  // out-of-order JSON with contiguous offsets is fine; empty tensors can sit anywhere
  const f = readSafetensors(craft({ y: u8(4, 8), e: { dtype: "F32", shape: [0, 3], data_offsets: [4, 4] }, x: u8(0, 4) }, eight));
  assert.deepEqual(f.names(), ["y", "e", "x"]);
  assert.equal(f.view("e").length, 0);
});

test("misaligned data (unpadded header, as MLX writes) is copied once into an aligned view", () => {
  const bytes = craft({ x: { dtype: "F32", shape: [2], data_offsets: [0, 8] } }, f32(1.5, -2));
  assert.notEqual((8 + JSON.stringify({ x: { dtype: "F32", shape: [2], data_offsets: [0, 8] } }).length) % 4, 0);
  const file = readSafetensors(bytes);
  const v = file.view("x") as Float32Array;
  assert.deepEqual([...v], [1.5, -2]);
  assert.notEqual(v.buffer, bytes.buffer); // copied
  const aligned = readSafetensors(craft({ x: { dtype: "F32", shape: [2], data_offsets: [0, 8] } }, f32(1.5, -2), true));
  assert.equal(aligned.view("x").buffer, aligned.bytes("x").buffer); // zero-copy
});

test("viewAs / toFloat32 dtype coverage", () => {
  assert.ok(viewAs("F16", new Uint8Array(new Uint16Array([0x3e00]).buffer)) instanceof Float16Array);
  assert.equal(viewAs("F16", new Uint8Array(new Uint16Array([0x3e00]).buffer))[0], 1.5);
  assert.ok(viewAs("BF16", new Uint8Array(2)) instanceof Uint16Array);
  assert.ok(viewAs("I64", new Uint8Array(8)) instanceof BigInt64Array);
  assert.ok(viewAs("BOOL", new Uint8Array(1)) instanceof Uint8Array);
  code(() => viewAs("F32", new Uint8Array(3)), "TensorInvalidInfo");
  const src = f32(1, 2);
  const out = toFloat32("F32", src);
  assert.notEqual(out.buffer, src.buffer); // always a new array
  assert.deepEqual([...toFloat32("BF16", new Uint8Array(new Uint16Array([0x3fc0, 0xc040]).buffer))], [1.5, -3]);
  assert.deepEqual([...toFloat32("I64", new Uint8Array(new BigInt64Array([-5n, 2n ** 40n]).buffer))], [-5, 2 ** 40]);
  assert.deepEqual([...toFloat32("F16", new Uint8Array(new Uint16Array([0x3e00, 0x7c00]).buffer))], [1.5, Infinity]);
});
