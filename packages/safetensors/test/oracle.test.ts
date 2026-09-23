/**
 * Differential tests against the Python reference (`safetensors` +
 * numpy/torch), both directions:
 *   Python writes (save_file / serialize / a hand-built MLX-style unpadded
 *   file) -> we read: bytes and float32 conversions must match bit-for-bit.
 *   We write -> Python's reader (safetensors.deserialize) reads it back, and
 *   our bytes must equal the reference serializer's output byte-for-byte.
 * Skips (never fails) without a Python that has numpy + safetensors.
 */
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { openSafetensors, readSafetensors, writeSafetensors, type SafeDType, type TensorInput } from "../src/index.ts";
import { fromB64, oracleSkip, runOracle, sameF32, toB64 } from "./helpers.ts";

interface Dump {
  metadata: Record<string, string> | null;
  tensors: Record<string, { dtype: SafeDType; shape: number[]; bytes_b64: string; f32_b64: string }>;
}

function f32(b64: string): Float32Array {
  const bytes = fromB64(b64);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

test("reads reference-written files exactly (memory, lazy file, lazy Blob)", { skip: oracleSkip }, async (t) => {
  const dir = mkdtempSync(join(tmpdir(), "safetensors-oracle-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const manifest = runOracle("make", dir) as { files: Record<string, string>; bf16: string | null };
  assert.ok(manifest.bf16, "bf16 fixture needs torch or ml_dtypes in the oracle python");
  assert.deepEqual(Object.keys(manifest.files).sort(), ["bf16", "misaligned", "mixed"]);

  for (const [label, path] of Object.entries(manifest.files)) {
    await t.test(label, async () => {
      const expected = runOracle("dump", path) as Dump;
      const bytes = new Uint8Array(readFileSync(path));
      const mem = readSafetensors(bytes);
      const lazyFile = await openSafetensors(path);
      const lazyBlob = await openSafetensors(new Blob([bytes]));
      try {
        assert.deepEqual(mem.metadata, expected.metadata ?? {});
        assert.deepEqual(new Set(mem.names()), new Set(Object.keys(expected.tensors)));
        if (label === "misaligned") assert.notEqual(mem.header.dataStart % 2, 0, "fixture really is misaligned");
        for (const [name, e] of Object.entries(expected.tensors)) {
          const info = mem.info(name);
          assert.equal(info.dtype, e.dtype, name);
          assert.deepEqual(info.shape, e.shape, name);
          const want = fromB64(e.bytes_b64);
          assert.deepEqual(mem.bytes(name), want, `${name}: bytes`);
          assert.deepEqual(await lazyFile.readBytes(name), want, `${name}: lazy file bytes`);
          assert.deepEqual(await lazyBlob.readBytes(name), want, `${name}: lazy blob bytes`);
          // typed view reads the same bytes back (also for misaligned data)
          const view = mem.view(name);
          assert.deepEqual(new Uint8Array(view.buffer, view.byteOffset, view.byteLength), want, `${name}: view`);
          // float32 conversion vs numpy astype(float32) / torch .float()
          const diff = sameF32(mem.toF32(name), f32(e.f32_b64)) ?? sameF32(await lazyFile.toF32(name), f32(e.f32_b64));
          assert.equal(diff, undefined, `${name}: toF32 ${diff}`);
        }
        const all = await lazyFile.readMany();
        for (const [name, e] of Object.entries(expected.tensors)) {
          const v = all.get(name)!;
          assert.deepEqual(new Uint8Array(v.buffer, v.byteOffset, v.byteLength), fromB64(e.bytes_b64), `${name}: readMany`);
        }
      } finally {
        await lazyFile.close();
      }
    });
  }
});

test("writer output is read back by Python and equals the reference serializer byte-for-byte", { skip: oracleSkip }, async (t) => {
  const dir = mkdtempSync(join(tmpdir(), "safetensors-oracle-w-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const tensors: Record<string, TensorInput> = {
    weight: { dtype: "F32", shape: [2, 3], data: new Float32Array([0.5, -1, 2, Number.NaN, Infinity, -0]) },
    half: { dtype: "F16", shape: [4], data: new Float16Array([1.5, -0.1, 65504, 6e-8]) },
    brain: { dtype: "BF16", shape: [2], data: new Uint16Array([0x3fc0, 0xff80]) },
    ids: { dtype: "I64", shape: [3], data: new BigInt64Array([-(2n ** 40n), 0n, 2n ** 52n]) },
    big: { dtype: "U64", shape: [1], data: new BigUint64Array([12345n]) },
    dbl: { dtype: "F64", shape: [1], data: new Float64Array([Math.E]) },
    mask: { dtype: "BOOL", shape: [2, 2], data: new Uint8Array([1, 0, 0, 1]) },
    i8: { dtype: "I8", shape: [2], data: new Int8Array([-128, 127]) },
    u16: { dtype: "U16", shape: [1], data: new Uint16Array([65535]) },
    i32: { dtype: "I32", shape: [1], data: new Int32Array([-7]) },
    "0": { dtype: "U8", shape: [3], data: new Uint8Array([1, 2, 3]) },
    "layers.10.attn": { dtype: "F32", shape: [0], data: new Float32Array(0) },
    scalar: { dtype: "F32", shape: [], data: new Float32Array([4.5]) },
  };
  for (const metadata of [undefined, { format: "pt" }]) {
    await t.test(metadata ? "with metadata" : "without metadata", () => {
      const ours = writeSafetensors(tensors, metadata);
      const ourPath = join(dir, "ours.safetensors");
      writeFileSync(ourPath, ours);
      const dumped = runOracle("dump", ourPath) as Dump;
      assert.deepEqual(dumped.metadata, metadata ?? null);
      for (const [name, input] of Object.entries(tensors)) {
        const e = dumped.tensors[name]!;
        assert.equal(e.dtype, input.dtype, name);
        assert.deepEqual(e.shape, [...input.shape], name);
        assert.deepEqual(fromB64(e.bytes_b64), new Uint8Array(input.data.buffer, input.data.byteOffset, input.data.byteLength), name);
      }
      const spec = {
        metadata: metadata ?? null,
        tensors: Object.fromEntries(
          Object.entries(tensors).map(([n, x]) => [n, { dtype: x.dtype, shape: x.shape, bytes_b64: toB64(new Uint8Array(x.data.buffer, x.data.byteOffset, x.data.byteLength)) }]),
        ),
      };
      const specPath = join(dir, "spec.json");
      const refPath = join(dir, "ref.safetensors");
      writeFileSync(specPath, JSON.stringify(spec));
      runOracle("serialize", specPath, refPath);
      const reference = new Uint8Array(readFileSync(refPath));
      assert.equal(new TextDecoder().decode(ours.subarray(8, 8 + 200)), new TextDecoder().decode(reference.subarray(8, 8 + 200)));
      assert.deepEqual(ours, reference, "byte-identical to safetensors.serialize");
    });
  }
});
