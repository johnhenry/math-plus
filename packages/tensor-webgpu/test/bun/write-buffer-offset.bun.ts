/**
 * Standalone Bun repro for the `queue.writeBuffer` byteOffset bug (issue
 * #126), run by test/write-buffer.test.ts as `bun test/bun/write-buffer-offset.bun.ts`
 * (the node:test suite itself runs under Node). Prints one JSON line:
 *
 *  - `runtime`: bytes uploaded through this package's paths (`writeBytes`,
 *    `GPUTensor.fromFloat32Array` / `fromFloat16Bits` on views with a
 *    non-zero byteOffset) — must equal the view's own elements.
 *  - `rawView`: what `queue.writeBuffer(buffer, 0, view)` uploads for the
 *    same view. Under Bun 1.2.x + Dawn (`webgpu` 0.6.x) this is the START of
 *    the underlying ArrayBuffer (`[0, 1, 2, 3]`), not the view (`[4, 5, 6, 7]`) —
 *    reported, not asserted, so an upstream fix doesn't break the suite.
 *
 * Exits 2 when Dawn has no adapter (the caller skips), 1 on a mismatch.
 */
import { requestDawnGPU } from "../../src/dawn.ts";
import { GPUTensor } from "../../src/device.ts";
import { readBackBytes, writeBytes } from "../../src/gpu-runtime.ts";

const gpu = await requestDawnGPU();
const adapter = await gpu?.requestAdapter();
if (!adapter) {
  console.log(JSON.stringify({ unavailable: true }));
  process.exit(2);
}
const device = await adapter.requestDevice();

const backing = new Float32Array(16).map((_, i) => i);
const view = backing.subarray(4, 8); // byteOffset 16

const fresh = (): GPUBuffer =>
  device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

const raw = fresh();
device.queue.writeBuffer(raw, 0, view);
const rawView = Array.from(new Float32Array(await readBackBytes(device, raw, 16)));

const viaWriteBytes = fresh();
writeBytes(device, viaWriteBytes, 0, view);
const writeBytesOut = Array.from(new Float32Array(await readBackBytes(device, viaWriteBytes, 16)));

const t32 = GPUTensor.fromFloat32Array(device, view, [4]);
const tensorF32 = Array.from(await t32.toFloat32Array());

const bits = new Uint16Array(12).map((_, i) => 0x3c00 + i);
const t16 = GPUTensor.fromFloat16Bits(device, bits.subarray(3, 7), [4]); // byteOffset 6: not even 4-aligned
const tensorF16 = Array.from(await t16.toUint16Array());

const expected = { f32: [4, 5, 6, 7], f16: [0x3c03, 0x3c04, 0x3c05, 0x3c06] };
const ok =
  JSON.stringify(writeBytesOut) === JSON.stringify(expected.f32) &&
  JSON.stringify(tensorF32) === JSON.stringify(expected.f32) &&
  JSON.stringify(tensorF16) === JSON.stringify(expected.f16);
console.log(
  JSON.stringify({
    runtime: typeof (globalThis as { Bun?: unknown }).Bun !== "undefined" ? "bun" : "node",
    ok,
    writeBytes: writeBytesOut,
    tensorF32,
    tensorF16,
    rawView,
    rawViewIgnoresByteOffset: JSON.stringify(rawView) !== JSON.stringify(expected.f32),
  }),
);
t32.free();
t16.free();
device.destroy();
process.exit(ok ? 0 : 1);
