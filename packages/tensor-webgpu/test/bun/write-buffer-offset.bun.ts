/**
 * Standalone Bun repro for the `queue.writeBuffer` byteOffset bug (issue
 * #126), run by test/write-buffer.test.ts as `bun test/bun/write-buffer-offset.bun.ts`
 * (the node:test suite itself runs under Node). Prints one JSON line:
 *
 *  - `runtime`: bytes uploaded through this package's paths (the facade's
 *    `fromHost`, i.e. backend-webgpu's `Runtime.write`, on views with a
 *    non-zero byteOffset: an f32 subarray, and an f16 one at a byteOffset
 *    that is not 4-aligned when the device has shader-f16) — must equal
 *    the view's own elements.
 *  - `rawView`: what `queue.writeBuffer(buffer, 0, view)` uploads for the
 *    same view. Under Bun 1.2.x + Dawn (`webgpu` 0.6.x) this is the START of
 *    the underlying ArrayBuffer (`[0, 1, 2, 3]`), not the view (`[4, 5, 6, 7]`) —
 *    reported, not asserted, so an upstream fix doesn't break the suite.
 *
 * Exits 2 when Dawn has no adapter (the caller skips), 1 on a mismatch.
 */
import { getGpu } from "@johnhenry/backend-webgpu";
import { detectWebGPU } from "../../src/device.ts";
import { createWebGpuDevice } from "../../src/facade.ts";

const gpuEntry = await getGpu();
const cap = gpuEntry ? await detectWebGPU({ gpu: gpuEntry }) : undefined;
if (!cap?.available || !cap.device) {
  console.log(JSON.stringify({ unavailable: true }));
  process.exit(2);
}
const device = cap.device;
const gpu = await createWebGpuDevice({ device });

const backing = new Float32Array(16).map((_, i) => i);
const view = backing.subarray(4, 8); // byteOffset 16

const fresh = (): GPUBuffer =>
  device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

const rt = gpu.backend.rt;
const raw = fresh();
device.queue.writeBuffer(raw, 0, view);
const rawView = Array.from(new Float32Array(await rt.readBytes(raw, 0, 16)));

const viaWrite = fresh();
rt.write(viaWrite, false, view);
const writeBytesOut = Array.from(new Float32Array(await rt.readBytes(viaWrite, 0, 16)));

const t32 = await gpu.fromHost({ dtype: "f32", shape: [4], data: view });
const tensorF32 = Array.from((await gpu.toHost(t32)).data);

const expected = { f32: [4, 5, 6, 7], f16: [0x3c03, 0x3c04, 0x3c05, 0x3c06] };
let tensorF16: number[] | null = null;
if (gpu.supports("f16")) {
  const halves = new Float16Array(12).map((_, i) => 1 + i / 1024); // bits 0x3c00 + i
  const t16 = await gpu.fromHost({ dtype: "f16", shape: [4], data: halves.subarray(3, 7) }); // byteOffset 6: not even 4-aligned
  const h = (await gpu.toHost(t16)).data as Float16Array;
  tensorF16 = Array.from(new Uint16Array(h.buffer, h.byteOffset, h.length));
  gpu.dispose(t16);
}

const ok =
  JSON.stringify(writeBytesOut) === JSON.stringify(expected.f32) &&
  JSON.stringify(tensorF32) === JSON.stringify(expected.f32) &&
  (tensorF16 === null || JSON.stringify(tensorF16) === JSON.stringify(expected.f16));
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
gpu.dispose(t32);
gpu.destroy();
device.destroy();
process.exit(ok ? 0 : 1);
