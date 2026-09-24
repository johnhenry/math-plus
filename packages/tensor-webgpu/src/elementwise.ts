/**
 * Dispatch side of the IR -> WGSL fusion (issue #12 / #11's IR), on
 * `@johnhenry/backend-webgpu`'s runtime since issue #146: the fused shader
 * from fusion-wgsl.ts's `compileIRToKernel` is compiled, cached and
 * batched by the same runtime that runs every other op on the device, so a
 * fused expression can consume and produce backend tensors without a host
 * round-trip (see `WebGpuDevice.fuse` in facade.ts).
 */
import type { WebGpuBackend, WebGpuTensor } from "@johnhenry/backend-webgpu";
import type { IRNode } from "@johnhenry/math-plus-tensor-compile";
import { allocate, backendFor, dispatchCustom, flatGrid, uploadSync } from "./bridge.ts";
import { compileIRToKernel, FUSED_WORKGROUP_SIZE } from "./fusion-wgsl.ts";

function numel(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * One fused dispatch of `node` over `inputs` (f32, all with the same
 * element count; views with an element offset are fine). Returns a new
 * untracked f32 tensor of `shape` (caller disposes).
 */
export function encodeFused(b: WebGpuBackend, node: IRNode, inputs: readonly WebGpuTensor[], shape: readonly number[]): WebGpuTensor {
  const n = numel(shape);
  for (const x of inputs) {
    if (x.dtype !== "f32") throw new TypeError(`fused elementwise: f32 inputs only (got ${x.dtype}); cast first`);
    if (x.disposed) throw new Error("fused elementwise: input used after dispose");
    if (numel(x.shape) !== n) {
      throw new RangeError(`fused elementwise: every input needs ${n} elements (the output's), got [${x.shape}] (broadcast first)`);
    }
  }
  const out = allocate(b, shape, "f32");
  if (n === 0) return out;
  const src = compileIRToKernel(node, inputs.length);
  const params: Record<string, number> = { n };
  inputs.forEach((x, j) => (params[`o${j}`] = x.offset));
  dispatchCustom(b, src.key, () => src, [...inputs.map((x) => x.storage.buffer), out.storage.buffer], params, flatGrid(Math.ceil(n / FUSED_WORKGROUP_SIZE)));
  return out;
}

/**
 * Run a compiled elementwise expression (a traced `IRNode` from
 * `@johnhenry/math-plus-tensor-compile`) on the GPU: one shader dispatch
 * touches every output element once, fusing however many ops the expression
 * chained — no intermediate GPU buffer per op. Host arrays in, host array
 * out; `inputs` must all have `elementCount` elements (broadcast first).
 *
 * @deprecated Kept through the deprecation window. New code:
 * `createWebGpuDevice()` and `gpu.fuse(node, tensors)` /
 * `gpu.compile(n, fn)`, which keep inputs and result on the GPU.
 */
export async function runElementwiseWGSL(
  device: GPUDevice,
  node: IRNode,
  inputs: readonly Float32Array[],
  elementCount: number,
): Promise<Float32Array> {
  if (inputs.some((a) => a.length !== elementCount)) {
    throw new RangeError(
      `runElementwiseWGSL: all inputs and the output must share elementCount ${elementCount} (broadcast first)`,
    );
  }
  compileIRToKernel(node, inputs.length); // surface IR errors before touching the GPU
  if (elementCount === 0) return new Float32Array(0);
  const b = backendFor(device);
  const ins = inputs.map((d) => uploadSync(b, d, [elementCount], "f32"));
  let out: WebGpuTensor | undefined;
  try {
    out = encodeFused(b, node, ins, [elementCount]);
    return new Float32Array(await b.rt.readBytes(out.storage.buffer, 0, elementCount * 4));
  } finally {
    for (const x of ins) b.dispose(x);
    if (out) b.dispose(out);
  }
}
