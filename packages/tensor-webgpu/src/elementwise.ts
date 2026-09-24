/**
 * Dispatch side of the IR -> WGSL fusion (issue #12 / #11's IR), on
 * `@johnhenry/backend-webgpu`'s runtime since issue #146: the traced
 * expression is lowered by fusion-wgsl.ts's `compileIRToElementwise` and
 * run by the backend's `elementwise` hook (0.3.1), so it is compiled,
 * cached and batched by the same runtime that runs every other op on the
 * device, broadcasts its inputs like every other elementwise op, and
 * consumes and produces backend tensors without a host round trip (see
 * `WebGpuDevice.fuse` in facade.ts).
 */
import type { WebGpuBackend, WebGpuTensor } from "@johnhenry/backend-webgpu";
import type { IRNode } from "@johnhenry/math-plus-tensor-compile";
import { backendFor, readRaw, uploadSync } from "./bridge.ts";
import { compileIRToElementwise } from "./fusion-wgsl.ts";

/**
 * One fused dispatch of `node` over `inputs` (f32; shapes broadcast
 * NumPy-style against each other; views with an element offset are fine).
 * Returns a new f32 tensor of the broadcast shape, tracked by the enclosing
 * backend `scope` (otherwise the caller disposes it).
 */
export function encodeFused(b: WebGpuBackend, node: IRNode, inputs: readonly WebGpuTensor[]): WebGpuTensor {
  if (!inputs.length) throw new RangeError("fused elementwise: needs at least one input");
  for (const x of inputs) {
    if (x.dtype !== "f32") throw new TypeError(`fused elementwise: f32 inputs only (got ${x.dtype}); cast first`);
    if (x.disposed) throw new Error("fused elementwise: input used after dispose");
  }
  const { expr, helpers } = compileIRToElementwise(node, inputs.length);
  return b.elementwise(expr, inputs, { helpers });
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
 * `gpu.compile(n, fn)`, which keep inputs and result on the GPU and
 * broadcast.
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
  compileIRToElementwise(node, inputs.length); // surface IR errors before touching the GPU
  if (elementCount === 0) return new Float32Array(0);
  const b = backendFor(device);
  const ins = inputs.map((d) => uploadSync(b, d, [elementCount], "f32"));
  let out: WebGpuTensor | undefined;
  try {
    out = encodeFused(b, node, ins);
    return new Float32Array(await readRaw(b, out));
  } finally {
    for (const x of ins) b.dispose(x);
    if (out) b.dispose(out);
  }
}
