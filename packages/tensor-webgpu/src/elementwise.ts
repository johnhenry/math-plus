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

