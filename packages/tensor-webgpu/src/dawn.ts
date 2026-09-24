/**
 * Node/Bun WebGPU via Dawn — the `./dawn` subpath.
 *
 * Since issue #146 the Dawn loader is `@johnhenry/backend-webgpu`'s
 * (`getGpu`), so the process has one Dawn instance per flag set no matter
 * which package asked for it. backend-webgpu depends on the `webgpu`
 * package (Dawn's official Node binding, prebuilt for darwin universal,
 * linux x64/arm64 and win32 x64/arm64) and hides it from browser bundles
 * behind its own `#dawn` import condition.
 *
 * ```ts
 * import { detectWebGPU } from "@johnhenry/math-plus-tensor-webgpu";
 * import { requestDawnGPU } from "@johnhenry/math-plus-tensor-webgpu/dawn";
 * const cap = await detectWebGPU({ gpu: (await requestDawnGPU({ unsafe: true }))! });
 * ```
 *
 * @deprecated Kept through the deprecation window (see the 0.2.0
 * changelog). `createWebGpuDevice()` finds the GPU itself (navigator.gpu,
 * else Dawn); `getGpu({ unsafe })` from `@johnhenry/backend-webgpu` is the
 * direct replacement.
 */
import { getGpu } from "@johnhenry/backend-webgpu";

export interface DawnOptions {
  /**
   * Create the Dawn instance with the `allow_unsafe_apis` toggle, which is
   * what exposes experimental features such as
   * `chromium-experimental-subgroup-matrix` (needed by the subgroup-matrix
   * GEMM kernel). It only unlocks experimental features/extensions; it does
   * not disable validation. Default false.
   */
  unsafe?: boolean;
}

/**
 * Returns a `GPU` entry point, or `null` when the `webgpu` package can't
 * load on this platform (never throws — callers skip/fall back on `null`).
 * In Node/Bun that is Dawn, whose WebGPU globals (`GPUBufferUsage`, …) are
 * installed on `globalThis` when missing; where `navigator.gpu` exists
 * (Deno, browsers) it is returned instead. One Dawn instance per option set
 * is created and reused.
 *
 * @deprecated See the module doc.
 */
export function requestDawnGPU(options: DawnOptions = {}): Promise<GPU | null> {
  return getGpu({ unsafe: options.unsafe ?? false });
}
