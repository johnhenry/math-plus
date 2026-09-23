/**
 * Node/Bun WebGPU via Dawn — the documented native-addon install step
 * docs/PLAN.md §6.3 calls for ("Node support ships as 'works with a
 * documented native-addon install step', not a first-class guarantee").
 *
 * Published as the separate `./dawn` subpath (never re-exported from the
 * package root) so browser bundles of the main entry never see the `webgpu`
 * specifier. `webgpu` (Dawn's official Node binding, prebuilt for darwin
 * universal, linux x64/arm64 and win32 x64/arm64) is an OPTIONAL peer
 * dependency: install it yourself (`npm install webgpu`) to use this.
 *
 * ```ts
 * import { detectWebGPU } from "@johnhenry/math-plus-tensor-webgpu";
 * import { requestDawnGPU } from "@johnhenry/math-plus-tensor-webgpu/dawn";
 * const cap = await detectWebGPU({ gpu: (await requestDawnGPU({ unsafe: true }))! });
 * ```
 */

const instances = new Map<string, Promise<GPU | null>>();

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
 * Returns Dawn's `GPU` entry point, or `null` when the `webgpu` package
 * isn't installed or its native addon can't load on this platform (never
 * throws — callers skip/fall back on `null`). Also installs Dawn's WebGPU
 * globals (`GPUBufferUsage`, `GPUMapMode`, …) on `globalThis` when they're
 * missing, since this package's kernels reference them. One instance per
 * option set is created and reused.
 */
export function requestDawnGPU(options: DawnOptions = {}): Promise<GPU | null> {
  const flags = options.unsafe ? ["enable-dawn-features=allow_unsafe_apis"] : [];
  const key = flags.join(" ");
  let p = instances.get(key);
  if (!p) {
    p = loadDawn(flags);
    instances.set(key, p);
  }
  return p;
}

async function loadDawn(flags: string[]): Promise<GPU | null> {
  // A variable specifier keeps TypeScript (and bundlers) from resolving
  // "webgpu" statically — it's an optional peer, absent in most installs.
  const specifier = "webgpu";
  try {
    const mod = (await import(specifier)) as { create(flags: string[]): GPU; globals: Record<string, unknown> };
    const g = globalThis as Record<string, unknown>;
    for (const [name, value] of Object.entries(mod.globals)) {
      if (g[name] === undefined) g[name] = value;
    }
    return mod.create(flags);
  } catch {
    return null;
  }
}
