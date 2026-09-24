/**
 * Per-device record of which optional GPU features GEMM may use on a device
 * you requested yourself (`detectWebGPU()`): bridge.ts reads it when it
 * creates that device's backend-webgpu `WebGpuBackend`, whose kernel
 * selector then uses subgroup matrices only if this says they are usable.
 *
 * Why a registry at all instead of just reading `device.features`: the
 * subgroup-matrix kernel needs more than the feature flag — it needs an
 * adapter that offers an **f32 8x8x8** subgroup-matrix configuration with a
 * **fixed subgroup size of 32** (the kernel's lane math assumes 32 lanes).
 * Those facts live on `GPUAdapter.info` (`subgroupMatrixConfigs`,
 * `subgroupMinSize`/`subgroupMaxSize`), and Dawn does NOT mirror
 * `subgroupMatrixConfigs` onto `GPUDevice.adapterInfo` (verified with the
 * `webgpu@0.6.1` npm package on Apple M2: the device-side list is empty even
 * when the feature is enabled). So the adapter has to be seen once, at
 * device creation — `detectWebGPU` does that for you; call
 * {@link registerGemmAdapter} yourself for a device you created directly.
 */

export interface GemmCapabilities {
  /** `shader-f16` is enabled on the device: f16 GEMM (f16 storage, f32 accumulation) is available. */
  f16: boolean;
  /** Subgroup-matrix GEMM usable: `chromium-experimental-subgroup-matrix` enabled, f32 8x8x8 config, subgroup size fixed at 32. */
  subgroupMatrix: boolean;
}

/** Dawn's experimental feature name for subgroup matrices. */
export const SUBGROUP_MATRIX_FEATURE = "chromium-experimental-subgroup-matrix";

const registered = new WeakMap<GPUDevice, GemmCapabilities>();
const adapters = new WeakMap<GPUDevice, GPUAdapter>();

interface SubgroupMatrixConfigLike {
  componentType: string;
  resultComponentType: string;
  M: number;
  N: number;
  K: number;
}

/** Pure predicate over an adapter's info + the device's enabled features (exported for unit tests). */
export function subgroupMatrixUsable(info: unknown, deviceFeatures: ReadonlySet<string>): boolean {
  if (!deviceFeatures.has(SUBGROUP_MATRIX_FEATURE)) return false;
  const i = info as
    | { subgroupMatrixConfigs?: readonly SubgroupMatrixConfigLike[]; subgroupMinSize?: number; subgroupMaxSize?: number }
    | undefined;
  if (!i || i.subgroupMinSize !== 32 || i.subgroupMaxSize !== 32) return false;
  for (const c of i.subgroupMatrixConfigs ?? []) {
    if (c.componentType === "f32" && c.resultComponentType === "f32" && c.M === 8 && c.N === 8 && c.K === 8) return true;
  }
  return false;
}

/**
 * Record what GEMM may use on `device`, derived from the `adapter` it was
 * requested from and the features actually enabled on it. Returns the
 * recorded capabilities. `detectWebGPU` calls this automatically.
 */
export function registerGemmAdapter(device: GPUDevice, adapter: GPUAdapter): GemmCapabilities {
  const features = device.features as unknown as ReadonlySet<string>;
  const caps: GemmCapabilities = {
    f16: features.has("shader-f16"),
    subgroupMatrix: subgroupMatrixUsable((adapter as { info?: unknown }).info, features),
  };
  registered.set(device, caps);
  adapters.set(device, adapter);
  return caps;
}

/** The adapter {@link registerGemmAdapter} saw for `device`, if any (bridge.ts passes it to backend-webgpu's `createWebGpuBackend({ device, adapter })`). */
export function gemmAdapter(device: GPUDevice): GPUAdapter | undefined {
  return adapters.get(device);
}

/**
 * What GEMM will use on `device`. For a device never passed through
 * {@link registerGemmAdapter}/`detectWebGPU`, f16 comes from
 * `device.features` and the subgroup-matrix kernel is off (it can't be
 * verified without the adapter — see this module's doc).
 */
export function gemmCapabilities(device: GPUDevice): GemmCapabilities {
  const hit = registered.get(device);
  if (hit) return hit;
  return { f16: (device.features as unknown as ReadonlySet<string>).has("shader-f16"), subgroupMatrix: false };
}
