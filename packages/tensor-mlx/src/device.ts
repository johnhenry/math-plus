/**
 * `MlxDevice` / `MlxArray` — the MLX device facade: the chainable
 * device-array API shared by every math-plus device
 * (`@johnhenry/math-plus-tensor-cpu`'s `ArrayDevice`/`DeviceArray`, one
 * implementation for CPU, MLX and WebGPU — AGENTS.md's canonical-implementation
 * rule) over `@johnhenry/backend-mlx` (Apple's mlx-c over FFI: koffi on Node,
 * bun:ffi on Bun, Deno.dlopen on Deno). This package binds no native symbols
 * and implements no op itself: `MlxArray` is a subclass of `DeviceArray` that
 * adds nothing, and `MlxDevice` adds only the MLX-specific extras (`kind`,
 * `info`, `liveArrays`, `memory`).
 *
 * The rules are the shared ones (see device-array.ts in tensor-cpu): no
 * global default device, explicit async transfers in both directions, no
 * implicit dtype promotion (`cast()` is the only way across dtypes — MLX
 * itself would promote), and lazy execution: ops append nodes to MLX's graph
 * and return at once; shape and dtype errors still throw at the call site,
 * and work runs at `eval()`, `toTensor()`/`toHost()`, or when MLX needs a
 * value. The general-numerics ops go through tensor-backend's compose
 * helpers, which use backend-mlx's native mlx-c kernels (every one is native
 * there).
 */
import { createMlxBackend, mlxPlatformSupported, resolveLib, type MlxBackend } from "@johnhenry/backend-mlx";
import { ArrayDevice, DeviceArray } from "@johnhenry/math-plus-tensor-cpu";

export interface MlxDeviceOptions {
  /** "gpu" (Metal, the default) or "cpu" (MLX's CPU backend). */
  device?: "gpu" | "cpu";
  /** Explicit path to libmlxc.dylib; otherwise @johnhenry/backend-mlx's resolution order applies. */
  libPath?: string;
  /** FinalizationRegistry safety net for leaked handles (default true). `dispose`/`scope` stay the real mechanism. */
  finalizers?: boolean;
}

/**
 * Why an MLX device cannot be created in this process, or `null` when it
 * can (darwin/arm64 and a libmlxc.dylib was found). Tests use this to skip,
 * never fail, where MLX is unavailable.
 */
export function mlxUnavailableReason(): string | null {
  if (!mlxPlatformSupported()) return "MLX needs macOS on Apple Silicon (darwin/arm64)";
  try {
    resolveLib(); // same resolution createMlxDevice uses, incl. a $LAYA_MLXC_PATH that points nowhere
  } catch (e) {
    return (e as Error).message.split("\n")[0]!;
  }
  return null;
}

/** Creates an MLX device. Throws off darwin/arm64 or when libmlxc cannot be found. */
export function createMlxDevice(opts: MlxDeviceOptions = {}): MlxDevice {
  return new MlxDevice(createMlxBackend(opts));
}

/**
 * An MLX device. Transfers (`fromTensor`/`fromHost`), `scope`, `eval`,
 * `where`, `wrap`, `supports` and `destroy` are the shared `ArrayDevice`
 * ones; `backend` is the underlying `@johnhenry/tensor-backend` `Backend`
 * (for the conformance suite, backend-generic code, and the ops the arrays
 * do not wrap).
 */
export class MlxDevice extends ArrayDevice<MlxBackend> {
  declare readonly name: "mlx";

  /** @internal Use `createMlxDevice()`. */
  constructor(backend: MlxBackend) {
    super(backend, { label: "tensor-mlx", device: "MlxDevice", array: "an MlxArray" }, MlxArray);
  }

  /** "gpu" or "cpu". */
  get kind(): "gpu" | "cpu" {
    return this.backend.device;
  }

  /** Which libmlxc loaded, through which FFI (node/bun/deno), and its mlx-c ABI. */
  get info(): MlxBackend["info"] {
    return this.backend.info;
  }

  /** Live (not yet disposed) MLX handles owned by this device. */
  liveArrays(): number {
    return this.backend.liveTensors();
  }

  /** MLX allocator statistics, in bytes. */
  memory(): { active: number; peak: number } {
    return this.backend.memory();
  }
}

/**
 * An array on an {@link MlxDevice}: the shared chainable `DeviceArray`
 * (`add`, `mul`, `exp`, `matmul`, `layerNorm`, `cast`, `reshape`,
 * `transpose`, the numerics ops, `toTensor`/`toHost`, `eval`, `dispose`, …),
 * so `x instanceof MlxArray` holds for every array an `MlxDevice` returns.
 */
export class MlxArray extends DeviceArray<MlxDevice> {}
