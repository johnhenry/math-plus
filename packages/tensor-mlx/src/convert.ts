/**
 * The host boundary: tensor-core `Tensor` <-> tensor-backend `HostTensor`.
 *
 * Pure TypeScript with no native dependency, so it runs (and is tested) on
 * every platform. Neither direction copies element data:
 *
 * - `hostFromTensor` views the tensor's own storage (a `subarray`, and for
 *   f16 a `Float16Array` over the same bytes). The one copy happens later,
 *   inside the device's `fromHost` (host memory -> MLX unified memory).
 * - `tensorFromHost` wraps the freshly read host buffer with
 *   `Tensor.fromTypedArray` via tensor-backend's `toMathPlusArgs` (the
 *   canonical copy of that mapping — not re-implemented here).
 *
 * Non-contiguous tensors are rejected instead of being packed silently
 * (docs/PLAN.md: no implicit copies) — call `.contiguous()` first.
 */
import { Tensor, type AnyTypedArray, type DType } from "@johnhenry/math-plus-tensor-core";
import { toMathPlusArgs, type DType as DeviceDType, type HostTensor } from "@johnhenry/tensor-backend";

export type { DeviceDType };

/** dtypes a device array can hold (the tensor-backend contract's set; same names as tensor-core). */
export const DEVICE_DTYPES: readonly DeviceDType[] = Object.freeze(["f32", "f16", "bf16", "i32", "bool"]);

export function isDeviceDType(d: string): d is DeviceDType {
  return (DEVICE_DTYPES as readonly string[]).includes(d);
}

function unsupported(dtype: DType): TypeError {
  const hint =
    dtype === "f64"
      ? 'MLX on Metal has no float64 — cast("f32") explicitly first'
      : dtype === "i64" || dtype === "u64"
        ? 'cast("i32") explicitly first (check the range yourself; nothing is narrowed implicitly)'
        : 'cast("i32") or cast("f32") explicitly first';
  return new TypeError(`tensor-mlx: dtype ${dtype} is not supported on the device (supported: ${DEVICE_DTYPES.join(", ")}); ${hint}`);
}

/**
 * A zero-copy `HostTensor` view of a tensor-core `Tensor`'s storage.
 * f16/bf16 storage is the raw IEEE-754 bit pattern in a `Uint16Array`
 * (tensor-core's layout); f16 is re-viewed as a `Float16Array`.
 */
export function hostFromTensor(t: Tensor): HostTensor {
  const dtype = t.dtype;
  if (!isDeviceDType(dtype)) throw unsupported(dtype);
  if (!t.isContiguous) {
    throw new Error("tensor-mlx: fromTensor needs a C-contiguous tensor; call .contiguous() first (no implicit copies)");
  }
  const data = t.data.subarray(t.offset, t.offset + t.size) as AnyTypedArray;
  const shape = [...t.shape];
  switch (dtype) {
    case "f16": {
      const bits = data as Uint16Array;
      if (typeof Float16Array !== "function") throw new Error("tensor-mlx: f16 needs a runtime with Float16Array (Node >= 24, Bun >= 1.2)");
      return { dtype: "f16", shape, data: new Float16Array(bits.buffer, bits.byteOffset, bits.length) };
    }
    case "f32":
      return { dtype: "f32", shape, data: data as Float32Array };
    case "bf16":
      return { dtype: "bf16", shape, data: data as Uint16Array };
    case "i32":
      return { dtype: "i32", shape, data: data as Int32Array };
    case "bool":
      return { dtype: "bool", shape, data: data as Uint8Array };
  }
}

/** Wraps a host buffer (just read from a device) as a tensor-core `Tensor` without copying. */
export function tensorFromHost(h: HostTensor): Tensor {
  const { data, shape, dtype } = toMathPlusArgs(h);
  return Tensor.fromTypedArray(data as AnyTypedArray, shape, { dtype });
}
