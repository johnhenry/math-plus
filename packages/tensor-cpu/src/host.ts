/**
 * The host boundary shared by every math-plus device package: tensor-core
 * `Tensor` <-> tensor-backend `HostTensor`.
 *
 * It lives here (not in each device package) because of AGENTS.md's
 * canonical-implementation rule: `@johnhenry/math-plus-tensor-mlx` and
 * `@johnhenry/math-plus-tensor-webgpu` both need exactly this mapping, and
 * this package is the pure-TypeScript, dependency-light home of math-plus's
 * side of the `@johnhenry/tensor-backend` contract. (It moved here from
 * tensor-mlx's `convert.ts`, which now re-exports it.)
 *
 * Pure TypeScript with no native dependency. Neither direction copies
 * element data:
 *
 * - `hostFromTensor` views the tensor's own storage (a `subarray`, and for
 *   f16 a `Float16Array` over the same bytes). The one copy happens later,
 *   inside the device's `fromHost`.
 * - `tensorFromHost` wraps a freshly read host buffer with
 *   `Tensor.fromTypedArray` via tensor-backend's `toMathPlusArgs` (the
 *   canonical copy of that mapping — not re-implemented here).
 *
 * Non-contiguous tensors are rejected instead of being packed silently
 * (docs/PLAN.md: no implicit copies) — call `.contiguous()` first.
 */
import { Tensor, type AnyTypedArray, type DType } from "@johnhenry/math-plus-tensor-core";
import { toMathPlusArgs, type DType as DeviceDType, type HostTensor } from "@johnhenry/tensor-backend";

export type { DeviceDType };

/**
 * dtypes a device tensor can hold at the host<->device transfer boundary
 * (the tensor-backend contract's set; same names as tensor-core). This is
 * the full 13-dtype set as of 2026-09-25 -- it is a storage/representation
 * question (can tensor-core even hand this dtype's data across the
 * boundary), not a "does this specific backend support it" question. A
 * backend that can't compute in a given dtype still rejects it via its own
 * `supports(dtype)` (e.g. WebGPU has no f64/i8/u8/i16/u16/i64/u64 -- see its
 * README's Limitations section for why, hardware/spec-capped not a gap).
 */
export const DEVICE_DTYPES: readonly DeviceDType[] = Object.freeze([
  "f32", "f16", "bf16", "i32", "bool", "u8", "i8", "u16", "i16", "u32", "u64", "i64", "f64",
]);

export function isDeviceDType(d: string): d is DeviceDType {
  return (DEVICE_DTYPES as readonly string[]).includes(d);
}

function unsupported(label: string, dtype: DType): TypeError {
  return new TypeError(`${label}: dtype ${dtype} is not supported at the host<->device boundary (supported: ${DEVICE_DTYPES.join(", ")})`);
}

/**
 * A zero-copy `HostTensor` view of a tensor-core `Tensor`'s storage.
 * f16/bf16 storage is the raw IEEE-754 bit pattern in a `Uint16Array`
 * (tensor-core's layout); f16 is re-viewed as a `Float16Array`. `label`
 * prefixes error messages (e.g. `"tensor-mlx"`).
 */
export function hostFromTensor(t: Tensor, label = "tensor-cpu"): HostTensor {
  const dtype = t.dtype;
  if (!isDeviceDType(dtype)) throw unsupported(label, dtype);
  if (!t.isContiguous) {
    throw new Error(`${label}: fromTensor needs a C-contiguous tensor; call .contiguous() first (no implicit copies)`);
  }
  const data = t.data.subarray(t.offset, t.offset + t.size) as AnyTypedArray;
  const shape = [...t.shape];
  switch (dtype) {
    case "f16": {
      const bits = data as Uint16Array;
      if (typeof Float16Array !== "function") throw new Error(`${label}: f16 needs a runtime with Float16Array (Node >= 24, Bun >= 1.2)`);
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
    case "u8":
      return { dtype: "u8", shape, data: data as Uint8Array };
    case "i8":
      return { dtype: "i8", shape, data: data as Int8Array };
    case "u16":
      return { dtype: "u16", shape, data: data as Uint16Array };
    case "i16":
      return { dtype: "i16", shape, data: data as Int16Array };
    case "u32":
      return { dtype: "u32", shape, data: data as Uint32Array };
    case "f64":
      return { dtype: "f64", shape, data: data as Float64Array };
    case "u64":
      return { dtype: "u64", shape, data: data as BigUint64Array };
    case "i64":
      return { dtype: "i64", shape, data: data as BigInt64Array };
  }
}

/** Wraps a host buffer (just read from a device) as a tensor-core `Tensor` without copying. */
export function tensorFromHost(h: HostTensor): Tensor {
  const { data, shape, dtype } = toMathPlusArgs(h);
  return Tensor.fromTypedArray(data as AnyTypedArray, shape, { dtype });
}
