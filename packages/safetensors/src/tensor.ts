/**
 * `@johnhenry/math-plus-safetensors/tensor` — interop with
 * @johnhenry/math-plus-tensor-core, kept in its own subpath so the main
 * entry has no tensor-core dependency. tensor-core is an OPTIONAL peer
 * dependency: install it if you import this subpath.
 *
 * dtype mapping is 1:1 (F16 -> "f16", BF16 -> "bf16", BOOL -> "bool", ...).
 * tensor-core stores f16/bf16 as Uint16Array bit patterns — exactly the
 * safetensors bytes — so no conversion happens here; use tensor-core's
 * `cast("f32")` for values.
 */
import { Tensor, type DType } from "@johnhenry/math-plus-tensor-core";
import { BYTES, SafetensorsError, type SafeDType } from "./header.ts";
import type { LazySafetensors } from "./lazy.ts";
import type { SafetensorsFile } from "./index.ts";
import { aligned, viewAs } from "./views.ts";
import type { TensorInput } from "./writer.ts";

const TO_TENSOR_DTYPE: Readonly<Record<SafeDType, DType>> = {
  BOOL: "bool", U8: "u8", I8: "i8", U16: "u16", I16: "i16", F16: "f16", BF16: "bf16",
  U32: "u32", I32: "i32", F32: "f32", U64: "u64", I64: "i64", F64: "f64",
};

const FROM_TENSOR_DTYPE: Readonly<Record<DType, SafeDType>> = Object.fromEntries(
  Object.entries(TO_TENSOR_DTYPE).map(([s, t]) => [t, s]),
) as Record<DType, SafeDType>;

/** tensor-core's storage view for raw bytes (f16/bf16 as Uint16Array bits). */
function storageView(dtype: SafeDType, bytes: Uint8Array) {
  if (dtype === "F16") {
    const a = aligned(bytes, 2);
    return new Uint16Array(a.buffer, a.byteOffset, a.byteLength / 2);
  }
  return viewAs(dtype, bytes) as Exclude<ReturnType<typeof viewAs>, Float16Array>;
}

/** Wraps raw tensor bytes as a Tensor. ALIASES `bytes` when aligned (tensor-core's `fromTypedArray` semantics). */
export function bytesToTensor(dtype: SafeDType, shape: readonly number[], bytes: Uint8Array): Tensor {
  if (!Object.hasOwn(TO_TENSOR_DTYPE, dtype)) throw new SafetensorsError("InvalidDtype", `unsupported dtype ${dtype}`);
  return Tensor.fromTypedArray(storageView(dtype, bytes), shape, { dtype: TO_TENSOR_DTYPE[dtype] });
}

/** One tensor of an in-memory file as a Tensor (zero-copy when aligned — shares the file's buffer). */
export function toTensor(file: SafetensorsFile, name: string): Tensor {
  const info = file.info(name);
  return bytesToTensor(info.dtype, info.shape, file.bytes(name));
}

/** Reads one tensor of a lazily-opened file as a Tensor (fresh buffer). */
export async function readTensor(file: LazySafetensors, name: string): Promise<Tensor> {
  const info = file.info(name);
  return bytesToTensor(info.dtype, info.shape, await file.readBytes(name));
}

/** A Tensor as a `writeSafetensors` input (packs non-contiguous views first). */
export function fromTensor(tensor: Tensor): TensorInput {
  const packed = tensor.contiguous();
  const bytesPer = BYTES[FROM_TENSOR_DTYPE[packed.dtype]];
  const data = new Uint8Array(packed.data.buffer, packed.data.byteOffset + packed.offset * bytesPer, packed.size * bytesPer);
  return { dtype: FROM_TENSOR_DTYPE[packed.dtype], shape: [...packed.shape], data };
}
