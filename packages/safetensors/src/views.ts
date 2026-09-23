/**
 * Typed views over one tensor's raw little-endian bytes, and float32
 * conversion.
 *
 * - F16  -> Float16Array (native in Node >= 24, Bun, Deno, current browsers)
 * - BF16 -> Uint16Array of raw bits (no native bfloat16 array exists)
 * - F32/F64/I8..I64/U8..U64 -> matching TypedArray; BOOL -> Uint8Array.
 *
 * Zero-copy when the bytes are aligned to the element size; otherwise ONE
 * copy is made (e.g. MLX writes unpadded headers, so every tensor of an
 * in-memory MLX checkpoint is misaligned). Lazy reads (`openSafetensors`)
 * land in fresh buffers and are always aligned.
 */
import { BYTES, SafetensorsError, type SafeDType } from "./header.ts";

export type SafeTypedArray =
  | Float16Array | Float32Array | Float64Array | Uint16Array | Int16Array
  | Uint8Array | Int8Array | Uint32Array | Int32Array | BigUint64Array | BigInt64Array;

/** `bytes`, or an aligned copy of it when its offset isn't a multiple of `size`. */
export function aligned(bytes: Uint8Array, size: number): Uint8Array {
  // Node's Buffer#slice is a view, so copy through the constructor.
  return bytes.byteOffset % size === 0 ? bytes : new Uint8Array(bytes);
}

function float16Ctor(): typeof Float16Array {
  const ctor = (globalThis as { Float16Array?: typeof Float16Array }).Float16Array;
  if (typeof ctor !== "function") {
    throw new SafetensorsError(
      "Float16Unavailable",
      "F16 needs a runtime with Float16Array (Node >= 24, Bun, Deno, Chrome 135+, Firefox 129+, Safari 18.2+)",
    );
  }
  return ctor;
}

/** Views `bytes` (exactly one tensor's data) as the dtype's TypedArray; copies only if misaligned. */
export function viewAs(dtype: SafeDType, bytes: Uint8Array): SafeTypedArray {
  const size = BYTES[dtype];
  if (bytes.byteLength % size !== 0) {
    throw new SafetensorsError("TensorInvalidInfo", `${bytes.byteLength} bytes is not a whole number of ${dtype} elements`);
  }
  const { buffer, byteOffset } = aligned(bytes, size);
  const n = bytes.byteLength / size;
  switch (dtype) {
    case "F16": return new (float16Ctor())(buffer, byteOffset, n);
    case "BF16": case "U16": return new Uint16Array(buffer, byteOffset, n);
    case "I16": return new Int16Array(buffer, byteOffset, n);
    case "F32": return new Float32Array(buffer, byteOffset, n);
    case "F64": return new Float64Array(buffer, byteOffset, n);
    case "I32": return new Int32Array(buffer, byteOffset, n);
    case "U32": return new Uint32Array(buffer, byteOffset, n);
    case "I64": return new BigInt64Array(buffer, byteOffset, n);
    case "U64": return new BigUint64Array(buffer, byteOffset, n);
    case "I8": return new Int8Array(buffer, byteOffset, n);
    case "U8": case "BOOL": return new Uint8Array(buffer, byteOffset, n);
  }
}

/**
 * Converts one tensor's bytes to a NEW Float32Array the caller owns (never
 * aliases the source). F16 widens exactly via the platform's Float16Array;
 * BF16 widens exactly (bfloat16 is the top half of a float32); F64 and the
 * integer dtypes round to nearest float32 (I64/U64 through Number, so
 * magnitudes above 2^53 are already rounded before the float32 rounding).
 */
export function toFloat32(dtype: SafeDType, bytes: Uint8Array): Float32Array {
  if (dtype === "BF16") {
    const bits = viewAs("BF16", bytes) as Uint16Array;
    const wide = new Uint32Array(bits.length);
    for (let i = 0; i < bits.length; i++) wide[i] = (bits[i] as number) << 16;
    return new Float32Array(wide.buffer);
  }
  const view = viewAs(dtype, bytes);
  if (view instanceof BigInt64Array || view instanceof BigUint64Array) {
    return Float32Array.from(view as ArrayLike<bigint>, Number);
  }
  // TypedArray -> TypedArray construction converts element-wise and always copies.
  return new Float32Array(view as ArrayLike<number>);
}
