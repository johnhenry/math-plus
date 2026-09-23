/**
 * Fixed-width dtypes for @johnhenry/math-plus-tensor-core.
 *
 * Includes i64/u64 (BigInt64Array/BigUint64Array-backed) — resolving the
 * source design's own inconsistency where the DType union stopped at 32-bit
 * ints while its ONNX example required int64 input_ids (docs/PLAN.md §9 #4,
 * decided at kickoff).
 *
 * f16/bf16 are STORAGE dtypes: each element is the raw IEEE-754 binary16 /
 * bfloat16 bit pattern in a Uint16Array (the layout safetensors, ONNX
 * Runtime's float16 tensors and WebGPU f16 buffers all use, so data moves
 * across those boundaries zero-copy). Values cross into/out of that bit
 * storage only through {@link encodeHalf}/{@link decodeHalf} — used by the
 * Tensor constructors (`full`/`from`/`arange`), element access
 * (`at`/`item`/`toArray`) and `cast()`. Arithmetic kernels do not operate on
 * half dtypes (they throw; `cast("f32")` first) — see tensor-core's README.
 */
export type DType =
  | "bool"
  | "u8"
  | "i8"
  | "u16"
  | "i16"
  | "u32"
  | "i32"
  | "u64"
  | "i64"
  | "f16"
  | "bf16"
  | "f32"
  | "f64";

export type TypedArrayFor<D extends DType> = D extends "f64"
  ? Float64Array
  : D extends "f32"
    ? Float32Array
    : D extends "i64"
      ? BigInt64Array
      : D extends "u64"
        ? BigUint64Array
        : D extends "i32"
          ? Int32Array
          : D extends "u32"
            ? Uint32Array
            : D extends "i16"
              ? Int16Array
              : D extends "u16" | "f16" | "bf16"
                ? Uint16Array
                : D extends "i8"
                  ? Int8Array
                  : Uint8Array; // u8, bool

export type AnyTypedArray =
  | Float64Array
  | Float32Array
  | BigInt64Array
  | BigUint64Array
  | Int32Array
  | Uint32Array
  | Int16Array
  | Uint16Array
  | Int8Array
  | Uint8Array;

const CONSTRUCTORS: Record<DType, new (length: number) => AnyTypedArray> = {
  bool: Uint8Array,
  u8: Uint8Array,
  i8: Int8Array,
  u16: Uint16Array,
  i16: Int16Array,
  u32: Uint32Array,
  i32: Int32Array,
  u64: BigUint64Array,
  i64: BigInt64Array,
  f16: Uint16Array,
  bf16: Uint16Array,
  f32: Float32Array,
  f64: Float64Array,
};

export const BYTES_PER_ELEMENT: Record<DType, number> = {
  bool: 1,
  u8: 1,
  i8: 1,
  u16: 2,
  i16: 2,
  u32: 4,
  i32: 4,
  u64: 8,
  i64: 8,
  f16: 2,
  bf16: 2,
  f32: 4,
  f64: 8,
};

/** True for i64/u64 — element access uses bigint, not number. */
export function isBigIntDType(dtype: DType): boolean {
  return dtype === "i64" || dtype === "u64";
}

export function allocate(dtype: DType, length: number): AnyTypedArray {
  return new CONSTRUCTORS[dtype](length);
}

/** True for f16/bf16 — Uint16Array bit storage, not directly numeric. */
export function isHalfDType(dtype: DType): dtype is "f16" | "bf16" {
  return dtype === "f16" || dtype === "bf16";
}

/** Round-half-to-even of a non-negative double (exact for the ranges used below). */
function roundHalfEven(v: number): number {
  const f = Math.floor(v);
  const d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * Encode a JS number (a double) as the bit pattern of a binary float with
 * `expBits` exponent bits and `mantBits` fraction bits, rounding to nearest,
 * ties to even, DIRECTLY from the double (no intermediate f32 rounding, so
 * no double-rounding). Overflow -> ±Infinity, underflow -> signed zero /
 * subnormals, NaN -> the canonical quiet NaN.
 */
function encodeBinaryFloat(x: number, expBits: number, mantBits: number): number {
  const signBit = 1 << (expBits + mantBits);
  const expMask = (1 << expBits) - 1;
  const bias = (1 << (expBits - 1)) - 1;
  if (Number.isNaN(x)) return (expMask << mantBits) | (1 << (mantBits - 1));
  const sign = x < 0 || Object.is(x, -0) ? signBit : 0;
  const a = Math.abs(x);
  if (a === 0) return sign;
  // Smallest value that rounds to infinity: halfway between max-finite and 2^(bias+1).
  const overflow = (2 - 2 ** -(mantBits + 1)) * 2 ** bias;
  if (a >= overflow) return sign | (expMask << mantBits);
  const emin = 1 - bias;
  if (a < 2 ** emin) {
    // Subnormal: a multiple of 2^(emin - mantBits); rounding up to
    // 2^mantBits correctly yields the smallest normal's bit pattern.
    return sign | roundHalfEven(a / 2 ** (emin - mantBits));
  }
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e--;
  else if (2 ** (e + 1) <= a) e++;
  const frac = roundHalfEven((a / 2 ** e - 1) * 2 ** mantBits);
  // frac === 2^mantBits carries into the exponent, which is exactly right
  // (and cannot overflow to Inf here thanks to the `overflow` check above).
  return sign | (((e + bias) << mantBits) + frac);
}

/** Decode a binary float bit pattern (see {@link encodeBinaryFloat}). */
function decodeBinaryFloat(bits: number, expBits: number, mantBits: number): number {
  const expMask = (1 << expBits) - 1;
  const bias = (1 << (expBits - 1)) - 1;
  const sign = (bits >> (expBits + mantBits)) & 1 ? -1 : 1;
  const e = (bits >> mantBits) & expMask;
  const m = bits & ((1 << mantBits) - 1);
  if (e === 0) return sign * m * 2 ** (1 - bias - mantBits);
  if (e === expMask) return m === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + m / 2 ** mantBits) * 2 ** (e - bias);
}

/**
 * The single f16/bf16 codec for Math Plus. `encodeHalf("f16", x)` returns the
 * IEEE binary16 bit pattern of `x` rounded to nearest-even (matching NumPy's
 * `astype(np.float16)` and `Math.f16round`); `"bf16"` returns the bfloat16
 * pattern (8 exponent / 7 fraction bits) rounded to nearest-even directly
 * from the double.
 */
export function encodeHalf(dtype: "f16" | "bf16", x: number): number {
  return dtype === "f16" ? encodeBinaryFloat(x, 5, 10) : encodeBinaryFloat(x, 8, 7);
}

/** Exact decode of an f16/bf16 bit pattern (every half value is a double). */
export function decodeHalf(dtype: "f16" | "bf16", bits: number): number {
  return dtype === "f16" ? decodeBinaryFloat(bits, 5, 10) : decodeBinaryFloat(bits, 8, 7);
}
