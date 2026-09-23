/**
 * safetensors writer — byte-for-byte the layout the reference Rust
 * implementation (`safetensors.serialize`, used by `safetensors.torch` /
 * `safetensors.numpy`'s `save_file`) produces:
 *
 * - tensors ordered by dtype, largest-alignment first (the Rust `Dtype`
 *   enum order, descending), then by name — so every tensor's data starts
 *   at an offset that is a multiple of its element size;
 * - compact JSON header: `__metadata__` first (when given), then each tensor
 *   as `{"dtype","shape","data_offsets"}`;
 * - header padded with spaces so the data section starts on an 8-byte
 *   boundary (8 + N ≡ 0 mod 8).
 *
 * Data is written verbatim (host byte order — little-endian on every
 * platform this package targets).
 */
import { BYTES, SafetensorsError, type SafeDType } from "./header.ts";

export interface TensorInput {
  readonly dtype: SafeDType;
  readonly shape: readonly number[];
  /**
   * The tensor's raw little-endian bytes as any ArrayBufferView — the
   * matching TypedArray (e.g. Float32Array for F32, Float16Array or a
   * Uint16Array of bits for F16, Uint16Array of bits for BF16,
   * BigInt64Array for I64), or a Uint8Array of bytes. Its byteLength must be
   * exactly `prod(shape) * BYTES[dtype]`.
   */
  readonly data: ArrayBufferView;
}

/** Position of each dtype in the reference implementation's `Dtype` enum (sort key). */
const DTYPE_ORDER: Readonly<Record<SafeDType, number>> = {
  BOOL: 0, U8: 1, I8: 2, I16: 3, U16: 4, F16: 5, BF16: 6,
  I32: 7, U32: 8, F32: 9, F64: 10, I64: 11, U64: 12,
};

/** Serializes tensors (and optional string metadata) to a safetensors file. */
export function writeSafetensors(
  tensors: Readonly<Record<string, TensorInput>> | ReadonlyMap<string, TensorInput>,
  metadata?: Readonly<Record<string, string>>,
): Uint8Array {
  const entries: Array<[string, TensorInput]> =
    tensors instanceof Map
      ? [...(tensors as ReadonlyMap<string, TensorInput>).entries()]
      : Object.entries(tensors as Record<string, TensorInput>);
  for (const [name, t] of entries) {
    if (name === "__metadata__") throw new SafetensorsError("InvalidHeaderDeserialization", `"__metadata__" is reserved and cannot name a tensor`);
    if (!Object.hasOwn(BYTES, t.dtype)) throw new SafetensorsError("InvalidDtype", `unsupported dtype ${JSON.stringify(t.dtype)} for ${name}`);
    if (!t.shape.every((d) => Number.isSafeInteger(d) && d >= 0)) {
      throw new SafetensorsError("InvalidShape", `${name}: shape must be non-negative integers, got [${t.shape}]`);
    }
    const expected = t.shape.reduce((a, d) => a * d, BYTES[t.dtype]);
    if (t.data.byteLength !== expected) {
      throw new SafetensorsError(
        "TensorInvalidInfo",
        `${name}: data has ${t.data.byteLength} bytes, expected ${expected} for ${t.dtype}[${t.shape}]`,
      );
    }
  }
  if (metadata !== undefined) {
    for (const [k, v] of Object.entries(metadata)) {
      if (typeof v !== "string") throw new SafetensorsError("InvalidHeaderDeserialization", `metadata.${k} must be a string`);
    }
  }
  entries.sort(([an, a], [bn, b]) => DTYPE_ORDER[b.dtype] - DTYPE_ORDER[a.dtype] || (an < bn ? -1 : an > bn ? 1 : 0));

  // Built as a string, not an object: JS objects hoist integer-like keys
  // ("0", "12") ahead of others, which would break the tensor order.
  const parts: string[] = [];
  if (metadata !== undefined) parts.push(`"__metadata__":${JSON.stringify(metadata)}`);
  let offset = 0;
  for (const [name, t] of entries) {
    const end = offset + t.data.byteLength;
    parts.push(`${JSON.stringify(name)}:${JSON.stringify({ dtype: t.dtype, shape: [...t.shape], data_offsets: [offset, end] })}`);
    offset = end;
  }
  const json = new TextEncoder().encode(`{${parts.join(",")}}`);
  const pad = (8 - (json.byteLength % 8)) % 8;
  const n = json.byteLength + pad;

  const out = new Uint8Array(8 + n + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(n), true);
  out.set(json, 8);
  out.fill(0x20, 8 + json.byteLength, 8 + n);
  let cursor = 8 + n;
  for (const [, t] of entries) {
    out.set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength), cursor);
    cursor += t.data.byteLength;
  }
  return out;
}
