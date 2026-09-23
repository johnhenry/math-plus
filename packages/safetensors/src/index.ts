/**
 * @johnhenry/math-plus-safetensors — read (and, later, write) the
 * safetensors format (https://github.com/huggingface/safetensors).
 *
 * Layout: u64 little-endian header length N, N bytes of UTF-8 JSON
 * `{name: {dtype, shape, data_offsets: [begin, end]}, "__metadata__"?: {...}}`,
 * then the raw little-endian tensor bytes (offsets relative to byte 8 + N).
 *
 * Typed views, zero-copy when aligned:
 * - F16  → Float16Array (native in Node >= 24, Bun, Deno, current browsers)
 * - BF16 → Uint16Array of raw bits (no native bf16 array exists)
 * - F32/F64/I8..I64/U8..U64 → matching TypedArray; BOOL → Uint8Array.
 *
 * Scope (v0): in-memory sources. Lazy sources (Blob, HTTP Range, file
 * handles) and the writer are tracked in the package README.
 */

export type SafeDType =
  | "BOOL" | "U8" | "I8" | "U16" | "I16" | "F16" | "BF16"
  | "U32" | "I32" | "F32" | "U64" | "I64" | "F64";

export const BYTES: Readonly<Record<SafeDType, number>> = {
  BOOL: 1, U8: 1, I8: 1, U16: 2, I16: 2, F16: 2, BF16: 2,
  U32: 4, I32: 4, F32: 4, U64: 8, I64: 8, F64: 8,
};

export interface TensorInfo {
  readonly name: string;
  readonly dtype: SafeDType;
  readonly shape: readonly number[];
  /** Byte range relative to the start of the data section. */
  readonly dataOffsets: readonly [number, number];
}

export interface SafetensorsHeader {
  readonly tensors: ReadonlyMap<string, TensorInfo>;
  readonly metadata: Readonly<Record<string, string>>;
  /** Absolute byte offset of the data section (8 + header length). */
  readonly dataStart: number;
}

export type SafeTypedArray =
  | Float16Array | Float32Array | Float64Array | Uint16Array | Int16Array
  | Uint8Array | Int8Array | Uint32Array | Int32Array | BigUint64Array | BigInt64Array;

const MAX_HEADER = 100 * 1024 * 1024;

/** Reads the header length from the first 8 bytes. */
export function headerLength(first8: Uint8Array): number {
  if (first8.byteLength < 8) throw new RangeError("safetensors: need 8 bytes for the header length");
  const dv = new DataView(first8.buffer, first8.byteOffset, 8);
  const n = dv.getBigUint64(0, true);
  if (n > BigInt(MAX_HEADER)) throw new RangeError(`safetensors: header length ${n} exceeds ${MAX_HEADER}`);
  return Number(n);
}

/** Parses a header from bytes that contain at least the first 8 + N bytes. */
export function parseHeader(bytes: Uint8Array): SafetensorsHeader {
  const n = headerLength(bytes);
  if (bytes.byteLength < 8 + n) throw new RangeError("safetensors: truncated header");
  const json = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + n))) as Record<string, unknown>;
  const tensors = new Map<string, TensorInfo>();
  let metadata: Record<string, string> = {};
  for (const [name, raw] of Object.entries(json)) {
    if (name === "__metadata__") {
      metadata = (raw ?? {}) as Record<string, string>;
      continue;
    }
    const r = raw as { dtype: SafeDType; shape: number[]; data_offsets: [number, number] };
    if (!(r.dtype in BYTES)) throw new TypeError(`safetensors: unsupported dtype ${r.dtype} for ${name}`);
    const count = r.shape.reduce((a, b) => a * b, 1);
    const [begin, end] = r.data_offsets;
    if (end - begin !== count * BYTES[r.dtype]) {
      throw new RangeError(`safetensors: ${name} has ${end - begin} bytes, expected ${count * BYTES[r.dtype]}`);
    }
    tensors.set(name, { name, dtype: r.dtype, shape: r.shape, dataOffsets: [begin, end] });
  }
  return { tensors, metadata, dataStart: 8 + n };
}

/** Views `bytes` (exactly one tensor's data) as the dtype's TypedArray; copies only if misaligned. */
export function viewAs(dtype: SafeDType, bytes: Uint8Array): SafeTypedArray {
  const size = BYTES[dtype];
  let b = bytes;
  // realign with a real copy (Node Buffer#slice is a view, so use the constructor)
  if (b.byteOffset % size !== 0) b = new Uint8Array(b);
  const { buffer, byteOffset } = b;
  const n = b.byteLength / size;
  switch (dtype) {
    case "F16": return new Float16Array(buffer, byteOffset, n);
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

/** An in-memory safetensors file. */
export class SafetensorsFile {
  readonly header: SafetensorsHeader;
  readonly #bytes: Uint8Array;

  constructor(source: ArrayBuffer | Uint8Array) {
    this.#bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
    this.header = parseHeader(this.#bytes);
  }

  get metadata(): Readonly<Record<string, string>> { return this.header.metadata; }
  names(): string[] { return [...this.header.tensors.keys()]; }
  has(name: string): boolean { return this.header.tensors.has(name); }

  info(name: string): TensorInfo {
    const t = this.header.tensors.get(name);
    if (!t) throw new RangeError(`safetensors: no tensor named ${name}`);
    return t;
  }

  /** Raw bytes of one tensor (a subarray; no copy). */
  bytes(name: string): Uint8Array {
    const { dataOffsets: [b, e] } = this.info(name);
    return this.#bytes.subarray(this.header.dataStart + b, this.header.dataStart + e);
  }

  /** Typed view of one tensor (zero-copy when aligned). */
  view(name: string): SafeTypedArray {
    return viewAs(this.info(name).dtype, this.bytes(name));
  }
}

export function readSafetensors(source: ArrayBuffer | Uint8Array): SafetensorsFile {
  return new SafetensorsFile(source);
}
