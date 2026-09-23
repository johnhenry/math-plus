/**
 * @johnhenry/math-plus-safetensors — read and write the safetensors format
 * (https://github.com/huggingface/safetensors) in any JS runtime.
 *
 * - `readSafetensors(bytes)` → `SafetensorsFile`: everything in memory,
 *   zero-copy typed views.
 * - `openSafetensors(source)` → `LazySafetensors`: header only, tensors on
 *   demand, from a Blob/File, an http(s) URL (Range requests), a file path /
 *   FileHandle (Node/Bun), in-memory bytes, or any custom `ByteSource`.
 * - `writeSafetensors(tensors, metadata?)` → bytes, laid out exactly like
 *   the reference implementation.
 * - `toFloat32` / `.toF32(name)`: any dtype → a new Float32Array.
 * - Interop with @johnhenry/math-plus-tensor-core lives in the separate
 *   `@johnhenry/math-plus-safetensors/tensor` subpath (optional peer).
 *
 * Typed views: F16 → Float16Array, BF16 → Uint16Array of raw bits,
 * everything else → the matching TypedArray (BOOL → Uint8Array).
 */
import { parseHeader, SafetensorsError, type SafetensorsHeader, type TensorInfo } from "./header.ts";
import { toFloat32, viewAs, type SafeTypedArray } from "./views.ts";

export {
  BYTES,
  MAX_HEADER,
  SafetensorsError,
  headerLength,
  parseHeader,
  type ParseHeaderOptions,
  type SafeDType,
  type SafetensorsErrorCode,
  type SafetensorsHeader,
  type TensorInfo,
} from "./header.ts";
export { aligned, toFloat32, viewAs, type SafeTypedArray } from "./views.ts";
export { writeSafetensors, type TensorInput } from "./writer.ts";
export { LazySafetensors, openSafetensors, type OpenOptions, type ReadManyOptions } from "./lazy.ts";
export {
  BlobSource,
  FileHandleSource,
  HttpSource,
  MemorySource,
  toByteSource,
  type ByteSource,
  type FileHandleLike,
  type HttpSourceOptions,
  type SafetensorsSource,
} from "./sources.ts";

/** An in-memory safetensors file (header validated against the buffer's length). */
export class SafetensorsFile {
  readonly header: SafetensorsHeader;
  readonly #bytes: Uint8Array;

  constructor(source: ArrayBuffer | Uint8Array) {
    this.#bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
    this.header = parseHeader(this.#bytes, { fileSize: this.#bytes.byteLength });
  }

  get metadata(): Readonly<Record<string, string>> { return this.header.metadata; }
  names(): string[] { return [...this.header.tensors.keys()]; }
  has(name: string): boolean { return this.header.tensors.has(name); }

  info(name: string): TensorInfo {
    const t = this.header.tensors.get(name);
    if (!t) throw new SafetensorsError("TensorNotFound", `no tensor named ${name}`);
    return t;
  }

  /** Raw bytes of one tensor (a subarray; no copy). */
  bytes(name: string): Uint8Array {
    const { dataOffsets: [b, e] } = this.info(name);
    return this.#bytes.subarray(this.header.dataStart + b, this.header.dataStart + e);
  }

  /** Typed view of one tensor (zero-copy when aligned, one copy otherwise). */
  view(name: string): SafeTypedArray {
    return viewAs(this.info(name).dtype, this.bytes(name));
  }

  /** One tensor converted to a new Float32Array (see `toFloat32`). */
  toF32(name: string): Float32Array {
    return toFloat32(this.info(name).dtype, this.bytes(name));
  }
}

export function readSafetensors(source: ArrayBuffer | Uint8Array): SafetensorsFile {
  return new SafetensorsFile(source);
}
