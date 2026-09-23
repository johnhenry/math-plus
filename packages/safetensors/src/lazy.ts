/**
 * Lazy safetensors: read the header only, then fetch tensors on demand.
 */
import { parseHeader, SafetensorsError, type SafetensorsHeader, type TensorInfo } from "./header.ts";
import { toByteSource, type ByteSource, type HttpSourceOptions, type SafetensorsSource } from "./sources.ts";
import { toFloat32, viewAs, type SafeTypedArray } from "./views.ts";

export interface OpenOptions extends HttpSourceOptions {
  /**
   * Bytes requested up front to cover the header in one round trip
   * (default 256 KiB — typical transformer checkpoints have 10–100 KB
   * headers). If the header is larger, one more read fetches the rest.
   */
  probeBytes?: number;
}

export interface ReadManyOptions {
  /** Adjacent tensors separated by at most this many bytes are fetched in one read (default 64 KiB). */
  maxGap?: number;
  /** Upper bound on one coalesced read (default 64 MiB). A single larger tensor is still read whole. */
  maxChunk?: number;
}

export class LazySafetensors {
  readonly header: SafetensorsHeader;
  readonly source: ByteSource;

  constructor(header: SafetensorsHeader, source: ByteSource) {
    this.header = header;
    this.source = source;
  }

  get metadata(): Readonly<Record<string, string>> { return this.header.metadata; }
  /** Total file size in bytes, when the source knows it. */
  get size(): number | undefined { return this.source.size; }
  names(): string[] { return [...this.header.tensors.keys()]; }
  has(name: string): boolean { return this.header.tensors.has(name); }

  info(name: string): TensorInfo {
    const t = this.header.tensors.get(name);
    if (!t) throw new SafetensorsError("TensorNotFound", `no tensor named ${name}`);
    return t;
  }

  /** Raw bytes of one tensor, in a fresh buffer (aligned). */
  async readBytes(name: string): Promise<Uint8Array> {
    const { dataOffsets: [b, e] } = this.info(name);
    const bytes = await this.source.read(this.header.dataStart + b, e - b);
    if (bytes.byteLength !== e - b) {
      throw new SafetensorsError("ReadError", `${name}: expected ${e - b} bytes, source returned ${bytes.byteLength} (truncated file?)`);
    }
    return bytes;
  }

  /** Typed view of one tensor (see `viewAs`). */
  async read(name: string): Promise<SafeTypedArray> {
    return viewAs(this.info(name).dtype, await this.readBytes(name));
  }

  /** One tensor converted to a new Float32Array (see `toFloat32`). */
  async toF32(name: string): Promise<Float32Array> {
    return toFloat32(this.info(name).dtype, await this.readBytes(name));
  }

  /**
   * Several tensors (default: all), coalescing nearby byte ranges into fewer
   * reads — fewer HTTP requests / syscalls. Results alias the coalesced
   * chunk buffers when aligned. Returned in the order requested.
   */
  async readMany(names: readonly string[] = this.names(), options: ReadManyOptions = {}): Promise<Map<string, SafeTypedArray>> {
    const maxGap = options.maxGap ?? 64 * 1024;
    const maxChunk = options.maxChunk ?? 64 * 1024 * 1024;
    const infos = [...new Set(names)].map((n) => this.info(n)).sort((a, b) => a.dataOffsets[0] - b.dataOffsets[0]);
    const groups: TensorInfo[][] = [];
    for (const t of infos) {
      const group = groups.at(-1);
      const first = group?.[0];
      const last = group?.at(-1);
      if (group && first && last && t.dataOffsets[0] - last.dataOffsets[1] <= maxGap && t.dataOffsets[1] - first.dataOffsets[0] <= maxChunk) {
        group.push(t);
      } else {
        groups.push([t]);
      }
    }
    const views = new Map<string, SafeTypedArray>();
    await Promise.all(
      groups.map(async (group) => {
        const start = (group[0] as TensorInfo).dataOffsets[0];
        const end = (group.at(-1) as TensorInfo).dataOffsets[1];
        const chunk = await this.source.read(this.header.dataStart + start, end - start);
        if (chunk.byteLength !== end - start) {
          throw new SafetensorsError("ReadError", `expected ${end - start} bytes at ${start}, source returned ${chunk.byteLength} (truncated file?)`);
        }
        for (const t of group) {
          views.set(t.name, viewAs(t.dtype, chunk.subarray(t.dataOffsets[0] - start, t.dataOffsets[1] - start)));
        }
      }),
    );
    return new Map([...new Set(names)].map((n) => [n, views.get(n) as SafeTypedArray]));
  }

  /** Releases the underlying source (closes files this package opened). */
  async close(): Promise<void> {
    await this.source.close?.();
  }
}

/**
 * Opens a safetensors file lazily: only the header is read (one or two
 * reads); tensors are fetched per `read`/`readMany`/`toF32` call. See
 * `toByteSource` for accepted sources (bytes, Blob/File, http(s) URL, file
 * path or FileHandle in Node/Bun, or any `ByteSource`).
 */
export async function openSafetensors(source: SafetensorsSource, options: OpenOptions = {}): Promise<LazySafetensors> {
  const src = await toByteSource(source, options);
  try {
    const probe = Math.max(8, options.probeBytes ?? 256 * 1024);
    let head = await src.read(0, src.size === undefined ? probe : Math.min(probe, src.size));
    if (head.byteLength < 8) {
      throw new SafetensorsError("HeaderTooSmall", `file is ${head.byteLength} bytes; a safetensors file has at least 8`);
    }
    const n = new DataView(head.buffer, head.byteOffset, 8).getBigUint64(0, true);
    const need = 8 + Number(n);
    if (n <= BigInt(Number.MAX_SAFE_INTEGER) && head.byteLength < need && (src.size === undefined || src.size >= need) && need <= 8 + 100 * 1024 * 1024) {
      const rest = await src.read(head.byteLength, need - head.byteLength);
      const joined = new Uint8Array(head.byteLength + rest.byteLength);
      joined.set(head);
      joined.set(rest, head.byteLength);
      head = joined;
    }
    const header = parseHeader(head, src.size === undefined ? {} : { fileSize: src.size });
    return new LazySafetensors(header, src);
  } catch (e) {
    await src.close?.();
    throw e;
  }
}
