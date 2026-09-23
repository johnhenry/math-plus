/**
 * Random-access byte sources for lazy reading: in-memory bytes, Blob/File,
 * HTTP(S) URLs (Range requests, falling back to one full download when the
 * server ignores ranges), and Node/Bun file paths or FileHandles.
 *
 * File access goes through a dynamic `import()` of `node:fs/promises` with a
 * non-literal specifier, so browser bundles never see a Node import; calling
 * it in a browser throws `UnsupportedSource`.
 */
import { SafetensorsError } from "./header.ts";

/** Anything that can serve byte ranges. Implement this for custom transports. */
export interface ByteSource {
  /** Total size in bytes, when known (enables end-of-file validation). */
  readonly size?: number | undefined;
  /** Exactly `length` bytes starting at `offset` (fewer only at end of file). */
  read(offset: number, length: number): Promise<Uint8Array>;
  close?(): Promise<void>;
}

export class MemorySource implements ByteSource {
  readonly #bytes: Uint8Array;
  readonly size: number;
  constructor(bytes: ArrayBuffer | Uint8Array) {
    this.#bytes = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    this.size = this.#bytes.byteLength;
  }
  /** Zero-copy subarray. */
  async read(offset: number, length: number): Promise<Uint8Array> {
    return this.#bytes.subarray(offset, offset + length);
  }
}

export class BlobSource implements ByteSource {
  readonly #blob: Blob;
  readonly size: number;
  constructor(blob: Blob) {
    this.#blob = blob;
    this.size = blob.size;
  }
  async read(offset: number, length: number): Promise<Uint8Array> {
    return new Uint8Array(await this.#blob.slice(offset, offset + length).arrayBuffer());
  }
}

export interface HttpSourceOptions {
  /** fetch implementation (default: globalThis.fetch). */
  fetch?: typeof fetch;
  /** Extra request headers, e.g. `{ Authorization: "Bearer hf_..." }`. */
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

function parseContentRangeTotal(value: string | null): number | undefined {
  const m = value?.match(/\/\s*(\d+)\s*$/);
  return m ? Number(m[1]) : undefined;
}

/**
 * HTTP(S) range reader. Each `read` is one `Range: bytes=a-b` request
 * (redirects are followed by fetch, e.g. Hugging Face `resolve/` -> CDN).
 * If the server answers a range request with 200 (ranges unsupported), the
 * WHOLE body is downloaded once and every later read is served from memory.
 */
export class HttpSource implements ByteSource {
  readonly url: string;
  size: number | undefined;
  /** False once the server has shown it ignores Range (everything is then in memory). */
  rangesSupported = true;
  readonly #fetch: typeof fetch;
  readonly #headers: Record<string, string>;
  readonly #signal: AbortSignal | undefined;
  #full: Uint8Array | undefined;
  /** Requests issued so far (observability/testing). */
  requests = 0;

  constructor(url: string | URL, options: HttpSourceOptions = {}) {
    this.url = String(url);
    this.#fetch = options.fetch ?? globalThis.fetch.bind(globalThis);
    this.#headers = options.headers ?? {};
    this.#signal = options.signal;
  }

  async read(offset: number, length: number): Promise<Uint8Array> {
    if (this.#full) return this.#full.subarray(offset, offset + length);
    if (length === 0) return new Uint8Array(0);
    this.requests++;
    const res = await this.#fetch(this.url, {
      headers: { ...this.#headers, Range: `bytes=${offset}-${offset + length - 1}` },
      ...(this.#signal ? { signal: this.#signal } : {}),
    });
    if (res.status === 206) {
      const total = parseContentRangeTotal(res.headers.get("content-range"));
      if (total !== undefined) this.size = total;
      const bytes = new Uint8Array(await res.arrayBuffer());
      if (bytes.byteLength !== length && (this.size === undefined || offset + bytes.byteLength !== this.size)) {
        throw new SafetensorsError("ReadError", `${this.url}: asked for ${length} bytes at ${offset}, got ${bytes.byteLength}`);
      }
      return bytes;
    }
    if (res.status === 200) {
      // Server ignored Range: this body IS the whole file.
      this.rangesSupported = false;
      this.#full = new Uint8Array(await res.arrayBuffer());
      this.size = this.#full.byteLength;
      return this.#full.subarray(offset, offset + length);
    }
    if (res.status === 416) {
      const total = parseContentRangeTotal(res.headers.get("content-range"));
      if (total !== undefined) this.size = total;
      await res.body?.cancel();
      return new Uint8Array(0);
    }
    await res.body?.cancel();
    throw new SafetensorsError("ReadError", `${this.url}: HTTP ${res.status} ${res.statusText}`);
  }
}

/** Minimal structural type of a Node/Bun `fs/promises` FileHandle. */
export interface FileHandleLike {
  read(buffer: Uint8Array, offset: number, length: number, position: number): Promise<{ bytesRead: number }>;
  stat(): Promise<{ size: number | bigint }>;
  close?(): Promise<void>;
}

export function isFileHandleLike(v: unknown): v is FileHandleLike {
  return typeof v === "object" && v !== null && typeof (v as FileHandleLike).read === "function" && typeof (v as FileHandleLike).stat === "function";
}

/** Reads from an open FileHandle with positional reads (no shared cursor, safe to run concurrently). */
export class FileHandleSource implements ByteSource {
  readonly #handle: FileHandleLike;
  readonly #owned: boolean;
  readonly size: number;
  private constructor(handle: FileHandleLike, size: number, owned: boolean) {
    this.#handle = handle;
    this.size = size;
    this.#owned = owned;
  }

  /** Wraps a caller-owned handle; `close()` will NOT close it. */
  static async fromHandle(handle: FileHandleLike): Promise<FileHandleSource> {
    return new FileHandleSource(handle, Number((await handle.stat()).size), false);
  }

  /** Opens `path` read-only (Node/Bun/Deno with node: compat); `close()` closes it. */
  static async open(path: string): Promise<FileHandleSource> {
    const specifier = "node:fs/promises";
    let fs: { open(path: string, flags: string): Promise<FileHandleLike> };
    try {
      fs = (await import(/* @vite-ignore */ /* webpackIgnore: true */ specifier)) as typeof fs;
    } catch (cause) {
      throw new SafetensorsError("UnsupportedSource", `file paths need node:fs (Node/Bun/Deno); got ${JSON.stringify(path)}. In a browser pass a Blob/File or an http(s) URL`, { cause });
    }
    const handle = await fs.open(path, "r");
    try {
      return new FileHandleSource(handle, Number((await handle.stat()).size), true);
    } catch (e) {
      await handle.close?.();
      throw e;
    }
  }

  async read(offset: number, length: number): Promise<Uint8Array> {
    const end = Math.min(offset + length, this.size);
    const out = new Uint8Array(Math.max(0, end - offset));
    let done = 0;
    while (done < out.byteLength) {
      const { bytesRead } = await this.#handle.read(out, done, out.byteLength - done, offset + done);
      if (bytesRead === 0) throw new SafetensorsError("ReadError", `unexpected end of file at ${offset + done}`);
      done += bytesRead;
    }
    return out;
  }

  async close(): Promise<void> {
    if (this.#owned) await this.#handle.close?.();
  }
}

/** What `openSafetensors` accepts. */
export type SafetensorsSource =
  | ArrayBuffer
  | Uint8Array
  | Blob
  | URL
  | string
  | FileHandleLike
  | ByteSource;

function isByteSource(v: unknown): v is ByteSource {
  return typeof v === "object" && v !== null && typeof (v as ByteSource).read === "function";
}

/**
 * Resolves a source to a ByteSource:
 * - ArrayBuffer/Uint8Array -> MemorySource (zero-copy)
 * - Blob/File (incl. Bun.file) -> BlobSource
 * - URL or string starting with http:, https:, blob:, data: -> HttpSource
 * - URL with file: or any other string -> file path (Node/Bun only)
 * - a Node FileHandle -> FileHandleSource (not closed by us)
 * - a ByteSource -> used as-is
 */
export async function toByteSource(source: SafetensorsSource, options: HttpSourceOptions = {}): Promise<ByteSource> {
  if (source instanceof Uint8Array || source instanceof ArrayBuffer) return new MemorySource(source);
  if (typeof Blob !== "undefined" && source instanceof Blob) return new BlobSource(source);
  if (source instanceof URL || typeof source === "string") {
    const text = String(source);
    if (/^(https?|blob|data):/i.test(text)) return new HttpSource(text, options);
    if (/^file:/i.test(text)) return FileHandleSource.open(decodeURIComponent(new URL(text).pathname));
    return FileHandleSource.open(text);
  }
  if (isFileHandleLike(source)) return FileHandleSource.fromHandle(source);
  if (isByteSource(source)) return source;
  throw new SafetensorsError("UnsupportedSource", "expected bytes, a Blob, a URL, a file path, a FileHandle or a ByteSource");
}
