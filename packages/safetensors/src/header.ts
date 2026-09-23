/**
 * safetensors header: dtype table, parsing and validation.
 *
 * Layout: u64 little-endian header length N, N bytes of UTF-8 JSON
 * `{name: {dtype, shape, data_offsets: [begin, end]}, "__metadata__"?: {...}}`,
 * then the raw little-endian tensor bytes (offsets relative to byte 8 + N).
 *
 * Validation mirrors the reference Rust implementation
 * (huggingface/safetensors `Metadata::validate`): the header must start with
 * `{`, offsets must tile the data section exactly — sorted by begin, the
 * first starts at 0, each begins where the previous ended (no gaps, no
 * overlaps), and each spans exactly `prod(shape) * sizeof(dtype)` bytes —
 * and, when the file size is known, the last tensor must end exactly at the
 * end of the file. Duplicate tensor names (which `JSON.parse` would silently
 * collapse) are rejected too.
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
  /** Length of the data section implied by the offsets (max `end`). */
  readonly dataLength: number;
}

/** Error codes follow the reference implementation's `SafeTensorError` variants where one exists. */
export type SafetensorsErrorCode =
  | "HeaderTooLarge"
  | "HeaderTooSmall"
  | "InvalidHeaderLength"
  | "InvalidHeader"
  | "InvalidHeaderStart"
  | "InvalidHeaderDeserialization"
  | "DuplicateTensor"
  | "InvalidDtype"
  | "InvalidShape"
  | "InvalidOffset"
  | "TensorInvalidInfo"
  | "ValidationOverflow"
  | "MetadataIncompleteBuffer"
  | "TensorNotFound"
  | "Float16Unavailable"
  | "UnsupportedSource"
  | "ReadError";

export class SafetensorsError extends Error {
  readonly code: SafetensorsErrorCode;
  constructor(code: SafetensorsErrorCode, message: string, options?: { cause?: unknown }) {
    super(`safetensors: ${message}`, options);
    this.name = "SafetensorsError";
    this.code = code;
  }
}

/** Same limit as the reference implementation (100 MB). */
export const MAX_HEADER = 100 * 1024 * 1024;

/** Reads the header length N from the first 8 bytes. */
export function headerLength(first8: Uint8Array): number {
  if (first8.byteLength < 8) {
    throw new SafetensorsError("HeaderTooSmall", `need 8 bytes for the header length, got ${first8.byteLength}`);
  }
  const dv = new DataView(first8.buffer, first8.byteOffset, 8);
  const n = dv.getBigUint64(0, true);
  if (n > BigInt(MAX_HEADER)) {
    throw new SafetensorsError("HeaderTooLarge", `header length ${n} exceeds ${MAX_HEADER}`);
  }
  return Number(n);
}

/**
 * Top-level object keys in source order, including duplicates. Only called
 * on text `JSON.parse` has already accepted, so a minimal scanner suffices.
 */
function topLevelKeys(json: string): string[] {
  const keys: string[] = [];
  let depth = 0;
  let expectKey = false;
  for (let i = 0; i < json.length; i++) {
    const c = json.charCodeAt(i);
    if (c === 0x22 /* " */) {
      let j = i + 1;
      while (json.charCodeAt(j) !== 0x22) j += json.charCodeAt(j) === 0x5c /* \ */ ? 2 : 1;
      if (depth === 1 && expectKey) {
        keys.push(JSON.parse(json.slice(i, j + 1)) as string);
        expectKey = false;
      }
      i = j;
    } else if (c === 0x7b /* { */ || c === 0x5b /* [ */) {
      depth++;
      if (depth === 1) expectKey = true;
    } else if (c === 0x7d /* } */ || c === 0x5d /* ] */) {
      depth--;
    } else if (c === 0x2c /* , */ && depth === 1) {
      expectKey = true;
    }
  }
  return keys;
}

function isNonNegativeSafeInt(v: unknown): v is number {
  return typeof v === "number" && Number.isSafeInteger(v) && v >= 0;
}

export interface ParseHeaderOptions {
  /**
   * Total file size in bytes, when known. Enables the reference
   * implementation's "offsets must cover the data section exactly" check
   * (no trailing bytes, no truncation). `SafetensorsFile` passes its buffer
   * length; lazy sources pass the size they discovered.
   */
  fileSize?: number;
}

/**
 * Parses and validates a header from bytes containing at least the first
 * 8 + N bytes of the file (the rest of the file is optional).
 */
export function parseHeader(bytes: Uint8Array, options: ParseHeaderOptions = {}): SafetensorsHeader {
  const n = headerLength(bytes);
  if (bytes.byteLength < 8 + n) {
    throw new SafetensorsError("InvalidHeaderLength", `truncated header: need ${8 + n} bytes, have ${bytes.byteLength}`);
  }
  if (options.fileSize !== undefined && options.fileSize < 8 + n) {
    throw new SafetensorsError("InvalidHeaderLength", `header length ${n} exceeds file size ${options.fileSize}`);
  }
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(8, 8 + n));
  } catch (cause) {
    throw new SafetensorsError("InvalidHeader", "header is not valid UTF-8", { cause });
  }
  if (!text.startsWith("{")) {
    throw new SafetensorsError("InvalidHeaderStart", "header must start with '{'");
  }
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (cause) {
    throw new SafetensorsError("InvalidHeaderDeserialization", `header is not valid JSON (${(cause as Error).message})`, { cause });
  }
  if (typeof json !== "object" || json === null || Array.isArray(json)) {
    throw new SafetensorsError("InvalidHeaderDeserialization", "header JSON must be an object");
  }
  // Scan keys ourselves: catches duplicates (JSON.parse keeps the last) and
  // preserves file order (JS objects hoist integer-like keys such as "0").
  const keys = topLevelKeys(text);
  const seen = new Set<string>();
  for (const key of keys) {
    if (seen.has(key)) throw new SafetensorsError("DuplicateTensor", `duplicate entry ${JSON.stringify(key)} in header`);
    seen.add(key);
  }

  const tensors: TensorInfo[] = [];
  let metadata: Record<string, string> = {};
  for (const name of keys) {
    const raw = (json as Record<string, unknown>)[name];
    if (name === "__metadata__") {
      // MLX writes `"__metadata__": null`; the reference accepts it (Option<HashMap>).
      if (raw === null) continue;
      if (typeof raw !== "object" || Array.isArray(raw)) {
        throw new SafetensorsError("InvalidHeaderDeserialization", "__metadata__ must be an object of strings");
      }
      for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
        if (typeof v !== "string") {
          throw new SafetensorsError("InvalidHeaderDeserialization", `__metadata__.${k} must be a string, got ${typeof v}`);
        }
      }
      metadata = { ...(raw as Record<string, string>) };
      continue;
    }
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
      throw new SafetensorsError("InvalidHeaderDeserialization", `entry ${name} must be an object`);
    }
    const r = raw as { dtype?: unknown; shape?: unknown; data_offsets?: unknown };
    if (typeof r.dtype !== "string" || !Object.hasOwn(BYTES, r.dtype)) {
      throw new SafetensorsError("InvalidDtype", `unsupported dtype ${JSON.stringify(r.dtype)} for ${name}`);
    }
    const dtype = r.dtype as SafeDType;
    if (!Array.isArray(r.shape) || !r.shape.every(isNonNegativeSafeInt)) {
      throw new SafetensorsError("InvalidShape", `${name}: shape must be an array of non-negative integers`);
    }
    const shape = [...(r.shape as number[])];
    const offsets = r.data_offsets;
    if (!Array.isArray(offsets) || offsets.length !== 2 || !offsets.every(isNonNegativeSafeInt)) {
      throw new SafetensorsError("InvalidOffset", `${name}: data_offsets must be [begin, end] non-negative integers`);
    }
    const [begin, end] = offsets as [number, number];
    if (end < begin) throw new SafetensorsError("InvalidOffset", `${name}: data_offsets end ${end} < begin ${begin}`);
    let expected = BYTES[dtype];
    for (const d of shape) {
      expected *= d;
      if (!Number.isSafeInteger(expected)) throw new SafetensorsError("ValidationOverflow", `${name}: byte size overflows`);
    }
    if (end - begin !== expected) {
      throw new SafetensorsError("TensorInvalidInfo", `${name} spans ${end - begin} bytes, expected ${expected} for ${dtype}[${shape}]`);
    }
    tensors.push(Object.freeze({ name, dtype, shape: Object.freeze(shape), dataOffsets: Object.freeze([begin, end] as [number, number]) }));
  }

  // Offsets must tile [0, dataLength) exactly: no gaps, no overlaps.
  const sorted = [...tensors].sort((a, b) => a.dataOffsets[0] - b.dataOffsets[0] || a.dataOffsets[1] - b.dataOffsets[1]);
  let cursor = 0;
  for (const t of sorted) {
    const [b, e] = t.dataOffsets;
    if (b !== cursor) {
      throw new SafetensorsError(
        "InvalidOffset",
        b < cursor ? `${t.name} overlaps the previous tensor (begins at ${b}, previous ends at ${cursor})` : `gap before ${t.name} (begins at ${b}, previous ends at ${cursor})`,
      );
    }
    cursor = e;
  }
  const dataStart = 8 + n;
  if (options.fileSize !== undefined && dataStart + cursor !== options.fileSize) {
    throw new SafetensorsError(
      "MetadataIncompleteBuffer",
      `data section is ${options.fileSize - dataStart} bytes but the header's offsets cover ${cursor}`,
    );
  }
  return {
    tensors: new Map(tensors.map((t) => [t.name, t])),
    metadata: Object.freeze(metadata),
    dataStart,
    dataLength: cursor,
  };
}
