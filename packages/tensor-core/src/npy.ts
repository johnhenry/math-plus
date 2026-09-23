/**
 * NPY v1.0 serialization (Tensor-agnostic: works on {data, shape, dtype}).
 *
 * Little-endian only, C-order only (fortran_order: True throws — M1 scope).
 * Dtype coverage matches tensor-core's fixed-width table, including f16
 * (`<f2`, IEEE binary16 — stored as the same Uint16Array bit patterns, so
 * no conversion is needed).
 *
 * bf16 follows the `ml_dtypes` convention (the bfloat16 dtype JAX, TF and
 * PyTorch-interop code use with NumPy): `np.save` of an `ml_dtypes.bfloat16`
 * array writes descr `'<V2'`, an UNTYPED 2-byte void, followed by the raw
 * bfloat16 bits. The file itself does not say "bfloat16"; Python reads it
 * back with `np.load(f).view(ml_dtypes.bfloat16)`. So:
 * - writing bf16 emits `'<V2'`, byte-identical to `np.save` + ml_dtypes;
 * - reading `'<V2'`/`'|V2'` needs an explicit `{ voidAs: "bf16" }` (the
 *   JS twin of that `.view()`), and throws without it, rather than
 *   silently guessing that arbitrary 2-byte void data is bfloat16.
 */
import {
  allocate,
  type AnyTypedArray,
  type DType,
} from "./dtype.ts";

const MAGIC = "\x93NUMPY";

const DTYPE_TO_DESCR: Partial<Record<DType, string>> = {
  bool: "|b1",
  u8: "|u1",
  i8: "|i1",
  u16: "<u2",
  i16: "<i2",
  u32: "<u4",
  i32: "<i4",
  u64: "<u8",
  i64: "<i8",
  f16: "<f2",
  f32: "<f4",
  f64: "<f8",
};

const DESCR_TO_DTYPE: Record<string, DType> = Object.fromEntries(
  Object.entries(DTYPE_TO_DESCR).map(([dtype, descr]) => [
    descr as string,
    dtype as DType,
  ]),
);
// Written only (see the module comment): '<V2' is not reverse-mapped, because
// reading it as bf16 needs the caller's explicit `voidAs`.
DTYPE_TO_DESCR.bf16 = "<V2";
// NumPy sometimes writes |i1 as <i1 etc.; endian-prefix variants of 1-byte types.
DESCR_TO_DTYPE["<i1"] = "i8";
DESCR_TO_DTYPE["<u1"] = "u8";
DESCR_TO_DTYPE["<b1"] = "bool";

export interface NpyPayload {
  data: AnyTypedArray;
  shape: number[];
  dtype: DType;
}

export function serializeNpy(payload: NpyPayload): Uint8Array {
  const descr = DTYPE_TO_DESCR[payload.dtype];
  if (!descr) {
    throw new TypeError(`dtype ${payload.dtype} has no .npy representation`);
  }
  const shapeTuple =
    payload.shape.length === 1
      ? `(${payload.shape[0]},)`
      : `(${payload.shape.join(", ")})`;
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeTuple}, }`;
  // Pad so magic(6) + version(2) + headerLen(2) + header is a multiple of 64.
  const unpadded = 10 + header.length + 1;
  header += " ".repeat((64 - (unpadded % 64)) % 64) + "\n";

  const body = new Uint8Array(
    payload.data.buffer,
    payload.data.byteOffset,
    payload.data.byteLength,
  );
  const out = new Uint8Array(10 + header.length + body.length);
  for (let i = 0; i < MAGIC.length; i++) out[i] = MAGIC.charCodeAt(i);
  out[6] = 1; // major version
  out[7] = 0; // minor version
  out[8] = header.length & 0xff;
  out[9] = header.length >> 8;
  for (let i = 0; i < header.length; i++) out[10 + i] = header.charCodeAt(i);
  out.set(body, 10 + header.length);
  return out;
}

export interface ParseNpyOptions {
  /**
   * How to type a 2-byte void descr (`'<V2'`/`'|V2'`), which is how NumPy
   * saves an `ml_dtypes.bfloat16` array. Only `"bf16"` is supported, and the
   * option has no effect on any other descr.
   */
  voidAs?: "bf16";
}

const VOID2 = /^[<>|=]V2$/;

export function parseNpy(bytes: Uint8Array, options: ParseNpyOptions = {}): NpyPayload {
  for (let i = 0; i < MAGIC.length; i++) {
    if (bytes[i] !== MAGIC.charCodeAt(i)) {
      throw new TypeError("not an NPY file (bad magic)");
    }
  }
  const major = bytes[6];
  let headerLength: number;
  let headerStart: number;
  if (major === 1) {
    headerLength = (bytes[8] as number) | ((bytes[9] as number) << 8);
    headerStart = 10;
  } else if (major === 2 || major === 3) {
    headerLength =
      (bytes[8] as number) |
      ((bytes[9] as number) << 8) |
      ((bytes[10] as number) << 16) |
      ((bytes[11] as number) << 24);
    headerStart = 12;
  } else {
    throw new TypeError(`unsupported NPY version ${major}`);
  }
  let header = "";
  for (let i = 0; i < headerLength; i++) {
    header += String.fromCharCode(bytes[headerStart + i] as number);
  }

  const descrMatch = header.match(/'descr':\s*'([^']+)'/);
  const fortranMatch = header.match(/'fortran_order':\s*(True|False)/);
  const shapeMatch = header.match(/'shape':\s*\(([^)]*)\)/);
  if (!descrMatch || !fortranMatch || !shapeMatch) {
    throw new TypeError(`unparseable NPY header: ${header}`);
  }
  if (fortranMatch[1] === "True") {
    throw new TypeError("fortran_order NPY files are not supported");
  }
  const descr = descrMatch[1] as string;
  if (VOID2.test(descr) && options.voidAs !== "bf16") {
    throw new TypeError(
      `NPY descr '${descr}' is an untyped 2-byte void, which is how NumPy saves ml_dtypes.bfloat16 arrays. ` +
        `If this file holds bfloat16, read it with fromNpy(bytes, { voidAs: "bf16" }).`,
    );
  }
  const dtype: DType | undefined = VOID2.test(descr) ? "bf16" : DESCR_TO_DTYPE[descr];
  if (!dtype) {
    throw new TypeError(`unsupported NPY descr ${descrMatch[1]}`);
  }
  const shape = (shapeMatch[1] as string)
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);

  const dataStart = headerStart + headerLength;
  const size = shape.reduce((a, b) => a * b, 1);
  const data = allocate(dtype, size);
  const bodyBytes = bytes.subarray(dataStart, dataStart + data.byteLength);
  new Uint8Array(data.buffer).set(bodyBytes);
  return { data, shape, dtype };
}
