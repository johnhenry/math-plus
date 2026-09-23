# @johnhenry/math-plus-safetensors

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fmath-plus-safetensors.svg)](https://www.npmjs.com/package/@johnhenry/math-plus-safetensors)
[![license](https://img.shields.io/npm/l/%40johnhenry%2Fmath-plus-safetensors.svg)](../../LICENSE)

Read and write [safetensors](https://github.com/huggingface/safetensors)
files in any JS runtime — Node, Bun, Deno, browsers. Headers are validated
like the reference implementation; tensors come back as typed arrays (F16 as
a native `Float16Array`); big checkpoints can be opened lazily from a file,
a `Blob`/`File`, or an HTTP URL with Range requests, reading only the tensors
you ask for. Zero dependencies. Differential-tested against Python's
`safetensors` (+ NumPy/PyTorch), in both directions.

## Install

```bash
npm install @johnhenry/math-plus-safetensors
```

## Quick start

```js
import { openSafetensors, readSafetensors, writeSafetensors } from "@johnhenry/math-plus-safetensors";

// Lazy: reads the header only (one read), then one read per tensor.
const model = await openSafetensors("/path/to/model.safetensors"); // Node/Bun path
// or: await openSafetensors("https://huggingface.co/org/repo/resolve/main/model.safetensors",
//                            { headers: { Authorization: `Bearer ${token}` } });
// or: await openSafetensors(fileInput.files[0]);                   // Blob/File in a browser
model.names();                         // ["embeddings.weight", ...]
model.info("embeddings.weight");       // { dtype: "F16", shape: [50368, 768], dataOffsets: [...] }
const w = await model.read("embeddings.weight");   // Float16Array
const f = await model.toF32("embeddings.weight");  // new Float32Array
const all = await model.readMany();                // Map<name, TypedArray>, coalesced reads
await model.close();                               // closes files this package opened

// In memory: everything already in an ArrayBuffer/Uint8Array.
const file = readSafetensors(bytes);
file.view("x");   // zero-copy typed view (copied once if misaligned)
file.toF32("x");  // new Float32Array

// Write (byte-identical to Python's safetensors.serialize / save_file):
const out = writeSafetensors(
  { x: { dtype: "F32", shape: [2, 2], data: new Float32Array([1, 2, 3, 4]) } },
  { format: "pt" },
);
```

With `@johnhenry/math-plus-tensor-core` installed (optional peer):

```js
import { toTensor, readTensor, fromTensor } from "@johnhenry/math-plus-safetensors/tensor";
const t = toTensor(file, "x");            // Tensor, aliases the file's bytes when aligned
const u = await readTensor(model, "x");   // Tensor from a lazy file
writeSafetensors({ x: fromTensor(t) });
```

## API surface

| Group | Exports |
|---|---|
| In memory | `readSafetensors(bytes)` → `SafetensorsFile` {`header`, `metadata`, `names()`, `has()`, `info()`, `bytes()`, `view()`, `toF32()`} |
| Lazy | `openSafetensors(source, options?)` → `LazySafetensors` {`header`, `metadata`, `size`, `names()`, `has()`, `info()`, `readBytes()`, `read()`, `readMany()`, `toF32()`, `close()`} |
| Sources | `toByteSource`, `MemorySource`, `BlobSource`, `HttpSource`, `FileHandleSource`, the `ByteSource` interface (implement it for any other transport) |
| Writer | `writeSafetensors(tensors, metadata?)` → `Uint8Array` |
| Low level | `parseHeader(bytes, {fileSize?})`, `headerLength`, `viewAs(dtype, bytes)`, `toFloat32(dtype, bytes)`, `aligned`, `BYTES`, `MAX_HEADER` |
| Errors | `SafetensorsError` with a `code` (`InvalidOffset`, `MetadataIncompleteBuffer`, `DuplicateTensor`, …; reference-implementation names where one exists) |
| `/tensor` subpath | `toTensor`, `readTensor`, `bytesToTensor`, `fromTensor` (needs `@johnhenry/math-plus-tensor-core`) |

`openSafetensors` sources: `Uint8Array`/`ArrayBuffer` (zero-copy), `Blob`/`File`
(incl. `Bun.file()`), `URL` or string starting with `http:`/`https:`/`blob:`/`data:`,
a `file:` URL or any other string (a filesystem path — Node/Bun/Deno only), a
Node `FileHandle` (not closed by us), or a custom `ByteSource`. Options:
`fetch`, `headers`, `signal`, `probeBytes` (default 256 KiB — the first read,
sized to cover the header in one round trip).

## Validation

Mirrors the reference Rust implementation: 8-byte little-endian header
length ≤ 100 MB; UTF-8 JSON that starts with `{`; string-only `__metadata__`
(or `null`, as MLX writes); known dtype, non-negative integer shape, and
`data_offsets` spanning exactly `prod(shape) × sizeof(dtype)`; offsets
sorted by begin must tile the data section with no gap or overlap, starting
at 0; and when the file size is known (in-memory, files, Blobs, HTTP with
`Content-Range`) the last tensor must end exactly at end of file. Duplicate
tensor names are rejected (plain `JSON.parse` would silently keep the last).
Tensor order is preserved exactly as written, even for integer-like names.

## Memory behaviour

- `readSafetensors` keeps the whole file in memory; views are zero-copy
  **when aligned**. Files whose header isn't padded to 8 bytes (MLX writes
  them that way — e.g. the Laya checkpoints) make every tensor misaligned,
  so each `view()` copies that tensor once.
- `openSafetensors` holds only the header. `read`/`readBytes`/`toF32` read
  each tensor into a fresh (aligned, so zero-copy-viewable) buffer;
  `readMany` coalesces adjacent tensors (gap ≤ 64 KiB, chunk ≤ 64 MiB) and its
  views share those chunk buffers. Reading every tensor of an 842 MB F16
  checkpoint therefore costs ~842 MB; `toF32` on all of it ~1.7 GB.
- HTTP: if the server ignores `Range` (answers 200), the whole body is
  downloaded once and served from memory (`source.rangesSupported === false`).
- `toFloat32`/`toF32` always return a new array you own (never aliasing the
  file), including for F32.

## Limitations

- **dtypes:** BOOL, U8/I8, U16/I16, F16, BF16, U32/I32, F32, U64/I64, F64.
  FP8 (`F8_E4M3`, `F8_E5M2`, …), `F4`/`F6`, and `C64` are rejected with
  `InvalidDtype` (the whole file, not just that tensor).
- **F16 needs `Float16Array`** (Node ≥ 24, Bun, Deno, Chrome 135+, Firefox
  129+, Safari 18.2+); without it `view`/`read`/`toF32` of F16 throw
  `Float16Unavailable`. BF16 has no native array: views are `Uint16Array`
  bit patterns; `toF32` widens exactly.
- `toF32` of I64/U64 goes through `Number`, so magnitudes above 2^53 round
  twice (to double, then to float32).
- **Host byte order is assumed little-endian** (true on every platform JS
  engines ship on today); data is never byte-swapped.
- The writer builds the whole file in one `Uint8Array` (no streaming
  writer) and follows the reference layout: tensors sorted by dtype
  alignment then name, header space-padded to 8 bytes.
- No resumable/parallel HTTP downloads, no retry policy, no caching —
  combine with a caching layer (e.g. `@johnhenry/hf-cache`) for that.
  Redirects are followed by `fetch`; the `Range` header is re-sent to the
  final URL (standard fetch behaviour).
- JSR publishes the main entry only (`./src/index.ts`); the `/tensor`
  subpath is npm-only for now.

## Tests

`npm test`. `test/oracle.test.ts` generates fixtures with Python
(`scripts/safetensors_oracle.py`: `safetensors.numpy.save_file`,
`safetensors.torch.save_file` for BF16, and a hand-built unpadded
MLX-style file) and checks bytes and float32 conversions bit-for-bit, then
checks our writer is read back by Python and is byte-identical to
`safetensors.serialize`. It skips (never fails) without a Python that has
`numpy` + `safetensors` (`$MATH_PLUS_ORACLE_PYTHON`, else `python3`); BF16
fixtures additionally need `torch` or `ml_dtypes`. `test/checkpoint.test.ts`
lazily opens the real `aac6fef/laya-mlx` checkpoint from the local Hugging
Face cache when present (skipped otherwise).
