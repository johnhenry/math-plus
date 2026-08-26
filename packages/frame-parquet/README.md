# @johnhenry/math-plus-frame-parquet

Parquet read/write for `@johnhenry/math-plus-frame-arrow`, built on
hyparquet — with *genuine* projection and predicate pushdown (statistics-based
row-group skipping and column-chunk-level fetch avoidance, proven at the byte
level in tests, not just claimed).

## Install

```bash
npm install @johnhenry/math-plus-frame-parquet
```

## Quick start

```js
import { readParquet, writeParquet, scanParquetLazy } from "@johnhenry/math-plus-frame-parquet";

// Projection + predicate pushdown (hyparquet's Mongo-style filter shapes)
const frame = await readParquet("data.parquet", {
  columns: ["id", "value"],
  filter: { value: { $gt: 5990.5 } },
});

// Glob scan, genuinely lazy row data (footers still read up front)
const lazy = await scanParquetLazy("parts/*.parquet", {});
const narrowed = await lazy.select("id", "part").collectAsync();

// Write: snappy (default), zstd, or uncompressed — verified against real pyarrow
await writeParquet(frame, "out.parquet", { compression: "zstd" });
```

## API surface

| Export | What it is |
|---|---|
| `readParquet(path, opts?)` | Local file → `Frame`, with `columns`/`filter`/`initialFetchSize` |
| `readParquetFile(asyncBuffer, opts?)` | The core — browser/edge/no-fs path |
| `scanParquet(glob, opts?)` | Multi-file, **eager** (reads and concats immediately; requires `{ columns }` for projection) |
| `scanParquetLazy(glob, opts?)` / `lazyParquetFrame(buf, opts?)` | Lazy row data via `Frame.fromLazySource`; `select()` prunes automatically; **must** use `collectAsync()` |
| `writeParquet(frame, path, opts?)` / `writeParquetBuffer(frame, opts?)` | Write with `compression` (`snappy`/`zstd`/`uncompressed`) and `rowGroupSize` |
| `UnsupportedParquetTypeError`, `mapLeafElement`, `mapTopLevelColumns` | Schema mapping utilities |

Free functions, not `Frame` methods — `Frame`'s constructor is a closed set
of entry points by design (deviation from issue #20's sketch, documented).

## Traps

- **The zstd write footgun (why this package exists in this shape):**
  hyparquet-writer with `codec: 'ZSTD'` and no registered compressor does
  not throw — it **silently writes uncompressed bytes labeled ZSTD**, which
  pyarrow then rejects. This package wires a real encoder
  (`@hpcc-js/wasm-zstd`); `hyparquet-compressors` alone is decompress-only.
- **`initialFetchSize`:** hyparquet speculatively fetches the last 512 KiB
  for the footer. For files smaller than that, "reading metadata" reads the
  whole file — swamping any pushdown savings. Set it small for small files.
- **Lazy scans read every matched footer up front** (schema can't be known
  otherwise), and `scanParquetLazy` type-maps *every* top-level column — a
  file containing an INT96/MAP column throws immediately even if you never
  select it. The eager `readParquet` path can select *around* an unsupported
  column.
- **No file-level skipping:** neither scan does Hive-partition or
  whole-file pruning; every matched file is read on collect.
- **Sync accessors on a lazy frame throw** — use `collectAsync()`.
- **Timestamps read back as raw bigints**, not `Date` (deliberately
  overriding hyparquet's lossy `Date` parsers).
- **Dictionary asymmetry:** a `dictionary` column writes as plain STRING and
  reads back `utf8` (Parquet dictionary encoding is page-level, not
  schema-level) — documented v1 simplification.

## Tests

`npm test` — byte-counting `AsyncBuffer` wrappers prove pushdown (filtered
read < 30% of full-read bytes), nested list/struct round-trips with nulls at
every level (issue #30), and write-compatibility checks against real pyarrow.

## Provenance

Built from the issue #20 Parquet bakeoff (hyparquet chosen over parquet-wasm)
and issue #32 (lazy scan). Part of the
[math-plus](https://github.com/johnhenry/math-plus) monorepo; family docs at
<https://opensource.johnhenry.me/math/>. Python-side interop lives in
`johnhenry-math-plus-interop` (PyPI).
