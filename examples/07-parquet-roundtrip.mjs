// frame-parquet: write a Frame to Parquet, read it back with genuine
// projection + predicate pushdown. Pure JS (hyparquet), no native deps.
//
// Run: npm install && npm run build && node examples/07-parquet-roundtrip.mjs
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Frame, stringifyRows } from "@johnhenry/math-plus-frame-arrow";
import { readParquet, writeParquet } from "@johnhenry/math-plus-frame-parquet";

const dir = await mkdtemp(join(tmpdir(), "math-plus-parquet-"));
const path = join(dir, "example.parquet");

const rows = ["id,label,value"];
for (let i = 1; i <= 100; i++) rows.push(`${i},item-${i},${i * 1.5}`);
const frame = Frame.fromCSV(rows.join("\n"));

// Write. Default compression is snappy; "zstd" works too — this package
// wires a real WASM zstd encoder (hyparquet-writer alone would silently
// write corrupt "ZSTD"-labeled bytes without one).
await writeParquet(frame, path, { compression: "snappy" });

// Read back with pushdown: `columns` avoids fetching unrequested column
// chunks; `filter` (Mongo-style shapes) skips row groups via statistics.
// TIP for small files: hyparquet's default footer fetch is 512 KiB — set
// initialFetchSize small or "reading metadata" reads the whole file.
const back = await readParquet(path, {
  columns: ["id", "value"],
  filter: { value: { $gt: 145 } },
  initialFetchSize: 4096,
});

console.log(back.columns); // ['id', 'value']
console.log(back.length); // 4   (values 145.5, 147, 148.5, 150)
console.log(stringifyRows(back.toRows())); // int64 ids are BigInt -> strings

await rm(dir, { recursive: true, force: true });
