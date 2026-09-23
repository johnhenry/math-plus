/**
 * Proves `columns`/`filter` pushdown is REAL — genuinely reduces bytes
 * fetched at the hyparquet I/O layer through frame-parquet's own
 * `readParquetFile`, not just that the final Frame happens to contain the
 * right rows (which would also be true of a full-read-then-filter
 * implementation and wouldn't prove pushdown at all).
 *
 * Methodology mirrors docs/spikes/parquet-bakeoff.md §2-3 (and its
 * `test_io_skipping.mjs`): wrap the real file's `AsyncBuffer` with a
 * byte-counting proxy around `.slice()`, and call `readParquetFile` — the
 * EXACT function `readParquet(path, opts)` delegates to — directly with
 * that instrumented buffer. test/fixtures/pushdown.parquet has 12 row
 * groups and a strictly monotonic `value` column, so a `$gt` filter has a
 * decisive, predictable row-group-skip outcome (statistics-based).
 *
 * `initialFetchSize: 4096` is passed throughout: hyparquet's own metadata
 * parser speculatively fetches the LAST 512 KiB of the file in one round
 * trip by default (tuned for remote/HTTP sources where round-trips are
 * expensive) — and test/fixtures/pushdown.parquet is deliberately kept
 * small (~70 KB, per the "small, committed fixtures" convention), so with
 * the default that speculative read pulls in the ENTIRE file as part of
 * "parsing metadata," swamping any data-fetch savings from pushdown in the
 * byte count. This isn't hypothetical — it was caught by this exact test
 * failing during development. A small `initialFetchSize` (real, documented
 * frame-parquet option — see read.ts) bounds the metadata fetch to roughly
 * its actual size (a few KB), so the byte-count comparison below measures
 * what it claims to: data-fetch reduction from columns/filter, not an
 * artifact of the fixture happening to be smaller than hyparquet's default
 * speculative-read window.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { asyncBufferFromFile, type AsyncBuffer } from "hyparquet";
import { readParquetFile } from "../src/index.ts";

const FIXTURE = new URL("fixtures/pushdown.parquet", import.meta.url).pathname;

interface IoStats {
  calls: number;
  bytes: number;
}

function countingBuffer(inner: AsyncBuffer): { file: AsyncBuffer; stats: IoStats } {
  const stats: IoStats = { calls: 0, bytes: 0 };
  const file: AsyncBuffer = {
    byteLength: inner.byteLength,
    async slice(start: number, end?: number): Promise<ArrayBuffer> {
      stats.calls++;
      stats.bytes += (end ?? inner.byteLength) - start;
      return inner.slice(start, end);
    },
  };
  return { file, stats };
}

const SMALL_FOOTER_FETCH = { initialFetchSize: 4096 };

test("readParquetFile: filter on a monotonic column fetches dramatically fewer bytes than a full read", async () => {
  const inner = await asyncBufferFromFile(FIXTURE);

  const full = countingBuffer(inner);
  const fullFrame = await readParquetFile(full.file, { columns: ["id", "value"], ...SMALL_FOOTER_FETCH });
  assert.equal(fullFrame.length, 6000);

  // Only the last row group's worth of rows can match (value is 0..5999,
  // strictly monotonic, row_group_size=500 -> row group 11 holds 5500..5999).
  const filtered = countingBuffer(inner);
  const filteredFrame = await readParquetFile(filtered.file, {
    columns: ["id", "value"],
    filter: { value: { $gt: 5990.5 } },
    ...SMALL_FOOTER_FETCH,
  });
  assert.equal(filteredFrame.length, 9);

  assert.ok(
    filtered.stats.bytes < full.stats.bytes * 0.3,
    `expected filtered read (${filtered.stats.bytes}B) to fetch well under 30% of the full read's bytes ` +
      `(${full.stats.bytes}B) via row-group statistics skipping — got ${((filtered.stats.bytes / full.stats.bytes) * 100).toFixed(1)}%`,
  );
});

test("readParquetFile: columns projection fetches fewer bytes than reading every column", async () => {
  const inner = await asyncBufferFromFile(FIXTURE);

  const full = countingBuffer(inner);
  await readParquetFile(full.file, { ...SMALL_FOOTER_FETCH });

  const projected = countingBuffer(inner);
  const projectedFrame = await readParquetFile(projected.file, { columns: ["id"], ...SMALL_FOOTER_FETCH });
  assert.equal(projectedFrame.length, 6000);
  assert.deepEqual(projectedFrame.columns, ["id"]);

  assert.ok(
    projected.stats.bytes < full.stats.bytes,
    `expected projecting to 1 of 3 columns (${projected.stats.bytes}B) to fetch fewer bytes than all columns (${full.stats.bytes}B)`,
  );
});

test("control case: a full read (no columns/filter) really does fetch the whole file's data", async () => {
  const inner = await asyncBufferFromFile(FIXTURE);
  const full = countingBuffer(inner);
  const frame = await readParquetFile(full.file, { ...SMALL_FOOTER_FETCH });
  assert.equal(frame.length, 6000);
  // Sanity: a full read should touch a substantial fraction of the file
  // (metadata + every row group's every column chunk) — this is the
  // baseline the two tests above compare against, so it must not itself be
  // some trivially-small number that would make the ratio comparisons
  // meaningless.
  assert.ok(full.stats.bytes > 20_000, `expected a substantial full-read byte count, got ${full.stats.bytes}`);
});

test("readParquetFile: WITHOUT a small initialFetchSize, a small file's default metadata fetch dominates byte accounting " +
  "(documents why the tests above override it — see this file's module doc)", async () => {
  const inner = await asyncBufferFromFile(FIXTURE);
  const full = countingBuffer(inner);
  await readParquetFile(full.file, { columns: ["id", "value"] });
  // hyparquet's default 512 KiB speculative footer fetch clamps to reading
  // the file's ENTIRE contents just to "parse metadata" when the file is
  // smaller than that (as pushdown.parquet is here) — on top of which the
  // subsequent data read still issues its own separate slice() calls for the
  // column chunks it needs, so total accounted bytes actually EXCEEDS the
  // file size (the metadata fetch and the data fetch overlap but aren't
  // deduped by byte range, matching what a real, non-caching remote source
  // would also pay for two separate range requests covering the same bytes).
  assert.ok(
    full.stats.bytes >= inner.byteLength,
    `expected the default-initialFetchSize read to fetch at least the whole file's bytes (${inner.byteLength}B) just for metadata, got ${full.stats.bytes}B`,
  );
});
