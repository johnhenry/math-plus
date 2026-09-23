/**
 * scanParquetLazy: genuinely lazy row-data reads on top of frame-arrow's
 * `"lazySource"`/`collectAsync()` (issue #32, part 2/2 — see scan-lazy.ts's
 * module doc for exactly what's deferred and what isn't).
 *
 * Byte-counting methodology matches test/pushdown.test.ts: wrap a real
 * file's `AsyncBuffer` with a counting proxy and call `lazyParquetFrame`
 * directly (the exact function `scanParquetLazy` delegates to per matched
 * file) — proves pruning is real at the I/O layer, not just that the final
 * Frame happens to contain the right columns.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { asyncBufferFromFile, type AsyncBuffer } from "hyparquet";
import { lazyParquetFrame, scanParquet, scanParquetLazy } from "../src/index.ts";

const SCAN_DIR = new URL("fixtures/scan/", import.meta.url).pathname;
const PUSHDOWN_FIXTURE = new URL("fixtures/pushdown.parquet", import.meta.url).pathname;

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

test("scanParquetLazy: schema/columns work without any row-data read (only footer metadata is fetched up front)", async () => {
  const inner = await asyncBufferFromFile(PUSHDOWN_FIXTURE);
  const counting = countingBuffer(inner);
  const frame = await lazyParquetFrame(counting.file, SMALL_FOOTER_FETCH);
  assert.deepEqual(frame.columns, ["id", "value", "label"]);
  assert.equal(frame.schema.length, 3);
  // A footer-only fetch is a small, bounded number of bytes -- nowhere near
  // what reading all 6000 rows' worth of column data would cost.
  assert.ok(
    counting.stats.bytes < 20_000,
    `expected schema resolution alone to fetch well under 20KB (metadata only), got ${counting.stats.bytes}B`,
  );
});

test("scanParquetLazy: collectAsync() materializes the SAME result as eager scanParquet", async () => {
  const eager = await scanParquet(`${SCAN_DIR}*.parquet`);
  const lazy = await scanParquetLazy(`${SCAN_DIR}*.parquet`);
  assert.deepEqual(lazy.columns, eager.columns);
  const collected = await lazy.collectAsync();
  const eagerRows = eager.toRows().sort((a, b) => Number(a.id) - Number(b.id));
  const lazyRows = collected.toRows().sort((a, b) => Number(a.id) - Number(b.id));
  assert.deepEqual(lazyRows, eagerRows);
});

test("scanParquetLazy: calling a sync terminal accessor before collectAsync() throws the documented clear error", async () => {
  const frame = await scanParquetLazy(`${SCAN_DIR}*.parquet`);
  assert.throws(() => frame.toRows(), /collectAsync/);
});

test("scanParquetLazy: select() automatically narrows the columns fetched per file, with NO explicit `columns` option needed", async () => {
  const inner = await asyncBufferFromFile(PUSHDOWN_FIXTURE);

  const full = countingBuffer(inner);
  const fullFrame = await lazyParquetFrame(full.file, SMALL_FOOTER_FETCH);
  await fullFrame.collectAsync();
  assert.equal(fullFrame.length, 6000);

  const projected = countingBuffer(inner);
  const projectedFrame = await lazyParquetFrame(projected.file, SMALL_FOOTER_FETCH);
  const narrowed = projectedFrame.select("id");
  await narrowed.collectAsync();
  assert.deepEqual(narrowed.columns, ["id"]);
  assert.equal(narrowed.length, 6000);

  assert.ok(
    projected.stats.bytes < full.stats.bytes,
    `expected select('id') to fetch fewer bytes (${projected.stats.bytes}B) than the full collect (${full.stats.bytes}B) ` +
      "-- pruning through the Frame API alone, no explicit `columns` option passed to lazyParquetFrame",
  );
});

test("scanParquetLazy: filter on a monotonic column fetches dramatically fewer bytes via row-group statistics skipping", async () => {
  const inner = await asyncBufferFromFile(PUSHDOWN_FIXTURE);

  const full = countingBuffer(inner);
  await (await lazyParquetFrame(full.file, SMALL_FOOTER_FETCH)).collectAsync();

  const filtered = countingBuffer(inner);
  const filteredFrame = await lazyParquetFrame(filtered.file, {
    ...SMALL_FOOTER_FETCH,
    filter: { value: { $gt: 5990.5 } },
  });
  const collected = await filteredFrame.collectAsync();
  assert.equal(collected.length, 9);

  assert.ok(
    filtered.stats.bytes < full.stats.bytes * 0.3,
    `expected filtered read (${filtered.stats.bytes}B) to fetch well under 30% of the full read's bytes (${full.stats.bytes}B)`,
  );
});

test("scanParquetLazy: concat of multiple lazy files resolves correctly, each getting its own read() call with its own pruning", async () => {
  const frame = await scanParquetLazy(`${SCAN_DIR}*.parquet`, {});
  const narrowed = frame.select("id", "part");
  const collected = await narrowed.collectAsync();
  assert.equal(collected.length, 30);
  assert.deepEqual(collected.columns, ["id", "part"]);
});

test("scanParquetLazy: a pattern matching nothing throws a clear error", async () => {
  await assert.rejects(() => scanParquetLazy(`${SCAN_DIR}nope-*.parquet`), /no files matched pattern/);
});

test("scanParquetLazy: a pattern matching a single file still works", async () => {
  const frame = await scanParquetLazy(`${SCAN_DIR}part-0.parquet`);
  const collected = await frame.collectAsync();
  assert.equal(collected.length, 10);
});
