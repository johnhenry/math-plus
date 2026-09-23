import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Table, TimestampMicrosecond, TimestampMillisecond } from "apache-arrow";
import { col, fn, Frame } from "../src/index.ts";
import { timestampVector } from "./helpers.ts";

test("timestamp[ms] naive and tz-aware round-trip through Series (exact) and toDates() (ms-truncated)", () => {
  const naive = timestampVector([1737000000000n, null, -1n], new TimestampMillisecond(null));
  const tzAware = timestampVector([1737000000000n, null, -1n], new TimestampMillisecond("UTC"));
  const table = new Table({ ts_naive: naive, ts_utc: tzAware });
  const frame = Frame.fromArrow(table);

  assert.equal(frame.schema[0]?.dtype, "timestamp_ms");
  assert.equal(frame.schema[0]?.timezone, null);
  assert.equal(frame.schema[1]?.timezone, "UTC");

  const series = frame.getSeries("ts_naive");
  assert.deepEqual(series.toArray(), [1737000000000n, null, -1n]); // exact bigint epoch-ms

  const dates = series.toDates();
  assert.equal(dates[0]?.getTime(), 1737000000000);
  assert.equal(dates[1], null);
});

test("timestamp[us] preserves EXACT microsecond values Series-side — the builder workaround from the spike (sharp edge #2)", () => {
  // A value that would throw through apache-arrow's own timestamp[us] builder
  // (BigInt(value * 1000) on a fractional-ms float) — this is exactly why
  // Frame never routes timestamp writes through vectorFromArray/a Builder.
  const exactMicros = 1768480496789123n; // has a fractional-ms component (.123 us beyond whole ms)
  const vector = timestampVector([exactMicros, null], new TimestampMicrosecond("America/Los_Angeles"));
  const frame = Frame.fromArrow(new Table({ ts: vector }));

  const series = frame.getSeries("ts");
  assert.equal(series.dtype, "timestamp_us");
  assert.equal(series.descriptor.timezone, "America/Los_Angeles");
  // Exact accessor: full microsecond precision, unlike Vector.get()'s epoch-ms number.
  assert.equal(series.toArray()[0], exactMicros);
  assert.equal(series.toArray()[1], null);
});

test("fn.month() extracts calendar month (UTC calendar day of the epoch value) inside withColumns", () => {
  // 2026-03-15 and 2026-11-02, both UTC midday to avoid any day-boundary ambiguity.
  const msValues = [Date.UTC(2026, 2, 15, 12), Date.UTC(2026, 10, 2, 12)];
  const vector = timestampVector(
    msValues.map((ms) => BigInt(ms)),
    new TimestampMillisecond(null),
  );
  const frame = Frame.fromArrow(new Table({ ts: vector }));
  const withMonth = frame.withColumns({ month: fn.month(col("ts")) });
  const rows = withMonth.toRows();
  assert.equal(rows[0]?.month, 3);
  assert.equal(rows[1]?.month, 11);
});

test("toIPC()/fromIPC() round-trips a timestamp[us, tz] column exactly (JS-only, no Python needed)", () => {
  const exactMicros = 1737000000123456n;
  const vector = timestampVector([exactMicros], new TimestampMicrosecond("America/Los_Angeles"));
  const frame = Frame.fromArrow(new Table({ ts: vector }));
  const bytes = frame.toIPC();
  const restored = Frame.fromIPC(bytes);
  assert.equal(restored.schema[0]?.dtype, "timestamp_us");
  assert.equal(restored.schema[0]?.timezone, "America/Los_Angeles");
  assert.equal(restored.getSeries("ts").toArray()[0], exactMicros);
});
