import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Float64, Table, vectorFromArray } from "apache-arrow";
import { col, Frame } from "../src/index.ts";

test("toIPC()/fromIPC() round-trips basic columns bit-exactly, full cycle, JS-only", () => {
  const table = new Table({
    v: vectorFromArray([1.5, null, 3.25, -0.5], new Float64()),
  });
  const frame = Frame.fromArrow(table);
  const bytes = frame.toIPC();
  assert.ok(bytes instanceof Uint8Array);
  const restored = Frame.fromIPC(bytes);
  assert.deepEqual(restored.toRows(), frame.toRows());
});

test("chunked (multi-record-batch) tables preserve batch structure through toIPC()/fromIPC()", () => {
  const t1 = new Table({ v: vectorFromArray([1, 2, 3], new Float64()) });
  const t2 = new Table({ v: vectorFromArray([4, 5], new Float64()) });
  const t3 = new Table({ v: vectorFromArray([6, 7, 8, 9], new Float64()) });
  const chunked = t1.concat(t2, t3);
  assert.equal(chunked.batches.length, 3);

  const frame = Frame.fromArrow(chunked);
  assert.equal(frame.length, 9);
  const bytes = frame.toIPC();
  const restored = Frame.fromIPC(bytes);
  assert.equal(restored.toArrow().batches.length, 3);
  assert.deepEqual(
    restored.toRows().map((r) => r.v),
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
  );
});

test("filter() across a chunked table is chunk-transparent (random access resolves correctly)", () => {
  const t1 = new Table({ v: vectorFromArray([1, 2, 3], new Float64()) });
  const t2 = new Table({ v: vectorFromArray([4, 5, 6], new Float64()) });
  const chunked = t1.concat(t2);
  const frame = Frame.fromArrow(chunked);
  const evens = frame.filter(col("v").gt(3));
  assert.deepEqual(
    evens.toRows().map((r) => r.v),
    [4, 5, 6],
  );
});
