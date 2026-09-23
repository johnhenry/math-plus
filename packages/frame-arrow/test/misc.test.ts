import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Float64, Int32, Table, Utf8, vectorFromArray } from "apache-arrow";
import { desc, Frame } from "../src/index.ts";

test("Frame.concat (static) and .concat (instance) are equivalent and require matching schemas", () => {
  const t1 = Frame.fromArrow(new Table({ v: vectorFromArray([1, 2], new Float64()) }));
  const t2 = Frame.fromArrow(new Table({ v: vectorFromArray([3, 4], new Float64()) }));

  const viaStatic = Frame.concat([t1, t2]);
  const viaInstance = t1.concat(t2);
  assert.deepEqual(viaStatic.toRows(), viaInstance.toRows());
  assert.deepEqual(
    viaStatic.toRows().map((r) => r.v),
    [1, 2, 3, 4],
  );

  const mismatched = Frame.fromArrow(new Table({ other: vectorFromArray([1], new Float64()) }));
  assert.throws(() => Frame.concat([t1, mismatched]).schema, /schema mismatch/);
});

test("Frame.concat with a zero-row input doesn't throw on read (issue #31 — apache-arrow's Table.concat breaks getChild() once any input has zero rows)", () => {
  const empty = Frame.fromArrow(new Table({ v: vectorFromArray([], new Int32()) }));
  const full = Frame.fromArrow(new Table({ v: vectorFromArray([1, 2, 3], new Int32()) }));

  // Zero-row input in the middle/edges — every position, since the bug is
  // triggered by ANY concatenated input having zero rows, not just the first.
  const combined = Frame.concat([empty, full, empty]);
  assert.equal(combined.length, 3);
  assert.deepEqual(
    combined.toRows().map((r) => r.v),
    [1, 2, 3],
  );
  assert.deepEqual(combined.toArrow().getChild("v")?.toArray(), Int32Array.from([1, 2, 3]));

  // Every input empty -> still returns a valid (empty) Frame with the right schema.
  const allEmpty = Frame.concat([empty, empty]);
  assert.equal(allEmpty.length, 0);
  assert.deepEqual(allEmpty.columns, ["v"]);
});

test("sortBy() is stable, ascending by default, descending via desc()", () => {
  const frame = Frame.fromArrow(
    new Table({
      k: vectorFromArray(["b", "a", "a", "c"], new Utf8()),
      seq: vectorFromArray([0, 1, 2, 3], new Int32()),
    }),
  );
  const asc = frame.sortBy("k");
  assert.deepEqual(
    asc.toRows().map((r) => [r.k, r.seq]),
    [
      ["a", 1],
      ["a", 2],
      ["b", 0],
      ["c", 3],
    ],
  );
  const descending = frame.sortBy(desc("k"));
  assert.deepEqual(
    descending.toRows().map((r) => r.k),
    ["c", "b", "a", "a"],
  );
});

test("sortBy() sorts nulls last regardless of direction", () => {
  const frame = Frame.fromArrow(new Table({ v: vectorFromArray([3, null, 1, null, 2], new Float64()) }));
  assert.deepEqual(frame.sortBy("v").toRows().map((r) => r.v), [1, 2, 3, null, null]);
  assert.deepEqual(frame.sortBy(desc("v")).toRows().map((r) => r.v), [3, 2, 1, null, null]);
});

test("toCSV() is bigint-safe and quotes fields containing commas", () => {
  const frame = Frame.fromArrow(
    new Table({
      name: vectorFromArray(["Smith, John", "Doe"], new Utf8()),
      score: vectorFromArray([1.5, null], new Float64()),
    }),
  );
  const csv = frame.toCSV();
  assert.equal(csv, 'name,score\n"Smith, John",1.5\nDoe,\n');
});

test("toArrow() returns the underlying apache-arrow Table", () => {
  const table = new Table({ v: vectorFromArray([1, 2], new Float64()) });
  const frame = Frame.fromArrow(table);
  const returned = frame.toArrow();
  assert.equal(returned.numRows, 2);
  assert.ok(returned.schema);
});

test("Series.unique() and Series.valueCounts()", () => {
  const frame = Frame.fromArrow(new Table({ v: vectorFromArray(["a", "b", "a", "a", "c", "b"], new Utf8()) }));
  const series = frame.getSeries("v");
  assert.deepEqual(series.unique().toArray(), ["a", "b", "c"]);
  const counts = series.valueCounts();
  assert.deepEqual(counts.toRows(), [
    { value: "a", count: 3n },
    { value: "b", count: 2n },
    { value: "c", count: 1n },
  ]);
});

test("Series.isNull() and Series.fillNull()", () => {
  const frame = Frame.fromArrow(new Table({ v: vectorFromArray([1.0, null, 3.0], new Float64()) }));
  const series = frame.getSeries("v");
  assert.deepEqual(series.isNull().toArray(), [false, true, false]);
  assert.deepEqual(series.fillNull(-1).toArray(), [1, -1, 3]);
});
