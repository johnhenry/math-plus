import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Bool, Float64, Int32, Table, Utf8, vectorFromArray } from "apache-arrow";
import { Frame } from "../src/index.ts";

function basicTable(): Table {
  return new Table({
    f64: vectorFromArray([1.5, null, 3.25, -0.5], new Float64()),
    i32: vectorFromArray([1, null, -3, 2147483647], new Int32()),
    str: vectorFromArray(["alpha", null, "gamma", ""], new Utf8()),
    flag: vectorFromArray([true, null, false, true], new Bool()),
  });
}

test("Frame.fromArrow exposes schema/columns/length without collecting extra data", () => {
  const frame = Frame.fromArrow(basicTable());
  assert.deepEqual(frame.columns, ["f64", "i32", "str", "flag"]);
  assert.equal(frame.schema.length, 4);
  assert.equal(frame.schema[0]?.dtype, "float64");
  assert.equal(frame.schema[1]?.dtype, "int32");
  assert.equal(frame.schema[2]?.dtype, "utf8");
  assert.equal(frame.schema[3]?.dtype, "bool");
  assert.equal(frame.length, 4);
});

test("toRows() preserves nulls and empty-string-vs-null distinction", () => {
  const frame = Frame.fromArrow(basicTable());
  const rows = frame.toRows();
  assert.equal(rows.length, 4);
  assert.deepEqual(rows[0], { f64: 1.5, i32: 1, str: "alpha", flag: true });
  assert.deepEqual(rows[1], { f64: null, i32: null, str: null, flag: null });
  assert.equal(rows[3]?.str, ""); // empty string, distinct from null
});

test("nullCount() reports per-column null counts", () => {
  const frame = Frame.fromArrow(basicTable());
  assert.deepEqual(frame.nullCount(), { f64: 1, i32: 1, str: 1, flag: 1 });
});

test("select/drop/rename are lazy and compose", () => {
  const frame = Frame.fromArrow(basicTable());
  const selected = frame.select("f64", "str");
  assert.deepEqual(selected.columns, ["f64", "str"]);

  const dropped = frame.drop("flag");
  assert.deepEqual(dropped.columns, ["f64", "i32", "str"]);

  const renamed = frame.rename({ f64: "value" });
  assert.deepEqual(renamed.columns, ["value", "i32", "str", "flag"]);
  assert.deepEqual(renamed.toRows()[0], { value: 1.5, i32: 1, str: "alpha", flag: true });
});

test("dropNull() without arguments drops rows null in ANY column", () => {
  const frame = Frame.fromArrow(basicTable());
  const clean = frame.dropNull();
  assert.equal(clean.length, 3); // rows 0, 2, 3 have no nulls; row 1 is all-null
});

test("dropNull(column) only checks the given column(s)", () => {
  const frame = Frame.fromArrow(basicTable());
  const clean = frame.dropNull("i32");
  assert.equal(clean.length, 3); // only row 1 has a null i32
});

test("fillNull(scalar) fills every column's nulls with the same value where type-compatible", () => {
  const frame = Frame.fromArrow(
    new Table({ a: vectorFromArray([1.0, null, 3.0], new Float64()) }),
  );
  const filled = frame.fillNull(0);
  assert.deepEqual(filled.toRows(), [{ a: 1 }, { a: 0 }, { a: 3 }]);
});

test("fillNull(map) only fills specified columns", () => {
  const frame = Frame.fromArrow(basicTable());
  const filled = frame.fillNull({ f64: -1, str: "MISSING" });
  const rows = filled.toRows();
  assert.equal(rows[1]?.f64, -1);
  assert.equal(rows[1]?.str, "MISSING");
  assert.equal(rows[1]?.i32, null); // untouched
});

test("collect() is idempotent and memoizes (repeated calls return the same Frame)", () => {
  const frame = Frame.fromArrow(basicTable()).select("f64");
  const collected1 = frame.collect();
  const collected2 = frame.collect();
  assert.equal(collected1, collected2);
  assert.equal(collected1, frame); // collect() returns `this`
});

test("limit() and slice() truncate rows without touching columns", () => {
  const frame = Frame.fromArrow(basicTable());
  assert.equal(frame.limit(2).length, 2);
  assert.deepEqual(frame.limit(2).toRows()[1], { f64: null, i32: null, str: null, flag: null });
  const sliced = frame.slice(1, 3);
  assert.equal(sliced.length, 2);
  assert.equal(sliced.toRows()[0]?.str, null);
  assert.equal(sliced.toRows()[1]?.str, "gamma");
});
