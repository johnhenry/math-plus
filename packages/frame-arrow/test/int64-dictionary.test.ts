import assert from "node:assert/strict";
import { test } from "node:test";
import { Dictionary, Int32, Int64, Table, Utf8, vectorFromArray } from "apache-arrow";
import { Frame } from "../src/index.ts";
import { stringifyRows } from "../src/safe-json.ts";

test("int64 round-trips as bigint, including values outside Number.MAX_SAFE_INTEGER", () => {
  const table = new Table({
    id: vectorFromArray([1n, null, -9007199254740993n, 9223372036854775807n], new Int64()),
  });
  const frame = Frame.fromArrow(table);
  assert.equal(frame.schema[0]?.dtype, "int64");
  const rows = frame.toRows();
  assert.equal(rows[0]?.id, 1n);
  assert.equal(rows[1]?.id, null);
  assert.equal(rows[2]?.id, -9007199254740993n); // exact — would be lossy as a float64
  assert.equal(rows[3]?.id, 9223372036854775807n);
  assert.equal(typeof rows[0]?.id, "bigint");
});

test("plain JSON.stringify throws on a bigint row, but stringifyRows() does not (sharp edge #1)", () => {
  const table = new Table({ id: vectorFromArray([1n], new Int64()) });
  const rows = Frame.fromArrow(table).toRows();
  assert.throws(() => JSON.stringify(rows));
  const json = stringifyRows(rows);
  assert.equal(json, '[{"id":"1"}]');
});

test("dictionary<utf8> decodes to plain strings, not proxies", () => {
  const dictType = new Dictionary(new Utf8(), new Int32());
  const table = new Table({
    color: vectorFromArray(["red", "green", null, "red", "blue", "green"], dictType),
  });
  const frame = Frame.fromArrow(table);
  assert.equal(frame.schema[0]?.dtype, "dictionary");
  const rows = frame.toRows();
  assert.deepEqual(
    rows.map((r) => r.color),
    ["red", "green", null, "red", "blue", "green"],
  );
});

test("Series.cast() converts int64 <-> float64 and utf8 <-> dictionary", () => {
  const table = new Table({
    id: vectorFromArray([1n, 2n, 3n], new Int64()),
    name: vectorFromArray(["a", "b", "a"], new Utf8()),
  });
  const frame = Frame.fromArrow(table);

  const idAsFloat = frame.getSeries("id").cast("float64");
  assert.deepEqual(idAsFloat.toArray(), [1, 2, 3]);

  const nameAsDict = frame.getSeries("name").cast("dictionary");
  assert.equal(nameAsDict.dtype, "dictionary");
  assert.deepEqual(nameAsDict.toArray(), ["a", "b", "a"]);

  const backToUtf8 = nameAsDict.cast("utf8");
  assert.equal(backToUtf8.dtype, "utf8");
  assert.deepEqual(backToUtf8.toArray(), ["a", "b", "a"]);
});

test("Series.cast() throws a clear error for unsupported casts", () => {
  const table = new Table({ name: vectorFromArray(["a"], new Utf8()) });
  const series = Frame.fromArrow(table).getSeries("name");
  assert.throws(() => series.cast("float64"), /unsupported cast/);
});

test("Series.cast() throws instead of silently losing precision on int64 -> float64 outside Number.MAX_SAFE_INTEGER", () => {
  const table = new Table({
    id: vectorFromArray([9007199254740993n], new Int64()), // MAX_SAFE_INTEGER + 2
  });
  const series = Frame.fromArrow(table).getSeries("id");
  assert.throws(() => series.cast("float64"), /precision loss/);
});

test("Series.cast() int64 -> float64 still casts normally for in-range values (no regression)", () => {
  const table = new Table({
    id: vectorFromArray([1n, -9007199254740991n, 9007199254740991n], new Int64()), // MIN/MAX_SAFE_INTEGER
  });
  const series = Frame.fromArrow(table).getSeries("id");
  assert.deepEqual(series.cast("float64").toArray(), [1, -9007199254740991, 9007199254740991]);
});
