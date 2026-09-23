import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Field, Float64, List, Struct, Table, Utf8, vectorFromArray } from "apache-arrow";
import { Frame } from "../src/index.ts";

test("list<float64> materializes to plain arrays (not Arrow Vector proxies), nested nulls preserved", () => {
  const listType = new List(new Field("item", new Float64(), true));
  const table = new Table({
    values: vectorFromArray([[1.5, 2.5], null, [], [3.5, null, 5.5]], listType),
  });
  const frame = Frame.fromArrow(table);
  assert.equal(frame.schema[0]?.dtype, "list");
  assert.equal(frame.schema[0]?.itemDType, "float64");

  const rows = frame.toRows();
  assert.deepEqual(rows[0]?.values, [1.5, 2.5]);
  assert.equal(rows[1]?.values, null);
  assert.deepEqual(rows[2]?.values, []);
  assert.deepEqual(rows[3]?.values, [3.5, null, 5.5]);
  assert.ok(Array.isArray(rows[0]?.values), "must be a plain Array, not a Vector");
});

test("flat struct<a: float64, b: utf8> materializes to plain objects, inner nulls preserved", () => {
  const structType = new Struct([
    new Field("a", new Float64(), true),
    new Field("b", new Utf8(), true),
  ]);
  const table = new Table({
    point: vectorFromArray(
      [{ a: 1.5, b: "one" }, null, { a: null, b: "three" }, { a: 4.5, b: null }],
      structType,
    ),
  });
  const frame = Frame.fromArrow(table);
  assert.equal(frame.schema[0]?.dtype, "struct");

  const rows = frame.toRows();
  assert.deepEqual(rows[0]?.point, { a: 1.5, b: "one" });
  assert.equal(rows[1]?.point, null);
  assert.deepEqual(rows[2]?.point, { a: null, b: "three" });
  assert.deepEqual(rows[3]?.point, { a: 4.5, b: null });
  assert.ok(
    rows[0]?.point && typeof rows[0].point === "object" && !("toJSON" in (rows[0].point as object)),
    "must be a plain object, not a StructRow proxy",
  );
});

test("deeply nested types (list<struct>) are explicitly rejected as unsupported, not silently mishandled", () => {
  const innerStruct = new Struct([new Field("a", new Float64(), true)]);
  const listOfStruct = new List(new Field("item", innerStruct, true));
  const table = new Table({
    bad: vectorFromArray([[{ a: 1 }]], listOfStruct),
  });
  const frame = Frame.fromArrow(table);
  assert.throws(() => frame.schema, /does not support|UnsupportedTypeError/i);
});
