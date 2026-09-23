/**
 * Proves column pruning is REAL, not cosmetic: a table carries a "poisoned"
 * column of a dtype this package doesn't support (decimal128 — explicitly
 * deferred past v1). If that column is never selected/required by a
 * predicate, it must never be read, type-checked, or throw — exactly the
 * proof plan.ts's module doc comment calls out ("an unreadable/malformed
 * column that's pruned away never causes an error"). If it's NOT pruned
 * (accessed directly, or via `.schema` on an unfiltered Frame), it's
 * expected to throw UnsupportedTypeError — that's the control case showing
 * the column really is unsupported, not that our test accidentally works
 * for some unrelated reason.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Decimal, Float64, makeData, Table, Utf8, Vector, vectorFromArray } from "apache-arrow";
import { col, fn, Frame, UnsupportedTypeError } from "../src/index.ts";

function poisonedTable(): Table {
  const decimalType = new Decimal(2, 38, 128);
  const poisoned = new Vector([
    makeData({ type: decimalType, length: 4, nullCount: 0, data: new Uint32Array(16) }),
  ]);
  return new Table({
    a: vectorFromArray([1.0, 2.0, 3.0, 4.0], new Float64()),
    b: vectorFromArray(["w", "x", "y", "z"], new Utf8()),
    poisoned,
  });
}

test("control case: the poisoned column really is unsupported when actually touched", () => {
  const frame = Frame.fromArrow(poisonedTable());
  assert.throws(() => frame.schema, UnsupportedTypeError);
  assert.throws(() => frame.toRows(), UnsupportedTypeError);
});

test("select() prunes the poisoned column away — toRows() never touches it, never throws", () => {
  const frame = Frame.fromArrow(poisonedTable()).select("a", "b");
  assert.deepEqual(frame.columns, ["a", "b"]);
  const rows = frame.toRows(); // must NOT throw
  assert.deepEqual(rows[0], { a: 1, b: "w" });
  assert.equal(rows.length, 4);
});

test("select().filter() prunes the poisoned column through a predicate that only needs selected columns", () => {
  const frame = Frame.fromArrow(poisonedTable())
    .select("a", "b")
    .filter(col("a").gt(2));
  const rows = frame.toRows(); // must NOT throw
  assert.deepEqual(
    rows.map((r) => r.b),
    ["y", "z"],
  );
});

test("drop() of the poisoned column also prunes it", () => {
  const frame = Frame.fromArrow(poisonedTable()).drop("poisoned");
  assert.deepEqual(frame.toRows().length > 0 ? Object.keys(frame.toRows()[0] as object) : [], ["a", "b"]);
});

test("groupBy().aggregate().select(keyOnly) prunes both the poisoned column AND the unselected aggregate", () => {
  const table = poisonedTable();
  const grouped = Frame.fromArrow(table)
    .groupBy("b")
    .aggregate({ total: fn.sum(col("a")) })
    .select("b");
  const rows = grouped.toRows(); // must NOT throw despite "poisoned" being present in the source table,
  // and "total" is never computed since it wasn't selected either.
  assert.equal(rows.length, 4);
  assert.deepEqual(Object.keys(rows[0] as object), ["b"]);
});
