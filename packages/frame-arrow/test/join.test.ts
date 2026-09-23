import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Float64, Int32, Int64, Table, Utf8, vectorFromArray } from "apache-arrow";
import { Frame } from "../src/index.ts";

function usersFrame(): Frame {
  return Frame.fromArrow(
    new Table({
      id: vectorFromArray([1, 2, 3], new Int32()),
      name: vectorFromArray(["alice", "bob", "carol"], new Utf8()),
    }),
  );
}

function ordersFrame(): Frame {
  return Frame.fromArrow(
    new Table({
      id: vectorFromArray([1, 1, 4], new Int32()),
      item: vectorFromArray(["widget", "gadget", "orphan"], new Utf8()),
    }),
  );
}

test("inner join keeps only matching keys, and duplicates left rows for multiple right matches", () => {
  const joined = usersFrame().join(ordersFrame(), { on: "id", how: "inner" }).sortBy("item");
  const rows = joined.toRows();
  assert.deepEqual(
    rows.map((r) => [r.id, r.name, r.item]),
    [
      [1, "alice", "gadget"],
      [1, "alice", "widget"],
    ],
  );
});

test("left join keeps every left row, nulling right-side columns when unmatched", () => {
  const joined = usersFrame().join(ordersFrame(), { on: "id", how: "left" }).sortBy("name", "item");
  const rows = joined.toRows();
  assert.deepEqual(
    rows.map((r) => [r.id, r.name, r.item]),
    [
      [1, "alice", "gadget"],
      [1, "alice", "widget"],
      [2, "bob", null],
      [3, "carol", null],
    ],
  );
});

test("right join keeps every right row, nulling left-side columns when unmatched", () => {
  const joined = usersFrame().join(ordersFrame(), { on: "id", how: "right" }).sortBy("item");
  const rows = joined.toRows();
  assert.deepEqual(
    rows.map((r) => [r.id, r.name, r.item]),
    [
      [1, "alice", "gadget"],
      [4, null, "orphan"],
      [1, "alice", "widget"],
    ],
  );
});

test("outer join keeps everything from both sides", () => {
  const joined = usersFrame().join(ordersFrame(), { on: "id", how: "outer" });
  assert.equal(joined.length, 5); // alice x2, bob, carol, orphan(id=4)
});

test("null join keys never match, including a null on both sides (issue #102)", () => {
  const left = Frame.fromArrow(
    new Table({
      id: vectorFromArray([1, null, 3], new Int32()),
      name: vectorFromArray(["alice", "nullguy", "carol"], new Utf8()),
    }),
  );
  const right = Frame.fromArrow(
    new Table({
      id: vectorFromArray([1, null, 4], new Int32()),
      item: vectorFromArray(["widget", "nullitem", "orphan"], new Utf8()),
    }),
  );
  const joined = left.join(right, { on: "id", how: "inner" });
  // Only id=1 matches. The two null-keyed rows (one per side) must NOT match
  // each other -- NULL never equals NULL in a join (standard SQL/Arrow semantics).
  assert.deepEqual(joined.toRows(), [{ id: 1, name: "alice", item: "widget" }]);
});

test("a null join key never matches another null even under outer join (both null rows stay unmatched)", () => {
  const left = Frame.fromArrow(
    new Table({ id: vectorFromArray([null], new Int32()), name: vectorFromArray(["left-null"], new Utf8()) }),
  );
  const right = Frame.fromArrow(
    new Table({ id: vectorFromArray([null], new Int32()), item: vectorFromArray(["right-null"], new Utf8()) }),
  );
  const joined = left.join(right, { on: "id", how: "outer" }).sortBy("name", "item");
  // If null==null spuriously matched, this would collapse to a single row
  // with both name and item populated. Each side's null row must instead
  // surface as its own unmatched row, with the other side's columns null.
  assert.deepEqual(joined.toRows(), [
    { id: null, name: "left-null", item: null },
    { id: null, name: null, item: "right-null" },
  ]);
});

test("int64 and float64 join keys holding the same logical value match (issue #102)", () => {
  const left = Frame.fromArrow(
    new Table({
      id: vectorFromArray([1n, 2n], new Int64()),
      name: vectorFromArray(["alice", "bob"], new Utf8()),
    }),
  );
  const right = Frame.fromArrow(
    new Table({
      id: vectorFromArray([1.0, 2.0], new Float64()),
      item: vectorFromArray(["widget", "gadget"], new Utf8()),
    }),
  );
  const joined = left.join(right, { on: "id", how: "inner" }).sortBy("name");
  const rows = joined.toRows();
  assert.deepEqual(
    rows.map((r) => [r.name, r.item]),
    [
      ["alice", "widget"],
      ["bob", "gadget"],
    ],
  );
});

test("overlapping non-key column names get the configured suffix", () => {
  const left = Frame.fromArrow(
    new Table({ id: vectorFromArray([1], new Int32()), label: vectorFromArray(["L"], new Utf8()) }),
  );
  const right = Frame.fromArrow(
    new Table({ id: vectorFromArray([1], new Int32()), label: vectorFromArray(["R"], new Utf8()) }),
  );
  const joined = left.join(right, { on: "id" });
  assert.deepEqual(joined.columns, ["id", "label", "label_right"]);
  assert.deepEqual(joined.toRows(), [{ id: 1, label: "L", label_right: "R" }]);
});
