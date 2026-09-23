import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Bool, Float64, Int32, Table, Utf8, vectorFromArray } from "apache-arrow";
import { col, fn, Frame, lit } from "../src/index.ts";

function peopleTable(): Table {
  return new Table({
    name: vectorFromArray(["alice", "bob", "carol", "dave"], new Utf8()),
    age: vectorFromArray([30, 17, 45, 22], new Int32()),
    active: vectorFromArray([true, true, false, true], new Bool()),
    score: vectorFromArray([1.0, 2.0, 3.0, 4.0], new Float64()),
  });
}

test("filter() with comparison + logical combinators reads naturally", () => {
  const frame = Frame.fromArrow(peopleTable());
  const adultsActive = frame.filter(col("age").gte(18).and(col("active").eq(true)));
  const names = adultsActive.toRows().map((r) => r.name);
  assert.deepEqual(names, ["alice", "dave"]);
});

test("filter() with .or() and .not()", () => {
  const frame = Frame.fromArrow(peopleTable());
  const minorsOrInactive = frame.filter(col("age").lt(18).or(col("active").eq(false)));
  assert.deepEqual(
    minorsOrInactive.toRows().map((r) => r.name),
    ["bob", "carol"],
  );
  const notActive = frame.filter(col("active").eq(true).not());
  assert.deepEqual(
    notActive.toRows().map((r) => r.name),
    ["carol"],
  );
});

test("withColumns() computes a new column from arithmetic combinators", () => {
  const frame = Frame.fromArrow(peopleTable());
  const withDoubled = frame.withColumns({ doubled: col("score").mul(2) });
  assert.deepEqual(
    withDoubled.toRows().map((r) => r.doubled),
    [2, 4, 6, 8],
  );
});

test("withColumns() can overwrite an existing column", () => {
  const frame = Frame.fromArrow(peopleTable());
  const bumped = frame.withColumns({ age: col("age").add(1) });
  assert.deepEqual(bumped.columns, ["name", "age", "active", "score"]); // no new column appended
  assert.deepEqual(
    bumped.toRows().map((r) => r.age),
    [31, 18, 46, 23],
  );
});

test("fn.mean(...).overAll() broadcasts a whole-column aggregate back as a same-length column", () => {
  const frame = Frame.fromArrow(peopleTable());
  const withDeviation = frame.withColumns({
    deviation: col("score").sub(fn.mean(col("score")).overAll()),
  });
  const deviations = withDeviation.toRows().map((r) => r.deviation);
  // mean of [1,2,3,4] = 2.5
  assert.deepEqual(deviations, [-1.5, -0.5, 0.5, 1.5]);
});

test("a bare aggregate expression used outside aggregate()/overAll() throws a clear error", () => {
  const frame = Frame.fromArrow(peopleTable());
  assert.throws(
    () => frame.withColumns({ bad: fn.sum(col("score")) }).toRows(),
    /outside groupBy\(\)\.aggregate\(\) or \.overAll\(\)/,
  );
});

test("arithmetic on a non-numeric column throws instead of silently writing NaN as a valid value (issue #102)", () => {
  const frame = Frame.fromArrow(
    new Table({ name: vectorFromArray(["alice", "bob"], new Utf8()) }),
  );
  assert.throws(() => frame.withColumns({ bad: col("name").sub(1) }).toRows(), /requires numeric operands/);
  assert.throws(() => frame.withColumns({ bad: col("name").mul(2) }).toRows(), /requires numeric operands/);
  assert.throws(() => frame.withColumns({ bad: col("name").add(1) }).toRows(), /requires numeric operands/);
});

test("lit() wraps a literal for use on the left-hand side, and comparisons handle null propagation", () => {
  const frame = Frame.fromArrow(
    new Table({ x: vectorFromArray([1, null, 3], new Float64()) }),
  );
  const gt1 = frame.filter(col("x").gt(1));
  assert.equal(gt1.length, 1); // null rows never pass a comparison
  assert.deepEqual(
    frame.withColumns({ y: lit(10) }).toRows().map((r) => r.y),
    [10, 10, 10],
  );
});
