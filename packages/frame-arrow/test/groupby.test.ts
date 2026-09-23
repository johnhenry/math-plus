import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Float64, Table, Utf8, vectorFromArray } from "apache-arrow";
import { col, fn, Frame } from "../src/index.ts";

function salesTable(): Table {
  return new Table({
    region: vectorFromArray(["west", "east", "west", "east", "west"], new Utf8()),
    amount: vectorFromArray([10, 20, 30, null, 50], new Float64()),
  });
}

test("groupBy(...).aggregate({...}) computes count/sum/mean/stddev per group", () => {
  const frame = Frame.fromArrow(salesTable());
  const grouped = frame
    .groupBy("region")
    .aggregate({
      n: fn.count(),
      total: fn.sum(col("amount")),
      avg: fn.mean(col("amount")),
    })
    .sortBy("region");

  const rows = grouped.toRows();
  assert.deepEqual(rows, [
    { region: "east", n: 2n, total: 20, avg: 20 },
    { region: "west", n: 3n, total: 90, avg: 30 },
  ]);
});

test("fn.count() with no argument counts rows including nulls; fn.sum/mean skip nulls", () => {
  const frame = Frame.fromArrow(salesTable());
  const grouped = frame.groupBy("region").aggregate({ n: fn.count() }).sortBy("region");
  assert.deepEqual(
    grouped.toRows().map((r) => r.n),
    [2n, 3n],
  );
});

test("fn.stddev() is sample stddev (ddof=1), 0 for single-element groups", () => {
  const frame = Frame.fromArrow(
    new Table({
      g: vectorFromArray(["a", "a", "a", "b"], new Utf8()),
      v: vectorFromArray([2, 4, 4, 100], new Float64()),
    }),
  );
  const grouped = frame.groupBy("g").aggregate({ sd: fn.stddev(col("v")) }).sortBy("g");
  const rows = grouped.toRows();
  // group a: values [2,4,4], mean=10/3, sample variance = ((2-10/3)^2+(4-10/3)^2+(4-10/3)^2)/2
  const mean = 10 / 3;
  const variance = ((2 - mean) ** 2 + (4 - mean) ** 2 + (4 - mean) ** 2) / 2;
  assert.ok(Math.abs((rows[0]?.sd as number) - Math.sqrt(variance)) < 1e-9);
  assert.equal(rows[1]?.sd, 0); // single-element group "b"
});

test("groupBy().aggregate() schema is computable WITHOUT collecting (keys + agg output types, no data touch)", () => {
  const frame = Frame.fromArrow(salesTable());
  const grouped = frame.groupBy("region").aggregate({ total: fn.sum(col("amount")) });
  const schema = grouped.schema; // must not throw / must not require .collect()
  assert.deepEqual(
    schema.map((f) => f.name),
    ["region", "total"],
  );
  assert.equal(schema[1]?.dtype, "float64");
});

test("selecting only the group key after aggregate() prunes away the never-materialized agg columns", () => {
  const frame = Frame.fromArrow(salesTable());
  const grouped = frame
    .groupBy("region")
    .aggregate({ total: fn.sum(col("amount")), avg: fn.mean(col("amount")) })
    .select("region")
    .sortBy("region");
  assert.deepEqual(grouped.toRows(), [{ region: "east" }, { region: "west" }]);
});
