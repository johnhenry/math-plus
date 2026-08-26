// frame-arrow: immutable, lazy, expression-oriented dataframes on Apache
// Arrow. Nothing executes until a materialization boundary (toRows, length,
// toArrow, ...); schema/columns never execute at all.
//
// Run: npm install && npm run build && node examples/06-dataframes.mjs
import assert from "node:assert/strict";
import { col, fn, Frame, stringifyRows } from "@johnhenry/math-plus-frame-arrow";

const frame = Frame.fromCSV(
  [
    "name,age,active,region,amount",
    "alice,34,true,west,30",
    "bob,15,true,west,40",
    "carol,42,false,west,20",
    "dave,29,true,east,10",
    "erin,51,false,east,10",
  ].join("\n"),
);

// Lazy plan building — none of this touches data yet
const adultsActive = frame.filter(col("age").gte(18).and(col("active").eq(true)));
console.log(adultsActive.toRows().map((r) => r.name)); // ['alice', 'dave']

// Whole-column broadcast aggregates
const withDeviation = frame.withColumns({
  deviation: col("amount").sub(fn.mean(col("amount")).overAll()),
});
console.log(withDeviation.toRows().map((r) => r.deviation)); // [8, 18, -2, -12, -12]

// groupBy/aggregate. NOTE: CSV inference makes integer columns int64, so
// count AND int64 aggregates come back as BigInt — plain JSON.stringify
// throws on them; use stringifyRows().
const grouped = frame
  .groupBy("region")
  .aggregate({ n: fn.count(), total: fn.sum(col("amount")), avg: fn.mean(col("amount")) })
  .sortBy("region");
console.log(stringifyRows(grouped.toRows()));
// [{"region":"east","n":"2","total":"20","avg":10},{"region":"west","n":"3","total":"90","avg":30}]

// Null semantics: an empty CSV cell is null; comparisons on null return
// null, so null rows never pass a filter — and fn.count() counts rows,
// nulls included, while sum/mean skip them.
const sparse = Frame.fromCSV("x,y\n1,a\n,b\n3,c");
assert.equal(sparse.filter(col("x").gt(0)).length, 2); // the null row is out
console.log("nulls:", sparse.nullCount()); // { x: 1, y: 0 }
