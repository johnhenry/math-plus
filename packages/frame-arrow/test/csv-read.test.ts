import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Frame } from "../src/index.ts";

test("Frame.fromCSV: infers int64 for an all-integer-looking column, hand-computed", () => {
  const frame = Frame.fromCSV("id,name\n1,alpha\n2,beta\n-3,gamma\n");
  assert.deepEqual(frame.columns, ["id", "name"]);
  assert.equal(frame.schema[0]?.dtype, "int64");
  assert.equal(frame.schema[1]?.dtype, "utf8");
  const rows = frame.toRows();
  assert.equal(rows.length, 3);
  assert.equal(rows[0]?.id, 1n);
  assert.equal(rows[2]?.id, -3n);
  assert.equal(rows[1]?.name, "beta");
});

test("Frame.fromCSV: a column mixing integer- and decimal-looking cells widens to float64", () => {
  const frame = Frame.fromCSV("value\n1\n2.5\n-3\n");
  assert.equal(frame.schema[0]?.dtype, "float64");
  const rows = frame.toRows();
  assert.equal(rows[0]?.value, 1);
  assert.equal(rows[1]?.value, 2.5);
});

test("Frame.fromCSV: infers bool only when every non-empty cell is exactly true/false (case-insensitive)", () => {
  const frame = Frame.fromCSV("active\nTrue\nfalse\nTRUE\n");
  assert.equal(frame.schema[0]?.dtype, "bool");
  const rows = frame.toRows();
  assert.deepEqual(rows.map((r) => r.active), [true, false, true]);
});

test("Frame.fromCSV: a column with a single non-bool/non-numeric cell falls back to utf8, not silently coerced", () => {
  const frame = Frame.fromCSV("mixed\n1\n2\nnotanumber\n");
  assert.equal(frame.schema[0]?.dtype, "utf8");
  assert.deepEqual(
    frame.toRows().map((r) => r.mixed),
    ["1", "2", "notanumber"],
  );
});

test("Frame.fromCSV: large integers beyond Number.MAX_SAFE_INTEGER stay exact via BigInt, unlike a Number()-based parser", () => {
  const big = "9007199254740993"; // MAX_SAFE_INTEGER + 2, unrepresentable exactly as a JS number
  const frame = Frame.fromCSV(`id\n${big}\n`);
  assert.equal(frame.schema[0]?.dtype, "int64");
  assert.equal(frame.toRows()[0]?.id, BigInt(big));
});

test("Frame.fromCSV: empty cells become null, not the empty string or NaN", () => {
  const frame = Frame.fromCSV("id,name\n1,\n,beta\n");
  const rows = frame.toRows();
  assert.equal(rows[0]?.name, null);
  assert.equal(rows[1]?.id, null);
});

test("Frame.fromCSV: an all-empty column infers utf8 (no numeric/bool evidence either way)", () => {
  const frame = Frame.fromCSV("id,blank\n1,\n2,\n");
  assert.equal(frame.schema[1]?.dtype, "utf8");
});

test("Frame.fromCSV: quoted fields with embedded commas and doubled-quote escapes parse per RFC 4180", () => {
  const frame = Frame.fromCSV('name,note\n"Smith, John","says ""hi"""\n');
  const rows = frame.toRows();
  assert.equal(rows[0]?.name, "Smith, John");
  assert.equal(rows[0]?.note, 'says "hi"');
});

test("Frame.fromCSV: CRLF and LF line endings both parse to the same rows", () => {
  const crlf = Frame.fromCSV("a,b\r\n1,2\r\n3,4\r\n");
  const lf = Frame.fromCSV("a,b\n1,2\n3,4\n");
  assert.deepEqual(crlf.toRows(), lf.toRows());
});

test("Frame.fromCSV: rejects a ragged row with a clear error rather than silently padding", () => {
  assert.throws(() => Frame.fromCSV("a,b\n1,2\n3\n"), /Row 2 has 1 fields, but the header has 2/);
});

test("Frame.fromCSV: rejects an unterminated quoted field", () => {
  assert.throws(() => Frame.fromCSV('a,b\n"unterminated,2\n'), /Unterminated quoted field/);
});

test("Frame.fromCSV -> .toCSV() round-trips a simple table", () => {
  const original = "id,name,active\n1,alpha,true\n2,beta,false\n";
  const frame = Frame.fromCSV(original);
  assert.equal(frame.toCSV(), original);
});
