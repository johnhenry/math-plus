/**
 * Read-path correctness against real pyarrow-written fixtures (not
 * hand-rolled buffers — test/fixtures/generate.py, run via nix-shell
 * pyarrow, see its module doc). Covers: value/null fidelity across every
 * v1-supported dtype, zstd-compressed input, column projection, filter
 * correctness, dictionary-column-decodes-as-utf8 (documented v1
 * simplification), list<T>/struct<...> columns with nulls at every level
 * (issue #30), and the clear-throw on a still-unsupported nested (MAP)
 * column.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { readParquet, UnsupportedParquetTypeError } from "../src/index.ts";
import { FIXTURES_DIR } from "./helpers.ts";

interface BasicExpected {
  f64: (number | null)[];
  i64: (number | null)[];
  i32: (number | null)[];
  u32: (number | null)[];
  f32: (number | null)[];
  utf8: (string | null)[];
  bool: (boolean | null)[];
  ts_ms: (number | null)[];
  ts_us: (number | null)[];
}

function loadExpected(name: string): BasicExpected {
  return JSON.parse(readFileSync(new URL(name, FIXTURES_DIR), "utf8")) as BasicExpected;
}

async function assertBasicFidelity(path: string) {
  const expected = loadExpected("basic_expected.json");
  const frame = await readParquet(path);
  assert.deepEqual(frame.columns, ["f64", "i64", "i32", "u32", "f32", "utf8", "bool", "ts_ms", "ts_us"]);
  assert.equal(frame.length, 100);
  const rows = frame.toRows();

  for (let i = 0; i < 100; i++) {
    const row = rows[i] as Record<string, unknown>;
    assert.equal(row.f64, expected.f64[i], `f64[${i}]`);
    assert.equal(row.i32, expected.i32[i], `i32[${i}]`);
    assert.equal(row.u32, expected.u32[i], `u32[${i}]`);
    assert.equal(row.utf8, expected.utf8[i], `utf8[${i}]`);
    assert.equal(row.bool, expected.bool[i], `bool[${i}]`);

    const wantI64 = expected.i64[i];
    assert.equal(row.i64, wantI64 === null ? null : BigInt(wantI64), `i64[${i}]`);

    const wantF32 = expected.f32[i];
    if (wantF32 === null) assert.equal(row.f32, null, `f32[${i}]`);
    else assert.ok(Math.abs((row.f32 as number) - wantF32) < 1e-4, `f32[${i}] ${row.f32} vs ${wantF32}`);

    const wantMs = expected.ts_ms[i];
    assert.equal(row.ts_ms, wantMs === null ? null : BigInt(wantMs), `ts_ms[${i}]`);
    const wantUs = expected.ts_us[i];
    assert.equal(row.ts_us, wantUs === null ? null : BigInt(wantUs), `ts_us[${i}]`);
  }
}

test("readParquet: value+null fidelity across every v1 dtype (snappy fixture)", async () => {
  await assertBasicFidelity(new URL("fixtures/basic.parquet", import.meta.url).pathname);
});

test("readParquet: zstd-compressed input reads transparently (compressors always passed)", async () => {
  await assertBasicFidelity(new URL("fixtures/basic_zstd.parquet", import.meta.url).pathname);
});

test("readParquet: schema dtypes match the v1 mapping", async () => {
  const frame = await readParquet(new URL("fixtures/basic.parquet", import.meta.url).pathname);
  const byName = Object.fromEntries(frame.schema.map((f) => [f.name, f.dtype]));
  assert.deepEqual(byName, {
    f64: "float64",
    i64: "int64",
    i32: "int32",
    u32: "uint32",
    f32: "float32",
    utf8: "utf8",
    bool: "bool",
    ts_ms: "timestamp_ms",
    ts_us: "timestamp_us",
  });
});

test("readParquet: columns option projects — only requested columns present", async () => {
  const frame = await readParquet(new URL("fixtures/basic.parquet", import.meta.url).pathname, {
    columns: ["utf8", "i64"],
  });
  assert.deepEqual(frame.columns, ["utf8", "i64"]);
  assert.equal(frame.length, 100);
});

test("readParquet: unknown column in `columns` throws a clear error", async () => {
  await assert.rejects(
    () => readParquet(new URL("fixtures/basic.parquet", import.meta.url).pathname, { columns: ["nope"] }),
    /no such column "nope"/,
  );
});

test("readParquet: filter option returns exactly the matching rows", async () => {
  const frame = await readParquet(new URL("fixtures/pushdown.parquet", import.meta.url).pathname, {
    columns: ["id", "value"],
    filter: { value: { $gt: 5990.5 } },
  });
  const rows = frame.toRows();
  assert.equal(rows.length, 9); // ids 5991..5999
  assert.deepEqual(
    rows.map((r) => Number(r.id)),
    [5991, 5992, 5993, 5994, 5995, 5996, 5997, 5998, 5999],
  );
});

test("readParquet: dictionary-encoded column decodes as plain utf8 (documented v1 simplification)", async () => {
  const expected = JSON.parse(readFileSync(new URL("fixtures/quirks_expected.json", import.meta.url), "utf8")) as {
    cat: string[];
  };
  const frame = await readParquet(new URL("fixtures/quirks.parquet", import.meta.url).pathname);
  const catField = frame.schema.find((f) => f.name === "cat");
  assert.equal(catField?.dtype, "utf8"); // NOT "dictionary" — see schema.ts's module doc
  const rows = frame.toRows();
  assert.deepEqual(
    rows.map((r) => r.cat),
    expected.cat,
  );
});

test("readParquet: a MAP column throws UnsupportedParquetTypeError naming the column (still unsupported — no frame-arrow map DType)", async () => {
  await assert.rejects(
    () => readParquet(new URL("fixtures/nested_map.parquet", import.meta.url).pathname),
    (err: unknown) => {
      assert.ok(err instanceof UnsupportedParquetTypeError);
      assert.match((err as Error).message, /"attrs"/);
      return true;
    },
  );
});

test("readParquet: selecting around a MAP column avoids the throw (projection happens before the schema check fails on it)", async () => {
  const frame = await readParquet(new URL("fixtures/nested_map.parquet", import.meta.url).pathname, {
    columns: ["id"],
  });
  assert.deepEqual(frame.columns, ["id"]);
  assert.equal(frame.length, 10);
});

test("readParquet: list<float64> column maps to frame-arrow's list<T>, with null list / empty list / null element all distinguished", async () => {
  const expected = JSON.parse(readFileSync(new URL("fixtures/nested_list_expected.json", import.meta.url), "utf8")) as {
    values: ((number | null)[] | null)[];
  };
  const frame = await readParquet(new URL("fixtures/nested_list.parquet", import.meta.url).pathname);
  const field = frame.schema.find((f) => f.name === "values");
  assert.equal(field?.dtype, "list");
  assert.equal(field?.itemDType, "float64");
  assert.equal(frame.length, 60);

  const rows = frame.toRows();
  for (let i = 0; i < 60; i++) {
    assert.deepEqual(rows[i]?.values, expected.values[i], `values[${i}]`);
  }
  // Sanity: the fixture actually exercises all three null shapes, not just one.
  assert.ok(expected.values.some((v) => v === null), "fixture must include a null list");
  assert.ok(expected.values.some((v) => Array.isArray(v) && v.length === 0), "fixture must include an empty list");
  assert.ok(expected.values.some((v) => Array.isArray(v) && v.includes(null)), "fixture must include a null element inside a non-null list");
});

test("readParquet: struct<a: float64, b: utf8> column maps to frame-arrow's struct<...>, with null struct vs. a null field distinguished", async () => {
  const expected = JSON.parse(readFileSync(new URL("fixtures/nested_struct_expected.json", import.meta.url), "utf8")) as {
    point: ({ a: number | null; b: string | null } | null)[];
  };
  const frame = await readParquet(new URL("fixtures/nested_struct.parquet", import.meta.url).pathname);
  const field = frame.schema.find((f) => f.name === "point");
  assert.equal(field?.dtype, "struct");
  assert.equal(frame.length, 60);

  const rows = frame.toRows();
  for (let i = 0; i < 60; i++) {
    assert.deepEqual(rows[i]?.point, expected.point[i], `point[${i}]`);
  }
  // Sanity: the fixture actually exercises both null shapes.
  assert.ok(expected.point.some((v) => v === null), "fixture must include a null struct");
  assert.ok(expected.point.some((v) => v !== null && (v.a === null || v.b === null)), "fixture must include a null field inside a non-null struct");
});
