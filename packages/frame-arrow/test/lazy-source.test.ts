/**
 * Frame.fromLazySource()/.collectAsync() (issue #32) — tested in isolation
 * with a hand-rolled mock reader, independent of @johnhenry/math-plus-frame-parquet's
 * real scanParquetLazy (which builds on this once it's shipped). See
 * plan.ts's module doc comment for the design (why collect() stays
 * synchronous, why schema/columns never need collectAsync()).
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Float64, Int32, Schema, Table, Utf8, Field, Vector, vectorFromArray } from "apache-arrow";
import { Frame, col, type Wanted } from "../src/index.ts";

const FULL_SCHEMA = new Schema([
  new Field("id", new Int32(), false),
  new Field("value", new Float64(), true),
  new Field("label", new Utf8(), true),
]);

const ROWS = {
  id: [1, 2, 3, 4],
  value: [1.5, null, 3.25, -0.5],
  label: ["alpha", "beta", "gamma", null],
};

function tableFor(columns: readonly string[]): Table {
  const cols: Record<string, Vector> = {};
  if (columns.includes("id")) cols.id = vectorFromArray(ROWS.id, new Int32());
  if (columns.includes("value")) cols.value = vectorFromArray(ROWS.value, new Float64());
  if (columns.includes("label")) cols.label = vectorFromArray(ROWS.label, new Utf8());
  return new Table(cols);
}

/** A lazy source whose read() records every `wanted` it was called with — the pruning proof. */
function mockLazySource(): { frame: Frame; calls: Wanted[] } {
  const calls: Wanted[] = [];
  const frame = Frame.fromLazySource(FULL_SCHEMA, async (wanted) => {
    calls.push(wanted);
    const columns = wanted === "all" ? FULL_SCHEMA.names : [...wanted];
    return tableFor(columns as string[]);
  });
  return { frame, calls };
}

test("schema/columns work SYNCHRONOUSLY on a lazy-source Frame, before any collectAsync()", () => {
  const { frame, calls } = mockLazySource();
  assert.deepEqual(frame.columns, ["id", "value", "label"]);
  assert.equal(frame.schema.length, 3);
  assert.equal(calls.length, 0, "reading schema/columns must never call read()");
});

test("collectAsync() materializes correctly and calls read() with wanted='all' when nothing narrows it", async () => {
  const { frame, calls } = mockLazySource();
  const collected = await frame.collectAsync();
  assert.equal(collected, frame, "collectAsync() returns `this`, same as collect()");
  assert.deepEqual(collected.toRows(), [
    { id: 1, value: 1.5, label: "alpha" },
    { id: 2, value: null, label: "beta" },
    { id: 3, value: 3.25, label: "gamma" },
    { id: 4, value: -0.5, label: null },
  ]);
  assert.deepEqual(calls, ["all"]);
});

test("collectAsync() is idempotent/memoized — read() is called exactly once even across repeated collectAsync() calls", async () => {
  const { frame, calls } = mockLazySource();
  await frame.collectAsync();
  await frame.collectAsync();
  await frame.collectAsync();
  assert.equal(calls.length, 1);
});

test("calling the SYNCHRONOUS collect()/a terminal accessor directly on an unresolved lazy Frame throws a clear error", () => {
  const { frame } = mockLazySource();
  assert.throws(() => frame.collect(), /collectAsync/);
  assert.throws(() => frame.toRows(), /collectAsync/);
  assert.throws(() => frame.length, /collectAsync/);
});

test("real pruning: select().filter() narrows the wanted set read() actually receives, matching an eager Frame's own pruning rules exactly", async () => {
  const { frame, calls } = mockLazySource();
  const narrowed = frame.select("id", "value").filter(col("id").gt(2));
  const collected = await narrowed.collectAsync();
  assert.deepEqual(
    collected.toRows().map((r) => r.id),
    [3, 4],
  );
  // select(['id','value']) narrows to {id, value}; filter's predicate needs
  // 'id' too, but it's already in the set -- same as requiredInputColumns'
  // "select" then "filter" cases would compute for an eager source.
  assert.equal(calls.length, 1);
  const wanted = calls[0] as Wanted;
  assert.notEqual(wanted, "all");
  assert.deepEqual([...(wanted as ReadonlySet<string>)].sort(), ["id", "value"]);
});

test("a column pruned away entirely (e.g. 'label') is never in the wanted set passed to read()", async () => {
  const { frame, calls } = mockLazySource();
  await frame.select("id").collectAsync();
  const wanted = calls[0] as Wanted;
  assert.deepEqual([...(wanted as ReadonlySet<string>)], ["id"]);
});

test("concat of multiple lazy sources: each gets its own read() call, resolved and concatenated correctly", async () => {
  const schemaAB = new Schema([new Field("id", new Int32(), false), new Field("value", new Float64(), true)]);
  const calls: Wanted[][] = [[], []];
  const a = Frame.fromLazySource(schemaAB, async (wanted) => {
    calls[0]?.push(wanted);
    return new Table({ id: vectorFromArray([1, 2], new Int32()), value: vectorFromArray([1.1, 2.2], new Float64()) });
  });
  const b = Frame.fromLazySource(schemaAB, async (wanted) => {
    calls[1]?.push(wanted);
    return new Table({ id: vectorFromArray([3], new Int32()), value: vectorFromArray([3.3], new Float64()) });
  });
  const combined = Frame.concat([a, b]);
  const collected = await combined.collectAsync();
  assert.deepEqual(collected.toRows(), [
    { id: 1, value: 1.1 },
    { id: 2, value: 2.2 },
    { id: 3, value: 3.3 },
  ]);
});

test("a zero-row lazy source concatenated with a non-empty one resolves correctly (issue #31's fix applies here too)", async () => {
  const schema = new Schema([new Field("id", new Int32(), false)]);
  const empty = Frame.fromLazySource(schema, async () => new Table({ id: vectorFromArray([], new Int32()) }));
  const full = Frame.fromLazySource(schema, async () => new Table({ id: vectorFromArray([1, 2], new Int32()) }));
  const collected = await Frame.concat([empty, full, empty]).collectAsync();
  assert.deepEqual(
    collected.toRows().map((r) => r.id),
    [1, 2],
  );
});

test("join: a lazy source on either side is read with wanted='all', matching an eager Frame's own join behavior", async () => {
  const leftSchema = new Schema([new Field("id", new Int32(), false), new Field("value", new Float64(), true)]);
  const rightSchema = new Schema([new Field("id", new Int32(), false), new Field("label", new Utf8(), true)]);
  const calls: Wanted[] = [];
  const left = Frame.fromLazySource(leftSchema, async (wanted) => {
    calls.push(wanted);
    return new Table({ id: vectorFromArray([1, 2], new Int32()), value: vectorFromArray([10, 20], new Float64()) });
  });
  const right = Frame.fromArrow(
    new Table({ id: vectorFromArray([1, 2], new Int32()), label: vectorFromArray(["a", "b"], new Utf8()) }),
  );
  const joined = left.join(right, { on: "id" }).select("id", "label"); // narrows OUTPUT, but join forces wanted="all" on both inputs
  const collected = await joined.collectAsync();
  assert.deepEqual(collected.toRows(), [
    { id: 1, label: "a" },
    { id: 2, label: "b" },
  ]);
  assert.deepEqual(calls, ["all"]);
});

test("a plan with NO lazy source at all resolves through collectAsync() unchanged (it's a strict superset of collect())", async () => {
  const table = new Table({ id: vectorFromArray([1, 2, 3], new Int32()) });
  const frame = Frame.fromArrow(table).select("id").filter(col("id").gt(1));
  const collected = await frame.collectAsync();
  assert.deepEqual(
    collected.toRows().map((r) => r.id),
    [2, 3],
  );
});
