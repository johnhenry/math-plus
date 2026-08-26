# @johnhenry/math-plus-frame-arrow

Immutable, expression-oriented `Frame`/`Series` dataframes on Apache Arrow.
Build a lazy plan with a Polars-style expression algebra; nothing executes
until you cross a materialization boundary.

## Install

```bash
npm install @johnhenry/math-plus-frame-arrow
```

Depends on `apache-arrow` (exact-pinned). `@johnhenry/math-plus-tensor-core`
is an *optional* peer — only needed for `toTensor()`, loaded via dynamic
import.

## Quick start

```js
import { col, fn, Frame, lit } from "@johnhenry/math-plus-frame-arrow";

const frame = Frame.fromCSV(csvText); // or Frame.fromArrow(table) / Frame.fromIPC(bytes)

// Lazy: none of this touches data yet
const adults = frame.filter(col("age").gte(18).and(col("active").eq(true)));

// Whole-column broadcast aggregates
const withDeviation = frame.withColumns({
  deviation: col("score").sub(fn.mean(col("score")).overAll()),
});

// groupBy/aggregate
const grouped = frame
  .groupBy("region")
  .aggregate({ n: fn.count(), total: fn.sum(col("amount")), avg: fn.mean(col("amount")) })
  .sortBy("region");

grouped.toRows(); // materializes here
// [{ region: "east", n: 2n, total: 20, avg: 20 }, ...]  <- count is a bigint
```

## API surface

- **Construction:** `Frame.fromArrow` / `fromIPC` / `fromCSV` (RFC-4180 +
  dtype inference) / `fromLazySource` / `concat`. The constructor is private.
- **Lazy plan ops:** `select`, `drop`, `rename`, `withColumns`, `filter`,
  `sortBy` (+ `desc()`), `limit`, `slice`, `groupBy(...).aggregate(...)`,
  `join(other, { on, how, suffix })`, `dropNull`, `fillNull`.
- **Materialization:** `collect()` (memoized), `collectAsync()` (required for
  lazy sources, e.g. frame-parquet's lazy scan). `length`, `toRows`,
  `toArrow`, `toCSV`, `toIPC`, `nullCount`, `getSeries`, `toTensor` all
  collect implicitly; `schema`/`columns` never do (metadata-only).
- **Expressions:** `col`, `lit`, comparison/logical/arithmetic combinators,
  aggregates `fn.count/sum/mean/stddev` (+ `.overAll()`), `fn.month`, and 41
  scalar math functions (`sin`…`relu`; note `ln`, not `log`, matching
  `@johnhenry/math`'s `Symbolic` names 1:1).
- **`Series`:** `toArray`, `get`, `isNull`, `fillNull`, `unique`,
  `valueCounts`, `cast` (numeric↔numeric or utf8↔dictionary only),
  `toDates` (documented lossy), `toTensor`.
- **Safety:** unsupported Arrow types throw `UnsupportedTypeError` at
  inspection time; `stringifyRows`/`bigintSafeReplacer` for JSON with int64s.

## Traps

- **int64 is `bigint` everywhere.** `toRows()` yields `2n`; plain
  `JSON.stringify` on those rows **throws** — use `stringifyRows()`.
- **Nulls:** comparisons on null return null, so null rows never pass a
  filter. `sortBy` puts nulls last in *both* directions. `fn.count()` counts
  nulls; `sum`/`mean` skip them; `stddev` is sample (ddof=1). Null join keys
  never match — not even null-to-null, even under outer join (issue #102).
- **Column pruning is real** — `select()` before `toRows()` can avoid
  decoding an unsupported column entirely. But **`join` does not prune**:
  both sides execute with all columns regardless of downstream `select`.
- **Timestamps:** exact bigint epoch via `toRows()`/`Series.toArray()`, but
  *inside expression evaluation* they pass through epoch-ms numbers — a
  `timestamp[us]` predicate silently loses sub-ms precision. `fn.month` uses
  the UTC calendar, ignoring tz metadata.
- **Bare aggregates outside `groupBy`/`.overAll()`** throw — but only at
  collect time, not when you build the plan.
- **`toTensor()`** rejects nulls ("call `.fillNull(...)` first") and
  non-numeric dtypes; `Frame.toTensor()` promotes every column to one dtype
  (default `float64`, `[rows, cols]` row-major) without range re-validation.
- Arithmetic on a non-numeric column throws instead of writing NaN
  (issue #102). `Frame.concat` papers over apache-arrow's zero-row
  `Table.concat` breakage (issue #31).

## Tests

`npm test` — includes byte-level pruning proofs and null-semantics
regression tests for issue #102.

## Provenance

Part of the [math-plus](https://github.com/johnhenry/math-plus) monorepo —
the Arrow-backed tabular half of the family. Parquet I/O lives separately in
`@johnhenry/math-plus-frame-parquet`. Family docs:
<https://opensource.johnhenry.me/math/>.
