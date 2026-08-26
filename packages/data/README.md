# @johnhenry/math-plus-data

Async dataset pipelines for math-plus (issue #22): a curated `Dataset` facade
over [`@johnhenry/iteration`](https://github.com/johnhenry/math) —
chunk/batch/shuffle/epochs/mapConcurrent/prefetch/fold with `AbortSignal`
cancellation — producing `Tensor` batches shaped exactly for
`@johnhenry/math-plus-tensor-autograd`'s `trainer.fit`.

## Install

```bash
npm install @johnhenry/math-plus-data
```

## Quick start

```js
import { collate, fromAsync } from "@johnhenry/math-plus-data";

const ds = fromAsync([1, 2, 3, 4, 5, 6, 7, 8])
  .map((x) => x * 10)
  .filter((x) => x % 20 === 0)
  .drop(1)
  .take(2);
await ds.toArray(); // [40, 60] — and re-iterable, so again: [40, 60]

// Batching straight into the trainer's Batch shape
const samples = [{ x: [0, 1], y: [0] }, { x: [1, 2], y: [2] } /* ... */];
const pipeline = fromAsync(samples)
  .epochs(60, { reshuffle: { seed: 42 } })
  .batch(16, { collate: collate.xy({ dtype: "f64" }) });

const { lossHistory } = await trainer.fit(pipeline); // tensor-autograd
```

## API surface

- `fromAsync(source)` — source is an `Iterable`, `AsyncIterable`, or a
  **factory** returning one.
- `Dataset` methods: `map`, `filter`, `mapConcurrent(fn, { concurrency, ordered?, signal? })`,
  `prefetch(n)`, `chunk(n)`, `batch(n, { collate? })`,
  `shuffle({ seed?, bufferSize? })`, `take`, `drop`, `abortable(signal)`,
  `epochs(n, { reshuffle? })`, `fold`, `toArray`.
- `collate.vectors({ dtype? })` → `[batch, dim]` Tensor;
  `collate.scalars({ dtype? })` → `[batch]`;
  `collate.xy({ dtype? })` → `{ x, y }` — tensor-autograd's `Batch` exactly.

## Traps

- **One-shot vs re-iterable.** Arrays/Sets are re-iterable; a bare
  `AsyncIterable` is assumed one-shot — a second pass throws with a hint, and
  `.epochs()` refuses one-shot sources *up front*. Pass a factory
  (`fromAsync(() => stream())`) for multi-pass pipelines.
- **`collate` defaults to `f32`, but `nn.Linear` parameters are `f64`** and
  tensor-core has no implicit dtype promotion by design — pass
  `collate.xy({ dtype: "f64" })` when feeding the trainer.
- **Shuffle:** omitting `seed` is non-reproducible. `bufferSize` defaults to
  `Infinity` (full materialize + Fisher-Yates); a finite buffer is the
  tf.data streaming shuffle with mixing quality bounded by the buffer.
- **`shuffle()` on top of `epochs()` shuffles the concatenated stream.** For
  per-epoch reshuffling use `epochs(n, { reshuffle: { seed } })` (derives
  `seed + epochIndex`).
- **Ragged batches are loud:** `collate.vectors` throws `RangeError` on
  differing sample lengths — no silent padding.
- **Curated facade, enforced by a test:** no `count*`, no raw
  `group`/`reduce*` re-exports (`group` collides with dataframe `groupBy`
  vocabulary; `fold` is *the* terminal reduce). Power users can import
  `@johnhenry/iteration` directly — the facade is the supported surface, not
  a wall.

## Concurrency and cancellation

`mapConcurrent` delegates to iteration's `mapConcurrentAsync`: order
preserved by default, at most `concurrency` in flight, source pulled only
with spare capacity, source closed via `return()` on early exit or error.
`prefetch(n)` is a bounded read-ahead buffer. Cancellation is plain
`AbortSignal` end to end, rejecting with the signal's reason.

## Provenance

Part of the [math-plus](https://github.com/johnhenry/math-plus) monorepo;
family docs at <https://opensource.johnhenry.me/math/>.
