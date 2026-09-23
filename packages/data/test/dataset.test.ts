import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { collate, Dataset, fromAsync } from "../src/index.ts";

async function* slowNumbers(n: number, delayMs = 1): AsyncGenerator<number> {
  for (let i = 0; i < n; i++) {
    await new Promise((r) => setTimeout(r, delayMs));
    yield i;
  }
}

test("fromAsync over an array: map/filter/take/drop compose lazily and re-iterate", async () => {
  const ds = fromAsync([1, 2, 3, 4, 5, 6, 7, 8])
    .map((x) => x * 10)
    .filter((x) => x % 20 === 0)
    .drop(1)
    .take(2);
  assert.deepEqual(await ds.toArray(), [40, 60]);
  // Sync-iterable source -> re-iterable pipeline.
  assert.deepEqual(await ds.toArray(), [40, 60]);
});

test("chunk(n) yields fixed-size chunks with the trailing partial included (upstream group(n))", async () => {
  const chunks = await fromAsync([1, 2, 3, 4, 5, 6, 7]).chunk(3).toArray();
  assert.deepEqual(chunks, [[1, 2, 3], [4, 5, 6], [7]]);
  // Exact multiple: no empty trailing chunk.
  assert.deepEqual(await fromAsync([1, 2, 3, 4]).chunk(2).toArray(), [[1, 2], [3, 4]]);
});

test("batch(n, {collate}) produces Tensors; collate.xy produces trainer-shaped {x, y} batches", async () => {
  const samples = Array.from({ length: 5 }, (_, i) => ({ x: [i, i + 1], y: [i * 2] }));
  const batches = await fromAsync(samples).batch(2, { collate: collate.xy() }).toArray();
  assert.equal(batches.length, 3); // 2 + 2 + 1
  const first = batches[0]!;
  assert.ok(first.x instanceof Tensor && first.y instanceof Tensor);
  assert.deepEqual([...first.x.shape], [2, 2]);
  assert.deepEqual([...first.y.shape], [2, 1]);
  assert.deepEqual([...(first.x.contiguous().data as Float32Array)], [0, 1, 1, 2]);
  const last = batches[2]!;
  assert.deepEqual([...last.x.shape], [1, 2]); // trailing partial batch
  // Ragged batches are a loud error, not silent padding.
  await assert.rejects(
    fromAsync([[1, 2], [3]]).batch(2, { collate: collate.vectors() }).toArray(),
    /ragged/,
  );
});

test("shuffle: seeded shuffles are deterministic, permute (same multiset), and differ across seeds", async () => {
  const source = Array.from({ length: 50 }, (_, i) => i);
  const a1 = await fromAsync(source).shuffle({ seed: 7 }).toArray();
  const a2 = await fromAsync(source).shuffle({ seed: 7 }).toArray();
  const b = await fromAsync(source).shuffle({ seed: 8 }).toArray();
  assert.deepEqual(a1, a2); // deterministic
  assert.notDeepEqual(a1, source); // actually shuffled
  assert.notDeepEqual(a1, b); // seed matters
  assert.deepEqual([...a1].sort((x, y) => x - y), source); // a permutation, nothing lost/duplicated
});

test("shuffle with a finite bufferSize is a streaming permutation too", async () => {
  const source = Array.from({ length: 40 }, (_, i) => i);
  const out = await fromAsync(source).shuffle({ seed: 3, bufferSize: 8 }).toArray();
  assert.deepEqual([...out].sort((x, y) => x - y), source);
  assert.notDeepEqual(out, source);
});

test("epochs(n) repeats a re-iterable pipeline; reshuffle gives each epoch a different, reproducible order", async () => {
  const source = [0, 1, 2, 3, 4, 5];
  const stream = await fromAsync(source).epochs(3, { reshuffle: { seed: 11 } }).toArray();
  assert.equal(stream.length, 18);
  const epochs = [stream.slice(0, 6), stream.slice(6, 12), stream.slice(12)];
  for (const epoch of epochs) {
    assert.deepEqual([...epoch].sort((x, y) => x - y), source); // each epoch is a full permutation
  }
  assert.notDeepEqual(epochs[0], epochs[1]); // reshuffled between epochs
  // And the whole thing replays identically (seed + epochIndex derivation).
  const replay = await fromAsync(source).epochs(3, { reshuffle: { seed: 11 } }).toArray();
  assert.deepEqual(stream, replay);
});

test("a one-shot AsyncIterable source: first pass works, second pass throws the clear factory hint; epochs refuses up front", async () => {
  const oneShot = fromAsync(slowNumbers(3));
  assert.deepEqual(await oneShot.toArray(), [0, 1, 2]);
  await assert.rejects(oneShot.toArray(), /one-shot.*factory/s);
  assert.throws(() => fromAsync(slowNumbers(3)).epochs(2), /one-shot.*factory/s);
  // The factory form fixes both:
  const viaFactory = fromAsync(() => slowNumbers(3));
  assert.deepEqual(await viaFactory.toArray(), [0, 1, 2]);
  assert.deepEqual(await viaFactory.epochs(2).toArray(), [0, 1, 2, 0, 1, 2]);
});

test("mapConcurrent: bounded concurrency, order preserved, faster than sequential for I/O-shaped work", async () => {
  const started: number[] = [];
  let inFlight = 0;
  let maxInFlight = 0;
  const ds = fromAsync(() => slowNumbers(8, 0)).mapConcurrent(
    async (x) => {
      started.push(x);
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 5));
      inFlight--;
      return x * 2;
    },
    { concurrency: 3 },
  );
  assert.deepEqual(await ds.toArray(), [0, 2, 4, 6, 8, 10, 12, 14]); // ordered
  assert.ok(maxInFlight > 1, `expected real concurrency, saw max in-flight ${maxInFlight}`);
  assert.ok(maxInFlight <= 3, `concurrency bound violated: ${maxInFlight}`);
});

test("cancellation: an aborted signal rejects a mapConcurrent pipeline promptly with the signal's reason", async () => {
  const controller = new AbortController();
  const ds = fromAsync(() => slowNumbers(1000, 2)).mapConcurrent((x) => x, {
    concurrency: 2,
    signal: controller.signal,
  });
  const consuming = ds.toArray();
  setTimeout(() => controller.abort(new Error("stop the epoch")), 15);
  await assert.rejects(consuming, /stop the epoch/);
});

test("cancellation: abortable() view rejects plain pipelines too", async () => {
  const controller = new AbortController();
  const ds = fromAsync(() => slowNumbers(1000, 2)).abortable(controller.signal);
  const consuming = ds.toArray();
  setTimeout(() => controller.abort(), 15);
  await assert.rejects(consuming, /abort/i);
});

test("fold is the terminal reduce; prefetch preserves content", async () => {
  const sum = await fromAsync([1, 2, 3, 4]).fold((acc, x) => acc + x, 0);
  assert.equal(sum, 10);
  assert.deepEqual(await fromAsync(() => slowNumbers(5, 0)).prefetch(3).toArray(), [0, 1, 2, 3, 4]);
});

test("facade contract: curated names only — no count*, no raw group/reduce exports", async () => {
  const surface = await import("../src/index.ts");
  const names = Object.keys(surface);
  for (const banned of ["count", "countSync", "countAsync", "countBigSync", "countBigAsync", "group", "reduceSync", "reduceAsync"]) {
    assert.ok(!names.includes(banned), `facade must not export "${banned}" (see the module doc's curation rationale)`);
  }
  for (const required of ["fromAsync", "Dataset", "collate"]) {
    assert.ok(names.includes(required), `facade must export "${required}"`);
  }
  assert.equal(typeof Dataset.prototype.chunk, "function");
  assert.equal(typeof Dataset.prototype.fold, "function");
});
