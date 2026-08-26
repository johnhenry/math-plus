// data: async dataset pipelines — a curated Dataset facade over
// @johnhenry/iteration, producing Tensor batches shaped for
// tensor-autograd's trainer.
//
// Run: npm install && npm run build && node examples/08-data-pipelines.mjs
import assert from "node:assert/strict";
import { collate, fromAsync } from "@johnhenry/math-plus-data";
import { nn, optim, trainer } from "@johnhenry/math-plus-tensor-autograd";
import { random } from "@johnhenry/math-plus-tensor-core";

// Chainable, lazy, re-iterable (arrays are re-iterable; a bare
// AsyncIterable is one-shot — pass a factory for multi-pass pipelines).
const ds = fromAsync([1, 2, 3, 4, 5, 6, 7, 8])
  .map((x) => x * 10)
  .filter((x) => x % 20 === 0)
  .drop(1)
  .take(2);
console.log(await ds.toArray()); // [40, 60]
console.log(await ds.toArray()); // [40, 60] — again; the pipeline re-iterates

// Batching into the trainer's exact Batch shape. TRAP: collate defaults to
// f32, but nn.Linear's parameters are f64 and tensor-core has no implicit
// promotion — pass { dtype: "f64" } explicitly.
const samples = Array.from({ length: 64 }, (_, i) => {
  const x = i / 32 - 1;
  return { x: [x], y: [3 * x + 2] };
});

const pipeline = fromAsync(samples)
  .epochs(60, { reshuffle: { seed: 42 } }) // per-epoch reshuffle, reproducible
  .batch(16, { collate: collate.xy({ dtype: "f64" }) });

const model = new nn.Linear(1, 1, { rng: random.seed(7) });
const opt = new optim.SGD(model.parameters(), { lr: 0.05 });
const t = trainer.configure({ model, optimizer: opt, lossFn: nn.mseLoss });
// NOTE: for a dataLoader, epochs live in the PIPELINE (config.epochs is
// ignored — an arbitrary AsyncIterable isn't guaranteed re-iterable).
const { lossHistory } = await t.fit(pipeline);

console.log("batches trained:", lossHistory.length); // 60 epochs x 4 batches = 240
const finalLoss = lossHistory[lossHistory.length - 1];
console.log("final loss:", finalLoss);
assert.ok(finalLoss < 0.05);
