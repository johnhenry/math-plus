/**
 * End-to-end proof of the issue's design intent: a @johnhenry/math-plus-data pipeline
 * (shuffle -> epochs -> batch+collate) plugs straight into
 * tensor-autograd's `trainer.fit(dataLoader)` because `collate.xy()`
 * produces exactly its `Batch` shape — the `data` namespace and the
 * trainer facade (#43) were built for each other across two issues.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { nn, optim, trainer, Variable } from "@johnhenry/math-plus-tensor-autograd";
import { collate, fromAsync } from "../src/index.ts";

test("trainer.fit consumes a @johnhenry/math-plus-data pipeline: linear regression converges from shuffled, batched epochs", async () => {
  // y = 3x - 1, with x normalized to keep SGD well-behaved.
  const samples = Array.from({ length: 64 }, (_, i) => {
    const x = (i / 32) - 1;
    return { x: [x], y: [3 * x - 1] };
  });

  const pipeline = fromAsync(samples)
    .epochs(60, { reshuffle: { seed: 42 } })
    // f64 to match nn.Linear's parameter dtype (tensor-core has no implicit
    // promotion, by design) — collate's default stays f32, the family's
    // ML-oriented default.
    .batch(16, { collate: collate.xy({ dtype: "f64" }) });

  const model = new nn.Linear(1, 1);
  const t = trainer.configure({
    model,
    optimizer: new optim.SGD(model.parameters(), { lr: 0.1 }),
    lossFn: (prediction: Variable, target: Variable) => nn.mseLoss(prediction, target),
  });

  const { lossHistory } = await t.fit(pipeline);
  assert.equal(lossHistory.length, 60 * 4); // 60 epochs x ceil(64/16) batches
  const last = lossHistory[lossHistory.length - 1] as number;
  assert.ok(last < 1e-3, `expected convergence, final loss ${last}`);
  assert.ok(last < (lossHistory[0] as number), "loss should decrease");

  const weight = model.stateDict()["weight"];
  const bias = model.stateDict()["bias"];
  assert.ok(Math.abs((weight?.item() as number) - 3) < 0.05, `weight ≈ 3, got ${weight?.item()}`);
  assert.ok(Math.abs((bias?.item() as number) - -1) < 0.05, `bias ≈ -1, got ${bias?.item()}`);
});
