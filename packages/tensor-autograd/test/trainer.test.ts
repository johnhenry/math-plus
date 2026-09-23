/** trainer.configure/fit (issue #43). */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { random, Tensor } from "@johnhenry/math-plus-tensor-core";
import { nn, optim, trainer, Variable } from "../src/index.ts";

test("trainer.fit({x, y}): linear regression converges with plain SGD (same problem/threshold as the existing toy-training-loop test)", async () => {
  const rng = random.seed(3);
  const model = new nn.Linear(1, 1, { rng });
  const opt = new optim.SGD(model.parameters(), { lr: 0.01 });

  const xs = [1, 2, 3, 4, 5];
  const X = Tensor.from(xs, { dtype: "f64" }).reshape([5, 1]);
  const Y = Tensor.from(
    xs.map((x) => 3 * x + 2),
    { dtype: "f64" },
  ).reshape([5, 1]);

  const t = trainer.configure({ model, optimizer: opt, lossFn: nn.mseLoss, epochs: 2000 });
  const result = await t.fit({ x: X, y: Y });

  assert.equal(result.lossHistory.length, 2000);
  const lastLoss = result.lossHistory[result.lossHistory.length - 1] as number;
  assert.ok(lastLoss < 0.01, `linear regression did not converge: final loss ${lastLoss}`);
  // Loss should have actually decreased, not just ended low by luck.
  assert.ok(lastLoss < (result.lossHistory[0] as number));
});

test("trainer.fit({x, y}): XOR-MLP converges with AdamW", async () => {
  const rng = random.seed(7);
  class XorNet extends nn.Module {
    readonly fc1 = new nn.Linear(2, 8, { rng });
    readonly fc2 = new nn.Linear(8, 1, { rng });
    forward(x: Variable): Variable {
      return this.fc2.forward(this.fc1.forward(x).relu()).sigmoid();
    }
  }
  const net = new XorNet();
  const opt = new optim.AdamW(net.parameters(), { lr: 0.05 });

  const X = Tensor.from([0, 0, 0, 1, 1, 0, 1, 1], { dtype: "f64" }).reshape([4, 2]);
  const Y = Tensor.from([0, 1, 1, 0], { dtype: "f64" }).reshape([4, 1]);

  const t = trainer.configure({ model: net, optimizer: opt, lossFn: nn.mseLoss, epochs: 500 });
  const result = await t.fit({ x: X, y: Y });

  const lastLoss = result.lossHistory[result.lossHistory.length - 1] as number;
  assert.ok(lastLoss < 0.05, `XOR-MLP did not converge: final loss ${lastLoss}`);
});

test("trainer.fit(dataLoader): converges on the SAME linear-regression problem using a hand-rolled async generator, no real `data` package involved", async () => {
  const rng = random.seed(3);
  const model = new nn.Linear(1, 1, { rng });
  const opt = new optim.SGD(model.parameters(), { lr: 0.01 });

  const xs = [1, 2, 3, 4, 5];
  const ys = xs.map((x) => 3 * x + 2);

  async function* dataLoader(): AsyncGenerator<{ x: Tensor; y: Tensor }> {
    // 2000 repeated full-batch "steps", matching the {x,y} test's own epoch
    // count -- a plain async generator, deliberately not importing anything
    // from a `data` package (which doesn't exist yet, #22).
    for (let i = 0; i < 2000; i++) {
      yield {
        x: Tensor.from(xs, { dtype: "f64" }).reshape([5, 1]),
        y: Tensor.from(ys, { dtype: "f64" }).reshape([5, 1]),
      };
    }
  }

  const t = trainer.configure({ model, optimizer: opt, lossFn: nn.mseLoss });
  const result = await t.fit(dataLoader());

  assert.equal(result.lossHistory.length, 2000);
  const lastLoss = result.lossHistory[result.lossHistory.length - 1] as number;
  assert.ok(lastLoss < 0.01, `did not converge via dataLoader: final loss ${lastLoss}`);
});

test("trainer.fit(dataLoader): each yielded batch is a genuine, independent training step (loss decreases across mini-batches, not just repeats of one batch)", async () => {
  const rng = random.seed(3);
  const model = new nn.Linear(1, 1, { rng });
  const opt = new optim.SGD(model.parameters(), { lr: 0.02 });

  async function* dataLoader(): AsyncGenerator<{ x: Tensor; y: Tensor }> {
    for (let epoch = 0; epoch < 300; epoch++) {
      for (const x of [1, 2, 3, 4, 5]) {
        yield {
          x: Tensor.from([x], { dtype: "f64" }).reshape([1, 1]),
          y: Tensor.from([3 * x + 2], { dtype: "f64" }).reshape([1, 1]),
        };
      }
    }
  }

  const t = trainer.configure({ model, optimizer: opt, lossFn: nn.mseLoss });
  const result = await t.fit(dataLoader());

  assert.equal(result.lossHistory.length, 300 * 5);
  const lastLoss = result.lossHistory[result.lossHistory.length - 1] as number;
  assert.ok(lastLoss < 0.05, `mini-batch training did not converge: final loss ${lastLoss}`);
});
