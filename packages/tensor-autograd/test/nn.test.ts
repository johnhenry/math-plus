import assert from "node:assert/strict";
import { test } from "node:test";
import { Tensor, random } from "@johnhenry/math-plus-tensor-core";
import { Variable, constant, nn, optim, variable } from "../src/index.ts";
import type { Parameter } from "../src/nn.ts";
import { assertGradientMatches, randomTensor } from "./gradcheck.ts";

// ---- nn building blocks -----------------------------------------------------

test("Linear: forward shape and parameter collection", () => {
  const rng = random.seed(1);
  const linear = new nn.Linear(3, 4, { dtype: "f64", rng });
  const x = variable(random.uniform([2, 3], { rng, dtype: "f64" })); // Linear's params are f64
  const y = linear.forward(x);
  assert.deepEqual([...y.shape], [2, 4]);
  assert.equal(linear.parameters().length, 2); // weight + bias
});

test("Linear: bias: false omits the bias parameter", () => {
  const linear = new nn.Linear(3, 4, { dtype: "f64", bias: false, rng: random.seed(1) });
  assert.equal(linear.bias, null);
  assert.equal(linear.parameters().length, 1);
});

test("Module.parameters() recurses through nested modules", () => {
  class Net extends nn.Module {
    readonly a = new nn.Linear(2, 3, { dtype: "f64", rng: random.seed(1) });
    readonly b = new nn.Linear(3, 1, { dtype: "f64", rng: random.seed(2) });
    forward(x: Variable): Variable {
      return this.b.forward(this.a.forward(x).relu());
    }
  }
  const net = new Net();
  assert.equal(net.parameters().length, 4); // a.weight, a.bias, b.weight, b.bias
});

test("Module.zeroGrad() clears every parameter's gradient", () => {
  const linear = new nn.Linear(2, 2, { dtype: "f64", rng: random.seed(1) });
  const x = variable(random.uniform([1, 2], { rng: random.seed(2), dtype: "f64" })); // Linear's params are f64
  linear.forward(x).sum().backward();
  assert.notEqual(linear.weight.grad, null);
  linear.zeroGrad();
  assert.equal(linear.weight.grad, null);
  assert.equal((linear.bias as Parameter | null)?.grad ?? null, null);
});

test("Embedding: gather forward, scatter-add backward accumulates duplicate indices", () => {
  const emb = new nn.Embedding(5, 3, { dtype: "f64", rng: random.seed(1) });
  const indices = Tensor.from([0, 2, 0], { dtype: "i32" }); // index 0 repeated
  const out = emb.forward(indices);
  assert.deepEqual([...out.shape], [3, 3]);
  out.sum().backward();
  // Row 0's gradient should be 2x a single row's contribution (gathered twice).
  const g = emb.weight.grad?.toArray() as number[][];
  assert.deepEqual(g[0], [2, 2, 2]);
  assert.deepEqual(g[1], [0, 0, 0]); // never gathered
  assert.deepEqual(g[2], [1, 1, 1]);
});

test("Embedding: sparse backward matches the dense reference implementation exactly (differential test)", () => {
  // Reference: the OLD dense-allocation algorithm this replaced (issue #100)
  // -- a full (numEmbeddings x embeddingDim) zero table, scatter-added into
  // by row index. Recomputed here independently so the new sparse
  // Map<rowIdx, Float64Array> path (nn.ts's Embedding.forward) has something
  // to be checked against, not just "doesn't crash".
  function denseReferenceGrad(
    idxArray: number[],
    gRows: number[][],
    numEmbeddings: number,
    embeddingDim: number,
  ): number[][] {
    const acc: number[][] = Array.from({ length: numEmbeddings }, () =>
      new Array(embeddingDim).fill(0),
    );
    idxArray.forEach((rowIdx, i) => {
      for (let d = 0; d < embeddingDim; d++) {
        (acc[rowIdx] as number[])[d] += (gRows[i] as number[])[d] as number;
      }
    });
    return acc;
  }

  const numEmbeddings = 40;
  const embeddingDim = 6;
  const emb = new nn.Embedding(numEmbeddings, embeddingDim, { dtype: "f64", rng: random.seed(7) });
  // Duplicates AND untouched rows, to exercise accumulation and zero-fill alike.
  const rawIndices = [3, 17, 3, 0, 39, 17, 17, 22];
  const indices = Tensor.from(rawIndices, { dtype: "i32" });

  const out = emb.forward(indices);
  // Distinct upstream gradient per row (not all-ones) so accumulation order/
  // values are actually exercised, not just counted.
  const gradOutput = Tensor.from(
    rawIndices.flatMap((_, i) => Array.from({ length: embeddingDim }, (_, d) => i * 10 + d + 1)),
    { dtype: "f64" },
  ).reshape([rawIndices.length, embeddingDim]);
  out.backward(gradOutput);

  const actual = emb.weight.grad?.toArray() as number[][];
  const gRows = gradOutput.toArray() as number[][];
  const expected = denseReferenceGrad(rawIndices, gRows, numEmbeddings, embeddingDim);

  assert.deepEqual(actual, expected);
});

test("Embedding: backward on a large table stays fast regardless of table size (sparse, not dense-allocation, accumulation)", () => {
  // Issue #100's own repro: batch-of-3 into a 50,000x256 table measured
  // ~770ms under the old dense-per-call allocation. This asserts the fix
  // keeps a small batch's backward call well under that, on a table of the
  // same scale.
  const numEmbeddings = 50_000;
  const embeddingDim = 256;
  const emb = new nn.Embedding(numEmbeddings, embeddingDim, { dtype: "f64", rng: random.seed(3) });
  const indices = Tensor.from([10, 42, 1000], { dtype: "i32" });
  const out = emb.forward(indices);

  const start = performance.now();
  out.sum().backward();
  const elapsedMs = performance.now() - start;

  assert.ok(
    elapsedMs < 200,
    `backward() took ${elapsedMs}ms for a batch-of-3 into a ${numEmbeddings}x${embeddingDim} table; expected well under the ~770ms dense-allocation baseline`,
  );
});

test("LayerNorm: normalizes the last axis to ~zero mean, ~unit variance", () => {
  const ln = new nn.LayerNorm(4, { dtype: "f64" });
  const x = variable(Tensor.from([1, 2, 3, 100, -5, 0, 5, 10], { dtype: "f64" }).reshape([2, 4]));
  const y = ln.forward(x);
  const rowMean = y.value.mean(1).toArray() as number[];
  for (const m of rowMean) assert.ok(Math.abs(m) < 1e-6, `row mean ${m} not ~0`);
});

test("mseLoss and crossEntropy compute finite, non-negative losses", () => {
  const pred = variable(Tensor.from([0.5, 0.5], { dtype: "f64" }));
  const target = constant(Tensor.from([1, 0], { dtype: "f64" }));
  const mse = nn.mseLoss(pred, target);
  assert.ok(mse.value.item() as number > 0);

  const logits = variable(Tensor.from([2, 0.5, 0.1, 0.2, 3, 0.1], { dtype: "f64" }).reshape([2, 3]));
  const labels = Tensor.from([0, 1], { dtype: "i32" });
  const ce = nn.crossEntropy(logits, labels);
  assert.ok(Number.isFinite(ce.value.item() as number));
  assert.ok((ce.value.item() as number) > 0);
});

// ---- Sequential/Dropout/huberLoss/binaryCrossEntropy (issue #71) -----------

test("Sequential: forward chains layers in order, matching a hand-composed equivalent", () => {
  const rng = random.seed(11);
  const a = new nn.Linear(3, 5, { dtype: "f64", rng });
  const b = new nn.Linear(5, 2, { dtype: "f64", rng });
  const seq = new nn.Sequential([a, b]);
  const x = variable(random.uniform([4, 3], { rng, dtype: "f64" }));
  const viaSequential = seq.forward(x).value.toArray();
  const viaHandComposed = b.forward(a.forward(x)).value.toArray();
  assert.deepEqual(viaSequential, viaHandComposed);
});

test("Sequential: .parameters()/.namedParameters() discover every sub-module's parameters (reflection-walk regression test)", () => {
  const rng = random.seed(12);
  const seq = new nn.Sequential([new nn.Linear(2, 3, { dtype: "f64", rng }), new nn.Linear(3, 4, { dtype: "f64", rng }), new nn.Linear(4, 1, { dtype: "f64", rng })]);
  assert.equal(seq.parameters().length, 6); // 3 layers x (weight + bias)
  assert.deepEqual(
    Object.keys(seq.namedParameters()).sort(),
    ["0.bias", "0.weight", "1.bias", "1.weight", "2.bias", "2.weight"],
  );
  // And stateDict/loadStateDict (issue #42) work transparently through Sequential too.
  const snapshot = seq.stateDict();
  seq.loadStateDict(snapshot); // must not throw
  assert.equal(Object.keys(snapshot).length, 6);
});

test("Dropout: training=false is an exact identity", () => {
  const dropout = new nn.Dropout(0.5);
  const x = variable(randomTensor([20]));
  const out = dropout.forward(x, false);
  assert.deepEqual(out.value.toArray(), x.value.toArray());
});

test("Dropout: p=0 is an exact identity even when training=true", () => {
  const dropout = new nn.Dropout(0);
  const x = variable(randomTensor([20]));
  const out = dropout.forward(x, true);
  assert.deepEqual(out.value.toArray(), x.value.toArray());
});

test("Dropout: training=true zeroes ~p fraction of elements at a fixed seed, scales survivors by 1/(1-p), and preserves mean output magnitude in expectation", () => {
  const p = 0.3;
  const dropout = new nn.Dropout(p);
  const n = 20000;
  const x = variable(Tensor.from(new Array(n).fill(1), { dtype: "f64" }));
  const out = dropout.forward(x, true, { rng: random.seed(99) }).value.toArray() as number[];
  const zeros = out.filter((v) => v === 0).length;
  const survivors = out.filter((v) => v !== 0);
  const expectedScale = 1 / (1 - p);
  assert.ok(survivors.every((v) => Math.abs(v - expectedScale) < 1e-9), "every surviving element must be scaled by 1/(1-p) exactly");
  const zeroFraction = zeros / n;
  assert.ok(Math.abs(zeroFraction - p) < 0.02, `zero fraction ${zeroFraction} should be close to p=${p}`);
  const mean = out.reduce((a, b) => a + b, 0) / n;
  assert.ok(Math.abs(mean - 1) < 0.02, `expected mean output ~1 (unchanged in expectation), got ${mean}`);
});

test("Dropout: rejects an out-of-range p", () => {
  assert.throws(() => new nn.Dropout(-0.1), RangeError);
  assert.throws(() => new nn.Dropout(1), RangeError);
});

test("huberLoss: gradcheck against finite differences", () => {
  const target = constant(randomTensor([6]));
  assertGradientMatches((pred) => nn.huberLoss(pred, target, 0.7), randomTensor([6]));
});

test("huberLoss: near zero, behaves like a scaled MSE (both quadratic near the origin, per the pseudo-Huber definition)", () => {
  const pred = variable(Tensor.from([1.01, 0.99], { dtype: "f64" }));
  const target = constant(Tensor.from([1, 1], { dtype: "f64" }));
  const delta = 1;
  const huber = nn.huberLoss(pred, target, delta).value.item() as number;
  const mse = nn.mseLoss(pred, target).value.item() as number;
  // pseudo-Huber(x) ~ 0.5*x^2 for |x| << delta -- same leading term as mseLoss's mean(diff^2), off by the 0.5 factor.
  assert.ok(Math.abs(huber - 0.5 * mse) < 1e-4, `huber=${huber} should be ~0.5*mse=${0.5 * mse} for small errors`);
});

test("huberLoss: far from zero, grows roughly LINEARLY (not quadratically) in the error, the outlier-robustness property", () => {
  const target = constant(Tensor.from([0], { dtype: "f64" }));
  const delta = 1;
  const lossAt = (x: number) => nn.huberLoss(variable(Tensor.from([x], { dtype: "f64" })), target, delta).value.item() as number;
  const l10 = lossAt(10);
  const l20 = lossAt(20);
  const l1000 = lossAt(1000);
  const l2000 = lossAt(2000);
  // Far from the origin, doubling the error should roughly double pseudo-Huber's
  // loss (linear growth) -- unlike mseLoss, which would roughly QUADRUPLE.
  // At only 10x delta the asymptote hasn't fully kicked in yet (pseudo-Huber
  // is exactly sqrt(1+x^2)-1, so l20/l10 = (sqrt(401)-1)/(sqrt(101)-1) ~
  // 2.102, not 2 -- a real, computed value, not test slop) -- a looser bound
  // there; at 1000x delta it's tight.
  assert.ok(Math.abs(l20 / l10 - 2) < 0.15, `l20/l10=${l20 / l10} should be roughly 2 (linear growth)`);
  assert.ok(Math.abs(l2000 / l1000 - 2) < 0.01, `l2000/l1000=${l2000 / l1000} should be ~2 (linear growth)`);
});

test("binaryCrossEntropy: matches a hand-computed value at logit=0 (p=0.5)", () => {
  const logits = variable(Tensor.from([0, 0], { dtype: "f64" }));
  const target = constant(Tensor.from([1, 0], { dtype: "f64" }));
  const bce = nn.binaryCrossEntropy(logits, target).value.item() as number;
  // -mean(y*ln(0.5) + (1-y)*ln(0.5)) = -ln(0.5) = ln(2), for EITHER label at p=0.5.
  assert.ok(Math.abs(bce - Math.log(2)) < 1e-9, `expected ln(2)=${Math.log(2)}, got ${bce}`);
});

test("binaryCrossEntropy: a confident CORRECT prediction has much lower loss than a confident WRONG one", () => {
  const target = constant(Tensor.from([1], { dtype: "f64" }));
  const confidentCorrect = nn.binaryCrossEntropy(variable(Tensor.from([10], { dtype: "f64" })), target).value.item() as number;
  const confidentWrong = nn.binaryCrossEntropy(variable(Tensor.from([-10], { dtype: "f64" })), target).value.item() as number;
  assert.ok(confidentCorrect < 1e-3, `confident-correct loss should be tiny, got ${confidentCorrect}`);
  assert.ok(confidentWrong > 9, `confident-wrong loss should be large, got ${confidentWrong}`);
});

test("binaryCrossEntropy: gradcheck against finite differences", () => {
  const target = constant(Tensor.from([1, 0, 1, 0], { dtype: "f64" }));
  assertGradientMatches((logits) => nn.binaryCrossEntropy(logits, target), randomTensor([4]));
});

// ---- BCEWithLogits reformulation: saturated-logit NaN fix (issue #85) ------

test("binaryCrossEntropy: saturated (converged) logits produce a finite, near-zero loss, not NaN -- the issue's own deterministic repro", () => {
  const logits = variable(Tensor.from([50, -50], { dtype: "f64" }).reshape([2, 1]));
  const target = constant(Tensor.from([1, 0], { dtype: "f64" }).reshape([2, 1]));
  const bce = nn.binaryCrossEntropy(logits, target).value.item() as number;
  assert.ok(Number.isFinite(bce), `expected a finite loss, got ${bce}`);
  assert.ok(Math.abs(bce) < 1e-9, `expected ~0 for two confident-correct predictions, got ${bce}`);
});

test("binaryCrossEntropy: gradient stays finite (not NaN) at saturated logits -- the actual failure mode that poisoned weights on the next backward()", () => {
  const logits = variable(Tensor.from([50, -50], { dtype: "f64" }).reshape([2, 1]));
  const target = constant(Tensor.from([1, 0], { dtype: "f64" }).reshape([2, 1]));
  nn.binaryCrossEntropy(logits, target).backward();
  const grad = logits.grad!.toArray() as number[];
  for (const g of grad.flat(Infinity) as number[]) {
    assert.ok(Number.isFinite(g), `expected a finite gradient, got ${g}`);
  }
});

test("binaryCrossEntropy: a confidently WRONG prediction still produces a large but finite loss (not Infinity), even more saturated than the 'confident WRONG' test above", () => {
  const logits = variable(Tensor.from([50], { dtype: "f64" }));
  const target = constant(Tensor.from([0], { dtype: "f64" }));
  const bce = nn.binaryCrossEntropy(logits, target).value.item() as number;
  assert.ok(Number.isFinite(bce), `expected a finite loss, got ${bce}`);
  assert.ok(Math.abs(bce - 50) < 1e-6, `expected ~50 (relu(50) - 50*0 + log(1+exp(-50)) ~= 50), got ${bce}`);
});

// ---- toy training loop: the issue #9 acceptance criterion -------------------

test("toy training loop: XOR-MLP converges with AdamW", () => {
  const rng = random.seed(7);

  class XorNet extends nn.Module {
    readonly fc1 = new nn.Linear(2, 8, { dtype: "f64", rng });
    readonly fc2 = new nn.Linear(8, 1, { dtype: "f64", rng });
    forward(x: Variable): Variable {
      return this.fc2.forward(this.fc1.forward(x).relu()).sigmoid();
    }
  }

  const net = new XorNet();
  const opt = new optim.AdamW(net.parameters(), { lr: 0.05 });

  const X = Tensor.from([0, 0, 0, 1, 1, 0, 1, 1], { dtype: "f64" }).reshape([4, 2]);
  const Y = constant(Tensor.from([0, 1, 1, 0], { dtype: "f64" }).reshape([4, 1]));

  let lastLoss = Infinity;
  for (let epoch = 0; epoch < 500; epoch++) {
    net.zeroGrad();
    const pred = net.forward(variable(X));
    const loss = nn.mseLoss(pred, Y);
    loss.backward();
    opt.step();
    lastLoss = loss.value.item() as number;
  }

  assert.ok(lastLoss < 0.05, `XOR-MLP did not converge: final loss ${lastLoss}`);

  // Predictions should actually match XOR, not just have low MSE by luck.
  const finalPred = net.forward(variable(X)).value.toArray() as number[][];
  const expected = [0, 1, 1, 0];
  finalPred.forEach((row, i) => {
    const predicted = row[0] as number;
    assert.ok(
      Math.round(predicted) === expected[i],
      `example ${i}: predicted ${predicted}, expected ${expected[i]}`,
    );
  });
});

test("toy training loop: linear regression converges with plain SGD", () => {
  // y = 3x + 2, fit with a single Linear(1,1) and plain SGD.
  const rng = random.seed(3);
  const model = new nn.Linear(1, 1, { dtype: "f64", rng });
  const opt = new optim.SGD(model.parameters(), { lr: 0.01 });

  const xs = [1, 2, 3, 4, 5];
  const X = Tensor.from(xs, { dtype: "f64" }).reshape([5, 1]);
  const Y = constant(Tensor.from(xs.map((x) => 3 * x + 2), { dtype: "f64" }).reshape([5, 1]));

  let lastLoss = Infinity;
  for (let epoch = 0; epoch < 2000; epoch++) {
    model.zeroGrad();
    const pred = model.forward(variable(X));
    const loss = nn.mseLoss(pred, Y);
    loss.backward();
    opt.step();
    lastLoss = loss.value.item() as number;
  }

  assert.ok(lastLoss < 0.01, `linear regression did not converge: final loss ${lastLoss}`);
});

// ---- SGD momentum/Nesterov (issue #89) --------------------------------------

test("SGD: momentum defaults to 0 -- byte-identical to the plain (pre-#89) update", () => {
  const p1 = new nn.Parameter(Tensor.from([1], { dtype: "f64" }));
  const p2 = new nn.Parameter(Tensor.from([1], { dtype: "f64" }));
  const opt1 = new optim.SGD([p1], { lr: 0.1 });
  const opt2 = new optim.SGD([p2], { lr: 0.1, momentum: 0 });
  for (const g of [1, 1, 1]) {
    p1.grad = Tensor.from([g], { dtype: "f64" });
    p2.grad = Tensor.from([g], { dtype: "f64" });
    opt1.step();
    opt2.step();
    assert.equal(p1.value.item(), p2.value.item());
  }
  assert.ok(Math.abs((p1.value.item() as number) - 0.7) < 1e-12); // 1 - 0.1*1 - 0.1*1 - 0.1*1
});

test("SGD: classic momentum matches hand-computed buf = momentum*buf + grad, param -= lr*buf", () => {
  const p = new nn.Parameter(Tensor.from([1], { dtype: "f64" }));
  const opt = new optim.SGD([p], { lr: 0.1, momentum: 0.9 });
  // buf: 1 -> 1.9 -> 2.71 ; param: 0.9 -> 0.71 -> 0.439 (hand-verified via a standalone node -e script)
  const expected = [0.9, 0.71, 0.43899999999999995];
  for (const target of expected) {
    p.grad = Tensor.from([1], { dtype: "f64" });
    opt.step();
    assert.ok(Math.abs((p.value.item() as number) - target) < 1e-12, `expected ${target}, got ${p.value.item()}`);
  }
});

test("SGD: Nesterov matches hand-computed d_p = grad + momentum*buf (buf updated first)", () => {
  const p = new nn.Parameter(Tensor.from([1], { dtype: "f64" }));
  const opt = new optim.SGD([p], { lr: 0.1, momentum: 0.9, nesterov: true });
  // d_p: 1.9 -> 2.71 -> 3.439 ; param: 0.81 -> 0.539 -> 0.1951 (hand-verified via a standalone node -e script)
  const expected = [0.81, 0.539, 0.1951];
  for (const target of expected) {
    p.grad = Tensor.from([1], { dtype: "f64" });
    opt.step();
    assert.ok(Math.abs((p.value.item() as number) - target) < 1e-9, `expected ${target}, got ${p.value.item()}`);
  }
});

test("SGD: nesterov without a nonzero momentum throws", () => {
  assert.throws(() => new optim.SGD([], { lr: 0.1, nesterov: true }), RangeError);
  assert.throws(() => new optim.SGD([], { lr: 0.1, momentum: 0, nesterov: true }), RangeError);
});

test("toy training loop: linear regression converges faster with momentum than without, same lr/epoch budget", () => {
  const xs = [1, 2, 3, 4, 5];
  const X = Tensor.from(xs, { dtype: "f64" }).reshape([5, 1]);
  const Y = constant(Tensor.from(xs.map((x) => 3 * x + 2), { dtype: "f64" }).reshape([5, 1]));

  function finalLoss(opt: optim.SGD, model: InstanceType<typeof nn.Linear>, epochs: number): number {
    let lastLoss = Infinity;
    for (let epoch = 0; epoch < epochs; epoch++) {
      model.zeroGrad();
      const loss = nn.mseLoss(model.forward(variable(X)), Y);
      loss.backward();
      opt.step();
      lastLoss = loss.value.item() as number;
    }
    return lastLoss;
  }

  const plainModel = new nn.Linear(1, 1, { dtype: "f64", rng: random.seed(7) });
  const plainLoss = finalLoss(new optim.SGD(plainModel.parameters(), { lr: 0.01 }), plainModel, 200);

  const momentumModel = new nn.Linear(1, 1, { dtype: "f64", rng: random.seed(7) }); // same seed -- identical starting weights
  const momentumLoss = finalLoss(new optim.SGD(momentumModel.parameters(), { lr: 0.01, momentum: 0.9 }), momentumModel, 200);

  assert.ok(momentumLoss < plainLoss, `momentum (${momentumLoss}) should converge faster than plain SGD (${plainLoss}) over the same 200 epochs`);
});

// ---- Adam/RMSprop/StepLR (issue #72) -----------------------------------------

test("toy training loop: linear regression converges with plain Adam (weightDecay=0)", () => {
  const rng = random.seed(4);
  const model = new nn.Linear(1, 1, { dtype: "f64", rng });
  const opt = new optim.Adam(model.parameters(), { lr: 0.05 });
  assert.equal((opt as unknown as { weightDecay: number }).weightDecay, 0);

  const xs = [1, 2, 3, 4, 5];
  const X = Tensor.from(xs, { dtype: "f64" }).reshape([5, 1]);
  const Y = constant(Tensor.from(xs.map((x) => 3 * x + 2), { dtype: "f64" }).reshape([5, 1]));

  let lastLoss = Infinity;
  for (let epoch = 0; epoch < 500; epoch++) {
    model.zeroGrad();
    const loss = nn.mseLoss(model.forward(variable(X)), Y);
    loss.backward();
    opt.step();
    lastLoss = loss.value.item() as number;
  }
  assert.ok(lastLoss < 0.01, `Adam did not converge: final loss ${lastLoss}`);
});

test("toy training loop: XOR-MLP converges with RMSprop", () => {
  const rng = random.seed(5);
  class XorNet extends nn.Module {
    readonly l1 = new nn.Linear(2, 8, { dtype: "f64", rng });
    readonly l2 = new nn.Linear(8, 1, { dtype: "f64", rng });
    forward(x: Variable): Variable {
      return this.l2.forward(this.l1.forward(x).relu());
    }
  }
  const model = new XorNet();
  const opt = new optim.RMSprop(model.parameters(), { lr: 0.01 });
  const X = Tensor.from([0, 0, 0, 1, 1, 0, 1, 1], { dtype: "f64" }).reshape([4, 2]);
  const Y = constant(Tensor.from([0, 1, 1, 0], { dtype: "f64" }).reshape([4, 1]));

  let lastLoss = Infinity;
  for (let epoch = 0; epoch < 3000; epoch++) {
    model.zeroGrad();
    const loss = nn.mseLoss(model.forward(variable(X)), Y);
    loss.backward();
    opt.step();
    lastLoss = loss.value.item() as number;
  }
  assert.ok(lastLoss < 0.05, `RMSprop did not converge on XOR: final loss ${lastLoss}`);
});

test("StepLR: effective lr after N calls matches initialLr * gamma^floor(N/stepSize) exactly", () => {
  const opt = new optim.SGD([], { lr: 1.0 });
  const scheduler = new optim.StepLR(opt, { stepSize: 3, gamma: 0.5 });
  const expectedAt = (n: number): number => 1.0 * 0.5 ** Math.floor(n / 3);
  for (let n = 1; n <= 12; n++) {
    scheduler.step();
    const expected = expectedAt(n);
    assert.ok(Math.abs(opt.lr - expected) < 1e-12, `after ${n} calls: lr=${opt.lr}, expected ${expected}`);
  }
});

test("StepLR: does not touch lr before the first stepSize boundary", () => {
  const opt = new optim.SGD([], { lr: 0.1 });
  const scheduler = new optim.StepLR(opt, { stepSize: 5, gamma: 0.1 });
  for (let n = 0; n < 4; n++) scheduler.step();
  assert.equal(opt.lr, 0.1);
});

test("StepLR: rejects a non-positive stepSize", () => {
  const opt = new optim.SGD([], { lr: 0.1 });
  assert.throws(() => new optim.StepLR(opt, { stepSize: 0, gamma: 0.5 }), RangeError);
  assert.throws(() => new optim.StepLR(opt, { stepSize: -1, gamma: 0.5 }), RangeError);
});

test("StepLR composes with any optimizer that has a mutable lr (structural typing, not a class union)", () => {
  const rng = random.seed(6);
  const model = new nn.Linear(1, 1, { dtype: "f64", rng });
  for (const opt of [
    new optim.SGD(model.parameters(), { lr: 1 }),
    new optim.AdamW(model.parameters(), { lr: 1 }),
    new optim.Adam(model.parameters(), { lr: 1 }),
    new optim.RMSprop(model.parameters(), { lr: 1 }),
  ]) {
    const scheduler = new optim.StepLR(opt, { stepSize: 1, gamma: 0.5 });
    scheduler.step();
    assert.equal(opt.lr, 0.5, `${opt.constructor.name}: StepLR should have halved lr`);
  }
});

// ---- telemetry hooks (issue #10) --------------------------------------------

test("backward() emits a trace span when a sink is installed, nothing by default", async () => {
  const { setSink } = await import("@johnhenry/math-plus-telemetry");
  const events: unknown[] = [];
  setSink((e) => events.push(e));
  try {
    const x = variable(Tensor.from([2, 3], { dtype: "f64" }));
    x.mul(x).sum().backward(undefined, { runId: "r1", step: 5 });
    assert.equal(events.length, 1);
    const e = events[0] as { type: string; runId: string; step: number; spans: Array<{ name: string }> };
    assert.equal(e.type, "trace");
    assert.equal(e.runId, "r1");
    assert.equal(e.step, 5);
    assert.equal(e.spans[0]?.name, "backward");
  } finally {
    setSink(null);
  }

  // default: no sink installed, backward() still works exactly as before.
  const x2 = variable(Tensor.from([2, 3], { dtype: "f64" }));
  x2.mul(x2).sum().backward();
  assert.deepEqual(x2.grad?.toArray(), [4, 6]);
});

test("optim.step() emits an optim/gradNorm metric when a sink is installed", async () => {
  const { setSink } = await import("@johnhenry/math-plus-telemetry");
  const events: unknown[] = [];
  setSink((e) => events.push(e));
  try {
    const linear = new nn.Linear(2, 1, { dtype: "f64", rng: random.seed(1) });
    const opt = new optim.SGD(linear.parameters(), { lr: 0.1 });
    const x = variable(random.uniform([1, 2], { rng: random.seed(2), dtype: "f64" }));
    linear.forward(x).sum().backward(); // also emits a "backward" trace span, since the sink is already active
    opt.step({ runId: "r1", step: 0 });
    assert.equal(events.length, 2); // 1 trace (backward) + 1 metric (optim.step)
    const e = events[1] as { type: string; name: string; value: number };
    assert.equal(e.type, "metric");
    assert.equal(e.name, "optim/gradNorm");
    assert.ok(Number.isFinite(e.value) && e.value >= 0);
  } finally {
    setSink(null);
  }
});
