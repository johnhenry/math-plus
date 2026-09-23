/** Module.namedParameters()/stateDict()/loadStateDict() + io.writeCheckpoint/loadCheckpoint (issue #42). */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { random, Tensor } from "@johnhenry/math-plus-tensor-core";
import { io, nn, Variable, variable } from "../src/index.ts";

test("namedParameters(): dotted-path names for a nested module", () => {
  class Net extends nn.Module {
    readonly a = new nn.Linear(2, 3, { rng: random.seed(1) });
    readonly b = new nn.Linear(3, 1, { rng: random.seed(2) });
    forward(x: Variable): Variable {
      return this.b.forward(this.a.forward(x).relu());
    }
  }
  const net = new Net();
  const named = net.namedParameters();
  assert.deepEqual(new Set(Object.keys(named)), new Set(["a.weight", "a.bias", "b.weight", "b.bias"]));
});

test("stateDict()/loadStateDict() round-trip a Linear module's parameters exactly", () => {
  const src = new nn.Linear(3, 4, { rng: random.seed(1) });
  const dst = new nn.Linear(3, 4, { rng: random.seed(2) }); // different init -- must not already match

  const srcW = src.weight.value.toArray();
  const dstWBefore = dst.weight.value.toArray();
  assert.notDeepEqual(dstWBefore, srcW, "sanity: the two modules must start with different weights");

  dst.loadStateDict(src.stateDict());
  assert.deepEqual(dst.weight.value.toArray(), srcW);
  assert.deepEqual(dst.bias?.value.toArray(), src.bias?.value.toArray());
});

test("loadStateDict() produces identical forward output to the source module", () => {
  const src = new nn.Linear(3, 4, { rng: random.seed(1) });
  const dst = new nn.Linear(3, 4, { rng: random.seed(2) });
  dst.loadStateDict(src.stateDict());

  const x = variable(random.uniform([2, 3], { rng: random.seed(3), dtype: "f64" }));
  const ySrc = src.forward(x).value.toArray();
  const yDst = dst.forward(x).value.toArray();
  assert.deepEqual(ySrc, yDst);
});

test("loadStateDict() throws on a missing parameter", () => {
  const m = new nn.Linear(2, 2, { rng: random.seed(1) });
  const partial = { "weight": m.weight.value };
  assert.throws(() => m.loadStateDict(partial), /missing parameter "bias"/);
});

test("loadStateDict() throws on an unexpected extra key", () => {
  const m = new nn.Linear(2, 2, { bias: false, rng: random.seed(1) });
  const withExtra = { weight: m.weight.value, extra: m.weight.value };
  assert.throws(() => m.loadStateDict(withExtra), /unexpected parameter "extra"/);
});

test("writeCheckpoint/loadCheckpoint round-trip a multi-tensor state dict exactly, including mixed dtypes/shapes", () => {
  const stateDict = {
    "a.weight": Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]),
    "a.bias": Tensor.from([1, 2], { dtype: "f32" }),
    "counter": Tensor.from([1, 2, 3], { dtype: "i32" }),
  };
  const bytes = io.writeCheckpoint(stateDict);
  const loaded = io.loadCheckpoint(bytes);

  assert.deepEqual(new Set(Object.keys(loaded)), new Set(Object.keys(stateDict)));
  for (const [name, tensor] of Object.entries(stateDict)) {
    const loadedTensor = loaded[name] as Tensor;
    assert.equal(loadedTensor.dtype, tensor.dtype, `${name} dtype`);
    assert.deepEqual([...loadedTensor.shape], [...tensor.shape], `${name} shape`);
    assert.deepEqual(loadedTensor.toArray(), tensor.toArray(), `${name} values`);
  }
});

test("writeCheckpoint/loadCheckpoint round-trips end-to-end with a real Module via stateDict()/loadStateDict()", () => {
  const src = new nn.Linear(3, 4, { rng: random.seed(1) });
  const bytes = io.writeCheckpoint(src.stateDict());

  const dst = new nn.Linear(3, 4, { rng: random.seed(2) });
  dst.loadStateDict(io.loadCheckpoint(bytes));

  assert.deepEqual(dst.weight.value.toArray(), src.weight.value.toArray());
});

test("loadCheckpoint throws a clear error on malformed bytes (bad magic)", () => {
  assert.throws(() => io.loadCheckpoint(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9])), /not a math-plus checkpoint/);
});

test("loadCheckpoint throws a clear error on truncated bytes", () => {
  const bytes = io.writeCheckpoint({ x: Tensor.from([1, 2, 3], { dtype: "f64" }) });
  assert.throws(() => io.loadCheckpoint(bytes.slice(0, bytes.length - 5)), /truncated checkpoint/);
});

test("loadCheckpoint throws on an unsupported version byte", () => {
  const bytes = io.writeCheckpoint({ x: Tensor.from([1], { dtype: "f64" }) });
  const corrupted = bytes.slice();
  corrupted[4] = 99; // version byte
  assert.throws(() => io.loadCheckpoint(corrupted), /unsupported checkpoint version/);
});
