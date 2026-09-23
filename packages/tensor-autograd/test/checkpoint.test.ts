/** Module.namedParameters()/stateDict()/loadStateDict() + io.writeCheckpoint/loadCheckpoint (issue #42). */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { random, Tensor } from "@johnhenry/math-plus-tensor-core";
import { constant, io, nn, Variable, variable } from "../src/index.ts";

test("namedParameters(): dotted-path names for a nested module", () => {
  class Net extends nn.Module {
    readonly a = new nn.Linear(2, 3, { dtype: "f64", rng: random.seed(1) });
    readonly b = new nn.Linear(3, 1, { dtype: "f64", rng: random.seed(2) });
    forward(x: Variable): Variable {
      return this.b.forward(this.a.forward(x).relu());
    }
  }
  const net = new Net();
  const named = net.namedParameters();
  assert.deepEqual(new Set(Object.keys(named)), new Set(["a.weight", "a.bias", "b.weight", "b.bias"]));
});

test("stateDict()/loadStateDict() round-trip a Linear module's parameters exactly", () => {
  const src = new nn.Linear(3, 4, { dtype: "f64", rng: random.seed(1) });
  const dst = new nn.Linear(3, 4, { dtype: "f64", rng: random.seed(2) }); // different init -- must not already match

  const srcW = src.weight.value.toArray();
  const dstWBefore = dst.weight.value.toArray();
  assert.notDeepEqual(dstWBefore, srcW, "sanity: the two modules must start with different weights");

  dst.loadStateDict(src.stateDict());
  assert.deepEqual(dst.weight.value.toArray(), srcW);
  assert.deepEqual(dst.bias?.value.toArray(), src.bias?.value.toArray());
});

test("loadStateDict() produces identical forward output to the source module", () => {
  const src = new nn.Linear(3, 4, { dtype: "f64", rng: random.seed(1) });
  const dst = new nn.Linear(3, 4, { dtype: "f64", rng: random.seed(2) });
  dst.loadStateDict(src.stateDict());

  const x = variable(random.uniform([2, 3], { rng: random.seed(3), dtype: "f64" }));
  const ySrc = src.forward(x).value.toArray();
  const yDst = dst.forward(x).value.toArray();
  assert.deepEqual(ySrc, yDst);
});

test("loadStateDict() throws on a missing parameter", () => {
  const m = new nn.Linear(2, 2, { dtype: "f64", rng: random.seed(1) });
  const partial = { "weight": m.weight.value };
  assert.throws(() => m.loadStateDict(partial), /missing parameter "bias"/);
});

test("loadStateDict() throws on an unexpected extra key", () => {
  const m = new nn.Linear(2, 2, { dtype: "f64", bias: false, rng: random.seed(1) });
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
  const src = new nn.Linear(3, 4, { dtype: "f64", rng: random.seed(1) });
  const bytes = io.writeCheckpoint(src.stateDict());

  const dst = new nn.Linear(3, 4, { dtype: "f64", rng: random.seed(2) });
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

test("writeCheckpoint/loadCheckpoint round-trip f16 and bf16 tensors bit-for-bit (bf16 is stored as the ml_dtypes '<V2' .npy)", () => {
  const state = {
    "half.weight": Tensor.from([1.5, -0.1, 65504, 2 ** -24], { dtype: "f16" }).reshape([2, 2]),
    "bf16.weight": Tensor.from([1.5, -0.1, 3e38, 1e-40], { dtype: "bf16" }).reshape([2, 2]),
  };
  const loaded = io.loadCheckpoint(io.writeCheckpoint(state));
  for (const [name, t] of Object.entries(state)) {
    const back = loaded[name]!;
    assert.equal(back.dtype, t.dtype, name);
    assert.deepEqual(back.shape, t.shape, name);
    assert.deepEqual([...(back.data as Uint16Array)], [...(t.data as Uint16Array)], name);
  }
});

// ---- issue #123: [out, in] Linear, dtype-casting loadStateDict, legacy (v1) checkpoints ----

class TwoLayer extends nn.Module {
  readonly a = new nn.Linear(3, 4);
  readonly b = new nn.Linear(4, 4); // square: layout can't be guessed from the shape
  readonly norm = new nn.LayerNorm(4);
  forward(x: Variable): Variable {
    return this.norm.forward(this.b.forward(this.a.forward(x).relu()));
  }
}

/** A pre-#123 state dict: f64 everywhere, Linear weights stored [in, out]. */
function oldLayoutStateDict(): Record<string, Tensor> {
  const r = (shape: number[], seed: number) => random.uniform(shape, { min: -1, max: 1, dtype: "f64", rng: random.seed(seed) });
  return {
    "a.weight": r([3, 4], 1),
    "a.bias": r([4], 2),
    "b.weight": r([4, 4], 3),
    "b.bias": r([4], 4),
    "norm.weight": r([4], 5),
    "norm.bias": r([4], 6),
  };
}

/** The old math, computed directly in f64: x @ W_old + b. */
function oldForward(sd: Record<string, Tensor>, x: Tensor): Tensor {
  const h = x.matmul(sd["a.weight"] as Tensor).add(sd["a.bias"] as Tensor).relu();
  const y = h.matmul(sd["b.weight"] as Tensor).add(sd["b.bias"] as Tensor);
  const mean = y.mean(1).unsqueeze(1);
  const c = y.sub(mean);
  const std = c.mul(c).mean(1).unsqueeze(1).add(1e-5).sqrt();
  return c.div(std).mul(sd["norm.weight"] as Tensor).add(sd["norm.bias"] as Tensor);
}

function maxAbsDiff(a: Tensor, b: Tensor): number {
  const x = (a.toArray() as number[][]).flat();
  const y = (b.toArray() as number[][]).flat();
  return Math.max(...x.map((v, i) => Math.abs(v - (y[i] as number))));
}

test("Linear stores PyTorch's [out, in] weight layout, f32 by default", () => {
  const m = new nn.Linear(3, 5);
  assert.deepEqual([...m.weight.shape], [5, 3]);
  assert.equal(m.weight.dtype, "f32");
  assert.equal(new nn.Linear(3, 5, { dtype: "f64" }).weight.dtype, "f64");
});

test("writeCheckpoint writes format version 2", () => {
  assert.equal(io.writeCheckpoint({ x: Tensor.from([1], { dtype: "f32" }) })[4], 2);
});

test("a version-1 (pre-#123) MPCK checkpoint loads transparently: Linear weights transposed, f64 cast to the module's f32", () => {
  const old = oldLayoutStateDict();
  const bytes = io.writeCheckpoint(old);
  bytes[4] = 1; // the v1 container layout is byte-identical; only the Linear-layout meaning differs
  const loaded = io.loadCheckpoint(bytes);
  assert.equal((loaded as unknown as Record<symbol, unknown>)[io.LEGACY_LINEAR_LAYOUT], true);

  const m = new TwoLayer();
  m.loadStateDict(loaded);
  assert.deepEqual([...m.b.weight.shape], [4, 4]);
  assert.equal(m.b.weight.dtype, "f32");
  assert.deepEqual(m.b.weight.value.toArray(), (old["b.weight"] as Tensor).transpose().cast("f32").toArray());

  const x = random.uniform([2, 3], { min: -1, max: 1, dtype: "f64", rng: random.seed(9) });
  const got = m.forward(constant(x.cast("f32"))).value.cast("f64");
  assert.ok(maxAbsDiff(got, oldForward(old, x)) < 1e-5, "legacy checkpoint must reproduce the old model's outputs");
});

test("legacyLinearLayout: true handles an old in-memory state dict; without it a non-square old weight is a loud shape error", () => {
  const old = oldLayoutStateDict();
  const m = new TwoLayer();
  assert.throws(() => m.loadStateDict(old), /shape mismatch for "a.weight".*legacyLinearLayout/);
  m.loadStateDict(old, { legacyLinearLayout: true });
  const x = random.uniform([2, 3], { min: -1, max: 1, dtype: "f64", rng: random.seed(10) });
  assert.ok(maxAbsDiff(m.forward(constant(x.cast("f32"))).value.cast("f64"), oldForward(old, x)) < 1e-5);
});

test("a version-2 checkpoint is NOT transposed (round-trips as written)", () => {
  const src = new TwoLayer();
  const dst = new TwoLayer();
  dst.loadStateDict(io.loadCheckpoint(io.writeCheckpoint(src.stateDict())));
  assert.deepEqual(dst.b.weight.value.toArray(), src.b.weight.value.toArray());
});

test("loadStateDict casts to each parameter's dtype, and a failed load leaves the module untouched", () => {
  const m = new nn.Linear(2, 3, { rng: random.seed(1) });
  const before = m.weight.value.toArray();
  const f64 = { weight: random.uniform([3, 2], { dtype: "f64", rng: random.seed(2) }), bias: random.uniform([3], { dtype: "f64", rng: random.seed(3) }) };
  assert.throws(() => m.loadStateDict({ weight: f64.weight, bias: Tensor.zeros([4], { dtype: "f64" }) }), /shape mismatch for "bias"/);
  assert.deepEqual(m.weight.value.toArray(), before, "weight must not be half-updated by a failed load");
  m.loadStateDict(f64);
  assert.equal(m.weight.dtype, "f32");
  assert.deepEqual(m.weight.value.toArray(), f64.weight.cast("f32").toArray());
});

test("loadStateDict({ strict: false }) loads the intersection and ignores extra and missing keys", () => {
  const m = new nn.Linear(2, 2, { rng: random.seed(1) });
  const biasBefore = m.bias?.value.toArray();
  const w = Tensor.ones([2, 2], { dtype: "f32" });
  m.loadStateDict({ weight: w, "head.weight": w }, { strict: false });
  assert.deepEqual(m.weight.value.toArray(), w.toArray());
  assert.deepEqual(m.bias?.value.toArray(), biasBefore);
});

test("namedModules(): dotted paths, including the root as \"\"", () => {
  assert.deepEqual(Object.keys(new TwoLayer().namedModules()), ["", "a", "b", "norm"]);
});

test("MPCK round-trips f16 parameters bit-exactly", () => {
  const m = new nn.Linear(3, 2, { dtype: "f16", rng: random.seed(4) });
  const back = io.loadCheckpoint(io.writeCheckpoint(m.stateDict()));
  assert.equal(back.weight?.dtype, "f16");
  assert.deepEqual([...(back.weight as Tensor).data], [...m.weight.value.data]);
});
