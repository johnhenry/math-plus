/**
 * Issue #123: PyTorch parity (forward AND backward, inputs AND parameters)
 * for the new Variable ops and transformer layers. See ./torch-oracle.ts for
 * the protocol and the skip-don't-fail contract.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { Variable, nn } from "../src/index.ts";
import {
  boolMask,
  checkCase,
  prepareCases,
  rand,
  randomizeParams,
  runTorchOracle,
  torchSkip,
  type Case,
} from "./torch-oracle.ts";

const F64 = "f64" as const;
const F32 = "f32" as const;

function layer<M extends nn.Module>(m: M, seed: number): M {
  randomizeParams(m, seed);
  return m;
}

const cases: Case[] = [];

// ---- Variable view/structure ops ---------------------------------------------

cases.push(
  { id: "reshape", kind: "reshape", dtype: F64, config: { shape: [4, -1] }, inputs: { x: rand([2, 3, 4], F64, 1) },
    forward: ({ x }) => x!.reshape([4, -1]) },
  { id: "reshape of a non-contiguous (permuted) input", kind: "permuteReshape", dtype: F64,
    config: { axes: [2, 0, 1], shape: [4, 6] }, inputs: { x: rand([2, 3, 4], F64, 2) },
    forward: ({ x }) => x!.permute([2, 0, 1]).reshape([4, 6]) },
  { id: "permute", kind: "permute", dtype: F64, config: { axes: [2, 0, 1] }, inputs: { x: rand([2, 3, 4], F64, 3) },
    forward: ({ x }) => x!.permute([2, 0, 1]) },
  { id: "permute (negative axes)", kind: "permute", dtype: F64, config: { axes: [-1, 0, 1] }, inputs: { x: rand([2, 3, 4], F64, 4) },
    forward: ({ x }) => x!.permute([-1, 0, 1]) },
  { id: "transpose(0, 2)", kind: "transpose", dtype: F64, config: { dim0: 0, dim1: 2 }, inputs: { x: rand([2, 3, 4], F64, 5) },
    forward: ({ x }) => x!.transpose(0, 2) },
  { id: "transpose(-2, -1)", kind: "transpose", dtype: F64, config: { dim0: -2, dim1: -1 }, inputs: { x: rand([2, 3, 4], F64, 6) },
    forward: ({ x }) => x!.transpose(-2, -1) },
  { id: "slice (start/end/step, negative step, untouched axis)", kind: "slice", dtype: F64,
    config: { specs: [{ start: 1 }, { start: -1, end: 0, step: -2 }, null, { start: 0, end: 4, step: 3 }] },
    inputs: { x: rand([3, 5, 2, 4], F64, 7) },
    forward: ({ x }) => x!.slice({ start: 1 }, { start: -1, end: 0, step: -2 }, null, { start: 0, end: 4, step: 3 }) },
  { id: "narrow(-1, 1, 2)", kind: "narrow", dtype: F64, config: { dim: -1, start: 1, length: 2 }, inputs: { x: rand([2, 3, 4], F64, 8) },
    forward: ({ x }) => x!.narrow(-1, 1, 2) },
  { id: "concat (3 inputs, axis 1)", kind: "concat", dtype: F64, config: { order: ["a", "b", "c"], axis: 1 },
    inputs: { a: rand([2, 1, 3], F64, 9), b: rand([2, 3, 3], F64, 10), c: rand([2, 2, 3], F64, 11) },
    forward: ({ a, b, c }) => Variable.concat([a!, b!, c!], 1) },
  { id: "concat (axis -1)", kind: "concat", dtype: F64, config: { order: ["a", "b"], axis: -1 },
    inputs: { a: rand([2, 3], F64, 12), b: rand([2, 5], F64, 13) },
    forward: ({ a, b }) => Variable.concat([a!, b!], -1) },
  { id: "maskedFill (broadcast mask)", kind: "maskedFill", dtype: F64, config: { value: -5 },
    inputs: { x: rand([2, 3, 4], F64, 14) }, masks: { mask: boolMask([1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0], [3, 4]) },
    forward: ({ x }) => x!.maskedFill(boolMask([1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0], [3, 4]), -5) },
);

// ---- elementwise -----------------------------------------------------------------

cases.push(
  { id: "exp", kind: "exp", dtype: F64, inputs: { x: rand([3, 4], F64, 20, 2) }, forward: ({ x }) => x!.exp() },
  { id: "tanh", kind: "tanh", dtype: F64, inputs: { x: rand([3, 4], F64, 21, 3) }, forward: ({ x }) => x!.tanh() },
  { id: "gelu (exact)", kind: "gelu", dtype: F64, config: { approximate: "none" }, inputs: { x: rand([4, 8], F64, 22, 6) },
    forward: ({ x }) => x!.gelu({ approximate: "none" }) },
  { id: "gelu (tanh)", kind: "gelu", dtype: F64, config: { approximate: "tanh" }, inputs: { x: rand([4, 8], F64, 23, 6) },
    forward: ({ x }) => x!.gelu({ approximate: "tanh" }) },
  { id: "gelu (exact, f32)", kind: "gelu", dtype: F32, config: { approximate: "none" }, inputs: { x: rand([4, 8], F32, 24, 6) },
    forward: ({ x }) => x!.gelu({ approximate: "none" }) },
  { id: "div by scalar", kind: "divScalar", dtype: F64, config: { value: 3 }, inputs: { x: rand([2, 3], F64, 25) },
    forward: ({ x }) => x!.div(3) },
  { id: "cast f32 -> f64", kind: "cast", dtype: F64, config: { to: "f64" }, inputs: { x: rand([2, 3], F32, 26) },
    forward: ({ x }) => x!.cast("f64") },
);

// ---- batched matmul ----------------------------------------------------------------

cases.push(
  { id: "matmul (batched, broadcast batch axes)", kind: "matmul", dtype: F64,
    inputs: { a: rand([2, 1, 3, 4], F64, 30), b: rand([3, 4, 5], F64, 31) }, forward: ({ a, b }) => a!.matmul(b!) },
  { id: "matmul ([B, T, in] @ [in, out])", kind: "matmul", dtype: F64,
    inputs: { a: rand([2, 3, 4], F64, 32), b: rand([4, 5], F64, 33) }, forward: ({ a, b }) => a!.matmul(b!) },
);

// ---- functional attention / RoPE / GeGLU ------------------------------------------

const [q, k, v] = [rand([2, 2, 4, 8], F64, 40), rand([2, 2, 5, 8], F64, 41), rand([2, 2, 5, 6], F64, 42)];
// No row fully masked (that is NaN in PyTorch too).
const sdpaKeep = boolMask([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0], [4, 5]);
const sdpaFloat = rand([2, 1, 4, 5], F64, 43, 2);
cases.push(
  { id: "sdpa (no mask)", kind: "sdpa", dtype: F64, inputs: { q, k, v },
    forward: (i) => nn.scaledDotProductAttention(i.q!, i.k!, i.v!) },
  { id: "sdpa (bool mask, true = attend)", kind: "sdpa", dtype: F64, inputs: { q, k, v }, masks: { attnMask: sdpaKeep },
    forward: (i) => nn.scaledDotProductAttention(i.q!, i.k!, i.v!, { attnMask: sdpaKeep }) },
  { id: "sdpa (float additive mask)", kind: "sdpa", dtype: F64, inputs: { q, k, v }, masks: { attnMask: sdpaFloat },
    forward: (i) => nn.scaledDotProductAttention(i.q!, i.k!, i.v!, { attnMask: sdpaFloat }) },
  { id: "sdpa (isCausal, L != S)", kind: "sdpa", dtype: F64, config: { isCausal: true }, inputs: { q, k, v },
    forward: (i) => nn.scaledDotProductAttention(i.q!, i.k!, i.v!, { isCausal: true }) },
  { id: "sdpa (custom scale)", kind: "sdpa", dtype: F64, config: { scale: 0.3 }, inputs: { q, k, v },
    forward: (i) => nn.scaledDotProductAttention(i.q!, i.k!, i.v!, { scale: 0.3 }) },
  { id: "sdpa (f32, bool mask)", kind: "sdpa", dtype: F32, masks: { attnMask: sdpaKeep },
    inputs: { q: q.cast("f32"), k: k.cast("f32"), v: v.cast("f32") },
    forward: (i) => nn.scaledDotProductAttention(i.q!, i.k!, i.v!, { attnMask: sdpaKeep }) },
  { id: "RoPE (split-half, base 10000)", kind: "rope", dtype: F64, config: { base: 10000 }, inputs: { x: rand([2, 2, 5, 8], F64, 44) },
    forward: ({ x }) => new nn.RotaryEmbedding(8).forward(x!) },
  { id: "RoPE (base 160000, offset 3)", kind: "rope", dtype: F64, config: { base: 160000, offset: 3 }, inputs: { x: rand([3, 6, 4], F64, 45) },
    forward: ({ x }) => new nn.RotaryEmbedding(4, { base: 160000 }).forward(x!, { offset: 3 }) },
  { id: "RoPE (f32)", kind: "rope", dtype: F32, config: { base: 10000 }, inputs: { x: rand([2, 7, 16], F32, 46) },
    forward: ({ x }) => new nn.RotaryEmbedding(16).forward(x!) },
  { id: "geglu (exact)", kind: "geglu", dtype: F64, config: { approximate: "none" }, inputs: { x: rand([2, 3, 8], F64, 47, 3) },
    forward: ({ x }) => nn.geglu(x!) },
  { id: "geglu (tanh)", kind: "geglu", dtype: F64, config: { approximate: "tanh" }, inputs: { x: rand([2, 3, 8], F64, 48, 3) },
    forward: ({ x }) => nn.geglu(x!, { approximate: "tanh" }) },
);

// ---- layers --------------------------------------------------------------------------

function addLayer(c: Omit<Case, "forward"> & { forward: Case["forward"] }): void {
  cases.push(c);
}

{
  const m1 = layer(new nn.Linear(4, 5, { dtype: F64 }), 100);
  addLayer({ id: "Linear [out, in] (3-D input)", kind: "linear", dtype: F64, config: { in: 4, out: 5, bias: true },
    module: m1, inputs: { x: rand([2, 3, 4], F64, 101) }, forward: ({ x }) => m1.forward(x!) });
  const m2 = layer(new nn.Linear(4, 3, { dtype: F64, bias: false }), 102);
  addLayer({ id: "Linear (bias: false)", kind: "linear", dtype: F64, config: { in: 4, out: 3, bias: false },
    module: m2, inputs: { x: rand([5, 4], F64, 103) }, forward: ({ x }) => m2.forward(x!) });
  const m3 = layer(new nn.Linear(4, 3, { dtype: F64 }), 104);
  addLayer({ id: "Linear (1-D input)", kind: "linear", dtype: F64, config: { in: 4, out: 3, bias: true },
    module: m3, inputs: { x: rand([4], F64, 105) }, forward: ({ x }) => m3.forward(x!) });
  const m4 = layer(new nn.Linear(6, 5), 106); // default dtype: f32
  addLayer({ id: "Linear (default f32)", kind: "linear", dtype: F32, config: { in: 6, out: 5, bias: true },
    module: m4, inputs: { x: rand([2, 3, 6], F32, 107) }, forward: ({ x }) => m4.forward(x!) });
}
{
  const m1 = layer(new nn.LayerNorm(6, { dtype: F64 }), 110);
  addLayer({ id: "LayerNorm", kind: "layerNorm", dtype: F64, config: { dim: 6, eps: 1e-5, bias: true },
    module: m1, inputs: { x: rand([2, 3, 6], F64, 111, 2) }, forward: ({ x }) => m1.forward(x!) });
  const m2 = layer(new nn.LayerNorm(6, { dtype: F64, bias: false, eps: 1e-6 }), 112);
  addLayer({ id: "LayerNorm (bias: false)", kind: "layerNorm", dtype: F64, config: { dim: 6, eps: 1e-6, bias: false },
    module: m2, inputs: { x: rand([4, 6], F64, 113, 2) }, forward: ({ x }) => m2.forward(x!) });
  const m3 = layer(new nn.LayerNorm(8, { bias: false }), 114);
  addLayer({ id: "LayerNorm (bias: false, f32)", kind: "layerNorm", dtype: F32, config: { dim: 8, eps: 1e-5, bias: false },
    module: m3, inputs: { x: rand([3, 8], F32, 115, 2) }, forward: ({ x }) => m3.forward(x!) });
}
{
  const E = 8;
  const H = 2;
  const cfg = (o: Record<string, unknown>) => ({ embedDim: E, numHeads: H, bias: true, batchFirst: false, ...o });
  const a = layer(new nn.MultiheadAttention(E, H, { dtype: F64 }), 120);
  addLayer({ id: "MultiheadAttention (seq-first, self-attention)", kind: "mha", dtype: F64, config: cfg({}), module: a,
    inputs: { q: rand([4, 2, E], F64, 121) }, forward: ({ q: x }) => a.forward(x!, x!, x!) });

  const b = layer(new nn.MultiheadAttention(E, H, { dtype: F64, batchFirst: true }), 122);
  const kpm = boolMask([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [2, 5]); // batch 1: last two keys are padding
  const attnMaskTrueHides = boolMask([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [4, 5]);
  addLayer({ id: "MultiheadAttention (batch-first, cross-attn, bool attnMask + keyPaddingMask)", kind: "mha", dtype: F64,
    config: cfg({ batchFirst: true }), module: b, masks: { attnMask: attnMaskTrueHides, keyPaddingMask: kpm },
    inputs: { q: rand([2, 4, E], F64, 123), k: rand([2, 5, E], F64, 124), v: rand([2, 5, E], F64, 125) },
    forward: (i) => b.forward(i.q!, i.k!, i.v!, { attnMask: attnMaskTrueHides, keyPaddingMask: kpm }) });

  const c = layer(new nn.MultiheadAttention(E, H, { dtype: F64, bias: false }), 126);
  const floatMask3d = rand([2 * H, 3, 3], F64, 127, 2);
  addLayer({ id: "MultiheadAttention (bias: false, 3-D float attnMask)", kind: "mha", dtype: F64,
    config: cfg({ bias: false }), module: c, masks: { attnMask: floatMask3d },
    inputs: { q: rand([3, 2, E], F64, 128) }, forward: ({ q: x }) => c.forward(x!, x!, x!, { attnMask: floatMask3d }) });

  const d = layer(new nn.MultiheadAttention(E, H, { dtype: F64 }), 129);
  const kpm1 = boolMask([0, 0, 1], [3]);
  addLayer({ id: "MultiheadAttention (unbatched input, keyPaddingMask)", kind: "mha", dtype: F64, config: cfg({}), module: d,
    masks: { keyPaddingMask: kpm1 }, inputs: { q: rand([3, E], F64, 130) },
    forward: ({ q: x }) => d.forward(x!, x!, x!, { keyPaddingMask: kpm1 }) });

  const e = layer(new nn.MultiheadAttention(E, H, { batchFirst: true }), 131);
  addLayer({ id: "MultiheadAttention (f32)", kind: "mha", dtype: F32, config: cfg({ batchFirst: true }), module: e,
    inputs: { q: rand([2, 5, E], F32, 132) }, forward: ({ q: x }) => e.forward(x!, x!, x!) });

  const f = layer(
    new nn.MultiheadAttention(E, H, { dtype: F64, batchFirst: true, bias: false, rotary: new nn.RotaryEmbedding(E / H, { base: 10000 }) }),
    133,
  );
  addLayer({ id: "MultiheadAttention + rotary (ModernBERT-style)", kind: "mhaRope", dtype: F64,
    config: cfg({ batchFirst: true, bias: false, base: 10000 }), module: f,
    inputs: { q: rand([2, 5, E], F64, 134) }, forward: ({ q: x }) => f.forward(x!, x!, x!) });
}
{
  const d = 8;
  const base = { dModel: d, nhead: 2, dimFeedforward: 16, eps: 1e-5 };
  const mk = (o: nn.TransformerEncoderLayerOptions, seed: number) =>
    layer(new nn.TransformerEncoderLayer(d, 2, { dimFeedforward: 16, ...o }), seed);

  const a = mk({ dtype: F64, normFirst: true, batchFirst: true }, 140);
  const kpm = boolMask([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [2, 5]);
  addLayer({ id: "TransformerEncoderLayer (norm_first, relu, batch-first, key padding)", kind: "encoderLayer", dtype: F64,
    config: { ...base, activation: "relu", batchFirst: true, normFirst: true, bias: true }, module: a,
    masks: { srcKeyPaddingMask: kpm }, inputs: { x: rand([2, 5, d], F64, 141) },
    forward: ({ x }) => a.forward(x!, { srcKeyPaddingMask: kpm }) });

  const b = mk({ dtype: F64, activation: "gelu" }, 142);
  const causalHides = boolMask([0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0], [4, 4]);
  addLayer({ id: "TransformerEncoderLayer (post-norm, gelu, seq-first, causal srcMask)", kind: "encoderLayer", dtype: F64,
    config: { ...base, activation: "gelu", batchFirst: false, normFirst: false, bias: true }, module: b,
    masks: { srcMask: causalHides }, inputs: { x: rand([4, 2, d], F64, 143) },
    forward: ({ x }) => b.forward(x!, { srcMask: causalHides }) });

  const c = mk({ dtype: F64, normFirst: true, bias: false, batchFirst: true }, 144);
  addLayer({ id: "TransformerEncoderLayer (norm_first, bias: false)", kind: "encoderLayer", dtype: F64,
    config: { ...base, activation: "relu", batchFirst: true, normFirst: true, bias: false }, module: c,
    inputs: { x: rand([2, 3, d], F64, 145) }, forward: ({ x }) => c.forward(x!) });

  const e = mk({ normFirst: true, batchFirst: true, activation: "gelu" }, 146);
  addLayer({ id: "TransformerEncoderLayer (f32, norm_first, gelu)", kind: "encoderLayer", dtype: F32,
    config: { ...base, activation: "gelu", batchFirst: true, normFirst: true, bias: true }, module: e,
    inputs: { x: rand([2, 4, d], F32, 147) }, forward: ({ x }) => e.forward(x!) });
}
{
  const a = layer(new nn.GeGLU(6, 5, { dtype: F64 }), 150);
  addLayer({ id: "GeGLU (exact)", kind: "geGLU", dtype: F64, config: { dimIn: 6, dimOut: 5, bias: true, approximate: "none" },
    module: a, inputs: { x: rand([2, 3, 6], F64, 151, 2) }, forward: ({ x }) => a.forward(x!) });
  const b = layer(new nn.GeGLU(6, 4, { dtype: F64, bias: false, approximate: "tanh" }), 152);
  addLayer({ id: "GeGLU (tanh, bias: false)", kind: "geGLU", dtype: F64, config: { dimIn: 6, dimOut: 4, bias: false, approximate: "tanh" },
    module: b, inputs: { x: rand([3, 6], F64, 153, 2) }, forward: ({ x }) => b.forward(x!) });
  const c = layer(new nn.GeGLU(8, 4), 154);
  addLayer({ id: "GeGLU (f32)", kind: "geGLU", dtype: F32, config: { dimIn: 8, dimOut: 4, bias: true, approximate: "none" },
    module: c, inputs: { x: rand([2, 8], F32, 155, 2) }, forward: ({ x }) => c.forward(x!) });
}

const { prepared, requests } = prepareCases(cases);
const results = torchSkip ? {} : runTorchOracle(requests);

for (const [id, p] of prepared) {
  test(`PyTorch parity: ${id}`, { skip: torchSkip }, () => checkCase(p, results[id]));
}

// ---- non-oracle behavior -------------------------------------------------------------

test("maskedFill rejects a non-bool mask", () => {
  const x = Variable.variable(rand([2, 2], F64, 1));
  assert.throws(() => x.maskedFill(rand([2, 2], F64, 2), 0), /mask must be a bool tensor/);
});

test("scaledDotProductAttention rejects attnMask together with isCausal (as PyTorch does)", () => {
  const x = Variable.constant(rand([1, 2, 4], F64, 3));
  assert.throws(() => nn.scaledDotProductAttention(x, x, x, { attnMask: boolMask([1, 1, 1, 1], [2, 2]), isCausal: true }), /not both/);
});

test("MultiheadAttention: state-dict keys are PyTorch's", () => {
  const m = new nn.MultiheadAttention(8, 2);
  assert.deepEqual(Object.keys(m.stateDict()).sort(), ["in_proj_bias", "in_proj_weight", "out_proj.bias", "out_proj.weight"]);
  assert.deepEqual(Object.keys(new nn.MultiheadAttention(8, 2, { bias: false }).stateDict()).sort(), ["in_proj_weight", "out_proj.weight"]);
});

test("TransformerEncoderLayer: state-dict keys are PyTorch's (bias: false drops every bias)", () => {
  const keys = Object.keys(new nn.TransformerEncoderLayer(8, 2, { dimFeedforward: 16, bias: false }).stateDict()).sort();
  assert.deepEqual(keys, [
    "linear1.weight", "linear2.weight", "norm1.weight", "norm2.weight", "self_attn.in_proj_weight", "self_attn.out_proj.weight",
  ]);
});

test("half-precision parameter storage: f16/bf16 layers compute in the input's dtype and match an f32 copy", () => {
  for (const dtype of ["f16", "bf16"] as const) {
    const half = new nn.TransformerEncoderLayer(8, 2, { dimFeedforward: 16, dtype, normFirst: true, batchFirst: true });
    assert.equal(half.linear1.weight.dtype, dtype);
    const full = new nn.TransformerEncoderLayer(8, 2, { dimFeedforward: 16, normFirst: true, batchFirst: true });
    full.loadStateDict(half.stateDict()); // exact: every half value is representable in f32
    const x = Variable.constant(rand([2, 3, 8], F32, 5));
    assert.deepEqual(half.forward(x).value.toArray(), full.forward(x).value.toArray());
  }
});

test("half-precision parameters still receive gradients (in their storage dtype)", () => {
  const m = new nn.Linear(4, 3, { dtype: "f16" });
  m.forward(Variable.constant(rand([2, 4], F32, 6))).sum().backward();
  assert.equal(m.weight.grad?.dtype, "f16");
  assert.deepEqual([...(m.weight.grad as Tensor).shape], [3, 4]);
});

test("Embedding with f16 storage returns f32 rows", () => {
  const emb = new nn.Embedding(5, 3, { dtype: "f16" });
  const out = emb.forward(Tensor.from([0, 4], { dtype: "i32" }));
  assert.equal(out.dtype, "f32");
  assert.deepEqual(out.value.toArray(), emb.weight.value.cast("f32").take([0, 4], { axis: 0 }).toArray());
});

test("RotaryEmbedding rejects an odd dim and a mismatched input", () => {
  assert.throws(() => new nn.RotaryEmbedding(5), /must be even/);
  assert.throws(() => new nn.RotaryEmbedding(4).forward(Variable.constant(rand([2, 6], F64, 7))), /expected last axis 4/);
});
