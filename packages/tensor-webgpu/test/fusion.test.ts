/**
 * The correctness oracle the issue asks for: run the SAME traced `IRNode`
 * (built once via `@johnhenry/math-plus-tensor-compile`'s `Traced`, exactly as a real
 * `compile()` call would) through TWO independent backends —
 * `evalWithGrad` on the CPU (tensor-compile's own interpreter) and
 * `compileIRToWGSL`'s lowering run by `createWebGpuDevice().fuse` (backend-webgpu's
 * `elementwise` hook) on a live GPUAdapter — and assert
 * they agree elementwise. Same "two independently-implemented consumers of
 * one IR must agree" shape as this repo's DualNumber-vs-reverse-mode-tape
 * autograd cross-check (docs/TESTING.md), just with a GPU backend on one
 * side instead of a second CPU one.
 */
import assert from "node:assert/strict";
import path from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { evalWithGrad, Traced, type IRNode } from "@johnhenry/math-plus-tensor-compile";
import { erf, geluErf, geluTanh } from "@johnhenry/math-plus-tensor-core";
import { bundleForBrowser, closeHarness, FUSE_HOST, getHarness, SRC } from "./helpers.ts";

after(closeHarness);

function randomData(size: number, seed: number, scale = 1): Float32Array {
  let s = seed >>> 0;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = ((s / 0xffffffff) * 2 - 1) * scale;
  }
  return out;
}

/** CPU oracle: `evalWithGrad`'s `.value` at every element, for `numInputs` flat input arrays. */
function cpuForward(node: IRNode, inputs: readonly Float32Array[], elementCount: number): Float32Array {
  const out = new Float32Array(elementCount);
  const numInputs = inputs.length;
  const scratch = new Array<number>(numInputs);
  for (let i = 0; i < elementCount; i++) {
    for (let k = 0; k < numInputs; k++) scratch[k] = inputs[k]![i]!;
    out[i] = evalWithGrad(node, scratch, numInputs).value;
  }
  return out;
}

async function crossCheck(
  t: import("../../../test/harness.ts").TestContext,
  label: string,
  node: IRNode,
  inputs: readonly Float32Array[],
): Promise<void> {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const elementCount = inputs[0]!.length;
  const expected = cpuForward(node, inputs, elementCount);

  const bundle = bundleForBrowser([path.join(SRC, "facade.ts")]);
  const inputsLiteral = inputs.map((arr) => `new Float32Array(${JSON.stringify(Array.from(arr))})`).join(", ");
  const result = await harness.run<number[]>(
    `${FUSE_HOST}
    const node = ${JSON.stringify(node)};
    const inputs = [${inputsLiteral}];
    const out = await fuseHost(node, inputs);
    return Array.from(out);
    `,
    bundle,
  );

  assert.equal(result.length, expected.length, `${label}: length mismatch`);
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i] as number;
    const g = result[i] as number;
    if (Number.isNaN(e)) {
      assert.ok(Number.isNaN(g), `${label} at ${i}: CPU is NaN but GPU is ${g}`);
      continue;
    }
    const diff = Math.abs(g - e);
    // f32 GPU vs f64 JS Math — a looser tolerance than the GEMM/attention
    // tests since some ops (tan/sec/csc near their poles, inverse
    // hyperbolics near domain edges) amplify small input differences a lot;
    // inputs are kept away from those edges below specifically to avoid
    // needing an even looser bound.
    const tol = 1e-2 * Math.max(1, Math.abs(e));
    assert.ok(diff <= tol, `${label} at ${i}: GPU ${g} vs CPU ${e} (diff ${diff})`);
  }
}

test("fusion cross-check: single unary ops", async (t) => {
  const x = randomData(64, 1, 0.8); // kept in (-0.8, 0.8): safe domain for asin/acos/atanh/etc.
  const positiveX = randomData(64, 2, 0.8).map((v) => Math.abs(v) + 0.1); // safe domain for log/sqrt/acosh-ish ops
  const cases: Array<{ op: import("@johnhenry/math-plus-tensor-compile").UnaryOp; data: Float32Array }> = [
    { op: "neg", data: x },
    { op: "relu", data: x },
    { op: "sigmoid", data: x },
    { op: "gelu", data: x },
    { op: "gelu_tanh", data: x },
    { op: "exp", data: x },
    { op: "log", data: positiveX },
    { op: "sqrt", data: positiveX },
    { op: "sin", data: x },
    { op: "cos", data: x },
    { op: "tan", data: x },
    { op: "asin", data: x },
    { op: "acos", data: x },
    { op: "atan", data: x },
    { op: "sinh", data: x },
    { op: "cosh", data: x },
    { op: "tanh", data: x },
    { op: "abs", data: x },
    { op: "log2", data: positiveX },
    { op: "log10", data: positiveX },
    { op: "cbrt", data: x },
    { op: "floor", data: x },
    { op: "ceil", data: x },
    { op: "round", data: x },
    { op: "sign", data: x },
    { op: "trunc", data: x },
    { op: "expm1", data: x },
    { op: "log1p", data: positiveX },
    { op: "erf", data: x },
    { op: "asinh", data: x },
    { op: "acosh", data: positiveX.map((v) => v + 1) },
    { op: "atanh", data: x },
  ];
  for (const { op, data } of cases) {
    // Built directly (not via a Traced method) since "gelu_tanh" is reached
    // through Traced.gelu({ approximate: "tanh" }), not a same-named method.
    const node: IRNode = { kind: "unary", op, arg: { kind: "input", index: 0 } };
    await crossCheck(t, op, node, [data]);
  }
});

test("fusion cross-check: binary ops", async (t) => {
  const a = randomData(64, 3, 3);
  const b = randomData(64, 4, 3).map((v) => (v === 0 ? 0.5 : v)); // avoid exact zero divisor
  const positiveA = randomData(64, 5, 2).map((v) => Math.abs(v) + 0.1);
  const cases: Array<{ op: import("@johnhenry/math-plus-tensor-compile").BinaryOp; l: Float32Array; r: Float32Array }> = [
    { op: "add", l: a, r: b },
    { op: "sub", l: a, r: b },
    { op: "mul", l: a, r: b },
    { op: "div", l: a, r: b },
    { op: "pow", l: positiveA, r: randomData(64, 6, 2) },
    { op: "atan2", l: a, r: b },
    { op: "hypot", l: a, r: b },
    { op: "min", l: a, r: b },
    { op: "max", l: a, r: b },
  ];
  for (const { op, l, r } of cases) {
    const node = Traced.input(0)[op](Traced.input(1)).node;
    await crossCheck(t, op, node, [l, r]);
  }
});

test("fusion cross-check: a realistic fused chain (add -> mul -> sigmoid -> relu)", async (t) => {
  const a = randomData(256, 7, 2);
  const b = randomData(256, 8, 2);
  const c = randomData(256, 9, 2);
  const expr = Traced.input(0).add(Traced.input(1)).mul(Traced.input(2)).sigmoid().relu();
  await crossCheck(t, "fused chain", expr.node, [a, b, c]);
});

test("fusion cross-check: select() (piecewise) matches CPU short-circuit semantics", async (t) => {
  const cond = randomData(64, 10, 1); // some positive, some negative/zero
  const thenVals = randomData(64, 11, 5);
  const elseVals = randomData(64, 12, 5);
  const expr = Traced.input(0).select(Traced.input(1), Traced.input(2));
  await crossCheck(t, "select", expr.node, [cond, thenVals, elseVals]);
});

test("fusion cross-check: cmp ops produce 0.0/1.0 matching CPU", async (t) => {
  const a = randomData(64, 13, 3);
  const b = randomData(64, 14, 3);
  const cases: readonly import("@johnhenry/math-plus-tensor-compile").CmpOp[] = ["lt", "le", "gt", "ge", "eq", "ne"];
  for (const op of cases) {
    const expr = Traced.input(0).cmp(op, Traced.input(1));
    await crossCheck(t, `cmp ${op}`, expr.node, [a, b]);
  }
});

/**
 * Issue #122: the WGSL erf/erfc/GELU are an f32 lowering of tensor-core's
 * canonical f64 erf (@johnhenry/math-plus-special), so the GPU result is checked against
 * that canonical implementation directly, over the full range the f32
 * lowering claims ([-6, 6] for erf, both tails for GELU) and at a much tighter
 * bound than the generic 1e-2 cross-check above. The bound is f32 rounding
 * (inputs are exactly representable f32 values, the reference is f64) plus
 * WGSL's loosely-specified `exp` — see ERF_WGSL_FN's doc comment.
 */
test("fusion: WGSL erf / exact gelu / tanh gelu track the canonical f64 implementation (#122)", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }
  const n = 481;
  const xs = new Float32Array(n);
  for (let i = 0; i < n; i++) xs[i] = -6 + (12 * i) / (n - 1); // [-6, 6], includes both series/CF regions and the |x| = 1 seam
  const bundle = bundleForBrowser([path.join(SRC, "facade.ts")]);
  const run = async (op: "erf" | "gelu" | "gelu_tanh" | "tanh", data: Float32Array): Promise<number[]> =>
    harness.run<number[]>(
      `${FUSE_HOST}
      const node = { kind: "unary", op: ${JSON.stringify(op)}, arg: { kind: "input", index: 0 } };
      return Array.from(await fuseHost(node, [new Float32Array(${JSON.stringify(Array.from(data))})]));
      `,
      bundle,
    );

  const erfGpu = await run("erf", xs);
  let worstErf = 0;
  for (let i = 0; i < n; i++) worstErf = Math.max(worstErf, Math.abs(erfGpu[i]! - erf(xs[i]!)));
  assert.ok(worstErf < 1e-6, `WGSL erf worst absolute error ${worstErf} over [-6, 6]`);

  // GELU over [-12, 12]: relative bound (exact GELU's left tail is tiny but
  // must not collapse to 0 early -- the whole point of computing it via erfc).
  const gs = xs.map((v) => v * 2);
  const geluGpu = await run("gelu", gs);
  const geluTanhGpu = await run("gelu_tanh", gs);
  let worstGelu = 0;
  let worstGeluTanh = 0;
  for (let i = 0; i < n; i++) {
    const x = gs[i]!;
    const exact = geluErf(x);
    worstGelu = Math.max(worstGelu, Math.abs(geluGpu[i]! - exact) / Math.max(Math.abs(exact), 1e-30));
    worstGeluTanh = Math.max(worstGeluTanh, Math.abs(geluTanhGpu[i]! - geluTanh(x)) / Math.max(1, Math.abs(geluTanh(x))));
  }
  assert.ok(worstGelu < 1e-4, `WGSL exact gelu worst relative error ${worstGelu} over [-12, 12]`);
  assert.ok(worstGeluTanh < 1e-5, `WGSL tanh gelu worst error ${worstGeluTanh} over [-12, 12]`);
  // Regression: Metal (via Dawn) computes tanh through exp and returned NaN
  // once that overflowed (|x| > ~44); the generated WGSL clamps to ±15.
  const big = Float32Array.from([-1e4, -100, -45, -15, 0, 15, 45, 100, 1e4]);
  const tanhGpu = await run("tanh", big);
  big.forEach((v, i) => assert.equal(tanhGpu[i], Math.tanh(v) === 0 ? 0 : Math.fround(Math.tanh(v)), `WGSL tanh(${v})`));
  t.diagnostic(`WGSL erf max abs err ${worstErf.toExponential(2)}; exact gelu max rel err ${worstGelu.toExponential(2)}; tanh gelu ${worstGeluTanh.toExponential(2)}`);
});
