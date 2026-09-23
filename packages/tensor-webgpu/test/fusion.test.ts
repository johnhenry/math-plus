/**
 * The correctness oracle the issue asks for: run the SAME traced `IRNode`
 * (built once via `@johnhenry/math-plus-tensor-compile`'s `Traced`, exactly as a real
 * `compile()` call would) through TWO independent backends —
 * `evalWithGrad` on the CPU (tensor-compile's own interpreter) and
 * `compileIRToWGSL` + `runElementwiseWGSL` on a live GPUAdapter — and assert
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
import { bundleForBrowser, closeHarness, getHarness, SRC } from "./helpers.ts";

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

  const bundle = bundleForBrowser([path.join(SRC, "elementwise.ts")]);
  const inputsLiteral = inputs.map((arr) => `new Float32Array(${JSON.stringify(Array.from(arr))})`).join(", ");
  const result = await harness.run<number[]>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const node = ${JSON.stringify(node)};
    const inputs = [${inputsLiteral}];
    const out = await runElementwiseWGSL(device, node, inputs, ${elementCount});
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
    const node = Traced.input(0)[op]().node;
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
