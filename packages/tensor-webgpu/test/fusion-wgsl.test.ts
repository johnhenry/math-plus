/**
 * Pure logic tests for `compileIRToWGSL` (no GPU needed): shape of the
 * generated shader (binding count, entry point name), and that every
 * `UnaryOp`/`BinaryOp`/`CmpOp` the IR defines actually lowers to SOME WGSL
 * text (i.e. `compileIRToWGSL` never silently falls through for an op this
 * package is supposed to cover). The real correctness oracle — comparing
 * against `evalWithGrad`'s CPU value on a live GPUAdapter — is in
 * fusion.test.ts.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { Traced, type BinaryOp, type CmpOp, type UnaryOp } from "@johnhenry/math-plus-tensor-compile";
import { ERF_F32_PARAMS, ERF_SERIES_CUTOFF } from "@johnhenry/math-plus-tensor-core";
import { compileIRToWGSL } from "../src/fusion-wgsl.ts";

const ALL_UNARY: readonly UnaryOp[] = [
  "neg", "relu", "sigmoid", "gelu", "exp", "log", "sqrt", "sin", "cos", "tan",
  "asin", "acos", "atan", "sinh", "cosh", "tanh", "cot", "sec", "csc",
  "asinh", "acosh", "atanh", "coth", "sech", "csch", "acot", "asec", "acsc",
  "acoth", "asech", "acsch", "abs", "log10", "log2", "cbrt", "floor", "ceil",
  "round", "sign", "trunc", "expm1", "log1p", "erf", "gelu_tanh",
];

const ALL_BINARY: readonly BinaryOp[] = ["add", "sub", "mul", "div", "pow", "atan2", "hypot", "min", "max"];
const ALL_CMP: readonly CmpOp[] = ["lt", "le", "gt", "ge", "eq", "ne"];

test("compileIRToWGSL: every UnaryOp lowers to non-empty WGSL that references the input", () => {
  for (const op of ALL_UNARY) {
    const node = { kind: "unary" as const, op, arg: { kind: "input" as const, index: 0 } };
    const { code } = compileIRToWGSL(node, 1);
    assert.match(code, /fn main/, `${op}: missing entry point`);
    assert.match(code, /input0\[gid\.x\]/, `${op}: does not reference its input`);
    assert.equal((code.match(/@group\(0\) @binding\(0\)/g) ?? []).length, 1, `${op}: wrong binding count`);
  }
});

test("compileIRToWGSL: every BinaryOp lowers to non-empty WGSL referencing both inputs", () => {
  for (const op of ALL_BINARY) {
    const node = {
      kind: "binary" as const,
      op,
      left: { kind: "input" as const, index: 0 },
      right: { kind: "input" as const, index: 1 },
    };
    const { code, numInputs, outputBinding } = compileIRToWGSL(node, 2);
    assert.match(code, /input0\[gid\.x\]/, `${op}: missing left input`);
    assert.match(code, /input1\[gid\.x\]/, `${op}: missing right input`);
    assert.equal(numInputs, 2);
    assert.equal(outputBinding, 2);
  }
});

test("compileIRToWGSL: every CmpOp lowers using select()", () => {
  for (const op of ALL_CMP) {
    const node = {
      kind: "cmp" as const,
      op,
      left: { kind: "input" as const, index: 0 },
      right: { kind: "const" as const, value: 1 },
    };
    const { code } = compileIRToWGSL(node, 1);
    assert.match(code, /select\(0\.0, 1\.0,/, `${op}: expected a select() bool->f32 conversion`);
  }
});

test("compileIRToWGSL: select node lowers using WGSL select() with the cond compared against 0.0", () => {
  const node = {
    kind: "select" as const,
    cond: { kind: "input" as const, index: 0 },
    then: { kind: "input" as const, index: 1 },
    else: { kind: "const" as const, value: -1 },
  };
  const { code } = compileIRToWGSL(node, 2);
  assert.match(code, /select\(.*!= 0\.0\)/s);
});

test("compileIRToWGSL: emits the erf helper functions only when erf or exact gelu is used", () => {
  const withErf = compileIRToWGSL({ kind: "unary", op: "erf", arg: { kind: "input", index: 0 } }, 1);
  assert.match(withErf.code, /fn math_plus_erf/);

  // Exact gelu (the default since #122) lowers through math_plus_erfc; the
  // tanh approximation needs no helper.
  const withGelu = compileIRToWGSL({ kind: "unary", op: "gelu", arg: { kind: "input", index: 0 } }, 1);
  assert.match(withGelu.code, /fn math_plus_gelu/);
  assert.match(withGelu.code, /math_plus_gelu\(input0\[gid\.x\]\)/);
  const withGeluTanh = compileIRToWGSL({ kind: "unary", op: "gelu_tanh", arg: { kind: "input", index: 0 } }, 1);
  assert.doesNotMatch(withGeluTanh.code, /fn math_plus_erf/);
  assert.match(withGeluTanh.code, /tanh\(/);

  const withoutErf = compileIRToWGSL({ kind: "unary", op: "neg", arg: { kind: "input", index: 0 } }, 1);
  assert.doesNotMatch(withoutErf.code, /fn math_plus_erf/);
});

test("compileIRToWGSL: traced expression built via Traced (the same API CompiledFn uses) lowers without throwing", () => {
  const a = Traced.input(0);
  const b = Traced.input(1);
  const expr = a.add(b).mul(a.sigmoid()).relu();
  const { code } = compileIRToWGSL(expr.node, 2);
  assert.match(code, /fn main/);
});

test("compileIRToWGSL: an input the expression never references is still statically used (phony assignment) — the #58 fuzzer's silent-zeros finding", () => {
  // Passthrough of input 0 with TWO declared inputs: before the fix, input1
  // was absent from the shader body, "auto" layout dropped its binding,
  // createBindGroup failed async validation, and the dispatch silently
  // returned zeros.
  const node = { kind: "input" as const, index: 0 };
  const { code } = compileIRToWGSL(node, 2);
  assert.match(code, /_ = input0\[0\];/);
  assert.match(code, /_ = input1\[0\];/);
});

test("compileIRToWGSL: the WGSL erf is lowered from tensor-core's canonical erf parameters, not a separate approximation (#122)", () => {
  const { code } = compileIRToWGSL({ kind: "unary", op: "erf", arg: { kind: "input", index: 0 } }, 1);
  // Loop bounds come straight from ERF_F32_PARAMS, the region split from ERF_SERIES_CUTOFF.
  assert.ok(code.includes(`n <= ${ERF_F32_PARAMS.seriesTerms};`), "series term count not taken from ERF_F32_PARAMS");
  assert.ok(code.includes(`var n: i32 = ${ERF_F32_PARAMS.cfDepth};`), "continued-fraction depth not taken from ERF_F32_PARAMS");
  assert.ok(code.includes(`if (ax < ${ERF_SERIES_CUTOFF}.0)`), "series/CF cutoff not taken from ERF_SERIES_CUTOFF");
  // The Abramowitz & Stegun 7.1.26 coefficients it replaced must be gone.
  assert.doesNotMatch(code, /0\.3275911|1\.061405429/);
});
