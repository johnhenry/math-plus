/**
 * Direct unit coverage for `evalValue` (issue #99) — the value-only sibling
 * of `evalWithGrad` that `CompiledFn.forward()` now uses instead of
 * discarding a computed-but-unused gradient array. `compile.test.ts`,
 * `ir-extended.test.ts`, and `fuzz.test.ts` already exercise `evalValue`
 * indirectly (every `.forward()` call routes through it), but this file
 * checks it directly against `evalWithGrad`'s `.value` — one IRNode tree per
 * node kind/op, built by hand rather than through `Traced`/`compile()`, so a
 * future op added to only one of the two evaluators (or a copy-paste
 * mistake between them) fails here instead of only showing up as a subtle
 * forward-vs-backward-vs-forward-again value mismatch elsewhere.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { evalValue, evalWithGrad, type BinaryOp, type CmpOp, type IRNode, type UnaryOp } from "../src/index.ts";

function agree(node: IRNode, inputs: readonly number[], numInputs: number, label: string): void {
  const viaGrad = evalWithGrad(node, inputs, numInputs).value;
  const viaValue = evalValue(node, inputs);
  assert.ok(
    Number.isNaN(viaGrad) ? Number.isNaN(viaValue) : viaGrad === viaValue,
    `${label}: evalWithGrad=${viaGrad}, evalValue=${viaValue}`,
  );
}

test("evalValue agrees with evalWithGrad's .value: input/const", () => {
  agree({ kind: "input", index: 0 }, [3.5, -2], 2, "input(0)");
  agree({ kind: "input", index: 1 }, [3.5, -2], 2, "input(1)");
  agree({ kind: "const", value: 42 }, [], 0, "const");
});

test("evalValue agrees with evalWithGrad's .value: every UnaryOp", () => {
  const unaryOps: UnaryOp[] = [
    "neg", "relu", "sigmoid", "gelu", "exp", "log", "sqrt", "sin", "cos", "tan",
    "asin", "acos", "atan", "sinh", "cosh", "tanh", "cot", "sec", "csc",
    "asinh", "acosh", "atanh", "coth", "sech", "csch", "acot", "asec", "acsc",
    "acoth", "asech", "acsch", "abs", "log10", "log2", "cbrt", "floor",
    "ceil", "round", "sign", "trunc", "expm1", "log1p", "erf", "gelu_tanh",
  ];
  // A domain-safe-ish positive value works for every op above (acosh/asech
  // need x >= 1 / 0 < x <= 1 respectively; the two probes below cover both).
  for (const op of unaryOps) {
    for (const x of [0.5, 1.25]) {
      agree({ kind: "unary", op, arg: { kind: "const", value: x } }, [], 0, `${op}(${x})`);
    }
  }
});

test("evalValue agrees with evalWithGrad's .value: every BinaryOp", () => {
  const binaryOps: BinaryOp[] = ["add", "sub", "mul", "div", "pow", "atan2", "hypot", "min", "max"];
  const l: IRNode = { kind: "const", value: 3 };
  const r: IRNode = { kind: "const", value: 1.5 };
  for (const op of binaryOps) {
    agree({ kind: "binary", op, left: l, right: r }, [], 0, op);
  }
});

test("evalValue agrees with evalWithGrad's .value: every CmpOp, both outcomes", () => {
  const cmpOps: CmpOp[] = ["lt", "le", "gt", "ge", "eq", "ne"];
  for (const op of cmpOps) {
    for (const [a, b] of [
      [1, 2],
      [2, 1],
      [1, 1],
    ] as const) {
      agree(
        { kind: "cmp", op, left: { kind: "const", value: a }, right: { kind: "const", value: b } },
        [],
        0,
        `${a} ${op} ${b}`,
      );
    }
  }
});

test("evalValue short-circuits select exactly like evalWithGrad: an untaken branch's domain error never surfaces", () => {
  // else-branch would be sqrt(-1) = NaN if evaluated; cond is truthy so it must not be.
  const node: IRNode = {
    kind: "select",
    cond: { kind: "const", value: 1 },
    then: { kind: "const", value: 7 },
    else: { kind: "unary", op: "sqrt", arg: { kind: "const", value: -1 } },
  };
  assert.equal(evalValue(node, []), 7);
  assert.equal(evalWithGrad(node, [], 0).value, 7);

  const flipped: IRNode = { ...node, cond: { kind: "const", value: 0 } };
  // Now the (safe) then-branch is untaken and the unsafe one is taken —
  // both evaluators should agree it's NaN, not throw.
  assert.ok(Number.isNaN(evalValue(flipped, [])));
  assert.ok(Number.isNaN(evalWithGrad(flipped, [], 0).value));
});
