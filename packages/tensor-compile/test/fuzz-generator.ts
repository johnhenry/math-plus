/**
 * Shared generator/shrinker machinery for the IR differential fuzzers
 * (issue #48's CPU legs in this package; issue #58's real-GPU WGSL leg in
 * @johnhenry/math-plus-tensor-webgpu imports THIS module rather than duplicating it --
 * the canonical-implementation rule). Pure relocation from fuzz.test.ts;
 * no behavior change.
 */
import assert from "node:assert/strict";
import type { BinaryOp, IRNode, UnaryOp } from "../src/index.ts";

// ---- deterministic RNG (mulberry32 -- tiny, seedable, good enough for case
// generation; NOT crypto, NOT for statistics) --------------------------------

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type Rng = () => number;

export function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length)] as T;
}

// ---- op spec tables ---------------------------------------------------------

/** Smooth in-domain: safe for finite-difference gradient checking (domain
 * membership is enforced by rejection sampling, not by the table). */
export const SMOOTH_UNARY: readonly UnaryOp[] = [
  "neg", "sigmoid", "gelu", "exp", "log", "sqrt", "sin", "cos", "tan",
  "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "atanh",
  "log10", "log2", "cbrt", "expm1", "log1p", "erf", "gelu_tanh",
];
export const SMOOTH_BINARY: readonly BinaryOp[] = ["add", "sub", "mul", "div", "pow", "atan2", "hypot"];

/** Everything, including kinked/discontinuous ops -- value-leg only. */
export const ALL_UNARY: readonly UnaryOp[] = [
  ...SMOOTH_UNARY,
  "relu", "abs", "floor", "ceil", "round", "sign", "trunc",
  "cot", "sec", "csc", "coth", "sech", "csch",
  "acot", "asec", "acsc", "acoth", "asech", "acsch", "acosh",
];
export const ALL_BINARY: readonly BinaryOp[] = [...SMOOTH_BINARY, "min", "max"];

export interface GenSpec {
  unary: readonly UnaryOp[];
  binary: readonly BinaryOp[];
  /** allow cmp + select nodes (value leg only -- their gradient is defined as 0 / branch passthrough and FD at a branch boundary is meaningless) */
  cmpSelect: boolean;
  maxDepth: number;
}

export const GRAD_SPEC: GenSpec = { unary: SMOOTH_UNARY, binary: SMOOTH_BINARY, cmpSelect: false, maxDepth: 4 };
export const VALUE_SPEC: GenSpec = { unary: ALL_UNARY, binary: ALL_BINARY, cmpSelect: true, maxDepth: 5 };

// ---- graph generation -------------------------------------------------------

export function genNode(rng: Rng, spec: GenSpec, numInputs: number, depth: number): IRNode {
  if (depth >= spec.maxDepth || rng() < 0.25) {
    return rng() < 0.7
      ? { kind: "input", index: Math.floor(rng() * numInputs) }
      : { kind: "const", value: Math.round((rng() * 4 - 2) * 100) / 100 };
  }
  const r = rng();
  if (spec.cmpSelect && r < 0.12) {
    return {
      kind: "select",
      cond: {
        kind: "cmp",
        op: pick(rng, ["lt", "le", "gt", "ge", "eq", "ne"] as const),
        left: genNode(rng, spec, numInputs, depth + 1),
        right: genNode(rng, spec, numInputs, depth + 1),
      },
      then: genNode(rng, spec, numInputs, depth + 1),
      else: genNode(rng, spec, numInputs, depth + 1),
    };
  }
  if (r < 0.55) {
    return { kind: "unary", op: pick(rng, spec.unary), arg: genNode(rng, spec, numInputs, depth + 1) };
  }
  return {
    kind: "binary",
    op: pick(rng, spec.binary),
    left: genNode(rng, spec, numInputs, depth + 1),
    right: genNode(rng, spec, numInputs, depth + 1),
  };
}

export function usesAnyInput(node: IRNode): boolean {
  switch (node.kind) {
    case "input": return true;
    case "const": return false;
    case "unary": return usesAnyInput(node.arg);
    case "binary": return usesAnyInput(node.left) || usesAnyInput(node.right);
    case "cmp": return usesAnyInput(node.left) || usesAnyInput(node.right);
    case "select": return usesAnyInput(node.cond) || usesAnyInput(node.then) || usesAnyInput(node.else);
  }
}

export function nodeCount(node: IRNode): number {
  switch (node.kind) {
    case "input":
    case "const": return 1;
    case "unary": return 1 + nodeCount(node.arg);
    case "binary":
    case "cmp": return 1 + nodeCount(node.left) + nodeCount(node.right);
    case "select": return 1 + nodeCount(node.cond) + nodeCount(node.then) + nodeCount(node.else);
  }
}

/** Every subnode of `node`, including itself (used for domain rejection: a
 * point is only accepted when EVERY intermediate stays finite and bounded,
 * which keeps composed functions inside their domains and away from poles). */
export function allSubnodes(node: IRNode): IRNode[] {
  switch (node.kind) {
    case "input":
    case "const": return [node];
    case "unary": return [node, ...allSubnodes(node.arg)];
    case "binary":
    case "cmp": return [node, ...allSubnodes(node.left), ...allSubnodes(node.right)];
    case "select": return [node, ...allSubnodes(node.cond), ...allSubnodes(node.then), ...allSubnodes(node.else)];
  }
}

// ---- size-monotone greedy shrinking -----------------------------------------

/** One-step reductions: replace any subtree with one of its own children.
 * Every candidate is strictly smaller, so the greedy loop below terminates
 * (asserted). */
export function reductions(node: IRNode): IRNode[] {
  const out: IRNode[] = [];
  switch (node.kind) {
    case "input":
    case "const":
      return out;
    case "unary":
      out.push(node.arg);
      for (const r of reductions(node.arg)) out.push({ ...node, arg: r });
      return out;
    case "binary":
    case "cmp":
      out.push(node.left, node.right);
      for (const r of reductions(node.left)) out.push({ ...node, left: r });
      for (const r of reductions(node.right)) out.push({ ...node, right: r });
      return out;
    case "select":
      out.push(node.then, node.else, node.cond);
      for (const r of reductions(node.cond)) out.push({ ...node, cond: r });
      for (const r of reductions(node.then)) out.push({ ...node, then: r });
      for (const r of reductions(node.else)) out.push({ ...node, else: r });
      return out;
  }
}

/** Greedily shrink `node` while `stillDiverges` holds. `stillDiverges` must
 * return false (not throw) for candidates it cannot confirm -- e.g. a shrunk
 * graph that now trips a domain-rejection guard. */
export function shrink(node: IRNode, stillDiverges: (candidate: IRNode) => boolean): IRNode {
  let current = node;
  for (;;) {
    const smaller = reductions(current).find((c) => stillDiverges(c));
    if (!smaller) return current;
    assert.ok(nodeCount(smaller) < nodeCount(current), "shrink must be size-monotone (termination guarantee)");
    current = smaller;
  }
}

