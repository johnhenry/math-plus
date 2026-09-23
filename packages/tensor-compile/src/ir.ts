/**
 * A small, closed, typed IR (issue #11, extended for #15) — elementwise/
 * broadcast nodes only. Deliberately NOT a general compiler: the node kind
 * set is fixed, so both the forward evaluator and its derivative are total
 * functions over a closed enum. This is the shared lowering target the
 * future WebGPU kernel DSL (#12) reuses, and the target the Symbolic bridge
 * (#15, `@johnhenry/math-plus-adapter-math`'s `compileExpr`) compiles @johnhenry/math's
 * `Expr` AST down into — hence the unary op set matching @johnhenry/math's
 * `FuncName` 1:1 (`ln` -> `log` is the one rename).
 */

import {
  checkGeluApproximate,
  erf,
  geluDerivative,
  geluErf,
  geluTanh,
  type GeluApproximate,
} from "@johnhenry/math-plus-tensor-core";

export type UnaryOp =
  | "neg"
  | "relu"
  | "sigmoid"
  | "gelu"
  | "gelu_tanh"
  | "exp"
  | "log"
  | "sqrt"
  | "sin"
  | "cos"
  | "tan"
  | "asin"
  | "acos"
  | "atan"
  | "sinh"
  | "cosh"
  | "tanh"
  | "cot"
  | "sec"
  | "csc"
  | "asinh"
  | "acosh"
  | "atanh"
  | "coth"
  | "sech"
  | "csch"
  | "acot"
  | "asec"
  | "acsc"
  | "acoth"
  | "asech"
  | "acsch"
  | "abs"
  | "log10"
  | "log2"
  | "cbrt"
  | "floor"
  | "ceil"
  | "round"
  | "sign"
  | "trunc"
  | "expm1"
  | "log1p"
  | "erf";

export type BinaryOp = "add" | "sub" | "mul" | "div" | "pow" | "atan2" | "hypot" | "min" | "max";

/** Mirrors @johnhenry/math's `CmpOp` structurally (not imported — tensor-compile stays dependency-free of @johnhenry/math; adapter-math's `compileExpr` maps one onto the other by matching literal names). */
export type CmpOp = "lt" | "le" | "gt" | "ge" | "eq" | "ne";

export type IRNode =
  | { kind: "input"; index: number }
  | { kind: "const"; value: number }
  | { kind: "unary"; op: UnaryOp; arg: IRNode }
  | { kind: "binary"; op: BinaryOp; left: IRNode; right: IRNode }
  | { kind: "cmp"; op: CmpOp; left: IRNode; right: IRNode }
  | { kind: "select"; cond: IRNode; then: IRNode; else: IRNode };

/**
 * Forward value AND per-input local derivative, computed together in one
 * recursive pass (forward-mode AD over the IR) — a single source of truth
 * for the math, rather than a separate forward evaluator and backward
 * evaluator that could drift apart. `grad[k]` is d(this node)/d(input k).
 */
export interface ValueAndGrad {
  value: number;
  grad: number[];
}

function zeros(n: number): number[] {
  return new Array(n).fill(0);
}

/** value/derivative pairs for every UnaryOp — d(value)/d(a.value), evaluated at a.value. Derivatives cross-checked against @johnhenry/math's Symbolic.differentiate's "func" case (same formulas, written numerically instead of symbolically). */
function unaryValueAndDeriv(op: UnaryOp, x: number): { value: number; deriv: number } {
  switch (op) {
    case "neg":
      return { value: -x, deriv: -1 };
    case "relu":
      return { value: x > 0 ? x : 0, deriv: x > 0 ? 1 : 0 };
    case "sigmoid": {
      const s = 1 / (1 + Math.exp(-x));
      return { value: s, deriv: s * (1 - s) };
    }
    // "gelu" is EXACT erf-GELU (Tensor.gelu()'s default since #122);
    // "gelu_tanh" is the tanh approximation (Tensor.gelu({ approximate: "tanh" })).
    // Both value and derivative come from tensor-core's canonical src/special.ts.
    case "gelu":
      return { value: geluErf(x), deriv: geluDerivative(x, "none") };
    case "gelu_tanh":
      return { value: geluTanh(x), deriv: geluDerivative(x, "tanh") };
    case "exp": {
      const value = Math.exp(x);
      return { value, deriv: value };
    }
    case "log":
      return { value: Math.log(x), deriv: 1 / x };
    case "sqrt": {
      const value = Math.sqrt(x);
      return { value, deriv: 1 / (2 * value) };
    }
    case "sin":
      return { value: Math.sin(x), deriv: Math.cos(x) };
    case "cos":
      return { value: Math.cos(x), deriv: -Math.sin(x) };
    case "tan": {
      const value = Math.tan(x);
      return { value, deriv: 1 + value * value };
    }
    case "asin":
      return { value: Math.asin(x), deriv: 1 / Math.sqrt(1 - x * x) };
    case "acos":
      return { value: Math.acos(x), deriv: -1 / Math.sqrt(1 - x * x) };
    case "atan":
      return { value: Math.atan(x), deriv: 1 / (1 + x * x) };
    case "sinh":
      return { value: Math.sinh(x), deriv: Math.cosh(x) };
    case "cosh":
      return { value: Math.cosh(x), deriv: Math.sinh(x) };
    case "tanh": {
      const value = Math.tanh(x);
      return { value, deriv: 1 - value * value };
    }
    case "cot": {
      const value = 1 / Math.tan(x);
      return { value, deriv: -(1 + value * value) };
    }
    case "sec": {
      const value = 1 / Math.cos(x);
      return { value, deriv: value * Math.tan(x) };
    }
    case "csc": {
      const value = 1 / Math.sin(x);
      return { value, deriv: -value / Math.tan(x) };
    }
    case "asinh":
      return { value: Math.asinh(x), deriv: 1 / Math.sqrt(x * x + 1) };
    case "acosh":
      return { value: Math.acosh(x), deriv: 1 / Math.sqrt(x * x - 1) };
    case "atanh":
      return { value: Math.atanh(x), deriv: 1 / (1 - x * x) };
    case "coth":
      return { value: 1 / Math.tanh(x), deriv: -1 / (Math.sinh(x) * Math.sinh(x)) };
    case "sech": {
      const value = 1 / Math.cosh(x);
      return { value, deriv: -value * Math.tanh(x) };
    }
    case "csch": {
      const value = 1 / Math.sinh(x);
      return { value, deriv: -value / Math.tanh(x) };
    }
    case "acot":
      return { value: Math.atan(1 / x), deriv: -1 / (1 + x * x) };
    case "asec":
      return { value: Math.acos(1 / x), deriv: 1 / (Math.abs(x) * Math.sqrt(x * x - 1)) };
    case "acsc":
      return { value: Math.asin(1 / x), deriv: -1 / (Math.abs(x) * Math.sqrt(x * x - 1)) };
    case "acoth":
      return { value: 0.5 * Math.log((x + 1) / (x - 1)), deriv: 1 / (1 - x * x) };
    case "asech":
      return { value: Math.acosh(1 / x), deriv: -1 / (x * Math.sqrt(1 - x * x)) };
    case "acsch":
      return { value: Math.asinh(1 / x), deriv: -1 / (Math.abs(x) * Math.sqrt(1 + x * x)) };
    case "abs":
      return { value: Math.abs(x), deriv: Math.sign(x) };
    case "log10":
      return { value: Math.log10(x), deriv: 1 / (x * Math.LN10) };
    case "log2":
      return { value: Math.log2(x), deriv: 1 / (x * Math.LN2) };
    case "cbrt": {
      const value = Math.cbrt(x);
      return { value, deriv: 1 / (3 * value * value) };
    }
    case "floor":
      return { value: Math.floor(x), deriv: 0 };
    case "ceil":
      return { value: Math.ceil(x), deriv: 0 };
    case "round":
      return { value: Math.round(x), deriv: 0 };
    case "sign":
      return { value: Math.sign(x), deriv: 0 };
    case "trunc":
      return { value: Math.trunc(x), deriv: 0 };
    case "expm1": {
      const value = Math.expm1(x);
      return { value, deriv: Math.exp(x) };
    }
    case "log1p":
      return { value: Math.log1p(x), deriv: 1 / (1 + x) };
    // Canonical double-precision erf (tensor-core src/special.ts, #122) —
    // replaced the Abramowitz & Stegun 7.1.26 copy (~1.5e-7 absolute) that lived here.
    case "erf":
      return { value: erf(x), deriv: (2 / Math.sqrt(Math.PI)) * Math.exp(-x * x) };
  }
}

/** Shared with {@link evalValue} so the two evaluators can't drift on what counts as "true" for a comparison. */
const CMP_OPS: Record<CmpOp, (a: number, b: number) => boolean> = {
  lt: (a, b) => a < b,
  le: (a, b) => a <= b,
  gt: (a, b) => a > b,
  ge: (a, b) => a >= b,
  eq: (a, b) => a === b,
  ne: (a, b) => a !== b,
};

export function evalWithGrad(node: IRNode, inputs: readonly number[], numInputs: number): ValueAndGrad {
  switch (node.kind) {
    case "input": {
      const grad = zeros(numInputs);
      grad[node.index] = 1;
      return { value: inputs[node.index] as number, grad };
    }
    case "const":
      return { value: node.value, grad: zeros(numInputs) };
    case "unary": {
      const a = evalWithGrad(node.arg, inputs, numInputs);
      const { value, deriv } = unaryValueAndDeriv(node.op, a.value);
      return { value, grad: a.grad.map((g) => g * deriv) };
    }
    case "binary": {
      const l = evalWithGrad(node.left, inputs, numInputs);
      const r = evalWithGrad(node.right, inputs, numInputs);
      let value: number;
      let grad: number[];
      switch (node.op) {
        case "add":
          value = l.value + r.value;
          grad = l.grad.map((g, i) => g + (r.grad[i] as number));
          break;
        case "sub":
          value = l.value - r.value;
          grad = l.grad.map((g, i) => g - (r.grad[i] as number));
          break;
        case "mul":
          value = l.value * r.value;
          grad = l.grad.map((g, i) => g * r.value + (r.grad[i] as number) * l.value);
          break;
        case "div":
          value = l.value / r.value;
          grad = l.grad.map(
            (g, i) => (g * r.value - l.value * (r.grad[i] as number)) / (r.value * r.value),
          );
          break;
        case "pow": {
          value = Math.pow(l.value, r.value);
          const dl = r.value * Math.pow(l.value, r.value - 1);
          // ln(l.value) is only finite for l.value > 0; guarded per-element below
          // so a constant exponent (r.grad[i] === 0 everywhere) never lets a
          // NaN from ln(negative) leak in via 0 * NaN.
          const lnL = l.value > 0 ? Math.log(l.value) : NaN;
          grad = l.grad.map((g, i) => {
            const rg = r.grad[i] as number;
            const fromL = g * dl;
            const fromR = rg !== 0 ? rg * value * lnL : 0;
            return fromL + fromR;
          });
          break;
        }
        case "atan2": {
          // call2("atan2", left, right) matches @johnhenry/math's convention: Math.atan2(left, right).
          value = Math.atan2(l.value, r.value);
          const denom = l.value * l.value + r.value * r.value;
          const dl = r.value / denom;
          const dr = -l.value / denom;
          grad = l.grad.map((g, i) => g * dl + (r.grad[i] as number) * dr);
          break;
        }
        case "hypot": {
          value = Math.hypot(l.value, r.value);
          const dl = l.value / value;
          const dr = r.value / value;
          grad = l.grad.map((g, i) => g * dl + (r.grad[i] as number) * dr);
          break;
        }
        case "min": {
          const lSmaller = l.value <= r.value;
          value = lSmaller ? l.value : r.value;
          grad = lSmaller ? l.grad : r.grad;
          break;
        }
        case "max": {
          const lBigger = l.value >= r.value;
          value = lBigger ? l.value : r.value;
          grad = lBigger ? l.grad : r.grad;
          break;
        }
      }
      return { value, grad };
    }
    case "cmp": {
      const l = evalWithGrad(node.left, inputs, numInputs);
      const r = evalWithGrad(node.right, inputs, numInputs);
      // Locally constant almost everywhere: gradient is 0, same convention as floor/sign/etc.
      return { value: CMP_OPS[node.op](l.value, r.value) ? 1 : 0, grad: zeros(numInputs) };
    }
    case "select": {
      // Short-circuits: only the taken branch is evaluated, so (unlike a
      // naive elementwise Tensor.where, which always evaluates both sides)
      // an untaken branch's domain errors (e.g. sqrt of a negative in the
      // "else" of a piecewise) never produce a stray NaN.
      const cond = evalWithGrad(node.cond, inputs, numInputs);
      return cond.value !== 0
        ? evalWithGrad(node.then, inputs, numInputs)
        : evalWithGrad(node.else, inputs, numInputs);
    }
  }
}

/**
 * Forward value ONLY — same recursive structure as {@link evalWithGrad}
 * (and built from the exact same per-op formulas: `unaryValueAndDeriv` and
 * `CMP_OPS`), but never allocates or propagates a `grad` array. Gradient
 * bookkeeping is `O(numInputs)` extra work *per node* in `evalWithGrad`
 * (a fresh array plus a `.map()` at every "input"/"const"/binary/cmp node);
 * over a `CompiledFn.forward()` call — many elements x several IR nodes,
 * every one of which discarded the gradient it just computed — that dwarfs
 * the actual arithmetic (measured 571ms vs 37ms, ~15x, for a 6-node/
 * 3-input/200k-element compiled function; see issue #99). Callers that only
 * want the forward value (`CompiledFn.forward()`) should use this instead;
 * `evalWithGrad` remains the entry point for anything that also needs
 * gradients (`CompiledFn.forwardWithGrad()` / `asVariableOp()`).
 */
export function evalValue(node: IRNode, inputs: readonly number[]): number {
  switch (node.kind) {
    case "input":
      return inputs[node.index] as number;
    case "const":
      return node.value;
    case "unary":
      return unaryValueAndDeriv(node.op, evalValue(node.arg, inputs)).value;
    case "binary": {
      const l = evalValue(node.left, inputs);
      const r = evalValue(node.right, inputs);
      switch (node.op) {
        case "add":
          return l + r;
        case "sub":
          return l - r;
        case "mul":
          return l * r;
        case "div":
          return l / r;
        case "pow":
          return Math.pow(l, r);
        case "atan2":
          return Math.atan2(l, r);
        case "hypot":
          return Math.hypot(l, r);
        case "min":
          return l <= r ? l : r;
        case "max":
          return l >= r ? l : r;
      }
    }
    case "cmp": {
      const l = evalValue(node.left, inputs);
      const r = evalValue(node.right, inputs);
      return CMP_OPS[node.op](l, r) ? 1 : 0;
    }
    case "select": {
      // Same short-circuit as evalWithGrad's "select" case.
      const cond = evalValue(node.cond, inputs);
      return cond !== 0 ? evalValue(node.then, inputs) : evalValue(node.else, inputs);
    }
  }
}

/** Ergonomic IR-graph builder — mirrors Variable's method style, but builds a node tree instead of executing. */
export class Traced {
  readonly node: IRNode;

  constructor(node: IRNode) {
    this.node = node;
  }

  static input(index: number): Traced {
    return new Traced({ kind: "input", index });
  }

  #unary(op: UnaryOp): Traced {
    return new Traced({ kind: "unary", op, arg: this.node });
  }

  #binary(op: BinaryOp, other: Traced | number): Traced {
    return new Traced({ kind: "binary", op, left: this.node, right: toNode(other) });
  }

  add(other: Traced | number): Traced {
    return this.#binary("add", other);
  }
  sub(other: Traced | number): Traced {
    return this.#binary("sub", other);
  }
  mul(other: Traced | number): Traced {
    return this.#binary("mul", other);
  }
  div(other: Traced | number): Traced {
    return this.#binary("div", other);
  }
  pow(other: Traced | number): Traced {
    return this.#binary("pow", other);
  }
  atan2(other: Traced | number): Traced {
    return this.#binary("atan2", other);
  }
  hypot(other: Traced | number): Traced {
    return this.#binary("hypot", other);
  }
  min(other: Traced | number): Traced {
    return this.#binary("min", other);
  }
  max(other: Traced | number): Traced {
    return this.#binary("max", other);
  }

  cmp(op: CmpOp, other: Traced | number): Traced {
    return new Traced({ kind: "cmp", op, left: this.node, right: toNode(other) });
  }
  lt(other: Traced | number): Traced {
    return this.cmp("lt", other);
  }
  le(other: Traced | number): Traced {
    return this.cmp("le", other);
  }
  gt(other: Traced | number): Traced {
    return this.cmp("gt", other);
  }
  ge(other: Traced | number): Traced {
    return this.cmp("ge", other);
  }
  eq(other: Traced | number): Traced {
    return this.cmp("eq", other);
  }
  ne(other: Traced | number): Traced {
    return this.cmp("ne", other);
  }

  /** `this` is the condition (nonzero -> `then`, zero -> `else`) — mirrors `Tensor.where`'s argument order but short-circuits (see `evalWithGrad`'s "select" case). */
  select(then: Traced | number, elseValue: Traced | number): Traced {
    return new Traced({ kind: "select", cond: this.node, then: toNode(then), else: toNode(elseValue) });
  }

  neg(): Traced {
    return this.#unary("neg");
  }
  relu(): Traced {
    return this.#unary("relu");
  }
  sigmoid(): Traced {
    return this.#unary("sigmoid");
  }
  /**
   * GELU; `approximate` mirrors `Tensor.gelu()` / PyTorch: `"none"` (default,
   * exact erf-GELU, IR op `"gelu"`) or `"tanh"` (IR op `"gelu_tanh"`).
   */
  gelu(options: { approximate?: GeluApproximate } = {}): Traced {
    const approximate = checkGeluApproximate(options.approximate);
    return this.#unary(approximate === "tanh" ? "gelu_tanh" : "gelu");
  }
  exp(): Traced {
    return this.#unary("exp");
  }
  log(): Traced {
    return this.#unary("log");
  }
  sqrt(): Traced {
    return this.#unary("sqrt");
  }
  sin(): Traced {
    return this.#unary("sin");
  }
  cos(): Traced {
    return this.#unary("cos");
  }
  tan(): Traced {
    return this.#unary("tan");
  }
  asin(): Traced {
    return this.#unary("asin");
  }
  acos(): Traced {
    return this.#unary("acos");
  }
  atan(): Traced {
    return this.#unary("atan");
  }
  sinh(): Traced {
    return this.#unary("sinh");
  }
  cosh(): Traced {
    return this.#unary("cosh");
  }
  tanh(): Traced {
    return this.#unary("tanh");
  }
  cot(): Traced {
    return this.#unary("cot");
  }
  sec(): Traced {
    return this.#unary("sec");
  }
  csc(): Traced {
    return this.#unary("csc");
  }
  asinh(): Traced {
    return this.#unary("asinh");
  }
  acosh(): Traced {
    return this.#unary("acosh");
  }
  atanh(): Traced {
    return this.#unary("atanh");
  }
  coth(): Traced {
    return this.#unary("coth");
  }
  sech(): Traced {
    return this.#unary("sech");
  }
  csch(): Traced {
    return this.#unary("csch");
  }
  acot(): Traced {
    return this.#unary("acot");
  }
  asec(): Traced {
    return this.#unary("asec");
  }
  acsc(): Traced {
    return this.#unary("acsc");
  }
  acoth(): Traced {
    return this.#unary("acoth");
  }
  asech(): Traced {
    return this.#unary("asech");
  }
  acsch(): Traced {
    return this.#unary("acsch");
  }
  abs(): Traced {
    return this.#unary("abs");
  }
  log10(): Traced {
    return this.#unary("log10");
  }
  log2(): Traced {
    return this.#unary("log2");
  }
  cbrt(): Traced {
    return this.#unary("cbrt");
  }
  floor(): Traced {
    return this.#unary("floor");
  }
  ceil(): Traced {
    return this.#unary("ceil");
  }
  round(): Traced {
    return this.#unary("round");
  }
  sign(): Traced {
    return this.#unary("sign");
  }
  trunc(): Traced {
    return this.#unary("trunc");
  }
  expm1(): Traced {
    return this.#unary("expm1");
  }
  log1p(): Traced {
    return this.#unary("log1p");
  }
  erf(): Traced {
    return this.#unary("erf");
  }
}

function toNode(x: Traced | number): IRNode {
  return typeof x === "number" ? { kind: "const", value: x } : x.node;
}
