/**
 * Expr evaluation over an already-pruned Arrow Table.
 *
 * DOCUMENTED v1 SIMPLIFICATION: inside expression evaluation (filter
 * predicates, withColumns, aggregates), timestamp columns are read via
 * `Vector.get()` — i.e. as epoch-*millisecond* JS numbers, matching
 * apache-arrow's own accessor semantics (docs/spikes/arrow-parity.md sharp
 * edge #3). This means a `timestamp[us]` predicate/aggregate loses
 * sub-millisecond precision. Exact microsecond reads remain available
 * outside expression evaluation, via `Series` accessors and `Frame.toRows()`
 * (access.ts's `timestampExactAt`, which reads the raw int64 buffer). This
 * split keeps expression evaluation uniformly numeric (so `.gt()`/arithmetic/
 * `fn.month()` all just work on `number`) without threading timestamp-unit
 * metadata through every expression node.
 *
 * List/struct columns are NOT specially flattened here (unlike access.ts) —
 * they're only ever meaningful as passthrough columns (`select`), which
 * never routes through this module. Comparing/computing on a list/struct
 * column through an Expr is not a supported v1 operation.
 */
import type { Table, Vector } from "apache-arrow";
// The canonical double-precision erf (issue #122) — the same function
// tensor-core's Tensor.erf() and tensor-compile's IR evaluator use. It lives
// in the zero-dependency @johnhenry/math-plus-special precisely so this
// package can use it without a static dependency on the tensor track.
import { erf } from "@johnhenry/math-plus-special";
import {
  AggExpr,
  ArithExpr,
  ColumnExpr,
  CompareExpr,
  Expr,
  LiteralExpr,
  LogicalExpr,
  NotExpr,
  OverAllExpr,
  ScalarFnExpr,
  type AggOp,
  type ArithOp,
  type CompareOp,
  type LogicalOp,
  type ScalarMathFuncName,
} from "./expr.ts";

/** Value-only formulas for every {@link ScalarMathFuncName} — matches `tensor-compile`'s
 * `unaryValueAndDeriv` value branch (packages/tensor-compile/src/ir.ts) so a computed column's
 * `fn.*` result and a `Symbolic`-compiled-to-IR tensor evaluation of the same function agree. */
const SCALAR_MATH_FN: Record<ScalarMathFuncName, (x: number) => number> = {
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  exp: Math.exp,
  ln: Math.log,
  sqrt: Math.sqrt,
  asin: Math.asin,
  acos: Math.acos,
  atan: Math.atan,
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  cot: (x) => 1 / Math.tan(x),
  sec: (x) => 1 / Math.cos(x),
  csc: (x) => 1 / Math.sin(x),
  asinh: Math.asinh,
  acosh: Math.acosh,
  atanh: Math.atanh,
  coth: (x) => 1 / Math.tanh(x),
  sech: (x) => 1 / Math.cosh(x),
  csch: (x) => 1 / Math.sinh(x),
  acot: (x) => Math.atan(1 / x),
  asec: (x) => Math.acos(1 / x),
  acsc: (x) => Math.asin(1 / x),
  acoth: (x) => 0.5 * Math.log((x + 1) / (x - 1)),
  asech: (x) => Math.acosh(1 / x),
  acsch: (x) => Math.asinh(1 / x),
  abs: Math.abs,
  log10: Math.log10,
  log2: Math.log2,
  cbrt: Math.cbrt,
  floor: Math.floor,
  ceil: Math.ceil,
  round: Math.round,
  sign: Math.sign,
  trunc: Math.trunc,
  expm1: Math.expm1,
  log1p: Math.log1p,
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  erf,
  relu: (x) => (x > 0 ? x : 0),
};

function evalColumn(table: Table, name: string): unknown[] {
  const vector = table.getChild(name) as Vector | null;
  if (!vector) {
    throw new Error(`no such column "${name}" (available: ${(table.schema.names as string[]).join(", ")})`);
  }
  const out: unknown[] = new Array(vector.length);
  for (let i = 0; i < vector.length; i++) out[i] = vector.get(i);
  return out;
}

function compareOp(op: CompareOp, a: unknown, b: unknown): boolean | null {
  if (a === null || a === undefined || b === null || b === undefined) return null;
  switch (op) {
    case "eq":
      // eslint-disable-next-line eqeqeq -- intentional loose equality: reconciles bigint (int64) vs number literals
      return (a as never) == (b as never);
    case "ne":
      // eslint-disable-next-line eqeqeq
      return (a as never) != (b as never);
    case "lt":
      return (a as never) < (b as never);
    case "lte":
      return (a as never) <= (b as never);
    case "gt":
      return (a as never) > (b as never);
    case "gte":
      return (a as never) >= (b as never);
  }
}

function logicalOp(op: LogicalOp, a: boolean | null, b: boolean | null): boolean | null {
  if (op === "and") {
    if (a === false || b === false) return false;
    if (a === null || b === null) return null;
    return a && b;
  }
  if (a === true || b === true) return true;
  if (a === null || b === null) return null;
  return a || b;
}

function arithOp(op: ArithOp, a: unknown, b: unknown): unknown {
  if (a === null || a === undefined || b === null || b === undefined) return null;
  if (typeof a === "bigint" && typeof b === "bigint") {
    switch (op) {
      case "add":
        return a + b;
      case "sub":
        return a - b;
      case "mul":
        return a * b;
      case "div":
        return a / b; // bigint division truncates toward zero — documented
    }
  }
  // Non-numeric operands (e.g. a string column) would otherwise fall through to
  // JS's `+`/`-`/`*`/`/` coercion rules — "-"/"*"/"/" silently produce NaN, and
  // "+" silently concatenates strings — and that NaN/garbage then gets written
  // into the output Arrow column as a normal VALID value (nullCount() stays 0),
  // making the corruption invisible to any null-checking consumer. Throw instead,
  // consistent with this file's other hard type-misuse errors (AggExpr used
  // outside aggregate()/overAll(), ScalarFnExpr's unhandled op, evalColumn's
  // missing-column check).
  if (typeof a !== "number" && typeof a !== "bigint") {
    throw new Error(`arithmetic op "${op}" requires numeric operands, got ${typeof a} (${JSON.stringify(a)})`);
  }
  if (typeof b !== "number" && typeof b !== "bigint") {
    throw new Error(`arithmetic op "${op}" requires numeric operands, got ${typeof b} (${JSON.stringify(b)})`);
  }
  const an = typeof a === "bigint" ? Number(a) : a;
  const bn = typeof b === "bigint" ? Number(b) : b;
  switch (op) {
    case "add":
      return an + bn;
    case "sub":
      return an - bn;
    case "mul":
      return an * bn;
    case "div":
      return an / bn;
  }
}

export function evalRowwise(expr: Expr, table: Table): unknown[] {
  const n = table.numRows;
  if (expr instanceof ColumnExpr) return evalColumn(table, expr.name);
  if (expr instanceof LiteralExpr) return new Array(n).fill(expr.value);
  if (expr instanceof CompareExpr) {
    const l = evalRowwise(expr.left, table);
    const r = evalRowwise(expr.right, table);
    return l.map((lv, i) => compareOp(expr.op, lv, r[i]));
  }
  if (expr instanceof LogicalExpr) {
    const l = evalRowwise(expr.left, table) as (boolean | null)[];
    const r = evalRowwise(expr.right, table) as (boolean | null)[];
    return l.map((lv, i) => logicalOp(expr.op, lv, r[i] as boolean | null));
  }
  if (expr instanceof NotExpr) {
    const v = evalRowwise(expr.operand, table) as (boolean | null)[];
    return v.map((x) => (x === null ? null : !x));
  }
  if (expr instanceof ArithExpr) {
    const l = evalRowwise(expr.left, table);
    const r = evalRowwise(expr.right, table);
    return l.map((lv, i) => arithOp(expr.op, lv, r[i]));
  }
  if (expr instanceof OverAllExpr) {
    const scalar = evalAggregateScalar(expr.agg, table);
    return new Array(n).fill(scalar);
  }
  if (expr instanceof ScalarFnExpr) {
    const inner = evalRowwise(expr.inner, table);
    if (expr.op === "month") {
      return (inner as (number | null)[]).map((ms) => (ms === null ? null : new Date(ms).getUTCMonth() + 1));
    }
    const mathFn = SCALAR_MATH_FN[expr.op as ScalarMathFuncName];
    if (!mathFn) {
      throw new Error(`unhandled scalar fn "${expr.op}"`);
    }
    // Same numeric-type guard as arithOp: throw on non-numeric input rather than
    // silently writing NaN as a valid column value.
    return inner.map((v) => {
      if (v === null || v === undefined) return null;
      if (typeof v !== "number" && typeof v !== "bigint") {
        throw new Error(`fn.${expr.op}() requires a numeric operand, got ${typeof v} (${JSON.stringify(v)})`);
      }
      return mathFn(typeof v === "bigint" ? Number(v) : v);
    });
  }
  if (expr instanceof AggExpr) {
    throw new Error(
      `aggregate expression (fn.${expr.op}) used outside groupBy().aggregate() or .overAll() — ` +
        `wrap it in .overAll() to broadcast it as a column inside withColumns()`,
    );
  }
  throw new Error(`evalRowwise: unhandled expr kind "${expr.kind}"`);
}

/** Reduce a column's values at the given row indices with the given aggregate op.
 * `values === null` is only valid for "count" (counts rows, i.e. group size). */
export function reduceGroup(op: AggOp, values: readonly unknown[] | null, indices: readonly number[]): unknown {
  if (op === "count") {
    if (values === null) return BigInt(indices.length);
    let c = 0;
    for (const i of indices) if (values[i] !== null && values[i] !== undefined) c++;
    return BigInt(c);
  }
  const raw: unknown[] = [];
  for (const i of indices) {
    const v = (values as readonly unknown[])[i];
    if (v !== null && v !== undefined) raw.push(v);
  }
  const allBigint = raw.length > 0 && raw.every((v) => typeof v === "bigint");
  if (op === "sum") {
    if (raw.length === 0) return allBigint ? 0n : 0;
    if (allBigint) return (raw as bigint[]).reduce((a, b) => a + b, 0n);
    return (raw as number[]).reduce((a, b) => a + b, 0);
  }
  const nums = raw.map((v) => (typeof v === "bigint" ? Number(v) : (v as number)));
  if (op === "mean") return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
  if (op === "stddev") {
    if (nums.length === 0) return null;
    if (nums.length === 1) return 0;
    const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
    const variance = nums.reduce((acc, x) => acc + (x - mean) ** 2, 0) / (nums.length - 1); // sample stddev (ddof=1), matches pandas default
    return Math.sqrt(variance);
  }
  throw new Error(`reduceGroup: unhandled op "${op}"`);
}

export function evalAggregateScalar(agg: AggExpr, table: Table): unknown {
  const values = agg.inner ? evalRowwise(agg.inner, table) : null;
  const allIndices = Array.from({ length: table.numRows }, (_, i) => i);
  return reduceGroup(agg.op, values, allIndices);
}
