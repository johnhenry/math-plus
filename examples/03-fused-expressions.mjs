// tensor-compile: trace an elementwise expression once, execute it fused —
// and plug the fused op into the autograd tape. Strictly opt-in; pure JS.
//
// Run: npm install && npm run build && node examples/03-fused-expressions.mjs
import assert from "node:assert/strict";
import { Variable } from "@johnhenry/math-plus-tensor-autograd";
import { compile } from "@johnhenry/math-plus-tensor-compile";
import { Tensor } from "@johnhenry/math-plus-tensor-core";

const a = Tensor.from([1, -2, 3, -4]);
const b = Tensor.from([0.5, 0.5, -6, 10]);
const c = Tensor.from([2, 2, 2, 2]);

// Trace once over symbolic inputs (fn only ever sees Traced nodes)...
const fused = compile(3, (x, y, z) => x.add(y).mul(z).relu());

// ...then forward() executes the whole expression in one fused pass.
// (forward() skips gradient bookkeeping entirely — ~15x faster than the
// grad-carrying evaluator when you only want the value.)
const out = fused.forward(a, b, c);
console.log(out.toArray()); // [3, 0, 0, 12] — matches a.add(b).mul(c).relu()
assert.deepEqual(out.toArray(), a.add(b).mul(c).relu().toArray());

// Broadcasting works like eager Tensor ops
const m = Tensor.from([1, 2, 3, 4]).reshape([2, 2]);
const v = Tensor.from([10, 20]);
const fused2 = compile(2, (x, y) => x.mul(y).sigmoid());
console.log(fused2.forward(m, v).shape); // [2, 2]

// asVariableOp(): the fused op becomes a differentiable tape node
const op = fused.asVariableOp();
const va = Variable.variable(a);
const vb = Variable.variable(b);
const vc = Variable.variable(c);
op(va, vb, vc).sum().backward();
console.log(va.grad.toArray()); // matches the unfused Variable graph's gradient
