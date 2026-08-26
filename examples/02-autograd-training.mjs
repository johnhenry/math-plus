// tensor-autograd: reverse-mode gradients, a tiny training loop, and the
// opt-in telemetry hook. Pure JS.
//
// Run: npm install && npm run build && node examples/02-autograd-training.mjs
import assert from "node:assert/strict";
import { grad, nn, optim, trainer, variable } from "@johnhenry/math-plus-tensor-autograd";
import { random, Tensor } from "@johnhenry/math-plus-tensor-core";
import { setSink } from "@johnhenry/math-plus-telemetry";

// Tape + backward. Gradients ACCUMULATE across backward() calls — zeroGrad()
// resets. Only a scalar output may call backward() without a gradOutput.
const x = variable(Tensor.from([2, 3], { dtype: "f64" }));
const y = variable(Tensor.from([4, 5], { dtype: "f64" }));
x.mul(y).sum().backward();
console.log(x.grad.toArray()); // [4, 5] — d(sum(x*y))/dx = y

// Functional API: value and gradient in one pass
const vg = grad.valueAndGrad((v) => v.mul(v).sum());
const { value, grad: g } = vg(Tensor.from([3, 4], { dtype: "f64" }));
console.log(value.item(), g.toArray()); // 25 [6, 8]

// Train y = 3x + 2 with a Linear layer. NOTE: nn.Linear initializes at f64,
// and tensor-core has no implicit promotion — inputs must be f64 too.
const xs = [0, 1, 2, 3, 4];
const X = Tensor.from(xs, { dtype: "f64" }).reshape([5, 1]);
const Y = Tensor.from(xs.map((v) => 3 * v + 2), { dtype: "f64" }).reshape([5, 1]);

// Telemetry is opt-in: install a sink and optimizers emit optim/gradNorm
// (the norm isn't even computed when no sink is installed).
const metrics = [];
setSink((e) => { if (e.type === "metric") metrics.push(e.name); });

const model = new nn.Linear(1, 1, { rng: random.seed(3) });
const opt = new optim.SGD(model.parameters(), { lr: 0.01 });
const t = trainer.configure({ model, optimizer: opt, lossFn: nn.mseLoss, epochs: 2000 });
const result = await t.fit({ x: X, y: Y });

setSink(null); // single global slot — always restore the no-op
const finalLoss = result.lossHistory[result.lossHistory.length - 1];
console.log("final loss:", finalLoss); // < 0.01
assert.ok(finalLoss < 0.01);
console.log("telemetry events seen:", metrics.length > 0 ? metrics[0] : "none"); // optim/gradNorm
