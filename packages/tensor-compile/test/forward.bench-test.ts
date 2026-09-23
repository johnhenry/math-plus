/**
 * Benchmark-threshold test, isolated from the rest of the suite (issue #99,
 * following the convention set by tensor-wasm's benchmark.bench-test.ts —
 * see issue #49 there for why this pattern exists).
 *
 * `CompiledFn.forward()` used to always compute AND discard a full gradient
 * vector per IR node per element (routed through `evalWithGrad`, same as
 * `forwardWithGrad()`) even though a plain forward pass never looks at
 * `.grad`. It now routes through `evalValue`, a sibling evaluator with the
 * exact same recursive structure but no `grad` array allocation/propagation
 * at all. The original issue (#99) measured 571ms vs 37ms (~15x) on a
 * 6-node/3-input/200k-element compiled function; this test's baseline is a
 * plain per-element loop over the same IR node (not the full `forward()`
 * plumbing) so it isolates the `evalWithGrad`-vs-`evalValue` delta and
 * consistently clears a smaller, conservative threshold rather than chasing
 * the exact 15x (which also reflects other now-amortized overhead).
 *
 * This file does NOT match `test/*.test.ts`, so the package `test` script
 * runs it in its own `node --test` invocation, after the main suite (see
 * `package.json`'s `test`/`test:bench` scripts) — timing-sensitive tests
 * never share the process pool with the rest of this package's tests.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { compile, evalWithGrad, Traced } from "../src/index.ts";

const MAX_ROUNDS = 3;

function benchMs(fn: () => void, iters: number): number {
  for (let i = 0; i < 2; i++) fn(); // warm up
  const start = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) fn();
  return Number(process.hrtime.bigint() - start) / iters / 1e6; // ms/call
}

/** Run `measure` up to MAX_ROUNDS times; pass when any round's speedup clears `threshold`. */
function assertSpeedupEventually(
  measure: () => { speedup: number; detail: string },
  threshold: number,
  label: string,
): void {
  const rounds: string[] = [];
  for (let round = 1; round <= MAX_ROUNDS; round++) {
    const { speedup, detail } = measure();
    rounds.push(`round ${round}: ${speedup.toFixed(2)}x (${detail})`);
    if (speedup > threshold) {
      if (round > 1) {
        console.warn(`[bench] ${label}: cleared ${threshold}x on round ${round} — earlier rounds were contended: ${rounds.slice(0, -1).join("; ")}`);
      }
      return;
    }
  }
  assert.fail(
    `expected ${label} to exceed ${threshold}x in at least one of ${MAX_ROUNDS} rounds — a consistent miss across rounds indicates a real regression, not contention. ${rounds.join("; ")}`,
  );
}

test("CompiledFn.forward() skips gradient computation entirely: beats an evalWithGrad-based baseline of the SAME graph at N=200k", () => {
  const N = 200_000;
  const numInputs = 3;
  // 6-node graph: add, mul, sigmoid, sub, div, relu.
  const graph = (x: Traced, y: Traced, z: Traced) =>
    x.add(y).mul(z).sigmoid().sub(x).div(y.add(2)).relu();

  const fused = compile(numInputs, graph);
  // Same IR node compile() builds internally (Traced.node is public), so the
  // baseline below evaluates the identical computation — the only variable
  // is evalWithGrad (old mechanism) vs evalValue (new, via fused.forward()).
  const node = graph(Traced.input(0), Traced.input(1), Traced.input(2)).node;

  const aData = new Float64Array(N);
  const bData = new Float64Array(N);
  const cData = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    aData[i] = Math.sin(i) + 2; // keep values away from 0 (log/div-by-zero-adjacent ops upstream)
    bData[i] = Math.cos(i) + 2;
    cData[i] = (i % 7) + 1;
  }
  const a = Tensor.fromTypedArray(aData, [N], { dtype: "f64" });
  const b = Tensor.fromTypedArray(bData, [N], { dtype: "f64" });
  const c = Tensor.fromTypedArray(cData, [N], { dtype: "f64" });

  // Reproduces what forward() used to do: call evalWithGrad and keep only
  // `.value`, discarding the gradient array it just built.
  const oldForward = (): Float64Array => {
    const out = new Float64Array(N);
    const scratch = new Array<number>(numInputs);
    for (let i = 0; i < N; i++) {
      scratch[0] = aData[i] as number;
      scratch[1] = bData[i] as number;
      scratch[2] = cData[i] as number;
      out[i] = evalWithGrad(node, scratch, numInputs).value;
    }
    return out;
  };

  const newForward = (): Tensor => fused.forward(a, b, c);

  // Sanity: both compute the same values (guards against the benchmark
  // silently comparing two different computations).
  const oldOut = oldForward();
  const newOut = Array.from(newForward().data as Float64Array);
  for (let i = 0; i < N; i += 997) {
    // sparse spot-check is enough; full agreement is already covered by
    // compile.test.ts's correctness tests.
    assert.ok(
      Math.abs((oldOut[i] as number) - (newOut[i] as number)) < 1e-9,
      `value mismatch at ${i}: ${oldOut[i]} vs ${newOut[i]}`,
    );
  }

  // Locally this consistently measures ~2-2.5x; a conservative 1.5x keeps
  // this robust to slower/contended CI hardware while still catching a real
  // regression back toward the always-compute-gradient path (which would
  // land speedup at/below 1x).
  assertSpeedupEventually(
    () => {
      const oldMs = benchMs(oldForward, 5);
      const newMs = benchMs(newForward, 5);
      return { speedup: oldMs / newMs, detail: `old(evalWithGrad)=${oldMs.toFixed(3)}ms, new(evalValue)=${newMs.toFixed(3)}ms` };
    },
    1.5,
    "CompiledFn.forward() vs evalWithGrad-based baseline at N=200k",
  );
});
