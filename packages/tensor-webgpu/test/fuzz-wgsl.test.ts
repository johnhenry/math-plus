/**
 * Randomized differential fuzzing through the REAL GPU (issue #58): the same
 * seeded IR-graph generator as tensor-compile's CPU fuzzer (imported from
 * ../../tensor-compile/test/fuzz-generator.ts — shared, not duplicated) runs
 * random programs through `compileIRToWGSL` + `runElementwiseWGSL` on a live
 * GPUAdapter and compares against `evalWithGrad` on the CPU. Unlike the CPU
 * fuzzer's value leg (where both sides share the interpreter), WGSL codegen +
 * GPU f32 execution is a genuinely independent third implementation of the
 * IR's semantics — extending fusion.test.ts's FIXED cross-checks to random
 * programs.
 *
 * f32-vs-f64 realities shape the spec table (documented, not silent):
 * - `pow` excluded: WGSL `pow` is NaN for negative bases where JS isn't.
 * - cmp/select excluded: a comparison that lands within f32 epsilon of the
 *   boundary flips branches between backends — a tolerance can't absorb a
 *   taken-vs-untaken branch.
 * - floor/ceil/round/sign/trunc excluded for the same step-function reason.
 * - Poles/domain edges handled by rejection: every SUBNODE's CPU value must
 *   stay finite and small at every element, same guard as the CPU fuzzer.
 * - Comparison uses fusion.test.ts's established f32 tolerance, not
 *   Object.is.
 *
 * Runs under this package's --test-concurrency=1 + ~/gpu.lock conventions;
 * skips (never fails) when the harness is unavailable, with #49's
 * contention-vs-genuinely-absent distinction in the skip reason.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import * as path from "node:path";
import { evalWithGrad, type BinaryOp, type IRNode, type UnaryOp } from "@johnhenry/math-plus-tensor-compile";
import {
  allSubnodes,
  genNode,
  mulberry32,
  reductions,
  usesAnyInput,
  type GenSpec,
} from "../../tensor-compile/test/fuzz-generator.ts";
import { bundleForBrowser, closeHarness, getHarness, SRC } from "./helpers.ts";

const CASES = Number(process.env.MATH_PLUS_WGSL_FUZZ_CASES ?? 25);
const BASE_SEED = Number(process.env.MATH_PLUS_WGSL_FUZZ_SEED ?? 20260813);
const ELEMENTS = 64;

/** f32-tame subset: smooth, WGSL-pole-safe under the rejection guard below. */
const WGSL_UNARY: readonly UnaryOp[] = [
  "neg", "relu", "abs", "sigmoid", "gelu", "exp", "tanh", "sin", "cos",
  "atan", "asinh", "erf", "cbrt", "sqrt", "log", "log1p", "expm1",
];
const WGSL_BINARY: readonly BinaryOp[] = ["add", "sub", "mul", "div", "atan2", "hypot", "min", "max"];
const WGSL_SPEC: GenSpec = { unary: WGSL_UNARY, binary: WGSL_BINARY, cmpSelect: false, maxDepth: 4 };

/** Same tolerance rationale as fusion.test.ts's crossCheck. */
function closeF32(cpu: number, gpu: number): boolean {
  if (Number.isNaN(cpu)) return Number.isNaN(gpu);
  return Math.abs(gpu - cpu) <= 1e-2 * Math.max(1, Math.abs(cpu));
}

interface FuzzCase {
  seed: number;
  node: IRNode;
  numInputs: number;
  inputs: number[][];
  expected: number[];
}

/** CPU reference + rejection: keep a case only when every subnode stays
 * finite and modest at EVERY element (poles/domain edges out). */
function buildCase(seed: number): FuzzCase | undefined {
  const rng = mulberry32(seed);
  const numInputs = 1 + Math.floor(rng() * 2);
  const node = genNode(rng, WGSL_SPEC, numInputs, 0);
  if (!usesAnyInput(node)) return undefined;
  const inputs = Array.from({ length: numInputs }, () =>
    Array.from({ length: ELEMENTS }, () => Math.round((rng() * 1.6 - 0.8) * 1000) / 1000),
  );
  const expected: number[] = [];
  const scratch = new Array<number>(numInputs);
  for (let i = 0; i < ELEMENTS; i++) {
    for (let k = 0; k < numInputs; k++) scratch[k] = (inputs[k] as number[])[i] as number;
    for (const sub of allSubnodes(node)) {
      const v = evalWithGrad(sub, scratch, numInputs).value;
      if (!Number.isFinite(v) || Math.abs(v) > 50) return undefined;
    }
    expected.push(evalWithGrad(node, scratch, numInputs).value);
  }
  return { seed, node, numInputs, inputs, expected };
}

test(`WGSL fuzz: ${CASES} random IR programs agree between real-GPU WGSL and the CPU interpreter`, async (t) => {
  t.after(closeHarness);
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`headless WebGPU not available: ${harness.reason}`);
    return;
  }

  const cases: FuzzCase[] = [];
  let discarded = 0;
  for (let i = 0; cases.length < CASES && i < CASES * 8; i++) {
    const c = buildCase(BASE_SEED + i);
    if (c) cases.push(c);
    else discarded++;
  }
  assert.ok(cases.length >= CASES * 0.6, `only built ${cases.length}/${CASES} cases (${discarded} discarded) — generator/guard drift`);

  const bundle = bundleForBrowser([path.join(SRC, "elementwise.ts")]);
  const runBatch = async (batch: Array<{ node: IRNode; inputs: number[][] }>): Promise<number[][]> =>
    await harness.run<number[][]>(
      `
      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice();
      const cases = ${JSON.stringify(batch)};
      const results = [];
      for (const c of cases) {
        const inputs = c.inputs.map((a) => new Float32Array(a));
        const out = await runElementwiseWGSL(device, c.node, inputs, ${ELEMENTS});
        results.push(Array.from(out));
      }
      return results;
      `,
      bundle,
    );

  const gpuResults = await runBatch(cases.map((c) => ({ node: c.node, inputs: c.inputs })));

  for (let ci = 0; ci < cases.length; ci++) {
    const c = cases[ci] as FuzzCase;
    const gpu = gpuResults[ci] as number[];
    const badIndex = c.expected.findIndex((e, i) => !closeF32(e, gpu[i] as number));
    if (badIndex < 0) continue;

    // Divergence: greedily shrink (shared reductions; async re-check per
    // candidate against the live GPU) to a minimal diverging program.
    let minimal = c.node;
    for (;;) {
      let reduced: IRNode | undefined;
      for (const candidate of reductions(minimal)) {
        if (!usesAnyInput(candidate)) continue;
        const rebuilt = buildCaseWith(candidate, c);
        if (!rebuilt) continue;
        const [gpuOut] = await runBatch([{ node: candidate, inputs: c.inputs }]);
        if (rebuilt.expected.some((e, i) => !closeF32(e, (gpuOut as number[])[i] as number))) {
          reduced = candidate;
          break;
        }
      }
      if (!reduced) break;
      minimal = reduced;
    }
    assert.fail(
      `WGSL/CPU divergence (replay: MATH_PLUS_WGSL_FUZZ_SEED=${c.seed} MATH_PLUS_WGSL_FUZZ_CASES=1)\n` +
        `minimal diverging program: ${JSON.stringify(minimal)}\n` +
        `first divergent element #${badIndex}: cpu=${c.expected[badIndex]} gpu=${gpu[badIndex]}`,
    );
  }
});

/** Re-evaluate the CPU expectation for a shrunk candidate on the ORIGINAL
 * inputs; undefined if the candidate now trips the rejection guard. */
function buildCaseWith(node: IRNode, base: FuzzCase): { expected: number[] } | undefined {
  const expected: number[] = [];
  const scratch = new Array<number>(base.numInputs);
  for (let i = 0; i < ELEMENTS; i++) {
    for (let k = 0; k < base.numInputs; k++) scratch[k] = (base.inputs[k] as number[])[i] as number;
    const v = evalWithGrad(node, scratch, base.numInputs).value;
    if (!Number.isFinite(v) || Math.abs(v) > 50) return undefined;
    expected.push(v);
  }
  return { expected };
}
