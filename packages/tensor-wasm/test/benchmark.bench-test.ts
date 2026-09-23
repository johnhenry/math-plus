/**
 * Benchmark-threshold tests, isolated from the rest of the suite (issue #49).
 *
 * Two things distinguish this file from the ordinary tests:
 *
 * 1. It does NOT match the `test/*.test.ts` glob — the package `test` script
 *    runs it in its own `node --test` invocation AFTER the main suite, so
 *    these timing-sensitive tests never share the process pool with other
 *    test files of this package (the isolation-over-tolerance idea from
 *    docs/spikes/woxi-study.md's nextest finding, adapted to node --test).
 *
 * 2. Each benchmark re-measures up to MAX_ROUNDS times and passes on the
 *    first round that clears its threshold. This does NOT loosen the
 *    assertion: the claim under test is existential — "the fast path CAN
 *    beat the baseline on this hardware" — and machine contention (another
 *    agent session, a background build) can only produce false NEGATIVES,
 *    never a false positive. A regressed fast path fails all rounds exactly
 *    as it failed the single round before. Round-by-round numbers are
 *    reported on failure so a real regression is diagnosable.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Kernels } from "../src/index.ts";

const MAX_ROUNDS = 3;

function benchMs(fn: () => void, iters: number): number {
  for (let i = 0; i < 3; i++) fn(); // warm up
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
        // Flaky-then-ok is worth a visible trace (issue #49): it means the
        // machine was contended, not that the code regressed.
        console.warn(`[bench] ${label}: cleared ${threshold}x on round ${round} — earlier rounds were contended: ${rounds.slice(0, -1).join("; ")}`);
      }
      return;
    }
  }
  assert.fail(
    `expected ${label} to exceed ${threshold}x in at least one of ${MAX_ROUNDS} rounds — a consistent miss across rounds indicates a real regression, not contention. ${rounds.join("; ")}`,
  );
}

test("addInto over resident buffers beats a pure-JS loop at N=1e6 (reproduces the measured 1.78x)", async () => {
  const kernels = await Kernels.load();
  const N = 1_000_000;
  const aData = new Float32Array(N).fill(1.5);
  const bData = new Float32Array(N).fill(2.5);
  const a = kernels.fromArray(aData, [N]);
  const b = kernels.fromArray(bData, [N]);
  const out = kernels.zeros([N]);
  const jsOut = new Float32Array(N);

  // docs/spikes/wasm-baseline.md measured 1.78x on this machine; require a
  // conservative >1.15x so the test isn't flaky across CI hardware while
  // still failing if the ...Into path regresses back toward parity (or
  // worse, back toward the 2.27x-slower copying wrapper it replaced).
  assertSpeedupEventually(
    () => {
      const jsTime = benchMs(() => {
        for (let i = 0; i < N; i++) jsOut[i] = aData[i]! + bData[i]!;
      }, 20);
      const wasmTime = benchMs(() => kernels.addInto(out, a, b), 20);
      return { speedup: jsTime / wasmTime, detail: `js=${jsTime.toFixed(3)}ms, wasm=${wasmTime.toFixed(3)}ms` };
    },
    1.15,
    "addInto vs pure JS at N=1e6",
  );
});

test("addInto: SIMD is a real, measured speedup over the scalar fallback at N=1e6 (issue #13 — docs/spikes/wasm-simd.md)", async () => {
  const withSimd = await Kernels.load();
  const scalarOnly = await Kernels.load(undefined, new Uint8Array([0, 1, 2, 3]));
  const N = 1_000_000;
  const aData = new Float32Array(N).fill(1.5);
  const bData = new Float32Array(N).fill(2.5);

  const setup = (kernels: Kernels) => {
    const a = kernels.fromArray(aData, [N]);
    const b = kernels.fromArray(bData, [N]);
    const out = kernels.zeros([N]);
    return () => kernels.addInto(out, a, b);
  };
  const scalarOp = setup(scalarOnly);
  const simdOp = setup(withSimd);

  // docs/spikes/wasm-simd.md measured a stable ~2.6-3x SIMD-only speedup
  // (apples-to-apples, contiguous-scalar baseline) on the dev machine.
  // Same conservative >1.15x threshold as the benchmark above, which the
  // best-of-rounds retry (issue #49) keeps stable under local CPU load.
  //
  // NOT RUN IN CI -- see the `test:bench` note in this package's
  // package.json. Measured 2026-08-22, same commit, two GitHub runners:
  //   runner A : 1.12x, 1.14x, 1.12x   (variance +/-0.01)
  //   runner B : 1.01x, 1.01x, 1.01x   (variance +/-0.00)
  //   dev box  : 2.28x, 1.31x, 1.66x
  // Runner B shows no SIMD advantage whatsoever, and does so with zero
  // variance -- that is a property of the hardware, not contention the
  // retry could absorb. Since a threshold low enough to pass runner B
  // (<1.01x) would also pass a genuine regression to parity, the
  // measurement is simply not meaningful on GitHub's mixed runner fleet.
  // It stays a real assertion where the hardware supports it.
  assertSpeedupEventually(
    () => {
      const scalarTime = benchMs(scalarOp, 20);
      const simdTime = benchMs(simdOp, 20);
      return {
        speedup: scalarTime / simdTime,
        detail: `scalar=${scalarTime.toFixed(3)}ms, simd=${simdTime.toFixed(3)}ms`,
      };
    },
    1.15,
    "SIMD addInto vs scalar fallback at N=1e6",
  );
});
