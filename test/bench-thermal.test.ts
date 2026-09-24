/**
 * Unit tests for scripts/bench/thermal.ts (docs/BENCHMARKING.md). A fake
 * clock and sleep make them instant and deterministic: they check the
 * methodology itself (cooldown before every cell, bounded timing window,
 * backends alternated within one process), not any real timing.
 */
import assert from "node:assert/strict";
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { formatGrid, machineInfo, runGrid, timeCell } from "../scripts/bench/thermal.ts";

function fakeClock(costMs: number) {
  let t = 0;
  const log: string[] = [];
  return {
    log,
    now: () => t,
    sleep: async (ms: number) => {
      log.push(`sleep ${ms}`);
      t += ms;
    },
    fn: (tag = "run") => () => {
      log.push(tag);
      t += costMs;
    },
  };
}

test("timeCell: cools down first, warms up untimed, then stops at the window bound", async () => {
  const c = fakeClock(100);
  const r = await timeCell(c.fn(), { cooldownMs: 5000, windowMs: 1000, sleep: c.sleep, now: c.now });
  assert.equal(c.log[0], "sleep 5000");
  assert.equal(r.n, 10, "100 ms calls in a 1000 ms window");
  assert.equal(c.log.filter((x) => x === "run").length, 11, "1 warmup + 10 timed");
  assert.equal(r.windowMs, 1000);
  assert.equal(r.medianMs, 100);
});

test("timeCell: minRuns wins over the window for slow calls, maxRuns caps fast ones", async () => {
  const slow = fakeClock(2000);
  assert.equal((await timeCell(slow.fn(), { cooldownMs: 0, sleep: slow.sleep, now: slow.now })).n, 3);
  const fast = fakeClock(1);
  assert.equal((await timeCell(fast.fn(), { cooldownMs: 0, maxRuns: 30, sleep: fast.sleep, now: fast.now })).n, 30);
});

test("timeCell: a call that returns { selfTimedMs } supplies its own sample; the window still uses the caller's clock", async () => {
  const c = fakeClock(100); // 100 ms per call on the caller's clock (e.g. including a CDP round trip)
  let i = 0;
  const r = await timeCell(
    () => {
      c.fn()();
      return { selfTimedMs: 10 + (i++ % 3) }; // what the page measured itself
    },
    { cooldownMs: 0, windowMs: 1000, sleep: c.sleep, now: c.now },
  );
  assert.equal(r.n, 10, "window bounded by the caller's 100 ms per call");
  assert.ok(r.medianMs >= 10 && r.medianMs <= 12, `median ${r.medianMs} comes from selfTimedMs`);
  assert.equal(r.maxMs, 12);
  // Anything else a call returns is ignored.
  const d = fakeClock(50);
  const r2 = await timeCell(() => (d.fn()(), { selfTimedMs: "fast" }), { cooldownMs: 0, windowMs: 200, sleep: d.sleep, now: d.now });
  assert.equal(r2.medianMs, 50);
});

test("runGrid: every cell cools down, backends alternate and swap order on every other cell", async () => {
  const c = fakeClock(10);
  const rows = await runGrid({
    cells: [1, 2, 3],
    backends: [
      { name: "a", run: c.fn("a") },
      { name: "b", run: c.fn("b") },
    ],
    cooldownMs: 5000,
    windowMs: 30,
    sleep: c.sleep,
    now: c.now,
    onResult: null,
  });
  assert.deepEqual(
    rows.map((r) => `${r.cell}:${r.backend}`),
    ["1:a", "1:b", "2:b", "2:a", "3:a", "3:b"],
  );
  assert.equal(c.log.filter((x) => x === "sleep 5000").length, 6, "one cooldown per (cell, backend)");
  assert.match(formatGrid(rows), /\| 2 \| 10\.000 \| 10\.000 \|/);
});

test("machineInfo: records runtime, platform and the cooldown/window settings", () => {
  const info = machineInfo();
  assert.match(String(info.runtime), /^(node|bun) /);
  assert.equal(typeof info.platform, "string");
  assert.equal(typeof info.cooldownS, "number");
  assert.equal(typeof info.windowMs, "number");
});
