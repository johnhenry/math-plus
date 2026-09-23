import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, afterEach } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import {
  artifact,
  hasSink,
  metric,
  setSink,
  startRun,
  tensorSummary,
  timed,
  trace,
  type TrainingEvent,
} from "../src/index.ts";

afterEach(() => setSink(null)); // always restore the no-op default between tests

test("default sink is a no-op — nothing throws, nothing is observable", () => {
  assert.equal(hasSink(), false);
  assert.doesNotThrow(() => {
    startRun("r1", {});
    metric("r1", 0, "loss", 1.5);
  });
});

test("setSink installs a collector; hasSink() reflects it", () => {
  const events: TrainingEvent[] = [];
  setSink((e) => events.push(e));
  assert.equal(hasSink(), true);
  metric("r1", 3, "loss", 0.42);
  assert.equal(events.length, 1);
  const e = events[0] as Extract<TrainingEvent, { type: "metric" }>;
  assert.equal(e.type, "metric");
  assert.equal(e.runId, "r1");
  assert.equal(e.step, 3);
  assert.equal(e.name, "loss");
  assert.equal(e.value, 0.42);
  assert.equal(typeof e.time, "number");
});

test("setSink(null) restores the no-op default", () => {
  const events: TrainingEvent[] = [];
  setSink((e) => events.push(e));
  setSink(null);
  metric("r1", 0, "loss", 1);
  assert.equal(events.length, 0);
  assert.equal(hasSink(), false);
});

test("startRun emits a run.start event with the given config", () => {
  const events: TrainingEvent[] = [];
  setSink((e) => events.push(e));
  startRun("r1", { lr: 0.01 }, { device: "wasm" });
  assert.equal(events.length, 1);
  const e = events[0] as Extract<TrainingEvent, { type: "run.start" }>;
  assert.equal(e.type, "run.start");
  assert.deepEqual(e.config, { lr: 0.01 });
  assert.deepEqual(e.environment, { device: "wasm" });
});

test("tensorSummary emits shape/dtype/stats, never raw data", () => {
  const events: TrainingEvent[] = [];
  setSink((e) => events.push(e));
  tensorSummary("r1", 0, "layer1.weight", {
    shape: [4, 3],
    dtype: "f32",
    device: "wasm",
    stats: { min: -1, max: 1, mean: 0, std: 0.5, finite: 1 },
  });
  const e = events[0] as Extract<TrainingEvent, { type: "tensor.summary" }>;
  assert.deepEqual(e.tensor.shape, [4, 3]);
  assert.equal(e.tensor.stats.finite, 1);
  assert.ok(!("data" in e.tensor)); // structurally cannot carry raw values
});

test("artifact and trace emit their respective event shapes", () => {
  const events: TrainingEvent[] = [];
  setSink((e) => events.push(e));
  artifact("r1", "checkpoint-epoch-5", "checkpoint", "/tmp/ckpt.bin", 5);
  trace("r1", 0, [{ name: "backward", start: 0, duration: 1.2, category: "autograd" }]);
  assert.equal(events[0]?.type, "artifact");
  assert.equal(events[1]?.type, "trace");
});

test("timed() skips performance.now() entirely when no sink is installed", () => {
  let called = false;
  const result = timed("r1", 0, "step", "test", () => {
    called = true;
    return 42;
  });
  assert.equal(result, 42);
  assert.equal(called, true);
});

test("timed() emits exactly one trace span when a sink is installed", () => {
  const events: TrainingEvent[] = [];
  setSink((e) => events.push(e));
  const result = timed("r1", 2, "backward", "autograd", () => 7);
  assert.equal(result, 7);
  assert.equal(events.length, 1);
  const e = events[0] as Extract<TrainingEvent, { type: "trace" }>;
  assert.equal(e.spans.length, 1);
  assert.equal(e.spans[0]?.name, "backward");
  assert.ok(e.spans[0]!.duration >= 0);
});
