/**
 * Tests for test/harness.ts's Bun shim, driven through a FAKE bun:test module
 * so they run (and guard the shim) under plain `node --test` too.
 */
import assert from "node:assert/strict";
import { makeTest, spyMethod } from "./harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);

function fakeBun() {
  const registered: Array<{ kind: string; name: string; fn?: () => unknown; timeout?: number }> = [];
  const t = Object.assign(
    (name: string, fn: () => unknown, timeout?: number) => registered.push({ kind: "test", name, fn, timeout }),
    {
      skip: (name: string) => registered.push({ kind: "skip", name }),
      todo: (name: string) => registered.push({ kind: "todo", name }),
    },
  );
  const noop = () => {};
  return { registered, mod: { test: t, beforeAll: noop, afterAll: noop, afterEach: noop } };
}

test("bun shim: skip/todo options map to bun's, and no default 5 s timeout is left in place", () => {
  const { registered, mod } = fakeBun();
  const h = makeTest(mod);
  h.test("a", () => {});
  h.test("b", { skip: "no oracle" }, () => {});
  h.test("c", { todo: true }, () => {});
  h.test("d", { timeout: 50 }, () => {});
  assert.deepEqual(
    registered.map((r) => [r.kind, r.name, r.timeout]),
    [["test", "a", 2 ** 31 - 1], ["skip", "b (no oracle)", undefined], ["todo", "c", undefined], ["test", "d", 50]],
  );
});

test("bun shim: subtests run in order, t.after runs last, and a failing subtest fails the parent", async () => {
  const { registered, mod } = fakeBun();
  const log: string[] = [];
  makeTest(mod).test("parent", (t) => {
    t.after(() => log.push("after"));
    // Deliberately not awaited, as several suites do: the parent still waits.
    void t.test("one", async () => {
      await new Promise((r) => setTimeout(r, 5));
      log.push("one");
    });
    void t.test("two", () => {
      log.push("two");
      throw new Error("boom");
    });
    void t.test("three", () => log.push(`three:${t.name}`));
  });
  await assert.rejects(async () => registered[0]!.fn!(), /boom/);
  assert.deepEqual(log, ["one", "two", "three:parent", "after"]);
});

test("bun shim: t.skip is reported once per test and does not fail it", async () => {
  const { registered, mod } = fakeBun();
  makeTest(mod).test("skipper", (t) => {
    t.skip("x");
    t.skip("x");
  });
  const lines: string[] = [];
  const orig = console.log;
  console.log = (s: string) => lines.push(s);
  try {
    await registered[0]!.fn!();
  } finally {
    console.log = orig;
  }
  assert.deepEqual(lines, ["[skip] skipper: x"]);
});

test("spyMethod counts calls, forwards them, and restores the original", () => {
  class C {
    v = 2;
    f(x: number) {
      return x * this.v;
    }
  }
  const c = new C();
  const spy = spyMethod(c, "f");
  assert.equal(c.f(3), 6);
  assert.equal(spy.callCount(), 1);
  spy.restore();
  assert.equal(Object.prototype.hasOwnProperty.call(c, "f"), false);
  assert.equal(c.f(1), 2);
});
