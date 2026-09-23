/**
 * One `test()` that runs under both `node --test` and `bun test`.
 *
 * Why this exists: Bun 1.2's `node:test` shim registers tests from the FIRST
 * file of a multi-file `bun test` run only. Every other file's `node:test`
 * tests are silently dropped and the run still reports success (observed with
 * Bun 1.2.17: `packages/fft` ran 8 of its 27 tests). The same happens if a
 * shared module imports `bun:test` on the files' behalf — registrations bind
 * to whichever file loaded it first. So under Bun, EACH test file must import
 * `bun:test` itself and hand the module to {@link makeTest}:
 *
 *     import { makeTest } from "../../../test/harness.ts";
 *     // @ts-ignore -- bun types are not installed; only evaluated under Bun
 *     const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
 *
 * Under Node, `makeTest(null)` returns `node:test`'s own functions unchanged,
 * so `npm test` behaves exactly as before. Under Bun, it returns a thin shim
 * implementing the subset of `node:test` this repo's suites use (the
 * {@link TestContext} type below is deliberately that subset, so `tsc` flags
 * a test that reaches for something the Bun shim does not implement):
 *
 * - `test(name, [options], fn)` with `options.skip` / `options.todo` /
 *   `options.timeout` (`concurrency` is accepted and ignored).
 * - `t.name`, `t.skip(reason)`, `t.after(fn)`, `t.diagnostic(msg)`,
 *   `t.test(name, [options], fn)` (subtests; `options.skip`/`todo` honoured;
 *   run sequentially inside the parent; a failing subtest fails the parent).
 * - file-level `before` / `after` / `afterEach` (Bun's `beforeAll` /
 *   `afterAll` / `afterEach`).
 *
 * Known Bun-side differences (scope boundaries, not bugs):
 * - A runtime `t.skip(reason)` cannot be expressed in `bun:test` 1.2, so the
 *   test is reported as PASSED and a `[skip] name: reason` line is printed
 *   instead. Oracle-gated suites therefore need the same "0 skipped" scrutiny
 *   under Bun that docs/TESTING.md asks for under Node — grep for `[skip]`.
 * - Subtests are not reported individually; the parent's pass/fail covers them.
 * - No per-test default timeout (matching `node --test`): Bun's own 5 s
 *   default would kill the NumPy/pyarrow/scipy oracle tests.
 *
 * Also exports {@link spyMethod}, a runtime-neutral stand-in for
 * `node:test`'s `mock.method` (whose Bun shim does not count calls).
 */
import * as nodeTest from "node:test";

/** The subset of `node:test`'s `TestContext` that works on both runtimes. */
export interface TestContext {
  readonly name: string;
  skip(reason?: string): void;
  after(fn: () => unknown): void;
  /** Informational message (node:test's `t.diagnostic`; printed under Bun). */
  diagnostic(message: string): void;
  test(name: string, fn: (t: TestContext) => unknown): Promise<void>;
  test(name: string, options: TestOptions, fn: (t: TestContext) => unknown): Promise<void>;
}

export interface TestOptions {
  skip?: boolean | string;
  todo?: boolean | string;
  timeout?: number;
  concurrency?: number | boolean;
}

export type TestBody = (t: TestContext) => unknown;

export interface TestFn {
  (name: string, fn: TestBody): void;
  (name: string, options: TestOptions, fn: TestBody): void;
}

export interface Harness {
  test: TestFn;
  before(fn: () => unknown): void;
  after(fn: () => unknown): void;
  afterEach(fn: () => unknown): void;
}

type BunTestFn = ((name: string, fn: () => unknown, timeout?: number) => void) & {
  skip(name: string, fn: () => unknown): void;
  todo(name: string, fn?: () => unknown): void;
};

interface BunTestModule {
  test: BunTestFn;
  beforeAll(fn: () => unknown): void;
  afterAll(fn: () => unknown): void;
  afterEach(fn: () => unknown): void;
}

/**
 * Pass the calling file's own `bun:test` module under Bun, `null` under Node.
 * Must be called once per test file (see the module comment for why).
 */
export function makeTest(bunTest: unknown): Harness {
  if (!bunTest) {
    return {
      test: nodeTest.test as unknown as TestFn,
      before: nodeTest.before,
      after: nodeTest.after,
      afterEach: nodeTest.afterEach,
    };
  }
  const bun = bunTest as BunTestModule;
  const test = ((name: string, a: TestOptions | TestBody, b?: TestBody) => {
    const [opts, fn] = typeof a === "function" ? [{} as TestOptions, a] : [a ?? {}, b as TestBody];
    if (opts.todo) return bun.test.todo(name);
    if (opts.skip) {
      const reason = typeof opts.skip === "string" ? ` (${opts.skip})` : "";
      return bun.test.skip(`${name}${reason}`, () => {});
    }
    // Bun treats a 0 timeout as "use the default" (5 s); ~24 days is "none".
    bun.test(name, () => runWithContext(name, fn), opts.timeout ?? 2 ** 31 - 1);
  }) as TestFn;
  return { test, before: bun.beforeAll, after: bun.afterAll, afterEach: bun.afterEach };
}

async function runWithContext(name: string, fn: TestBody): Promise<void> {
  const afters: Array<() => unknown> = [];
  const subtests: Array<Promise<void>> = [];
  let queue: Promise<unknown> = Promise.resolve();
  let skipped = false;
  const t: TestContext = {
    name,
    skip(reason) {
      // node:test records one skip per test however often t.skip() is called.
      if (skipped) return;
      skipped = true;
      console.log(`[skip] ${name}${reason ? `: ${reason}` : ""}`);
    },
    after(f) {
      afters.push(f);
    },
    diagnostic(message) {
      console.log(`# ${name}: ${message}`);
    },
    test(subName: string, a: TestOptions | TestBody, b?: TestBody) {
      const options: TestOptions = typeof a === "function" ? {} : a;
      const subFn = (typeof a === "function" ? a : b) as TestBody;
      const off = options.skip || options.todo;
      if (off) {
        console.log(`[skip] ${subName}${typeof off === "string" ? `: ${off}` : ""}`);
        return Promise.resolve();
      }
      // node:test runs a parent's subtests one at a time; keep that ordering.
      const p = queue.then(() => runWithContext(subName, subFn));
      queue = p.catch(() => {});
      subtests.push(p);
      return p;
    },
  };
  try {
    await fn(t);
    const results = await Promise.allSettled(subtests);
    const failed = results.filter((r): r is PromiseRejectedResult => r.status === "rejected");
    if (failed.length === 1) throw failed[0]!.reason;
    if (failed.length > 1) throw new AggregateError(failed.map((r) => r.reason), `${failed.length} subtests of "${name}" failed`);
  } finally {
    for (const f of afters.reverse()) await f();
  }
}

/**
 * Wrap `obj[key]` so calls are counted (and still forwarded). Runtime-neutral
 * replacement for `node:test`'s `mock.method(obj, key)` + `.mock.callCount()`.
 */
export function spyMethod<T extends object, K extends keyof T>(
  obj: T,
  key: K,
): { callCount(): number; restore(): void } {
  const hadOwn = Object.prototype.hasOwnProperty.call(obj, key);
  const original = obj[key] as unknown as (...args: unknown[]) => unknown;
  let calls = 0;
  (obj as Record<K, unknown>)[key] = function (this: unknown, ...args: unknown[]) {
    calls++;
    return original.apply(this, args);
  };
  return {
    callCount: () => calls,
    restore() {
      if (hadOwn) (obj as Record<K, unknown>)[key] = original;
      else delete (obj as Record<K, unknown>)[key];
    },
  };
}
