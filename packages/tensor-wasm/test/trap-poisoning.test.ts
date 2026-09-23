/**
 * Trap poisoning (issue #46, from the Woxi study's WASM-panic finding): a
 * Rust panic on wasm32-unknown-unknown becomes an `unreachable` trap, and a
 * trapped instance's memory state is undefined -- so the first trap must
 * permanently poison the Kernels instance and every later use must fail
 * LOUDLY, never silently compute on corrupt memory.
 *
 * The deterministic trap trigger: an out-of-bounds load. Reading via a
 * pointer far beyond linear memory's extent traps with "memory access out
 * of bounds" — a genuine trap that exists on any kernel taking pointers.
 * (The original trigger, `alloc` with a non-power-of-two align, stopped
 * panicking in issue #55 Phase 2: invalid layouts are now a DEFINED null
 * return so the native FFI build has a catchable failure signal instead of
 * a process abort — see the "defined failures do not poison" test below.)
 * This goes through the raw exports surface, which is exactly the point:
 * the guard lives on the rebound exports, so even direct exports usage is
 * covered.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Kernels } from "../src/index.ts";

/** Trigger a real WASM trap on `kernels` and return the error it surfaced as. */
function poisonViaOobLoad(kernels: Kernels): Error {
  try {
    // Read + write a single "element" at a pointer far past linear memory.
    kernels.exports.add_f32_strided(2 ** 30, 0, 1, 2 ** 30, 0, 1, 2 ** 30, 0, 1, 1);
  } catch (err) {
    return err as Error;
  }
  throw new Error("expected the out-of-bounds load to trap");
}

test("a real WASM trap surfaces as a wrapped error with the trap as its cause, and poisons the instance", async () => {
  const kernels = await Kernels.load();
  assert.equal(kernels.poisoned, false);
  assert.equal(kernels.poisonedBy, undefined);

  const err = poisonViaOobLoad(kernels);
  assert.match(err.message, /trapped/);
  assert.match(err.message, /Kernels\.load\(\)/);
  assert.ok(err.cause instanceof WebAssembly.RuntimeError, `cause should be the original trap, got ${err.cause}`);

  assert.equal(kernels.poisoned, true);
  assert.equal(kernels.poisonedBy, "add_f32_strided");
});

test("every subsequent kernel call on a poisoned instance throws the clear poisoned error", async () => {
  const kernels = await Kernels.load();
  // Allocate BEFORE poisoning so we can also exercise an ...Into call path.
  const a = kernels.fromArray(new Float32Array([1, 2, 3]), [3]);
  const b = kernels.fromArray(new Float32Array([4, 5, 6]), [3]);
  const out = kernels.zeros([3]);

  poisonViaOobLoad(kernels);

  assert.throws(() => kernels.zeros([4]), /poisoned/);
  assert.throws(() => kernels.fromArray(new Float32Array([1]), [1]), /poisoned/);
  // addInto hits whichever guarded export applies (SIMD fast path or the
  // scalar strided kernel) -- both share the same poison state.
  assert.throws(() => kernels.addInto(out, a, b), /poisoned/);
});

test("toFloat32Array on a pre-existing tensor refuses to read a poisoned instance's memory", async () => {
  const kernels = await Kernels.load();
  const t = kernels.fromArray(new Float32Array([1, 2, 3, 4]), [4]);
  assert.deepEqual([...t.toFloat32Array()], [1, 2, 3, 4]); // healthy read first

  poisonViaOobLoad(kernels);
  assert.throws(() => t.toFloat32Array(), /poisoned|untrustworthy/);
});

test("free() on a poisoned instance is a silent no-op (cleanup paths must not throw), not a call into the corrupt allocator", async () => {
  const kernels = await Kernels.load();
  const t = kernels.fromArray(new Float32Array([1, 2]), [2]);
  poisonViaOobLoad(kernels);
  t.free(); // must not throw
  assert.equal(kernels.poisonedBy, "add_f32_strided"); // and must not have re-entered wasm (poisonedBy unchanged)
});

test("poison is per-instance: a fresh Kernels.load() after poisoning works normally", async () => {
  const poisonedKernels = await Kernels.load();
  poisonViaOobLoad(poisonedKernels);
  assert.equal(poisonedKernels.poisoned, true);

  const fresh = await Kernels.load();
  assert.equal(fresh.poisoned, false);
  const a = fresh.fromArray(new Float32Array([1, 2, 3]), [3]);
  const b = fresh.fromArray(new Float32Array([10, 20, 30]), [3]);
  const out = fresh.zeros([3]);
  fresh.addInto(out, a, b);
  assert.deepEqual([...out.toFloat32Array()], [11, 22, 33]);
});

test("ordinary JS validation errors do NOT poison the instance", async () => {
  const kernels = await Kernels.load();
  const nonSquare = kernels.zeros([2, 3]);
  const b = kernels.zeros([2]);
  const out = kernels.zeros([2]);
  assert.throws(() => kernels.solveInto(out, nonSquare, b), RangeError);
  assert.equal(kernels.poisoned, false);

  // And the instance still works fine afterward.
  const x = kernels.fromArray(new Float32Array([5]), [1]);
  const y = kernels.fromArray(new Float32Array([7]), [1]);
  const sum = kernels.zeros([1]);
  kernels.addInto(sum, x, y);
  assert.deepEqual([...sum.toFloat32Array()], [12]);
});

test("defined failures do NOT poison: alloc with an invalid layout returns null (issue #55 Phase 2), surfaced as a plain JS error", async () => {
  const kernels = await Kernels.load();
  // Non-power-of-two align: previously a Rust panic (a trap here, a process
  // abort on the native FFI build); now a defined null return on both.
  assert.equal(kernels.exports.alloc(16, 3), 0);
  assert.equal(kernels.poisoned, false);
  // The public API surfaces null as "allocation failed", instance unharmed.
  const t = kernels.zeros([4]);
  assert.deepEqual([...t.toFloat32Array()], [0, 0, 0, 0]);
});
