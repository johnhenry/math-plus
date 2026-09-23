/**
 * toTensor() exercises the lazy dynamic import of @johnhenry/math-plus-tensor-core.
 * tensor-core IS present in this monorepo (a sibling workspace package), so
 * these tests confirm the dynamic import succeeds and produces a correct
 * Tensor — they can't prove "tensor-core absent" from inside the repo (see
 * the issue's own note on this); that half of the constraint is instead
 * verified structurally: package.json lists @johnhenry/math-plus-tensor-core ONLY under
 * peerDependencies + peerDependenciesMeta.optional, never dependencies.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Float64, Int32, Table, vectorFromArray } from "apache-arrow";
import { Frame } from "../src/index.ts";

test("Series.toTensor(): float64 column -> f64 Tensor with matching shape/values", async () => {
  const frame = Frame.fromArrow(new Table({ v: vectorFromArray([1.5, 2.5, 3.5], new Float64()) }));
  const tensor = (await frame.getSeries("v").toTensor()) as {
    shape: readonly number[];
    dtype: string;
    data: Float64Array;
  };
  assert.deepEqual(tensor.shape, [3]);
  assert.equal(tensor.dtype, "f64");
  assert.deepEqual(Array.from(tensor.data), [1.5, 2.5, 3.5]);
});

test("Series.toTensor(): int32 column -> i32 Tensor", async () => {
  const frame = Frame.fromArrow(new Table({ v: vectorFromArray([1, -2, 3], new Int32()) }));
  const tensor = (await frame.getSeries("v").toTensor()) as { dtype: string; data: Int32Array };
  assert.equal(tensor.dtype, "i32");
  assert.deepEqual(Array.from(tensor.data), [1, -2, 3]);
});

test("Series.toTensor() throws a clear error for a dtype Tensor can't represent (utf8)", async () => {
  const { Utf8 } = await import("apache-arrow");
  const frame = Frame.fromArrow(new Table({ name: vectorFromArray(["a", "b"], new Utf8()) }));
  await assert.rejects(() => frame.getSeries("name").toTensor(), /cannot represent/);
});

test("Series.toTensor() throws when the column contains nulls (no null representation in Tensor)", async () => {
  const frame = Frame.fromArrow(new Table({ v: vectorFromArray([1.0, null, 3.0], new Float64()) }));
  await assert.rejects(() => frame.getSeries("v").toTensor(), /null/);
});

test("Frame.toTensor(): all-numeric Frame -> 2D row-major float64 Tensor by default", async () => {
  const frame = Frame.fromArrow(
    new Table({
      a: vectorFromArray([1, 2, 3], new Float64()),
      b: vectorFromArray([10, 20, 30], new Float64()),
    }),
  );
  const tensor = (await frame.toTensor()) as { shape: readonly number[]; dtype: string; data: Float64Array };
  assert.deepEqual(tensor.shape, [3, 2]);
  assert.equal(tensor.dtype, "f64");
  assert.deepEqual(Array.from(tensor.data), [1, 10, 2, 20, 3, 30]);
});

test("@johnhenry/math-plus-frame-arrow's package.json lists @johnhenry/math-plus-tensor-core only as an optional peerDependency, as a caret RANGE covering that package's ACTUAL current version (not an exact pin)", async () => {
  const fs = await import("node:fs/promises");
  const url = await import("node:url");
  const pkgPath = url.fileURLToPath(new URL("../package.json", import.meta.url));
  const pkg = JSON.parse(await fs.readFile(pkgPath, "utf8"));
  // Read tensor-core's own version rather than hardcoding a literal here --
  // this test's job is to enforce the CARET-RANGE invariant, not to pin
  // itself to a version number that a release bump would silently stale.
  //
  // v1 exact-pinned this peer, which sounded safer but wasn't: it made
  // `npm install` ERESOLVE for any consumer already on a newer compatible
  // tensor-core (johnhenry/math-plus#86, filed by mallory, which
  // depends on tensor-core ^0.1.0). frame-arrow's actual coupling to
  // tensor-core is a small, stable structural surface (see tensor.ts's own
  // doc comment -- no static or type-level dependency at all, only a
  // dynamic import at call time), so a caret range is the right contract:
  // wide enough to resolve against sibling packages' own ranges, narrow
  // enough to still reject an actual breaking (0.x -> 0.(x+1)) bump.
  const tensorCorePkgPath = url.fileURLToPath(
    new URL("../../tensor-core/package.json", import.meta.url),
  );
  const tensorCorePkg = JSON.parse(await fs.readFile(tensorCorePkgPath, "utf8"));
  assert.equal(pkg.dependencies?.["@johnhenry/math-plus-tensor-core"], undefined);
  // A caret range (possibly one of several `||` alternatives, so a 0.x minor
  // bump of tensor-core doesn't force a major bump here) covering the ACTUAL version.
  const peerRange: string = pkg.peerDependencies?.["@johnhenry/math-plus-tensor-core"] ?? "";
  assert.ok(
    peerRange.split("||").map((r) => r.trim()).includes(`^${tensorCorePkg.version}`),
    `peer range ${JSON.stringify(peerRange)} must include ^${tensorCorePkg.version}`,
  );
  assert.equal(pkg.peerDependenciesMeta?.["@johnhenry/math-plus-tensor-core"]?.optional, true);
});
