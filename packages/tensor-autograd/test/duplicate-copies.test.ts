import assert from "node:assert/strict";
import { cpSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { random } from "@johnhenry/math-plus-tensor-core";
import { variable, nn } from "../src/index.ts";

// Regression for issue #170: on esm.sh the package's entries are bundled
// separately, so `nn.Parameter` extended a SECOND copy of `Variable` with its
// own id counter. Ids then collided with the tape's and backward() failed.
// Simulate that by loading a byte-identical second copy of src/ as separate
// module instances and mixing it with the primary copy.

const here = dirname(fileURLToPath(import.meta.url));
const copyDir = join(here, "..", `.dup-copy-${process.pid}`);

test("a second bundled copy of the module graph shares one id counter (#170)", async () => {
  cpSync(join(here, "..", "src"), copyDir, { recursive: true });
  try {
    const copy = (await import(pathToFileURL(join(copyDir, "index.ts")).href)) as typeof import("../src/index.ts");
    assert.notEqual(copy.Variable, (await import("../src/index.ts")).Variable, "copies must be distinct classes");

    const rng = random.seed(7);
    // Parameters come from the duplicate copy, the data and tape ops from the primary.
    const l1 = new copy.nn.Linear(3, 4, { dtype: "f64", rng });
    const l2 = new copy.nn.Linear(4, 1, { dtype: "f64", rng });
    const x = variable(random.uniform([5, 3], { rng, dtype: "f64" }));

    const ids = new Set<number>();
    for (const v of [x, ...l1.parameters(), ...l2.parameters()]) {
      assert.ok(!ids.has(v.id), `duplicate Variable id ${v.id}`);
      ids.add(v.id);
    }

    const out = l2.forward(l1.forward(x).relu());
    out.mul(out).sum().backward();
    for (const p of [...l1.parameters(), ...l2.parameters()]) assert.ok(p.grad, "parameter received a gradient");

    // And a primary-copy nn model still trains alongside.
    const m = new nn.Linear(3, 1, { dtype: "f64", rng });
    m.forward(x).sum().backward();
    assert.ok(m.weight.grad);
  } finally {
    rmSync(copyDir, { recursive: true, force: true });
  }
});
