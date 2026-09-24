/**
 * Backend behaviour the conformance fixtures don't pin: dtype support and
 * widening, lifetime (scope/dispose/destroy), error paths, the undefined-
 * by-contract fully-masked attention row, and drop-in compatibility with
 * laya-js's `@johnhenry/backend-cpu@0.2.0` (the implementation this package
 * replaces; pinned as a devDependency for this comparison and the benchmark).
 */
import assert from "node:assert/strict";
import { host, toF32, type Backend, type DType, type HostTensor, type Tensor } from "@johnhenry/tensor-backend";
import { createCpuBackend as createLayaCpuBackend } from "@johnhenry/backend-cpu";
import { erf as canonicalErf, geluErf } from "@johnhenry/math-plus-tensor-core";
import { CpuTensor, createCpuBackend, erf, geluScalar } from "../src/index.ts";
import { seeded, testFns } from "./helpers.ts";
import { makeTest } from "../../../test/harness.ts";

// Each file imports bun:test itself (see test/harness.ts).
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const harness = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
const { it } = testFns(harness);

it("supports f32/i32/bool; f16/bf16 are widened on upload and cast to them throws", async () => {
  const b = createCpuBackend();
  assert.equal(b.name, "cpu");
  for (const d of ["f32", "i32", "bool"] as const) assert.equal(b.supports(d), true, d);
  for (const d of ["f16", "bf16"] as const) assert.equal(b.supports(d), false, d);
  const f16 = await b.fromHost(host("f16", [3], [1.5, -2, 0.0999755859375]));
  const bf16 = await b.fromHost(host("bf16", [2], [1.5, -3]));
  assert.equal(f16.dtype, "f32");
  assert.equal(bf16.dtype, "f32");
  assert.deepEqual([...toF32(await b.read(f16))], [1.5, -2, 0.0999755859375]);
  assert.deepEqual([...toF32(await b.read(bf16))], [1.5, -3]);
  assert.throws(() => b.cast(f16, "f16"), /not supported/);
  assert.throws(() => b.cast(f16, "bf16"), /not supported/);
});

it("fromHost is async, copies at call time, validates length and normalizes bool to 0/1", async () => {
  const b = createCpuBackend();
  const data = new Float32Array([1, 2]);
  const p = b.fromHost(host("f32", [2], data));
  assert.ok(p instanceof Promise);
  data[0] = 9;
  assert.deepEqual([...toF32(await b.read(await p))], [1, 2]);
  await assert.rejects(b.fromHost({ dtype: "f32", shape: [3], data: new Float32Array(2) }), /2 values for shape \[3\]/);
  const m = await b.fromHost({ dtype: "bool", shape: [3], data: Uint8Array.from([0, 7, 255]) });
  assert.deepEqual([...(await b.read(m)).data], [0, 1, 1]);
});

it("nested scopes keep returned tensors (array and object) and free the rest; dispose is idempotent", async () => {
  const b = createCpuBackend();
  const x = await b.fromHost(host("f32", [2], [1, 2]));
  let inner: CpuTensor | undefined, tmp: CpuTensor | undefined;
  const out = b.scope(() => {
    const kept = b.scope(() => {
      tmp = b.exp(x);
      inner = b.add(tmp, x);
      return { y: inner, n: 3 };
    });
    assert.equal(tmp!.disposed, true);
    assert.equal(kept.y.disposed, false);
    return [b.mul(kept.y, x)];
  });
  assert.equal(inner!.disposed, true, "inner result is freed by the outer scope");
  assert.equal(out[0]!.disposed, false);
  assert.equal(x.disposed, false);
  assert.throws(() => b.scope(() => { b.exp(x); throw new Error("boom"); }), /boom/);
  b.dispose(x);
  b.dispose(x);
  assert.throws(() => b.exp(x), /after dispose/);
});

it("destroy releases tensors of open scopes", async () => {
  const b = createCpuBackend();
  const x = await b.fromHost(host("f32", [1], [1]));
  let y: CpuTensor | undefined;
  b.scope(() => {
    y = b.exp(x);
    b.destroy();
    return null;
  });
  assert.equal(y!.disposed, true);
});

it("reshape(-1), negative slice bounds, where broadcasting, sort, split", async () => {
  const b = createCpuBackend();
  const a = await b.fromHost(host("f32", [2, 3], [1, 2, 3, 4, 5, 6]));
  assert.deepEqual(b.reshape(a, [-1, 2]).shape, [3, 2]);
  assert.throws(() => b.reshape(a, [4, -1]), /reshape/);
  assert.deepEqual([...toF32(await b.read(b.slice(a, [0, -2], [2, 3])))], [2, 3, 5, 6]);
  const [cond, neg1] = await Promise.all([b.fromHost(host("bool", [1, 3], [1, 0, 1])), b.fromHost(host("f32", [1], [-1]))]);
  assert.deepEqual([...toF32(await b.read(b.where(cond, a, neg1)))], [1, -1, 3, 4, -1, 6]);
  const srt = b.sort(await b.fromHost(host("f32", [2, 2], [3, 1, -1, -5])), 0);
  assert.deepEqual([...toF32(await b.read(srt))], [-1, -5, 3, 1]);
  const parts = b.split(a, 3, 1);
  assert.deepEqual(await Promise.all(parts.map(async (p) => [...toF32(await b.read(p))])), [[1, 4], [2, 5], [3, 6]]);
  assert.throws(() => b.split(a, 2, 1), /cannot split 3 into 2/);
});

it("error paths name the problem", async () => {
  const b = createCpuBackend();
  const x = await b.fromHost(host("f32", [2, 3], [1, 2, 3, 4, 5, 6]));
  assert.throws(() => b.sum(x, 2), /axis 2 out of range/);
  assert.throws(() => b.transpose(x, [0, 0]), /not a permutation/);
  assert.throws(() => b.add(x, b.reshape(x, [3, 2])), /broadcast/);
  assert.throws(() => b.linear(x, b.reshape(x, [3, 2])), /linear \[2,3\] with weight \[3,2\]/);
  const ids = await b.fromHost(host("i32", [2], [0, 5]));
  assert.throws(() => b.embedding(x, ids), /out of range/);
  const gi = await b.fromHost(host("i32", [1, 1], [-3]));
  assert.throws(() => b.gatherRows(b.reshape(x, [1, 2, 3]), gi), /index -3 out of range for length 2/);
  assert.throws(() => b.matmul(b.reshape(x, [6]), x), /rank >= 2/);
});

it("sdpa: a fully-masked query row (undefined by contract) yields zeros, not NaN", async () => {
  const b = createCpuBackend();
  const [q, k, v] = await Promise.all([1, 2, 3].map((s) => b.fromHost(host("f32", [1, 1, 2, 4], seeded(8, s)))));
  const mask = await b.fromHost(host("bool", [1, 1, 2, 2], [1, 0, 0, 0]));
  const out = toF32(await b.read(b.sdpa(q!, k!, v!, mask, 0.5)));
  assert.deepEqual([...out.subarray(4)], [0, 0, 0, 0]);
  assert.ok(out.subarray(0, 4).every(Number.isFinite));
});

it("erf / gelu are math-plus's canonical scalar functions, rounded once to f32", async () => {
  assert.equal(erf, canonicalErf);
  assert.equal(geluScalar, geluErf);
  const b = createCpuBackend();
  const xs = [-3, -0.5, 0, 1e-3, 0.7, 2.5];
  const x = await b.fromHost(host("f32", [xs.length], xs));
  const e = toF32(await b.read(b.erf(x)));
  const g = toF32(await b.read(b.gelu(x)));
  xs.forEach((v, i) => {
    const f = Math.fround(v);
    assert.equal(e[i], Math.fround(canonicalErf(f)));
    assert.equal(g[i], Math.fround(geluErf(f)));
  });
});

// ---- drop-in compatibility with laya-js @johnhenry/backend-cpu@0.2.0 ------------------

type Op = (b: Backend<Tensor>, xs: Tensor[]) => Tensor | Tensor[];
const COMPAT: Array<[string, HostTensor[], Op]> = [
  ["add f32+i32", [host("f32", [2, 3], seeded(6, 1)), host("i32", [3], [1, -2, 3])], (b, [x, y]) => b.add(x!, y!)],
  ["sub i32-bool", [host("i32", [3], [4, 5, 6]), host("bool", [3], [1, 0, 1])], (b, [x, y]) => b.sub(x!, y!)],
  ["mul bool*bool", [host("bool", [2], [1, 1]), host("bool", [2], [1, 0])], (b, [x, y]) => b.mul(x!, y!)],
  ["div i32/i32", [host("i32", [2], [7, -7]), host("i32", [2], [2, 2])], (b, [x, y]) => b.div(x!, y!)],
  ["maximum i32", [host("i32", [2, 2], [1, 5, -3, 0]), host("i32", [2], [2, -1])], (b, [x, y]) => b.maximum(x!, y!)],
  ["where i32", [host("bool", [3], [1, 0, 1]), host("i32", [3], [1, 2, 3]), host("i32", [1], [-9])], (b, [c, x, y]) => b.where(c!, x!, y!)],
  ["scale i32", [host("i32", [2], [3, -4])], (b, [x]) => b.scale(x!, 0.5)],
  ["sum i32", [host("i32", [2, 3], [1, 2, 3, 4, 5, 6])], (b, [x]) => b.sum(x!, 0)],
  ["sum bool keepDims", [host("bool", [2, 3], [1, 0, 1, 1, 1, 0])], (b, [x]) => b.sum(x!, 1, true)],
  ["max i32", [host("i32", [2, 3], [1, 9, 3, 4, -5, 6])], (b, [x]) => b.max(x!, -1)],
  ["cast f32->i32 truncates", [host("f32", [4], [1.9, -1.9, 2.5, -0.5])], (b, [x]) => b.cast(x!, "i32")],
  ["cast i32->bool", [host("i32", [3], [0, 2, -1])], (b, [x]) => b.cast(x!, "bool")],
  ["cast bool->f32", [host("bool", [2], [1, 0])], (b, [x]) => b.cast(x!, "f32")],
  ["concat i32+f32", [host("i32", [1, 2], [1, 2]), host("f32", [1, 2], [0.5, 1.5])], (b, xs) => b.concat(xs, 0)],
  ["neg i32", [host("i32", [3], [1, -2, 2 ** 31 - 1])], (b, [x]) => b.neg!(x!)],
  ["abs bool", [host("bool", [2], [1, 0])], (b, [x]) => b.abs!(x!)],
  ["cumsum i32", [host("i32", [4], [1, 2, 3, 4])], (b, [x]) => b.cumsum!(x!, 0)],
  ["mean i32", [host("i32", [2, 2], [1, 2, 4, 4])], (b, [x]) => b.mean!(x!, 1)],
  ["min bool", [host("bool", [2, 2], [1, 0, 1, 1])], (b, [x]) => b.min!(x!, 1)],
  ["argmax ties -> first", [host("f32", [2, 3], [1, 3, 3, -1, -1, -2])], (b, [x]) => b.argmax!(x!, 1)],
  ["logicalNot f32", [host("f32", [3], [0, 2, -0])], (b, [x]) => b.logicalNot!(x!)],
  ["equal i32 vs f32", [host("i32", [2], [1, 2]), host("f32", [2], [1, 2.5])], (b, [x, y]) => b.equal!(x!, y!)],
  ["sqrt i32", [host("i32", [2], [4, 2])], (b, [x]) => b.sqrt!(x!)],
  ["meanPool", [host("f32", [1, 3, 2], [1, 2, 3, 4, 5, 6]), host("bool", [1, 3], [1, 0, 1])], (b, [x, m]) => b.meanPool!(x!, m!)],
  ["geglu", [host("f32", [2, 4], seeded(8, 3))], (b, [x]) => b.geglu!(x!)],
  ["rope", [host("f32", [1, 2, 3, 4], seeded(24, 4))], (b, [x]) => b.rope(x!, 10000)],
  ["layerNorm", [host("f32", [2, 5], seeded(10, 5)), host("f32", [5], seeded(5, 6))], (b, [x, w]) => b.layerNorm(x!, w!, null, 1e-5)],
  ["gatherRows", [host("f32", [1, 3, 2], [1, 2, 3, 4, 5, 6]), host("i32", [1, 2], [-1, 0])], (b, [x, i]) => b.gatherRows(x!, i!)],
];

it("drop-in compatible with @johnhenry/backend-cpu@0.2.0: same result dtypes, shapes and values", async () => {
  const ours = createCpuBackend() as unknown as Backend<Tensor>;
  const theirs = createLayaCpuBackend() as unknown as Backend<Tensor>;
  const failures: string[] = [];
  for (const [name, inputs, op] of COMPAT) {
    const run = async (b: Backend<Tensor>) => {
      const xs = await Promise.all(inputs.map((h) => b.fromHost(h)));
      const out = op(b, xs);
      return Promise.all((Array.isArray(out) ? out : [out]).map((t) => b.read(t)));
    };
    const [a, c] = [await run(ours), await run(theirs)];
    a.forEach((got, i) => {
      const want = c[i]!;
      if (got.dtype !== want.dtype) failures.push(`${name}: dtype ${got.dtype} vs ${want.dtype}`);
      else if (JSON.stringify(got.shape) !== JSON.stringify(want.shape)) failures.push(`${name}: shape [${got.shape}] vs [${want.shape}]`);
      else {
        const g = toF32(got), w = toF32(want);
        const bad = g.findIndex((v, k) => Math.abs(v - w[k]!) > 1e-6 * Math.max(1, Math.abs(w[k]!)));
        if (bad >= 0) failures.push(`${name}: [${bad}] ${g[bad]} vs ${w[bad]}`);
      }
    });
  }
  assert.deepEqual(failures, []);
});

it("same dtype table as backend-cpu@0.2.0 for every supported input dtype (supports/cast)", () => {
  const ours = createCpuBackend();
  const theirs = createLayaCpuBackend();
  for (const d of ["f32", "f16", "bf16", "i32", "bool"] as DType[]) assert.equal(ours.supports(d), theirs.supports(d), d);
});
