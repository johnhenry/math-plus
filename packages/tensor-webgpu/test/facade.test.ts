/**
 * `createWebGpuDevice()` (issue #146): the device facade over
 * @johnhenry/backend-webgpu — explicit async transfers to and from
 * tensor-core `Tensor`s as chainable `WebGpuArray`s (every device dtype,
 * views with offsets, refusal of implicit conversions), backend ops on
 * uploaded arrays (via `.handle` / `wrap`) cross-checked against
 * tensor-core on the CPU, the IR -> WGSL fusion on the backend's
 * runtime (`fuse`/`compile`) cross-checked against tensor-compile's CPU
 * `forward`, scope tracking of fused results, sharing a device from
 * `detectWebGPU()`, `destroy`, and the absence of the API removed in 0.3.0.
 * In-process on Dawn (Node and Bun); skips, never fails, without a WebGPU
 * adapter.
 */
import assert from "node:assert/strict";
import { compile, Traced } from "@johnhenry/math-plus-tensor-compile";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { readFileSync } from "node:fs";
import { getGpu } from "@johnhenry/backend-webgpu";
import { DeviceArray } from "@johnhenry/math-plus-tensor-cpu";
import * as api from "../src/index.ts";
import { createWebGpuDevice, detectWebGPU, webGpuUnavailableReason, type WebGpuDevice } from "../src/index.ts";
import { lookupBackend } from "../src/bridge.ts";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);

const skip = await webGpuUnavailableReason();
let shared: WebGpuDevice | undefined;
const device = async (): Promise<WebGpuDevice> => (shared ??= await createWebGpuDevice());
after(() => shared?.destroy());

function lcg(n: number, seed: number): Float32Array {
  let s = seed >>> 0;
  return Float32Array.from({ length: n }, () => ((s = (s * 1664525 + 1013904223) >>> 0) / 0xffffffff) * 2 - 1);
}

function assertClose(got: ArrayLike<number>, want: ArrayLike<number>, tol: number, label: string): void {
  assert.equal(got.length, want.length, `${label}: length`);
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i]! - want[i]!);
    if (!(d <= tol * Math.max(1, Math.abs(want[i]!)))) assert.fail(`${label}[${i}]: got ${got[i]}, want ${want[i]}`);
  }
}

test("createWebGpuDevice: a real device with no global default; info and dtype support are reported", { skip: skip ?? false }, async () => {
  const gpu = await device();
  assert.equal(gpu.name, "webgpu");
  assert.equal(gpu.backend.name, "webgpu");
  assert.ok(gpu.device, "the GPUDevice is exposed for sharing");
  assert.ok(gpu.info.source.length > 0);
  for (const d of ["f32", "bf16", "i32", "bool"] as const) assert.equal(gpu.supports(d), true, d);
  assert.equal(gpu.supports("f16"), gpu.device.features.has("shader-f16"));
  const other = await createWebGpuDevice();
  assert.notEqual(other.device, gpu.device, "each call creates its own device");
  other.destroy();
});

test("a backend createWebGpuDevice() creates under Dawn sleeps before long readbacks, with the 15 ms threshold (bridge.ts SLEEP_THRESHOLD_MS_DEFAULT)", { skip: skip ?? false }, async () => {
  const gpu = await device();
  assert.equal(gpu.backend.rt.sleepThresholdMs, 15);
  // backend-webgpu's default: on under Dawn (this process has no navigator.gpu), off for navigator.gpu.
  const hasNavigatorGpu = Boolean((globalThis as { navigator?: { gpu?: unknown } }).navigator?.gpu);
  assert.equal(gpu.backend.rt.sleepWhileWaiting, !hasNavigatorGpu);
});

test("fromTensor/toTensor round-trip every device dtype exactly, including a view with an offset", { skip: skip ?? false }, async () => {
  const gpu = await device();
  const cases: Tensor[] = [
    Tensor.from([1.5, -2, 3.25, 0, 1e-3, -7], { dtype: "f32" }).reshape([2, 3]),
    Tensor.from([1, -2, 3, 2 ** 30, -(2 ** 31)], { dtype: "i32" }),
    Tensor.from([1, 0, 1, 1], { dtype: "bool" }),
    Tensor.from([0.5, -1.25, 3], { dtype: "bf16" }),
    Tensor.from(Array.from(lcg(12, 3)), { dtype: "f32" }).reshape([4, 3]).slice({ start: 1, end: 3 }), // rows 1..2: offset 3, contiguous
  ];
  if (gpu.supports("f16")) cases.push(Tensor.from([0.5, -1.25, 65504, 6e-5], { dtype: "f16" }));
  for (const t of cases) {
    const x = await gpu.fromTensor(t);
    assert.deepEqual([...x.shape], [...t.shape]);
    assert.equal(x.dtype, t.dtype);
    const back = await gpu.toTensor(x);
    assert.equal(back.dtype, t.dtype);
    assert.deepEqual(back.toArray(), t.toArray(), `${t.dtype} [${t.shape}]`);
    gpu.dispose(x);
  }
});

test("fromTensor refuses implicit conversions synchronously, before any Promise: non-contiguous views and non-device dtypes (labelled tensor-webgpu)", { skip: skip ?? false }, async () => {
  const gpu = await device();
  assert.throws(() => gpu.fromTensor(Tensor.zeros([3, 4], { dtype: "f32" }).transpose()), /tensor-webgpu: .*contiguous\(\) first/);
  // f64/i64 are permanently unsupported on WebGPU (real WGSL spec limits --
  // see backend-webgpu's README "Limitations"), rejected by #checkDtype's
  // generic message now, not a dtype-specific "cast(...)" hint.
  assert.throws(() => gpu.fromTensor(Tensor.zeros([2], { dtype: "f64" })), /does not support f64/);
  assert.throws(() => gpu.fromTensor(Tensor.zeros([2], { dtype: "i64" })), /does not support i64/);
  if (!gpu.supports("f16")) assert.throws(() => gpu.fromTensor(Tensor.from([1], { dtype: "f16" })), /does not support f16/);
});

test("fromTensor/fromHost return chainable arrays; the transfer/dispose methods also take raw backend tensors; handle/wrap bridge to gpu.backend", { skip: skip ?? false }, async () => {
  const gpu = await device();
  const x = await gpu.fromTensor(Tensor.from([1, 4, 9], { dtype: "f32" }));
  assert.ok(x instanceof DeviceArray);
  assert.equal(x.device, gpu);
  assert.deepEqual([...(await x.sqrt().mul(2).toTensor()).data], [2, 4, 6]);
  const raw = await gpu.backend.fromHost({ dtype: "f32", shape: [3], data: Float32Array.from([1, 2, 3]) });
  assert.deepEqual([...(await gpu.toTensor(raw)).data], [1, 2, 3], "gpu.toTensor takes a raw backend tensor");
  const sum = gpu.wrap(gpu.backend.add(x.handle, raw));
  assert.deepEqual([...(await gpu.toHost(sum)).data], [2, 6, 12], "gpu.toHost takes an array");
  assert.throws(() => x.add(raw as never), /expected a WebGpuArray/);
  gpu.dispose(raw);
  gpu.dispose(sum);
  assert.equal(sum.disposed, true);
  x.dispose();
});

test("array and backend ops on uploaded arrays match tensor-core on the CPU (matmul, linear with bias, softmax, layerNorm)", { skip: skip ?? false }, async () => {
  const gpu = await device();
  const b = gpu.backend;
  const A = Tensor.fromTypedArray(lcg(96 * 64, 1), [96, 64], { dtype: "f32" });
  const B = Tensor.fromTypedArray(lcg(64 * 72, 2), [64, 72], { dtype: "f32" });
  const W = Tensor.fromTypedArray(lcg(40 * 64, 3), [40, 64], { dtype: "f32" });
  const bias = Tensor.fromTypedArray(lcg(40, 4), [40], { dtype: "f32" });
  const [a, bb, w, bi] = await Promise.all([A, B, W, bias].map((t) => gpu.fromTensor(t)));
  const outs = gpu.scope(() => ({ ab: a!.matmul(bb!), lin: gpu.wrap(b.linear(a!.handle, w!.handle, bi!.handle)), sm: a!.softmax(-1), ln: a!.layerNorm(null, null, 1e-5) }));
  const ab = await gpu.toTensor(outs.ab);
  assertClose(ab.data as Float32Array, A.matmul(B).data as Float32Array, 1e-4, "matmul");
  const lin = await gpu.toTensor(outs.lin);
  const linRef = A.matmul(W.transpose()).add(bias);
  assertClose(lin.data as Float32Array, linRef.contiguous().data as Float32Array, 1e-4, "linear");
  const sm = await gpu.toTensor(outs.sm);
  assertClose(sm.data as Float32Array, A.softmax(-1).contiguous().data as Float32Array, 1e-5, "softmax");
  const ln = await gpu.toTensor(outs.ln);
  const lnRef = new Float32Array(96 * 64);
  const ad = A.data as Float32Array;
  for (let r = 0; r < 96; r++) {
    const row = ad.subarray(r * 64, r * 64 + 64);
    const mu = row.reduce((p, v) => p + v, 0) / 64;
    const varc = row.reduce((p, v) => p + (v - mu) ** 2, 0) / 64;
    for (let c = 0; c < 64; c++) lnRef[r * 64 + c] = (row[c]! - mu) / Math.sqrt(varc + 1e-5);
  }
  assertClose(ln.data as Float32Array, lnRef, 1e-4, "layerNorm");
  for (const t of [a, bb, w, bi, outs.ab, outs.lin, outs.sm, outs.ln]) gpu.dispose(t!);
});

test("fuse/compile: a traced expression runs as ONE dispatch on the backend runtime and matches tensor-compile's CPU forward", { skip: skip ?? false }, async () => {
  const gpu = await device();
  const fn = (x: Traced, y: Traced, z: Traced): Traced => x.mul(y).add(z.exp()).gelu().select(x.gt(0), y.neg());
  const n = 3000; // > one 256-thread workgroup, not a multiple of it
  const X = Tensor.fromTypedArray(lcg(n, 1), [30, 100], { dtype: "f32" });
  const Y = Tensor.fromTypedArray(lcg(n, 2), [30, 100], { dtype: "f32" });
  const Z = Tensor.fromTypedArray(lcg(n, 3), [30, 100], { dtype: "f32" });
  const want = compile(3, fn).forward(X, Y, Z);
  const [x, y, z] = await Promise.all([X, Y, Z].map((t) => gpu.fromTensor(t)));
  const before = gpu.backend.rt.stats.dispatches;
  const f = gpu.compile(3, fn);
  const out = f(x!, y!, z!);
  assert.equal(gpu.backend.rt.stats.dispatches - before, 1, "one fused dispatch");
  assert.deepEqual([...out.shape], [30, 100]);
  const got = await gpu.toTensor(out);
  assertClose(got.data as Float32Array, want.contiguous().data as Float32Array, 1e-5, "fused");
  // Same expression via fuse() with a Traced and with a raw IRNode.
  const traced = fn(Traced.input(0), Traced.input(1), Traced.input(2));
  const viaNode = await gpu.toTensor(gpu.fuse(traced.node, [x!, y!, z!]));
  const viaTraced = await gpu.toTensor(gpu.fuse(traced, [x!, y!, z!]));
  assert.deepEqual(viaNode.toArray(), got.toArray());
  assert.deepEqual(viaTraced.toArray(), got.toArray());
});

test("fuse: works on backend views at an element offset; results are tracked by scope; bad inputs are refused", { skip: skip ?? false }, async () => {
  const gpu = await device();
  const b = gpu.backend;
  const X = Tensor.fromTypedArray(lcg(4 * 50, 9), [4, 50], { dtype: "f32" });
  const x = await gpu.fromTensor(X);
  const row2 = gpu.wrap(b.slice(x.handle, [2, 0], [3, 50])); // contiguous view, offset 100
  const twice = new Traced({ kind: "input", index: 0 }).mul(2);
  let inner: ReturnType<typeof gpu.fuse> | undefined;
  const kept = gpu.scope(() => {
    inner = gpu.fuse(twice, [row2]);
    return gpu.fuse(twice, [inner]);
  });
  assert.equal(inner!.disposed, true, "an intermediate fused result is freed by the scope");
  assert.equal(kept.disposed, false);
  const got = (await gpu.toTensor(kept)).data as Float32Array;
  assertClose(got, (X.data as Float32Array).slice(100, 150).map((v) => 4 * v), 0, "offset view");
  assert.throws(() => gpu.fuse(twice.add(new Traced({ kind: "input", index: 1 })), [x, gpu.wrap(b.slice(x.handle, [0, 0], [3, 50]))]), /cannot broadcast/);
  const rawOut = gpu.fuse(twice, [row2.handle]); // raw backend tensors in, a raw tensor out
  assert.ok(!(rawOut instanceof DeviceArray));
  assertClose((await gpu.toHost(rawOut)).data as Float32Array, (X.data as Float32Array).slice(100, 150).map((v) => 2 * v), 0, "raw fuse");
  gpu.dispose(rawOut);
  const i32 = await gpu.fromTensor(Tensor.from([1, 2], { dtype: "i32" }));
  assert.throws(() => gpu.fuse(twice, [i32]), /f32 inputs only/);
  assert.throws(() => gpu.fuse(twice, []), /at least one input/);
  for (const t of [x, row2, kept, i32]) gpu.dispose(t);
});

test("fuse/compile broadcast their inputs like tensor-compile's CPU forward (NumPy rules): trailing axes, size-1 axes, lower rank, offset views, an input the expression ignores; still ONE dispatch", { skip: skip ?? false }, async () => {
  const gpu = await device();
  const b = gpu.backend;
  const fn = (x: Traced, y: Traced, z: Traced): Traced => x.mul(y).add(z).erf().select(x.gt(y), z.neg().exp());
  const cases: [string, number[], number[], number[]][] = [
    ["[4,5,6] · [6] + [5,1]", [4, 5, 6], [6], [5, 1]],
    ["[4,1,6] · [1,5,1] + [4,5,6] (both sides broadcast)", [4, 1, 6], [1, 5, 1], [4, 5, 6]],
    ["[1] · [3,7] + [7]", [1], [3, 7], [7]],
    ["[2,3,1,5] · [3,4,1] + [1] (rank 4 against lower ranks)", [2, 3, 1, 5], [3, 4, 1], [1]],
    ["[300,257] · [257] + [300,1] (more than one workgroup, odd sizes)", [300, 257], [257], [300, 1]],
  ];
  let seed = 40;
  for (const [label, sx, sy, sz] of cases) {
    const numel = (sh: number[]): number => sh.reduce((a, d) => a * d, 1);
    const [X, Y, Z] = [sx, sy, sz].map((sh) => Tensor.fromTypedArray(lcg(numel(sh), seed++), sh, { dtype: "f32" }));
    const want = compile(3, fn).forward(X!, Y!, Z!);
    const [x, y, z] = await Promise.all([X!, Y!, Z!].map((t) => gpu.fromTensor(t)));
    const before = b.rt.stats.dispatches;
    const out = gpu.compile(3, fn)(x!, y!, z!);
    assert.equal(b.rt.stats.dispatches - before, 1, `${label}: one fused dispatch`);
    assert.deepEqual([...out.shape], [...want.shape], `${label}: broadcast shape`);
    assertClose((await gpu.toTensor(out)).data as Float32Array, want.contiguous().data as Float32Array, 1e-5, label);
    for (const t of [x!, y!, z!, out]) gpu.dispose(t);
  }

  // A row view at an element offset broadcast against a matrix, and an input the IR never reads
  // (it still takes part in the broadcast, as in tensor-compile's forward).
  const M = Tensor.fromTypedArray(lcg(5 * 6, 90), [5, 6], { dtype: "f32" });
  const R = Tensor.fromTypedArray(lcg(4 * 6, 91), [4, 6], { dtype: "f32" });
  const U = Tensor.fromTypedArray(lcg(3 * 1 * 1, 92), [3, 1, 1], { dtype: "f32" });
  const [m, r, u] = await Promise.all([M, R, U].map((t) => gpu.fromTensor(t)));
  const row2 = gpu.wrap(b.slice(r!.handle, [2, 0], [3, 6])); // [1, 6] at offset 12
  const expr = (a: Traced, c: Traced, _unused: Traced): Traced => a.sub(c).mul(a);
  const want = compile(3, expr).forward(M, R.slice({ start: 2, end: 3 }), U);
  const got = gpu.fuse(expr(Traced.input(0), Traced.input(1), Traced.input(2)), [m!, row2, u!]);
  assert.deepEqual([...got.shape], [3, 5, 6]);
  assert.deepEqual([...want.shape], [3, 5, 6]);
  assertClose((await gpu.toTensor(got)).data as Float32Array, want.contiguous().data as Float32Array, 1e-6, "offset view + ignored input");
  for (const t of [m!, r!, u!, row2, got]) gpu.dispose(t);
});

test("createWebGpuDevice({ device }) on a detectWebGPU() device: one shared backend, subgroup matrices follow detectWebGPU's adapter check, and the device survives destroy()", { skip: skip ?? false }, async () => {
  const cap = await detectWebGPU({ gpu: (await getGpu({ unsafe: true }))! });
  assert.ok(cap.available, cap.reason);
  const dev = cap.device!;
  const gpu = await createWebGpuDevice({ device: dev });
  const again = await createWebGpuDevice({ device: dev });
  assert.equal(again.backend, gpu.backend, "one backend per device");
  assert.equal(lookupBackend(dev), gpu.backend);
  assert.equal(gpu.backend.hasSubgroupMatrix, cap.gemm!.subgroupMatrix, "subgroup matrices follow detectWebGPU's adapter check");
  const A = await gpu.fromHost({ dtype: "f32", shape: [8, 16], data: lcg(8 * 16, 5) });
  const B = await gpu.fromHost({ dtype: "f32", shape: [16, 4], data: lcg(16 * 4, 6) });
  const C = A.matmul(B);
  const want = Tensor.fromTypedArray(lcg(8 * 16, 5), [8, 16], { dtype: "f32" }).matmul(Tensor.fromTypedArray(lcg(16 * 4, 6), [16, 4], { dtype: "f32" }));
  assertClose((await gpu.toHost(C)).data as Float32Array, want.data as Float32Array, 1e-5, "matmul on a shared device");
  for (const t of [A, B, C]) gpu.dispose(t);
  gpu.destroy();
  assert.equal(lookupBackend(dev), again.backend, "a device passed in keeps its backend after destroy()");
  dev.destroy();
});

test("destroy() of a device createWebGpuDevice requested releases it and unregisters its backend", { skip: skip ?? false }, async () => {
  const gpu = await createWebGpuDevice();
  const dev = gpu.device;
  const backend = gpu.backend;
  assert.equal(lookupBackend(dev), backend);
  gpu.destroy();
  assert.equal(lookupBackend(dev), undefined, "the destroyed backend is no longer handed out");
});

test("the API removed in 0.3.0 is gone: no deprecated exports, no ./dawn subpath", () => {
  const removed = [
    "toWebGPU", "GPUTensor", "runGemm", "runGemmWGSL", "runGemmF16WGSL", "gemmKernelApplicable",
    "runAttention", "runQKT", "runSoftmax", "runWeightedSum", "runElementwiseWGSL",
    "startProfiling", "stopProfiling", "configureGPURuntime", "backendFor", "requestDawnGPU",
  ];
  assert.deepEqual(removed.filter((name) => name in api), []);
  for (const manifest of ["../package.json", "../jsr.json"]) {
    const exp = (JSON.parse(readFileSync(new URL(manifest, import.meta.url), "utf8")) as { exports: unknown }).exports;
    assert.ok(typeof exp === "string" || !("./dawn" in (exp as object)), `${manifest} still exports ./dawn`);
  }
});
