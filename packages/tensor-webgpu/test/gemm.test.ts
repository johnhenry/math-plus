/**
 * GEMM correctness: every kernel family (tiled, skinny, subgroup-matrix,
 * and backend-webgpu's automatic choice) x dtype (f32, f16) x B layout
 * (`[K,N]`, `transB`) x alignment case, through the deprecated
 * `runGemm*` shims, which run on @johnhenry/backend-webgpu's GEMM since
 * issue #146 — on a real adapter (test/helpers.ts: Dawn in-process, or
 * headless Chrome), checked
 * against a NumPy float64 oracle (scripts/gemm_oracle.py) — the repo's
 * differential-oracle convention (docs/TESTING.md), skip-don't-fail when
 * either the adapter or NumPy is unavailable.
 *
 * Error bounds are derived, not hand-tuned per case: with exact inputs, an
 * f32-accumulated dot product of length K is off by at most ~K·u·(|A||B|)
 * (u = 2^-24), and realistically ~sqrt(K)·u·(|A||B|); `absprod` = |A|@|B|
 * comes from the oracle. f16 adds one rounding of the result to binary16
 * (<= 2^-11 relative). A wrong index, a missing K panel, or an unguarded
 * tail tile produces errors orders of magnitude above these bounds.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { gemmKernelApplicable } from "../src/gemm.ts";
import { subgroupMatrixUsable } from "../src/gemm-caps.ts";
import { bundleForBrowser, closeHarness, getHarness, SRC } from "./helpers.ts";

after(closeHarness);

// ---- oracle -------------------------------------------------------------------

const ORACLE_SCRIPT = new URL("../scripts/gemm_oracle.py", import.meta.url).pathname;

function findOraclePython(): string | undefined {
  for (const candidate of [process.env.MATH_PLUS_ORACLE_PYTHON, "python3"].filter((c): c is string => Boolean(c))) {
    try {
      execFileSync(candidate, ["-c", "import numpy"], { stdio: "ignore" });
      return candidate;
    } catch {
      // try next
    }
  }
  return undefined;
}
const PYTHON = findOraclePython();
const NO_ORACLE = "no python with numpy found (set MATH_PLUS_ORACLE_PYTHON)";

type DType = "f32" | "f16";
interface Case {
  m: number;
  k: number;
  n: number;
  transB: boolean;
  dtype: DType;
  a: Uint8Array;
  b: Uint8Array;
}
interface OracleResult {
  c: Float64Array;
  absprod: Float64Array;
}

const b64 = (u8: Uint8Array): string => Buffer.from(u8.buffer, u8.byteOffset, u8.byteLength).toString("base64");
const fromB64 = (s: string): Uint8Array => new Uint8Array(Buffer.from(s, "base64"));

function runOracle(cases: readonly Case[]): OracleResult[] {
  const out = execFileSync(PYTHON as string, [ORACLE_SCRIPT], {
    input: JSON.stringify({
      cases: cases.map((c) => ({ a: b64(c.a), b: b64(c.b), dtype: c.dtype, m: c.m, k: c.k, n: c.n, transB: c.transB })),
    }),
    encoding: "utf8",
    maxBuffer: 1 << 30,
  });
  return (JSON.parse(out) as { results: { c: string; absprod: string }[] }).results.map((r) => {
    const c = fromB64(r.c).slice();
    const p = fromB64(r.absprod).slice();
    return { c: new Float64Array(c.buffer), absprod: new Float64Array(p.buffer) };
  });
}

// ---- inputs -------------------------------------------------------------------

function lcg(size: number, seed: number): Float32Array {
  let s = seed >>> 0;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

type F16Ctor = { from(v: ArrayLike<number>): { buffer: ArrayBuffer; byteLength: number } } & (new (b: ArrayBuffer) => ArrayLike<number>);
function float16Array(): F16Ctor {
  const F16 = (globalThis as { Float16Array?: F16Ctor }).Float16Array;
  if (!F16) throw new Error("this test needs a runtime with Float16Array (Node >= 24) to build/decode f16 data");
  return F16;
}

/** Operand bytes as the GPU sees them: f32, or binary16 bits via the platform's Float16Array. */
function operandBytes(values: Float32Array, dtype: DType): Uint8Array {
  if (dtype === "f32") return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  const h = float16Array().from(values);
  return new Uint8Array(h.buffer, 0, h.byteLength);
}

function makeCase(m: number, k: number, n: number, transB: boolean, dtype: DType): Case {
  const seed = m * 7919 + k * 104729 + n * 1299709 + (transB ? 17 : 0);
  return { m, k, n, transB, dtype, a: operandBytes(lcg(m * k, seed), dtype), b: operandBytes(lcg(k * n, seed + 1), dtype) };
}

function toF32(bytes: Uint8Array, dtype: DType): Float32Array {
  const copy = bytes.slice();
  if (dtype === "f32") return new Float32Array(copy.buffer);
  return Float32Array.from(new (float16Array())(copy.buffer));
}

function assertWithinBound(got: Float32Array, ref: OracleResult, dtype: DType, label: string): void {
  assert.equal(got.length, ref.c.length, `${label}: element count`);
  for (let i = 0; i < got.length; i++) {
    const want = ref.c[i] as number;
    const scale = ref.absprod[i] as number;
    const bound = dtype === "f32" ? 1e-5 * scale + 1e-7 : 4.9e-4 * Math.abs(want) + 2e-5 * scale + 1e-7;
    const err = Math.abs((got[i] as number) - want);
    if (!(err <= bound)) assert.fail(`${label}: element ${i}: got ${got[i]}, want ${want} (err ${err} > bound ${bound})`);
  }
}

// ---- pure logic (no GPU) --------------------------------------------------------
//
// The kernel generators, planGemm and selectGemmKernel were removed with the
// duplicated kernels (issue #146); backend-webgpu's own kernel tests cover
// its WGSL, and the oracle test below covers every family end to end.

const NO_SG = { f16: true, subgroupMatrix: false };
const SG = { f16: true, subgroupMatrix: true };

test("gemmKernelApplicable: forced kernel families are refused outside their preconditions", () => {
  assert.equal(gemmKernelApplicable("tiled", 5, 3, 7, false, NO_SG), true);
  assert.equal(gemmKernelApplicable("skinny", 64, 64, 64, true, SG), true);
  assert.equal(gemmKernelApplicable("skinny", 65, 64, 64, true, SG), false, "backend-webgpu's skinny configs stop at M = 64");
  assert.equal(gemmKernelApplicable("skinny", 8, 64, 64, false, SG), false, "skinny is transB-only");
  assert.equal(gemmKernelApplicable("skinny", 8, 66, 64, true, SG), false, "K % 4 != 0");
  assert.equal(gemmKernelApplicable("subgroup-matrix", 128, 64, 64, false, NO_SG), false);
  assert.equal(gemmKernelApplicable("subgroup-matrix", 128, 64, 66, false, SG), true, "[K,N] B is transposed first, so N % 4 no longer matters");
  assert.equal(gemmKernelApplicable("subgroup-matrix", 128, 66, 64, true, SG), false, "K % 4 != 0");
});

test("subgroupMatrixUsable: needs the feature, an f32 8x8x8 config, and a fixed subgroup size of 32", () => {
  const feat = new Set(["chromium-experimental-subgroup-matrix"]);
  const f16cfg = { componentType: "f16", resultComponentType: "f16", M: 8, N: 8, K: 8 };
  const cfgs = [f16cfg, { componentType: "f32", resultComponentType: "f32", M: 8, N: 8, K: 8 }];
  assert.equal(subgroupMatrixUsable({ subgroupMinSize: 32, subgroupMaxSize: 32, subgroupMatrixConfigs: cfgs }, feat), true);
  assert.equal(subgroupMatrixUsable({ subgroupMinSize: 32, subgroupMaxSize: 32, subgroupMatrixConfigs: cfgs }, new Set()), false);
  assert.equal(subgroupMatrixUsable({ subgroupMinSize: 16, subgroupMaxSize: 32, subgroupMatrixConfigs: cfgs }, feat), false);
  assert.equal(subgroupMatrixUsable({ subgroupMinSize: 32, subgroupMaxSize: 32, subgroupMatrixConfigs: [f16cfg] }, feat), false);
  assert.equal(subgroupMatrixUsable(undefined, feat), false);
});

// ---- real GPU vs NumPy ------------------------------------------------------------

/** [m, k, n, transB] — each chosen to hit a specific kernel/alignment/tail path. */
const SHAPES: [number, number, number, boolean][] = [
  [4, 3, 5, false], // tiny, scalar loads everywhere, one partial tile
  [17, 33, 9, false], // odd everything (odd f16 byte counts -> padded buffers)
  [128, 128, 128, false], // multiple full tiles in both dims, vec4 loads
  [1, 64, 256, false], // M = 1, [K,N] layout
  [37, 70, 45, true], // transB with K%4 != 0 -> tiled scalar
  [1, 1024, 300, true], // skinny bucket 1 (M <= 40), long K
  [33, 256, 100, true], // skinny bucket 1, partial column block
  [50, 128, 96, true], // skinny bucket 2 (40 < M <= 64)
  [65, 64, 64, true], // first subgroup-matrix size, exact tiles
  [93, 256, 200, true], // subgroup-matrix transB, partial M and N tiles
  [300, 268, 130, true], // subgroup-matrix, K % 8 != 0 (partial last K panel)
  [130, 256, 200, false], // subgroup-matrix with [K,N] B (row-major right fragments)
  [257, 100, 132, false], // [K,N] B, partial tiles, K % 8 != 0
  [130, 256, 202, false], // [K,N] B with N%4 != 0 -> tiled, scalar B
  [256, 512, 256, false], // many K panels
];

const PAGE_CODECS = `
  const dec = (s) => { const bin = atob(s); const u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return u8.buffer; };
  const enc = (ta) => { const u8 = new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength); let s = ""; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
`;

interface PageResult {
  caseIndex: number;
  kernel: string;
  resolved: string;
  out: string;
}
interface Caps {
  f16: boolean;
  subgroupMatrix: boolean;
}

const gemmBundle = (): string => bundleForBrowser([path.join(SRC, "gemm.ts"), path.join(SRC, "device.ts")]);

for (const dtype of ["f32", "f16"] as const) {
  test(`GEMM ${dtype}: every applicable kernel matches NumPy across ${SHAPES.length} shapes (both B layouts, aligned + unaligned, partial tiles)`, async (t) => {
    const harness = await getHarness();
    if ("unavailable" in harness) {
      t.skip(`WebGPU not available: ${harness.reason}`);
      return;
    }
    if (!PYTHON) {
      t.skip(NO_ORACLE);
      return;
    }
    const cases = SHAPES.map(([m, k, n, transB]) => makeCase(m, k, n, transB, dtype));
    const { results, caps } = await harness.run<{ results: PageResult[]; caps: Caps }>(
      `
      const cap = await detectWebGPU({ gpu: navigator.gpu });
      if (!cap.available) throw new Error(cap.reason);
      const device = cap.device;
      const caps = cap.gemm;
      ${PAGE_CODECS}
      const CASES = ${JSON.stringify(cases.map((c) => ({ m: c.m, k: c.k, n: c.n, transB: c.transB, a: b64(c.a), b: b64(c.b) })))};
      const results = [];
      if (${JSON.stringify(dtype)} === "f16" && !caps.f16) return { results, caps };
      for (let i = 0; i < CASES.length; i++) {
        const c = CASES[i];
        const kernels = ["auto", "tiled", "skinny", "subgroup-matrix"].filter((kk) => kk === "auto" || gemmKernelApplicable(kk, c.m, c.k, c.n, c.transB, caps));
        for (const kernel of kernels) {
          const opts = { transB: c.transB, kernel };
          const out = ${JSON.stringify(dtype)} === "f32"
            ? await runGemmWGSL(device, new Float32Array(dec(c.a)), new Float32Array(dec(c.b)), c.m, c.k, c.n, opts)
            : await runGemmF16WGSL(device, new Uint16Array(dec(c.a)), new Uint16Array(dec(c.b)), c.m, c.k, c.n, opts);
          const resolved = kernel;
          results.push({ caseIndex: i, kernel, resolved, out: enc(out) });
        }
      }
      return { results, caps };
      `,
      gemmBundle(),
    );
    if (dtype === "f16" && !caps.f16) {
      t.skip("adapter has no shader-f16");
      return;
    }
    const oracle = runOracle(cases);
    const kernelsSeen = new Set<string>();
    for (const r of results) {
      const c = cases[r.caseIndex] as Case;
      kernelsSeen.add(r.resolved);
      const label = `${dtype} ${c.m}x${c.k}x${c.n}${c.transB ? " transB" : ""} ${r.kernel}->${r.resolved}`;
      assertWithinBound(toF32(fromB64(r.out), dtype), oracle[r.caseIndex] as OracleResult, dtype, label);
    }
    t.diagnostic(`harness=${harness.kind}; ${results.length} kernel runs; kernels exercised: ${[...kernelsSeen].sort().join(", ")}`);
    assert.ok(kernelsSeen.has("auto") && kernelsSeen.has("tiled") && kernelsSeen.has("skinny"), "the automatic choice and both portable families must be exercised");
    if (caps.subgroupMatrix) assert.ok(kernelsSeen.has("subgroup-matrix"), "the subgroup-matrix kernel must be exercised when the device supports it");
  });
}

test("GEMM: this adapter's subgroup-matrix support is detected (Apple/Metal: Dawn with allow_unsafe_apis, or Chrome --enable-unsafe-webgpu)", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`WebGPU not available: ${harness.reason}`);
    return;
  }
  const caps = await harness.run<Caps>(`const cap = await detectWebGPU({ gpu: navigator.gpu }); if (!cap.available) throw new Error(cap.reason); return cap.gemm;`, gemmBundle());
  if (!caps.subgroupMatrix) {
    t.skip(`adapter (${harness.kind}) offers no f32 8x8x8 subgroup matrices at subgroup size 32 — portable kernels only`);
    return;
  }
  assert.equal(caps.subgroupMatrix, true);
});

test("GEMM: f16 without shader-f16 throws instead of silently widening; bad shapes and inapplicable forced kernels throw", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`WebGPU not available: ${harness.reason}`);
    return;
  }
  const msgs = await harness.run<string[]>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice(); // no shader-f16 requested
    const out = [];
    const attempt = async (f) => { try { await f(); out.push("no error"); } catch (e) { out.push(e.message); } };
    await attempt(() => runGemmF16WGSL(device, new Uint16Array(4), new Uint16Array(4), 2, 2, 2));
    await attempt(() => runGemmWGSL(device, new Float32Array(6), new Float32Array(4), 2, 2, 2));
    const a = GPUTensor.fromFloat32Array(device, new Float32Array(6), [2, 3]);
    const b = GPUTensor.fromFloat32Array(device, new Float32Array(8), [2, 4]);
    await attempt(() => runGemm(device, a, b));
    await attempt(() => runGemmWGSL(device, new Float32Array(4), new Float32Array(4), 2, 2, 2, { kernel: "skinny" }));
    return out;
    `,
    gemmBundle(),
  );
  assert.match(msgs[0] as string, /shader-f16/);
  assert.match(msgs[1] as string, /a\.length 6 !== m\*k 4/);
  assert.match(msgs[2] as string, /inner dimensions differ/);
  assert.match(msgs[3] as string, /not applicable/);
});

test("runGemm: GPU-resident f32 and f16 chains (A·B, then a transB Linear on the result) match NumPy", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) {
    t.skip(`WebGPU not available: ${harness.reason}`);
    return;
  }
  if (!PYTHON) {
    t.skip(NO_ORACLE);
    return;
  }
  for (const dtype of ["f32", "f16"] as const) {
    const [m, k, n, p] = [96, 64, 72, 40];
    const first = makeCase(m, k, n, false, dtype);
    const weight = makeCase(1, p, n, false, dtype).b; // p*n values, used as a [p, n] transB weight
    type ChainResult = { ab: string; abw: string; dtypes: string[]; shapes: number[][] } | null;
    const res: ChainResult = await harness.run<ChainResult>(
      `
      const cap = await detectWebGPU({ gpu: navigator.gpu });
      if (!cap.available) throw new Error(cap.reason);
      const device = cap.device;
      if (${JSON.stringify(dtype)} === "f16" && !cap.gemm.f16) return null;
      ${PAGE_CODECS}
      const up = (s, shape) => ${JSON.stringify(dtype)} === "f32"
        ? GPUTensor.fromFloat32Array(device, new Float32Array(dec(s)), shape)
        : GPUTensor.fromFloat16Bits(device, new Uint16Array(dec(s)), shape);
      const A = up(${JSON.stringify(b64(first.a))}, [${m}, ${k}]);
      const B = up(${JSON.stringify(b64(first.b))}, [${k}, ${n}]);
      const W = up(${JSON.stringify(b64(weight))}, [${p}, ${n}]);
      const AB = await runGemm(device, A, B);
      const ABW = await runGemm(device, AB, W, { transB: true });
      const read = (t) => ${JSON.stringify(dtype)} === "f32" ? t.toFloat32Array() : t.toUint16Array();
      const r = { ab: enc(await read(AB)), abw: enc(await read(ABW)), dtypes: [AB.dtype, ABW.dtype], shapes: [[...AB.shape], [...ABW.shape]] };
      for (const x of [A, B, W, AB, ABW]) x.free();
      return r;
      `,
      gemmBundle(),
    );
    if (!res) {
      t.diagnostic("adapter has no shader-f16: f16 chain not run");
      assert.equal(dtype, "f16");
      continue;
    }
    assert.deepEqual(res.dtypes, [dtype, dtype]);
    assert.deepEqual(res.shapes, [[m, n], [m, p]]);
    const abBytes = fromB64(res.ab);
    const [refAB] = runOracle([first]);
    assertWithinBound(toF32(abBytes, dtype), refAB as OracleResult, dtype, `${dtype} resident A·B`);
    // Second hop is checked against NumPy on the ACTUAL intermediate the GPU produced, so the oracle's inputs are exact.
    const [refABW] = runOracle([{ m, k: n, n: p, transB: true, dtype, a: abBytes, b: weight }]);
    assertWithinBound(toF32(fromB64(res.abw), dtype), refABW as OracleResult, dtype, `${dtype} resident (A·B)·Wᵀ`);
  }
});
