/**
 * Fused (flash) attention, `runAttention` (issue #126): unmasked and masked
 * (sliding window, key padding, causal, arbitrary per-element), both
 * kernels (`fast` for head dim 32/64, `generic` for any head dim), with and
 * without masked-key-tile skipping — on a real adapter (test/helpers.ts),
 * checked against a NumPy float64 oracle (scripts/attention_oracle.py),
 * skip-don't-fail when either is unavailable (docs/TESTING.md).
 *
 * Error bound: outputs are convex combinations of V rows in [-1, 1]; f32
 * online softmax + f32 accumulation over <= 128 keys is good to ~1e-6. The
 * 1e-4 tolerance is loose enough for every backend and orders of magnitude
 * below what a wrong mask, a dropped tile or a mis-scaled row produces.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import test, { after } from "node:test";
import { planAttention } from "../src/attention.ts";
import { fastAttentionBytes, genericAttentionBytes, genericAttentionConfig } from "../src/attention-kernels.ts";
import { bundleForBrowser, closeHarness, getHarness, SRC } from "./helpers.ts";

after(closeHarness);

const ORACLE_SCRIPT = new URL("../scripts/attention_oracle.py", import.meta.url).pathname;

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

const b64 = (a: Float32Array): string => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64");

function lcg(size: number, seed: number): Float32Array {
  let s = seed >>> 0;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

interface Case {
  name: string;
  batch: number;
  seqQ: number;
  seqK: number;
  dim: number;
  scale: number;
  q: Float32Array;
  k: Float32Array;
  v: Float32Array;
  mask?: Float32Array;
  maskShape?: number[];
  kernel: "fast" | "generic" | "auto";
  /** Use the adapter's max workgroup memory (fast D=64 needs it) instead of the 16 KiB default. */
  raisedLimits: boolean;
  skipMaskedTiles?: boolean;
  /** Expected kernel after planning. */
  expectKernel: "fast" | "generic";
}

function makeCase(
  name: string,
  batch: number,
  seqQ: number,
  seqK: number,
  dim: number,
  extra: Partial<Case> & Pick<Case, "expectKernel">,
  seed: number,
): Case {
  return {
    name,
    batch,
    seqQ,
    seqK,
    dim,
    scale: 1 / Math.sqrt(dim),
    q: lcg(batch * seqQ * dim, seed),
    k: lcg(batch * seqK * dim, seed + 1),
    v: lcg(batch * seqK * dim, seed + 2),
    kernel: "auto",
    raisedLimits: true,
    ...extra,
  };
}

function slidingWindow(seqQ: number, seqK: number, radius: number): Float32Array {
  const m = new Float32Array(seqQ * seqK);
  for (let i = 0; i < seqQ; i++) {
    const c = Math.round((i * (seqK - 1)) / Math.max(1, seqQ - 1));
    for (let j = Math.max(0, c - radius); j <= Math.min(seqK - 1, c + radius); j++) m[i * seqK + j] = 1;
  }
  return m;
}

function padding(batch: number, seqK: number, lens: readonly number[]): Float32Array {
  const m = new Float32Array(batch * seqK);
  for (let b = 0; b < batch; b++) for (let j = 0; j < (lens[b] as number); j++) m[b * seqK + j] = 1;
  return m;
}

function causal(seqQ: number, seqK: number): Float32Array {
  const m = new Float32Array(seqQ * seqK);
  for (let i = 0; i < seqQ; i++) for (let j = 0; j <= Math.min(i, seqK - 1); j++) m[i * seqK + j] = 1;
  return m;
}

/** Random per-element mask (~30% visible), with query row 3 of batch 1 fully masked. */
function randomMask(batch: number, seqQ: number, seqK: number, seed: number): Float32Array {
  const r = lcg(batch * seqQ * seqK, seed);
  const m = r.map((x) => (x > 0.4 ? 1 : 0));
  m.fill(0, (1 * seqQ + 3) * seqK, (1 * seqQ + 4) * seqK);
  return m;
}

function runOracle(cases: readonly Case[]): Float64Array[] {
  const out = execFileSync(PYTHON as string, [ORACLE_SCRIPT], {
    input: JSON.stringify({
      cases: cases.map((c) => ({
        q: b64(c.q),
        k: b64(c.k),
        v: b64(c.v),
        batch: c.batch,
        seqQ: c.seqQ,
        seqK: c.seqK,
        dim: c.dim,
        scale: c.scale,
        mask: c.mask ? b64(c.mask) : null,
        maskShape: c.maskShape ?? null,
      })),
    }),
    encoding: "utf8",
    maxBuffer: 1 << 30,
  });
  return (JSON.parse(out) as { results: { out: string }[] }).results.map((r) => {
    const bytes = new Uint8Array(Buffer.from(r.out, "base64")).slice();
    return new Float64Array(bytes.buffer);
  });
}

/** Runs every case on one device in the harness; returns each output as f32 (NaN-preserving, via base64) plus the planned kernel. */
async function runOnGPU(
  harness: Exclude<Awaited<ReturnType<typeof getHarness>>, { unavailable: true }>,
  cases: readonly Case[],
): Promise<{ out: Float32Array; kernel: string; limit: number }[]> {
  const bundle = bundleForBrowser([path.join(SRC, "attention.ts")]);
  const payload = cases.map((c) => ({
    batch: c.batch,
    seqQ: c.seqQ,
    seqK: c.seqK,
    dim: c.dim,
    scale: c.scale,
    q: b64(c.q),
    k: b64(c.k),
    v: b64(c.v),
    mask: c.mask ? b64(c.mask) : null,
    maskShape: c.maskShape ?? null,
    kernel: c.kernel,
    raisedLimits: c.raisedLimits,
    skipMaskedTiles: c.skipMaskedTiles ?? true,
  }));
  const results = await harness.run<{ out: string; kernel: string; limit: number }[]>(
    `
    const cases = ${JSON.stringify(payload)};
    const dec = (s) => new Float32Array(Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer);
    const enc = (f) => { const u = new Uint8Array(f.buffer, f.byteOffset, f.byteLength); let s = ""; for (let i = 0; i < u.length; i += 8192) s += String.fromCharCode(...u.subarray(i, i + 8192)); return btoa(s); };
    const adapter = await navigator.gpu.requestAdapter();
    const raised = await adapter.requestDevice({ requiredLimits: { maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize } });
    const plain = await (await navigator.gpu.requestAdapter()).requestDevice(); // an adapter creates one device
    const out = [];
    for (const c of cases) {
      const device = c.raisedLimits ? raised : plain;
      const q = GPUTensor.fromFloat32Array(device, dec(c.q), [c.batch, c.seqQ, c.dim]);
      const k = GPUTensor.fromFloat32Array(device, dec(c.k), [c.batch, c.seqK, c.dim]);
      const v = GPUTensor.fromFloat32Array(device, dec(c.v), [c.batch, c.seqK, c.dim]);
      const mask = c.mask ? GPUTensor.fromFloat32Array(device, dec(c.mask), c.maskShape) : undefined;
      const opts = { scale: c.scale, mask, kernel: c.kernel, skipMaskedTiles: c.skipMaskedTiles };
      const plan = planAttention(c.batch, c.seqQ, c.dim, !!mask, device.limits.maxComputeWorkgroupStorageSize, opts);
      device.pushErrorScope("validation");
      const o = await runAttention(device, q, k, v, opts);
      const data = await o.toFloat32Array();
      const err = await device.popErrorScope();
      if (err) throw new Error("validation error: " + err.message);
      out.push({ out: enc(data), kernel: plan.kernel, limit: device.limits.maxComputeWorkgroupStorageSize });
      for (const t of [q, k, v, o, mask]) t?.free();
    }
    return out;
    `,
    bundle,
  );
  return results.map((r) => ({
    out: new Float32Array(new Uint8Array(Buffer.from(r.out, "base64")).slice().buffer),
    kernel: r.kernel,
    limit: r.limit,
  }));
}

function assertClose(actual: Float32Array, expected: Float64Array, label: string): void {
  assert.equal(actual.length, expected.length, `${label}: length`);
  let worst = 0;
  let at = -1;
  for (let i = 0; i < expected.length; i++) {
    const d = Math.abs((actual[i] as number) - (expected[i] as number));
    if (!(d <= worst)) {
      worst = d;
      at = i;
    }
  }
  assert.ok(worst <= 1e-4, `${label}: max |err| ${worst} at ${at} (got ${actual[at]}, expected ${expected[at]})`);
}

test("planAttention: fast kernel for head dim 32/64 when its workgroup memory fits, generic otherwise; generic tiles shrink to the limit", () => {
  const DEFAULT = 16384;
  assert.equal(planAttention(1, 10, 32, false, DEFAULT).kernel, "fast");
  assert.ok(fastAttentionBytes(64, true) > DEFAULT, "fast D=64 needs raised limits");
  assert.equal(planAttention(1, 10, 64, true, DEFAULT).kernel, "generic");
  assert.equal(planAttention(1, 10, 64, true, 32768).kernel, "fast");
  assert.equal(planAttention(1, 10, 48, false, 32768).kernel, "generic");
  assert.throws(() => planAttention(1, 10, 48, false, 32768, { kernel: "fast" }), /fast kernel/);
  assert.equal(planAttention(3, 70, 32, false, DEFAULT).groups.join(), "3,3");
  for (const D of [4, 64, 128, 256]) {
    const cfg = genericAttentionConfig(D, DEFAULT, true);
    assert.ok(cfg && genericAttentionBytes(D, cfg, true) <= DEFAULT, `D=${D} fits the default limit`);
  }
  assert.equal(genericAttentionConfig(128, DEFAULT, false)?.BQ, 8);
  // Skipping is a code-generation choice: the masked kernels with and without it differ.
  assert.notEqual(
    planAttention(1, 10, 32, true, DEFAULT, { skipMaskedTiles: true }).code,
    planAttention(1, 10, 32, true, DEFAULT, { skipMaskedTiles: false }).code,
  );
});

test("runAttention: unmasked and masked (sliding window, padding, causal, per-element), both kernels, tile skipping on and off, match NumPy", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  if (!PYTHON) return t.skip(NO_ORACLE);

  const cases: Case[] = [];
  cases.push(makeCase("fast D=64 unmasked", 2, 37, 45, 64, { expectKernel: "fast" }, 1));
  cases.push(makeCase("fast D=32 unmasked, odd scale", 1, 5, 3, 32, { expectKernel: "fast", scale: 0.7 }, 4));
  for (const skip of [true, false]) {
    cases.push(
      makeCase(`fast D=32 sliding window skip=${skip}`, 2, 40, 70, 32, {
        expectKernel: "fast",
        mask: slidingWindow(40, 70, 6),
        maskShape: [40, 70],
        skipMaskedTiles: skip,
      }, 7),
    );
    cases.push(
      makeCase(`generic D=48 per-element mask with a fully masked row skip=${skip}`, 2, 20, 50, 48, {
        expectKernel: "generic",
        mask: randomMask(2, 20, 50, 99),
        maskShape: [2, 20, 50],
        skipMaskedTiles: skip,
      }, 10),
    );
  }
  cases.push(
    makeCase("fast D=64 key padding [B,1,Lk]", 3, 33, 90, 64, {
      expectKernel: "fast",
      mask: padding(3, 90, [90, 50, 17]),
      maskShape: [3, 1, 90],
    }, 13),
  );
  cases.push(
    makeCase("generic D=64 (default 16 KiB limit) key padding", 3, 33, 90, 64, {
      expectKernel: "generic",
      raisedLimits: false,
      mask: padding(3, 90, [90, 50, 17]),
      maskShape: [3, 1, 90],
    }, 16),
  );
  cases.push(
    makeCase("generic D=128 (default limit) causal", 2, 40, 40, 128, {
      expectKernel: "generic",
      raisedLimits: false,
      mask: causal(40, 40),
      maskShape: [40, 40],
    }, 19),
  );
  cases.push(makeCase("generic D=5 unmasked", 2, 9, 11, 5, { expectKernel: "generic", scale: 0.7 }, 22));
  cases.push(
    makeCase("forced generic D=32 padding [Lk] broadcast to every batch and query", 2, 17, 64, 32, {
      expectKernel: "generic",
      kernel: "generic",
      mask: padding(1, 64, [30]),
      maskShape: [64],
    }, 25),
  );

  const expected = runOracle(cases);
  const got = await runOnGPU(harness, cases);
  cases.forEach((c, i) => {
    // An adapter whose maximum workgroup memory is still 16 KiB (e.g. a software one) can't run fast D=64.
    const fits = c.kernel !== "auto" || fastAttentionBytes(c.dim, c.mask !== undefined) <= (got[i]?.limit ?? 0);
    assert.equal(got[i]?.kernel, c.expectKernel === "fast" && !fits ? "generic" : c.expectKernel, `${c.name}: kernel`);
    assertClose(got[i]?.out as Float32Array, expected[i] as Float64Array, c.name);
  });
});

test("runAttention: masked key tiles are actually skipped — non-finite V rows in tiles no query can see don't reach the output with skipping on, and do with it off", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  if (!PYTHON) return t.skip(NO_ORACLE);
  // Key padding: batch b sees keys [0, len_b). Every key from the next
  // 32-aligned boundary on lives in a tile no query of that batch entry can
  // see; poison those V rows with NaN. A skipped tile is never loaded; a
  // walked one multiplies its zero weights by NaN.
  const batch = 2;
  const seqQ = 24;
  const seqK = 128;
  const dim = 32;
  const lens = [40, 70];
  const cases: Case[] = [];
  for (const [kernel, skip] of [["fast", true], ["generic", true], ["fast", false], ["generic", false]] as const) {
    const c = makeCase(`${kernel} skip=${skip}`, batch, seqQ, seqK, dim, {
      expectKernel: kernel,
      kernel,
      mask: padding(batch, seqK, lens),
      maskShape: [batch, 1, seqK],
      skipMaskedTiles: skip,
    }, 31);
    for (let b = 0; b < batch; b++) {
      const from = Math.ceil((lens[b] as number) / 32) * 32;
      c.v.fill(Number.NaN, (b * seqK + from) * dim, (b + 1) * seqK * dim);
    }
    cases.push(c);
  }
  const expected = runOracle(cases);
  const got = await runOnGPU(harness, cases);
  for (let i = 0; i < 2; i++) assertClose(got[i]?.out as Float32Array, expected[i] as Float64Array, (cases[i] as Case).name);
  for (let i = 2; i < 4; i++) {
    assert.ok((got[i]?.out as Float32Array).some(Number.isNaN), `${(cases[i] as Case).name}: walking every tile should reach the NaN rows`);
  }
});

test("runAttention: rejects f16 operands, mismatched shapes, and non-broadcastable masks", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const bundle = bundleForBrowser([path.join(SRC, "attention.ts")]);
  const errors = await harness.run<string[]>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const t = (shape) => GPUTensor.fromFloat32Array(device, new Float32Array(shape.reduce((a, b) => a * b, 1)), shape);
    const q = t([2, 4, 8]), k = t([2, 6, 8]), v = t([2, 6, 8]);
    const msgs = [];
    const attempt = async (f) => { try { await f(); msgs.push("no error"); } catch (e) { msgs.push(e.message); } };
    await attempt(() => runAttention(device, q, t([2, 5, 8]), v));
    await attempt(() => runAttention(device, q, k, v, { mask: t([3, 4, 6]) }));
    await attempt(() => runAttention(device, q, k, v, { mask: t([4, 5]) }));
    await attempt(() => runAttention(device, GPUTensor.fromFloat16Bits(device, new Uint16Array(64), [2, 4, 8]), k, v));
    return msgs;
    `,
    bundle,
  );
  assert.match(errors[0] as string, /shapes do not agree/);
  assert.match(errors[1] as string, /not broadcastable/);
  assert.match(errors[2] as string, /not broadcastable/);
  assert.match(errors[3] as string, /f32 GPUTensors only/);
});
