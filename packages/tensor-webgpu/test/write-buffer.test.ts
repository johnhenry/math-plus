/**
 * The `queue.writeBuffer` byteOffset bug (issue #126): Bun's Dawn binding
 * ignores a TypedArray view's own `byteOffset` and uploads bytes from the
 * start of the underlying ArrayBuffer. Since issue #146 every upload goes
 * through @johnhenry/backend-webgpu's `Runtime.write`, which always passes
 * `(arrayBuffer, byteOffset, byteLength)`.
 *
 *  1. Static audit: src/ has no `queue.writeBuffer` call of its own.
 *  2. Non-zero-offset views upload correctly on the harness's adapter
 *     (Dawn under Node, or Chrome).
 *  3. The same under Bun, where the bug actually lives: the node:test suite
 *     runs under Node, so this spawns the standalone repro
 *     test/bun/write-buffer-offset.bun.ts with `bun` (skipped only when no
 *     `bun` binary is on PATH or Dawn has no adapter there).
 */
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { bundleForBrowser, closeHarness, getHarness, SRC } from "./helpers.ts";

after(closeHarness);

test("src/ makes no queue.writeBuffer call of its own (uploads go through backend-webgpu's Runtime.write)", () => {
  const hits: string[] = [];
  for (const f of readdirSync(SRC).filter((n) => n.endsWith(".ts"))) {
    readFileSync(path.join(SRC, f), "utf8")
      .split("\n")
      .forEach((line, i) => {
        if (/\.writeBuffer\(/.test(line) && !/^\s*(\*|\/\/)/.test(line)) hits.push(`${f}:${i + 1}: ${line.trim()}`);
      });
  }
  assert.deepEqual(hits, []);
});

test("uploads of views with a non-zero byteOffset (f32 subarray, f16 subarray at a 2-byte offset, odd-length f16, a tensor-core view through the facade) round-trip exactly", async (t) => {
  const harness = await getHarness();
  if ("unavailable" in harness) return t.skip(`headless WebGPU not available: ${harness.reason}`);
  const r = await harness.run<{ f32: number[]; f16: number[]; f16odd: number[]; facade: number[] }>(
    `
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const backing = new Float32Array(16).map((_, i) => i);
    const a = GPUTensor.fromFloat32Array(device, backing.subarray(4, 8), [4]);
    const bits = new Uint16Array(12).map((_, i) => 0x3c00 + i);
    const b = GPUTensor.fromFloat16Bits(device, bits.subarray(3, 7), [4]);
    const c = GPUTensor.fromFloat16Bits(device, bits.subarray(5, 8), [3]);
    const gpu = await createWebGpuDevice({ device });
    const d = await gpu.fromHost({ dtype: "f32", shape: [4], data: backing.subarray(8, 12) });
    const facade = Array.from((await gpu.toHost(d)).data);
    const out = { f32: Array.from(await a.toFloat32Array()), f16: Array.from(await b.toUint16Array()), f16odd: Array.from(await c.toUint16Array()), facade };
    a.free(); b.free(); c.free(); gpu.dispose(d);
    return out;
    `,
    bundleForBrowser([path.join(SRC, "index.ts")]),
  );
  assert.deepEqual(r.f32, [4, 5, 6, 7]);
  assert.deepEqual(r.f16, [0x3c03, 0x3c04, 0x3c05, 0x3c06]);
  assert.deepEqual(r.f16odd, [0x3c05, 0x3c06, 0x3c07]);
  assert.deepEqual(r.facade, [8, 9, 10, 11]);
});

function findBun(): string | undefined {
  try {
    return execFileSync("which", ["bun"], { encoding: "utf8" }).trim() || undefined;
  } catch {
    return undefined;
  }
}

test("under Bun (where writeBuffer ignores a view's byteOffset), this package's upload paths still upload the view's bytes", (t) => {
  const bun = findBun();
  if (!bun) return t.skip("no `bun` on PATH");
  const script = path.resolve(SRC, "../test/bun/write-buffer-offset.bun.ts");
  const res = spawnSync(bun, [script], { encoding: "utf8", timeout: 60_000 });
  if (res.status === 2) return t.skip("Dawn has no adapter under Bun");
  const line = res.stdout.trim().split("\n").at(-1) ?? "";
  assert.equal(res.status, 0, `bun repro failed (status ${res.status}): ${line}\n${res.stderr}`);
  const r = JSON.parse(line) as { runtime: string; ok: boolean; rawView: number[]; rawViewIgnoresByteOffset: boolean };
  assert.equal(r.runtime, "bun");
  assert.ok(r.ok);
  t.diagnostic(
    `raw queue.writeBuffer(buffer, 0, view) under Bun uploaded [${r.rawView}] for a view of [4,5,6,7]` +
      (r.rawViewIgnoresByteOffset ? " (the binding bug is still present)" : " (the binding bug appears fixed upstream)"),
  );
});
