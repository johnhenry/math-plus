/**
 * Lazily opens the real 842 MB Laya checkpoint from the local Hugging Face
 * cache (skipped when absent): header only, then a few tensors, compared
 * with Python `safetensors.safe_open` on the same file (skipped without the
 * oracle). Guards the MLX layout: `"__metadata__": null` and an unpadded
 * header, so every tensor is misaligned in memory.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { openSafetensors } from "../src/index.ts";
import { PYTHON, fromB64, oracleSkip, sameF32 } from "./helpers.ts";

function findCheckpoint(): string | undefined {
  const hub = process.env.HF_HUB_CACHE ?? join(process.env.HF_HOME ?? join(homedir(), ".cache", "huggingface"), "hub");
  const snapshots = join(hub, "models--aac6fef--laya-mlx", "snapshots");
  if (!existsSync(snapshots)) return undefined;
  for (const rev of readdirSync(snapshots)) {
    const p = join(snapshots, rev, "model.safetensors");
    if (existsSync(p)) return p;
  }
  return undefined;
}

const CHECKPOINT = findCheckpoint();
const skip = CHECKPOINT ? false : "aac6fef/laya-mlx model.safetensors not in the local HF cache";

test("real checkpoint: header read lazily, a few tensors match Python", { skip }, async (t) => {
  const path = CHECKPOINT as string;
  const size = statSync(path).size;
  const before = process.memoryUsage().arrayBuffers;
  const lazy = await openSafetensors(path);
  t.after(() => lazy.close());
  assert.equal(lazy.size, size);
  assert.ok(lazy.names().length > 100);
  assert.equal(lazy.header.dataStart + lazy.header.dataLength, size, "offsets cover the whole file");
  assert.ok(process.memoryUsage().arrayBuffers - before < 16 * 1024 * 1024, "opening did not load the weights");
  const dtypes = new Set(lazy.names().map((n) => lazy.info(n).dtype));
  assert.deepEqual([...dtypes], ["F16"]);

  const small = lazy.names().filter((n) => lazy.info(n).dataOffsets[1] - lazy.info(n).dataOffsets[0] <= 8192).slice(0, 5);
  assert.ok(small.length > 0);
  await t.test("values vs safetensors.safe_open", { skip: oracleSkip }, async () => {
    const script = [
      "import sys, json, base64, numpy as np",
      "from safetensors import safe_open",
      "out = {}",
      "with safe_open(sys.argv[1], framework='np') as f:",
      "    for n in sys.argv[2:]:",
      "        out[n] = base64.b64encode(f.get_tensor(n).astype(np.float32).tobytes()).decode()",
      "print(json.dumps(out))",
    ].join("\n");
    const expected = JSON.parse(execFileSync(PYTHON as string, ["-c", script, path, ...small], { encoding: "utf8" })) as Record<string, string>;
    const many = await lazy.readMany(small);
    for (const name of small) {
      const want = fromB64(expected[name] as string);
      const wantF32 = new Float32Array(want.buffer, want.byteOffset, want.byteLength / 4);
      assert.equal(sameF32(await lazy.toF32(name), wantF32), undefined, name);
      assert.equal(sameF32(new Float32Array(many.get(name) as Float16Array), wantF32), undefined, `${name} (readMany)`);
    }
  });
});
