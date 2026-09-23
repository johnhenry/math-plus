import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { mkdtempSync, open as openCb, rmSync, writeFileSync } from "node:fs";
import { open } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { after, before, test } from "node:test";
import {
  HttpSource,
  SafetensorsError,
  openSafetensors,
  writeSafetensors,
  type ByteSource,
  type LazySafetensors,
} from "../src/index.ts";

const FILE = writeSafetensors(
  {
    w: { dtype: "F32", shape: [2, 3], data: new Float32Array([1, 2, 3, 4, 5, 6]) },
    h: { dtype: "F16", shape: [3], data: new Float16Array([0.5, -1, 65504]) },
    b: { dtype: "BF16", shape: [2], data: new Uint16Array([0x3fc0, 0xc040]) },
    i: { dtype: "I64", shape: [2], data: new BigInt64Array([-1n, 2n ** 50n]) },
    e: { dtype: "F32", shape: [0], data: new Float32Array(0) },
  },
  { format: "test" },
);

async function checkAll(lazy: LazySafetensors): Promise<void> {
  assert.deepEqual(lazy.metadata, { format: "test" });
  assert.deepEqual(lazy.names().sort(), ["b", "e", "h", "i", "w"]);
  assert.equal(lazy.size, FILE.byteLength);
  assert.deepEqual([...(await lazy.read("w") as Float32Array)], [1, 2, 3, 4, 5, 6]);
  assert.deepEqual([...(await lazy.read("h") as Float16Array)], [0.5, -1, 65504]);
  assert.deepEqual([...(await lazy.toF32("b"))], [1.5, -3]);
  assert.deepEqual([...(await lazy.read("i") as BigInt64Array)], [-1n, 2n ** 50n]);
  assert.equal((await lazy.read("e")).length, 0);
  const many = await lazy.readMany(["h", "w", "e"]);
  assert.deepEqual([...many.keys()], ["h", "w", "e"]);
  assert.deepEqual([...(many.get("w") as Float32Array)], [1, 2, 3, 4, 5, 6]);
  assert.deepEqual([...(many.get("h") as Float16Array)], [0.5, -1, 65504]);
  await assert.rejects(lazy.read("nope"), (e: unknown) => e instanceof SafetensorsError && e.code === "TensorNotFound");
}

let dir: string;
let path: string;
before(() => {
  dir = mkdtempSync(join(tmpdir(), "safetensors-lazy-"));
  path = join(dir, "model.safetensors");
  writeFileSync(path, FILE);
});
after(() => rmSync(dir, { recursive: true, force: true }));

test("in-memory bytes and ArrayBuffer", async () => {
  await checkAll(await openSafetensors(FILE));
  await checkAll(await openSafetensors(FILE.slice().buffer));
});

test("Blob / File", async () => {
  await checkAll(await openSafetensors(new Blob([FILE])));
  await checkAll(await openSafetensors(new File([FILE], "model.safetensors")));
});

test("file path, file: URL and caller-owned FileHandle (Node/Bun)", async () => {
  const lazy = await openSafetensors(path);
  await checkAll(lazy);
  await lazy.close();
  const viaUrl = await openSafetensors(pathToFileURL(path));
  await checkAll(viaUrl);
  await viaUrl.close();
  const handle = await open(path, "r");
  const lazyHandle = await openSafetensors(handle);
  await checkAll(lazyHandle);
  await lazyHandle.close(); // must NOT close the caller's handle
  assert.equal((await handle.stat()).size, FILE.byteLength);
  await handle.close();
});

test("missing file path rejects; file with trailing bytes rejects", async () => {
  await assert.rejects(openSafetensors(join(dir, "missing.safetensors")), /ENOENT/);
  const trailing = join(dir, "trailing.safetensors");
  writeFileSync(trailing, new Uint8Array([...FILE, 0, 0]));
  await assert.rejects(openSafetensors(trailing), (e: unknown) => e instanceof SafetensorsError && e.code === "MetadataIncompleteBuffer");
  // no file descriptor leaked by the failed open (best effort: the fd table still accepts opens)
  await new Promise<void>((resolve, reject) => openCb(trailing, "r", (err) => (err ? reject(err) : resolve())));
});

test("header larger than the probe is fetched with a second read", async () => {
  let reads = 0;
  const src: ByteSource = {
    size: FILE.byteLength,
    async read(offset, length) {
      reads++;
      return FILE.slice(offset, offset + length);
    },
  };
  const lazy = await openSafetensors(src, { probeBytes: 16 });
  assert.equal(reads, 2);
  await checkAll(lazy);
});

test("truncated source: reading a tensor past the end rejects", async () => {
  const cut = FILE.subarray(0, FILE.byteLength - 4);
  const src: ByteSource = { async read(offset, length) { return cut.slice(offset, offset + length); } };
  const lazy = await openSafetensors(src); // size unknown: header alone is fine
  assert.equal(lazy.size, undefined);
  const last = lazy.names().map((n) => lazy.info(n)).sort((a, b) => b.dataOffsets[1] - a.dataOffsets[1])[0];
  await assert.rejects(lazy.read(last!.name), (e: unknown) => e instanceof SafetensorsError && e.code === "ReadError");
});

// ---- HTTP -------------------------------------------------------------------

interface Served { url: string; server: Server; ranges: string[] }

function serve(opts: { ranges: boolean; redirect?: boolean }): Promise<Served> {
  const ranges: string[] = [];
  const server = createServer((req, res) => {
    if (opts.redirect && req.url === "/resolve/main/model.safetensors") {
      res.writeHead(302, { Location: "/cdn/model.safetensors" });
      res.end();
      return;
    }
    if (req.headers.authorization !== "Bearer t0k") {
      res.writeHead(401);
      res.end();
      return;
    }
    const range = req.headers.range;
    if (opts.ranges && range) {
      ranges.push(range);
      const m = /^bytes=(\d+)-(\d+)$/.exec(range);
      const start = Number(m![1]);
      const end = Math.min(Number(m![2]), FILE.byteLength - 1);
      if (start >= FILE.byteLength) {
        res.writeHead(416, { "Content-Range": `bytes */${FILE.byteLength}` });
        res.end();
        return;
      }
      res.writeHead(206, { "Content-Range": `bytes ${start}-${end}/${FILE.byteLength}`, "Content-Length": end - start + 1 });
      res.end(FILE.subarray(start, end + 1));
      return;
    }
    ranges.push("full");
    res.writeHead(200, { "Content-Length": FILE.byteLength });
    res.end(FILE);
  });
  return new Promise((resolve) =>
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as { port: number };
      resolve({ url: `http://127.0.0.1:${port}${opts.redirect ? "/resolve/main" : ""}/model.safetensors`, server, ranges });
    }),
  );
}

const auth = { headers: { Authorization: "Bearer t0k" } };

test("HTTP with Range support: header + per-tensor range requests, redirects followed", async () => {
  const s = await serve({ ranges: true, redirect: true });
  try {
    const lazy = await openSafetensors(s.url, auth);
    assert.equal(s.ranges.length, 1, "one request covers the header");
    assert.equal((lazy.source as HttpSource).rangesSupported, true);
    await checkAll(lazy);
    assert.ok(s.ranges.every((r) => r.startsWith("bytes=")));
    const before = s.ranges.length;
    await lazy.readMany(); // all tensors are adjacent: coalesced into one request
    assert.equal(s.ranges.length - before, 1);
  } finally {
    s.server.close();
  }
});

test("HTTP without Range support falls back to one full download", async () => {
  const s = await serve({ ranges: false });
  try {
    const lazy = await openSafetensors(new URL(s.url), auth);
    assert.equal((lazy.source as HttpSource).rangesSupported, false);
    await checkAll(lazy);
    assert.deepEqual(s.ranges, ["full"], "everything served from the single full body");
  } finally {
    s.server.close();
  }
});

test("HTTP errors surface as ReadError", async () => {
  const s = await serve({ ranges: true });
  try {
    await assert.rejects(openSafetensors(s.url), (e: unknown) => e instanceof SafetensorsError && e.code === "ReadError" && /401/.test(e.message));
  } finally {
    s.server.close();
  }
});

test("custom fetch is used (no network)", async () => {
  const calls: string[] = [];
  const fakeFetch = (async (_url: string | URL | Request, init?: RequestInit) => {
    const range = new Headers(init?.headers).get("range") as string;
    calls.push(range);
    const [a, b] = range.slice(6).split("-").map(Number) as [number, number];
    const end = Math.min(b, FILE.byteLength - 1);
    return new Response(FILE.slice(a, end + 1), { status: 206, headers: { "Content-Range": `bytes ${a}-${end}/${FILE.byteLength}` } });
  }) as typeof fetch;
  const lazy = await openSafetensors("https://example.invalid/model.safetensors", { fetch: fakeFetch });
  await checkAll(lazy);
  assert.ok(calls.length >= 2);
});
