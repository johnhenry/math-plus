/**
 * Builds main.ts for the browser (`bun build`) and serves the GEMM threshold
 * page for a real, visible browser, cross-origin isolated (COOP/COEP) so the
 * page's `performance.now()` has 5 µs rather than 100 µs resolution.
 *
 *   node scripts/gemm-threshold-page/serve.ts            # PORT=8765 by default
 *   open http://127.0.0.1:8765/                           # full run, ~10 min
 *   open "http://127.0.0.1:8765/?groups=square&sizes=64,128&cool=0"   # smoke
 *
 * Needs `bun` on PATH and tensor-wasm built (`npm run build` at the root).
 * The page imports Node-only modules only in code it never runs
 * (tensor-wasm's `readFile` fallback, thermal.ts's machine probes); an
 * import map points them at a stub.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WASM = path.resolve(HERE, "../../../tensor-wasm/wasm");
const OUT = path.join(tmpdir(), "math-plus-gemm-threshold-page");
mkdirSync(OUT, { recursive: true });

const NODE_STUB = `const unavailable = () => { throw new Error("Node API called in the browser page"); };
export const readFile = unavailable, execFileSync = unavailable, existsSync = () => false, readdirSync = () => [], readFileSync = unavailable;
export default { cpus: () => [], totalmem: () => 0, platform: () => "browser", release: () => "", arch: () => "", loadavg: () => [] };`;

execFileSync("bun", ["build", path.join(HERE, "main.ts"), "--target", "browser", "--format", "esm", "--external", "node:*", "--outfile", path.join(OUT, "main.js")], {
  stdio: "inherit",
});
writeFileSync(path.join(OUT, "node-stub.js"), NODE_STUB);
const importMap = { imports: Object.fromEntries(["node:fs/promises", "node:fs", "node:child_process", "node:os"].map((m) => [m, "./node-stub.js"])) };
writeFileSync(
  path.join(OUT, "index.html"),
  `<!doctype html><meta charset="utf-8"><title>tensor-webgpu GEMM threshold</title>
<script type="importmap">${JSON.stringify(importMap)}</script>
<button id="start">start</button>
<pre id="out"></pre>
<script type="module" src="./main.js"></script>`,
);

const TYPES: Record<string, string> = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm" };
const port = Number(process.env.PORT ?? 8765);
createServer((req, res) => {
  const url = new URL(req.url ?? "/", "http://x");
  const rel = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
  const file = rel.startsWith("wasm/") ? path.join(WASM, path.basename(rel)) : path.join(OUT, path.basename(rel));
  try {
    const body = readFileSync(file);
    res.writeHead(200, {
      "content-type": TYPES[path.extname(file)] ?? "application/octet-stream",
      "cross-origin-opener-policy": "same-origin",
      "cross-origin-embedder-policy": "require-corp",
      "cache-control": "no-store",
    });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
}).listen(port, "127.0.0.1", () => console.log(`GEMM threshold page: http://127.0.0.1:${port}/`));
