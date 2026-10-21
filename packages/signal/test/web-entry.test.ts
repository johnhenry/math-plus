import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import { builtinModules } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = join(here, "..");
const pkgJson = JSON.parse(readFileSync(join(pkgRoot, "package.json"), "utf8"));

const EXPORTS = ["rfft", "welch", "stft", "istft", "getWindow", "checkNOLA", "checkCOLA", "butter", "sosfiltfilt", "findPeaks", "resamplePoly"];

test("package.json exposes a ./web entry with types", () => {
  assert.ok(pkgJson.exports["./web"], "exports['./web'] missing");
  assert.ok(pkgJson.exports["./web"].types);
  assert.ok(pkgJson.exports["./web"].default);
});

test("./web exports rfft, welch, stft, butter, sosfiltfilt, findPeaks and resamplePoly", async () => {
  const mod = await import("../src/web.ts");
  for (const name of EXPORTS) assert.equal(typeof (mod as Record<string, unknown>)[name], "function", name);
});

// Collect every module reachable from the built ./web entry (relative and bare
// imports resolved through node_modules) and check none is a Node built-in.
function importsOf(src: string): string[] {
  const out: string[] = [];
  const re = /(?:^|[\n;])\s*(?:import|export)\s[^"'`]*?from\s*["']([^"']+)["']|import\s*["']([^"']+)["']|import\(\s*["']([^"']+)["']\s*\)/g;
  for (let m = re.exec(src); m; m = re.exec(src)) out.push((m[1] ?? m[2] ?? m[3]) as string);
  return out;
}

test("the built ./web graph imports no Node built-ins and uses no Node globals", () => {
  const webPath = join(pkgRoot, pkgJson.exports["./web"].default);
  if (!existsSync(webPath)) return; // not built (e.g. bare `bun test` without build): covered by turbo's build dependency
  const req = createRequire(webPath);
  const builtins = new Set([...builtinModules, ...builtinModules.map((b) => `node:${b}`)]);
  const seen = new Set<string>();
  const stack = [webPath];
  while (stack.length) {
    const file = stack.pop() as string;
    if (seen.has(file)) continue;
    seen.add(file);
    const src = readFileSync(file, "utf8");
    for (const spec of importsOf(src)) {
      assert.ok(!builtins.has(spec), `${file} imports Node built-in ${spec}`);
      stack.push(spec.startsWith(".") ? join(dirname(file), spec) : resolveBare(spec, file));
    }
    const code = src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");
    assert.ok(!/\b(?:process\.|Buffer\.|require\(|__dirname|__filename)/.test(code), `${file} uses a Node global`);
  }
  assert.ok(seen.size > 5);
  void req;
  void statSync;
  void readdirSync;
});

function resolveBare(spec: string, from: string): string {
  const url = import.meta.resolve(spec, pathToFileURL(from).href);
  return fileURLToPath(url);
}

test("the built ./web entry loads and runs with process, Buffer and require removed (Worker-like)", () => {
  const webPath = join(pkgRoot, pkgJson.exports["./web"].default);
  if (!existsSync(webPath)) return;
  const script = `
    const g = globalThis;
    for (const k of ["process", "Buffer", "require", "module", "exports", "__dirname", "__filename"]) {
      try { delete g[k]; } catch {}
      try { Object.defineProperty(g, k, { value: undefined, configurable: true }); } catch {}
    }
    const m = await import(${JSON.stringify(pathToFileURL(webPath).href)});
    const names = ${JSON.stringify(EXPORTS)};
    for (const n of names) if (typeof m[n] !== "function") throw new Error("missing " + n);
    const N = 256;
    const x = Array.from({ length: N }, (_, i) => Math.sin(2 * Math.PI * 16 * i / N));
    const T = m.Tensor;
    const t = T.from(x, { dtype: "f64" });
    const sos = m.butter(4, 0.2, { btype: "lowpass" });
    const y = m.sosfiltfilt(sos, t);
    if (y.shape[0] !== N) throw new Error("sosfiltfilt length");
    const spec = m.rfft(t);
    if (!spec) throw new Error("rfft");
    m.welch(t, { nperseg: 64 });
    m.stft(t, { nperseg: 64 });
    m.findPeaks(t);
    m.resamplePoly(t, 1, 2);
    console.log("OK");
  `;
  const out = execFileSync(process.execPath, ["--input-type=module", "-e", script], { encoding: "utf8" });
  assert.match(out, /OK/);
});
