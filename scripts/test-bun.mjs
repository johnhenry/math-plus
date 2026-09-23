#!/usr/bin/env node
/**
 * Run every workspace's `test:bun` script (plus the root `test/` suite) under
 * Bun and print one summary line per suite. Exits non-zero if any suite
 * failed, OR if Bun ran fewer files than the suite has — the signature of
 * Bun 1.2's node:test multi-file registration bug (test/harness.ts), which
 * otherwise reports a silently truncated run as a pass.
 *
 *   npm run test:bun                 # every suite
 *   npm run test:bun -- tensor-core  # suites whose dir contains a filter
 *
 * Workspaces are discovered, not listed (test/manifest-drift.test.ts asserts
 * each one has a `test:bun` script), so there is no third manifest to drift.
 * Requires the workspace to be built first (`npm run build`), same as `npm test`.
 */
import { spawnSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const filters = process.argv.slice(2);

const suites = [{ dir: ".", name: "(root manifest checks)" }];
for (const globRoot of ["packages", "adapters", "scalars"]) {
  for (const entry of readdirSync(join(ROOT, globRoot), { withFileTypes: true })) {
    const dir = `${globRoot}/${entry.name}`;
    const pkgPath = join(ROOT, dir, "package.json");
    if (!entry.isDirectory() || !existsSync(pkgPath)) continue;
    const pkg = JSON.parse(readFileSync(pkgPath, "utf8"));
    if (pkg.scripts?.["test:bun"]) suites.push({ dir, name: pkg.name });
  }
}

const selected = filters.length ? suites.filter((s) => filters.some((f) => s.dir.includes(f))) : suites;
const rows = [];
let failed = false;
for (const suite of selected) {
  const cwd = join(ROOT, suite.dir);
  const expectedFiles = readdirSync(join(cwd, "test")).filter((f) => f.endsWith(".test.ts")).length;
  const cmd = suite.dir === "." ? ["bun", ["test", "./test/"]] : ["npm", ["run", "test:bun", "--silent"]];
  process.stdout.write(`\n== ${suite.name} (${suite.dir})\n`);
  const res = spawnSync(cmd[0], cmd[1], { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  const out = `${res.stdout ?? ""}${res.stderr ?? ""}`;
  process.stdout.write(out);
  const num = (re) => Number(out.match(re)?.[1] ?? 0);
  const pass = num(/^\s*(\d+) pass$/m);
  const fail = num(/^\s*(\d+) fail$/m);
  const skip = num(/^\s*(\d+) skip$/m);
  const runtimeSkips = (out.match(/^\[skip\] /gm) ?? []).length;
  const files = num(/across (\d+) files?/);
  const ok = res.status === 0 && fail === 0 && files === expectedFiles;
  if (!ok) failed = true;
  rows.push({ suite: suite.name, ok, pass, fail, skip, runtimeSkips, files: `${files}/${expectedFiles}` });
}

console.log("\n== Bun summary");
for (const r of rows) {
  const skips = r.skip || r.runtimeSkips ? `  skip ${r.skip} (+${r.runtimeSkips} runtime [skip], counted as pass)` : "";
  console.log(`${r.ok ? "ok  " : "FAIL"} ${r.suite.padEnd(42)} pass ${String(r.pass).padStart(4)}  fail ${r.fail}  files ${r.files}${skips}`);
}
const total = rows.reduce((a, r) => ({ pass: a.pass + r.pass, fail: a.fail + r.fail }), { pass: 0, fail: 0 });
console.log(`suites ${rows.length}  failing suites ${rows.filter((r) => !r.ok).length}  tests pass ${total.pass}  fail ${total.fail}`);
process.exit(failed ? 1 : 0);
