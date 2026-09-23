/**
 * Guards against the "functions.csv rot" failure mode documented in
 * docs/spikes/woxi-study.md: a hand-maintained manifest with no validating
 * test silently drifts from reality. This repo has two such manifests --
 * root package.json's build/test `-w` script lists, and
 * scripts/sync-jsr-configs.mjs's PACKAGE_DIRS -- and both have already
 * required a manual edit (easy to forget) for every new package added.
 * A forgotten root-script entry means `npm run build`/`npm test` silently
 * stops covering that package while its own package-level scripts still
 * work -- invisible until something depends on a stale dist. A forgotten
 * PACKAGE_DIRS entry means the package ships to npm but never to JSR.
 */
import assert from "node:assert/strict";
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/**
 * Packages intentionally excluded because they aren't npm workspace
 * members at all (no package.json -- e.g. a PyPI-only package). Keep this
 * list exact and justified, never a blanket category skip (see AGENTS.md's
 * canonical-implementation/exception-discipline notes).
 */
const NOT_NPM_WORKSPACES = new Set(["packages/interop-python"]);

interface WorkspacePackage {
  dir: string;
  name: string;
}

function discoverWorkspacePackages(): WorkspacePackage[] {
  const found: WorkspacePackage[] = [];
  for (const globRoot of ["packages", "adapters", "scalars"]) {
    for (const entry of readdirSync(join(ROOT, globRoot), { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const dir = `${globRoot}/${entry.name}`;
      if (NOT_NPM_WORKSPACES.has(dir)) continue;
      const pkgJsonPath = join(ROOT, dir, "package.json");
      if (!existsSync(pkgJsonPath)) continue;
      const pkg = JSON.parse(readFileSync(pkgJsonPath, "utf8")) as { name: string };
      found.push({ dir, name: pkg.name });
    }
  }
  return found;
}

function readRootPackageJson(): { scripts: Record<string, string> } {
  return JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
}

test("every npm workspace package appears in root package.json's build AND test scripts", () => {
  const rootPkg = readRootPackageJson();
  const packages = discoverWorkspacePackages();
  const missingFromBuild = packages.filter((p) => !rootPkg.scripts.build.includes(`-w ${p.name}`)).map((p) => p.name);
  const missingFromTest = packages.filter((p) => !rootPkg.scripts.test.includes(`-w ${p.name}`)).map((p) => p.name);

  assert.deepEqual(
    missingFromBuild,
    [],
    `package(s) missing from root "build" script: ${missingFromBuild.join(", ")} -- add "-w <name>" to package.json's scripts.build`,
  );
  assert.deepEqual(
    missingFromTest,
    [],
    `package(s) missing from root "test" script: ${missingFromTest.join(", ")} -- add "-w <name>" to package.json's scripts.test`,
  );
});

test('root build/test scripts contain no stale "-w" entries for packages that no longer exist', () => {
  const rootPkg = readRootPackageJson();
  const packages = discoverWorkspacePackages();
  const knownNames = new Set(packages.map((p) => p.name));

  for (const scriptName of ["build", "test"] as const) {
    const flagged = [...rootPkg.scripts[scriptName].matchAll(/-w (\S+)/g)].map((m) => m[1] as string);
    const stale = flagged.filter((name) => !knownNames.has(name));
    assert.deepEqual(
      stale,
      [],
      `stale "-w" entries in root "${scriptName}" script (package no longer exists): ${stale.join(", ")}`,
    );
  }
});

test("every npm workspace package's directory appears in scripts/sync-jsr-configs.mjs's PACKAGE_DIRS", () => {
  const syncScript = readFileSync(join(ROOT, "scripts/sync-jsr-configs.mjs"), "utf8");
  const match = syncScript.match(/const PACKAGE_DIRS = \[([\s\S]*?)\];/);
  assert.ok(match, "could not locate a PACKAGE_DIRS array in sync-jsr-configs.mjs -- has its shape changed?");
  const dirs = new Set([...(match as RegExpMatchArray)[1].matchAll(/"([^"]+)"/g)].map((m) => m[1] as string));

  const packages = discoverWorkspacePackages();
  const missing = packages.filter((p) => !dirs.has(p.dir)).map((p) => p.dir);
  assert.deepEqual(
    missing,
    [],
    `package dir(s) missing from PACKAGE_DIRS: ${missing.join(", ")} -- these ship to npm but never to JSR`,
  );

  const knownDirs = new Set(packages.map((p) => p.dir));
  const stale = [...dirs].filter((d) => !knownDirs.has(d));
  assert.deepEqual(stale, [], `stale entries in PACKAGE_DIRS (dir no longer exists): ${stale.join(", ")}`);
});

test('every npm workspace package has a "test:bun" script (scripts/test-bun.mjs discovers suites by it)', () => {
  const missing = discoverWorkspacePackages()
    .filter((p) => {
      const pkg = JSON.parse(readFileSync(join(ROOT, p.dir, "package.json"), "utf8")) as { scripts?: Record<string, string> };
      return !pkg.scripts?.["test:bun"];
    })
    .map((p) => p.name);
  assert.deepEqual(missing, [], `package(s) without a "test:bun" script: ${missing.join(", ")} -- add "test:bun": "bun test ./test/"`);
});

test("every test file registers through test/harness.ts with its OWN bun:test import (Bun 1.2 multi-file bug)", () => {
  // Under Bun 1.2, tests declared via node:test -- or via a shared module that
  // imports bun:test -- register only for the first file of a run; the rest are
  // silently dropped. See test/harness.ts. A file that regresses to
  // `import { test } from "node:test"` still passes `npm test`, so only this
  // check (or the Bun CI job's file count) would notice.
  const dirs = ["test", ...discoverWorkspacePackages().map((p) => `${p.dir}/test`)];
  const offenders: string[] = [];
  for (const dir of dirs) {
    if (!existsSync(join(ROOT, dir))) continue;
    for (const f of readdirSync(join(ROOT, dir))) {
      if (!/\.(bench-)?test\.ts$/.test(f)) continue;
      const src = readFileSync(join(ROOT, dir, f), "utf8");
      const ok =
        /harness\.ts["']/.test(src) &&
        src.includes('await import("bun:test")') &&
        !/^import[^;]*from ["']node:test["']/m.test(src);
      if (!ok) offenders.push(`${dir}/${f}`);
    }
  }
  assert.deepEqual(offenders, [], `test file(s) not using the per-file makeTest(bun:test) pattern: ${offenders.join(", ")}`);
});

test("engines.node is identical in root package.json and every workspace package", () => {
  const rootEngine = (JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")) as { engines: { node: string } }).engines.node;
  const mismatched = discoverWorkspacePackages()
    .map((p) => ({ p, pkg: JSON.parse(readFileSync(join(ROOT, p.dir, "package.json"), "utf8")) as { engines?: { node?: string } } }))
    .filter(({ pkg }) => pkg.engines?.node !== rootEngine)
    .map(({ p, pkg }) => `${p.name} (${pkg.engines?.node ?? "none"})`);
  assert.deepEqual(mismatched, [], `engines.node differs from root's "${rootEngine}": ${mismatched.join(", ")}`);
});
