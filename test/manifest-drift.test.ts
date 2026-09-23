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
 * PACKAGE_DIRS entry means the package ships to npm but never to JSR --
 * unless the package is listed, with a reason, in JSR_EXCLUDED_DIRS.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
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

function stringArray(source: string, name: string): Set<string> {
  const match = source.match(new RegExp(`const ${name} = \\[([\\s\\S]*?)\\];`));
  assert.ok(match, `could not locate a ${name} array in sync-jsr-configs.mjs -- has its shape changed?`);
  // Strip // comments so quoted words inside a justification don't count as entries.
  const body = (match as RegExpMatchArray)[1].replace(/\/\/.*$/gm, "");
  return new Set([...body.matchAll(/"([^"]+)"/g)].map((m) => m[1] as string));
}

test("every npm workspace package's directory is in exactly one of PACKAGE_DIRS / JSR_EXCLUDED_DIRS", () => {
  const syncScript = readFileSync(join(ROOT, "scripts/sync-jsr-configs.mjs"), "utf8");
  const dirs = stringArray(syncScript, "PACKAGE_DIRS");
  const excluded = stringArray(syncScript, "JSR_EXCLUDED_DIRS");

  const packages = discoverWorkspacePackages();
  const missing = packages.filter((p) => !dirs.has(p.dir) && !excluded.has(p.dir)).map((p) => p.dir);
  assert.deepEqual(
    missing,
    [],
    `package dir(s) missing from PACKAGE_DIRS: ${missing.join(", ")} -- these ship to npm but never to JSR (add to JSR_EXCLUDED_DIRS with a reason if that is deliberate)`,
  );

  const both = [...excluded].filter((d) => dirs.has(d));
  assert.deepEqual(both, [], `dir(s) in both PACKAGE_DIRS and JSR_EXCLUDED_DIRS: ${both.join(", ")}`);

  const knownDirs = new Set(packages.map((p) => p.dir));
  const stale = [...dirs, ...excluded].filter((d) => !knownDirs.has(d));
  assert.deepEqual(stale, [], `stale entries in PACKAGE_DIRS/JSR_EXCLUDED_DIRS (dir no longer exists): ${stale.join(", ")}`);
});
