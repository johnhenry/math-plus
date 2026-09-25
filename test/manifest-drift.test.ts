/**
 * Guards against the "functions.csv rot" failure mode documented in
 * docs/spikes/woxi-study.md: a hand-maintained manifest with no validating
 * test silently drifts from reality. This repo has one remaining such
 * manifest -- scripts/sync-jsr-configs.mjs's PACKAGE_DIRS -- which has
 * already required a manual edit (easy to forget) for every new package
 * added. A forgotten PACKAGE_DIRS entry means the package ships to npm but
 * never to JSR -- unless the package is listed, with a reason, in
 * JSR_EXCLUDED_DIRS.
 *
 * Root package.json's build/test scripts used to be a second such manifest
 * (a hand-maintained `-w <name>` list, one entry per package). Since
 * 2026-09-25 they call `turbo run build`/`turbo run test` instead, which
 * covers every workspace package with that script by construction -- so
 * that failure mode no longer applies to them; see the "routes through
 * Turborepo" test below for what replaced the old per-package check.
 */
import assert from "node:assert/strict";
import { makeTest } from "./harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
// @ts-expect-error -- plain .mjs script without type declarations (importing it does not write files)
import { PACKAGE_DIRS, buildImports, jsrRange } from "../scripts/sync-jsr-configs.mjs";
// @ts-expect-error -- plain .mjs script without type declarations (importing it does not rewrite files)
import { addSelfTypes, findMissingSelfTypes, findTsSpecifiers, rewriteSpecifiers } from "../scripts/rewrite-dts-extensions.mjs";
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

test("root build AND test scripts route through Turborepo, not a hand-maintained -w list", () => {
  // Since 2026-09-25 (turbo.json adoption), root build/test call `turbo run
  // <task>`, which runs a task for every workspace package that defines it
  // -- structurally, not via a list that can go stale. The old assertion
  // here checked for a literal "-w <name>" per package; that's no longer
  // the mechanism, so this now checks the new one is actually in place.
  const rootPkg = readRootPackageJson();
  assert.match(rootPkg.scripts.build, /turbo run build/, 'root "build" script must invoke "turbo run build"');
  assert.match(rootPkg.scripts.test, /turbo run test/, 'root "test" script must invoke "turbo run test"');
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

/**
 * Deno/JSR take ONE comparator per `jsr:`/`npm:` specifier: `^1.2.3`, `~1.2`,
 * `1.2.3`, `1.x`, `*` (verified with Deno 2.9.7; `a || b`, `>=a <b` and `<b`
 * fail with "Invalid package specifier ... Unexpected character"). Release run
 * 35967422091's JSR job failed on `jsr:@johnhenry/math-plus-tensor-core@^0.0.0 || ^0.1.0`
 * (safetensors, and tensor-autograd's safetensors peer), cascading to every
 * dependent package.
 */
const VALID_SPECIFIER = /^(?:jsr:@[a-z0-9-]+\/[a-z0-9-]+|npm:(?:@[a-z0-9._-]+\/)?[a-z0-9._-]+)@(?:\*|[\^~]?\d+(?:\.(?:\d+|x|\*)){0,2}(?:-[0-9A-Za-z.-]+)?)$/;

test("every committed jsr.json import specifier is one Deno/JSR accept (single comparator, no `||` or spaces)", () => {
  const bad: string[] = [];
  for (const dir of PACKAGE_DIRS as string[]) {
    const jsr = JSON.parse(readFileSync(join(ROOT, dir, "jsr.json"), "utf8")) as { imports?: Record<string, string> };
    for (const [name, spec] of Object.entries(jsr.imports ?? {})) if (!VALID_SPECIFIER.test(spec)) bad.push(`${dir}: ${name} -> ${spec}`);
  }
  assert.deepEqual(bad, [], `invalid JSR import specifiers (run node scripts/sync-jsr-configs.mjs):\n${bad.join("\n")}`);
});

test("sync-jsr-configs.mjs generates valid specifiers from every package.json (what the release job publishes)", () => {
  const bad: string[] = [];
  for (const dir of PACKAGE_DIRS as string[]) {
    const pkg = JSON.parse(readFileSync(join(ROOT, dir, "package.json"), "utf8"));
    for (const [name, spec] of Object.entries(buildImports(pkg) as Record<string, string>)) if (!VALID_SPECIFIER.test(spec)) bad.push(`${dir}: ${name} -> ${spec}`);
  }
  assert.deepEqual(bad, [], `generator emits invalid JSR import specifiers:\n${bad.join("\n")}`);
});

test("jsrRange turns npm unions into the highest single comparator and refuses what it cannot express", () => {
  assert.equal(jsrRange("^0.1.2"), "^0.1.2");
  assert.equal(jsrRange("^0.0.0 || ^0.1.0"), "^0.1.0");
  assert.equal(jsrRange("^0.1.0 || ^0.0.0"), "^0.1.0");
  assert.equal(jsrRange("^0.2.0 || ^0.10.0 || ^0.9.1"), "^0.10.0");
  assert.equal(jsrRange("21.2.0"), "21.2.0");
  assert.throws(() => jsrRange(">=1.0.0 <2.0.0"), /single/);
  assert.throws(() => jsrRange("^1.0.0 || >=3"), /single/);
});

/**
 * Issue #157: tsc rewrites `./x.ts` imports to `./x.js` in emitted JS but not
 * in emitted `.d.ts`, so published declarations pointed at files that don't
 * exist in `dist/` (Deno's type check fails on them). Every package's build
 * runs scripts/rewrite-dts-extensions.mjs after tsc.
 */
const DTS_STEP = "tsc -p tsconfig.json && node ../../scripts/rewrite-dts-extensions.mjs";

test('every npm workspace package\'s "build" runs scripts/rewrite-dts-extensions.mjs after tsc (issue #157)', () => {
  const missing = discoverWorkspacePackages()
    .filter((p) => {
      const pkg = JSON.parse(readFileSync(join(ROOT, p.dir, "package.json"), "utf8")) as { scripts?: Record<string, string> };
      return pkg.scripts?.build !== DTS_STEP;
    })
    .map((p) => p.name);
  assert.deepEqual(missing, [], `package(s) whose "build" is not "${DTS_STEP}": ${missing.join(", ")}`);
});

test("no built dist/**/*.d.ts references a relative .ts path (issue #157)", (t) => {
  const built = discoverWorkspacePackages().filter((p) => existsSync(join(ROOT, p.dir, "dist")));
  if (!built.length) return t.skip("no package is built (run npm run build first)");
  const hits = built.flatMap((p) => (findTsSpecifiers(join(ROOT, p.dir, "dist")) as string[]).map((h) => h.slice(ROOT.length + 1)));
  assert.deepEqual(hits, [], `declaration files still import .ts paths (rebuild; see scripts/rewrite-dts-extensions.mjs):\n${hits.join("\n")}`);
});

test("every built dist/**/*.js with a declaration file points Deno at it with a @ts-self-types first line (issue #157)", (t) => {
  const built = discoverWorkspacePackages().filter((p) => existsSync(join(ROOT, p.dir, "dist")));
  if (!built.length) return t.skip("no package is built (run npm run build first)");
  const missing = built.flatMap((p) => (findMissingSelfTypes(join(ROOT, p.dir, "dist")) as string[]).map((f) => f.slice(ROOT.length + 1)));
  assert.deepEqual(missing, [], `emitted JS without the @ts-self-types directive (rebuild; see scripts/rewrite-dts-extensions.mjs):\n${missing.join("\n")}`);
});

test("rewrite-dts-extensions rewrites relative .ts/.mts/.cts specifiers in every import/export form, and nothing else", () => {
  const src = [
    'import { a } from "./a.ts";',
    "export * from '../b/c.ts';",
    'export { type D } from "./d.mts";',
    'type E = import("./e.cts").E;',
    'import "./side-effect.ts";',
    'import type { F } from "./f.d.ts";',
    'import { G } from "pkg/g.ts";',
    'import { H } from "./h.js";',
    'const s = "./not-an-import.ts";',
  ].join("\n");
  const { out, n } = rewriteSpecifiers(src) as { out: string; n: number };
  assert.equal(n, 5);
  assert.equal(
    out,
    [
      'import { a } from "./a.js";',
      "export * from '../b/c.js';",
      'export { type D } from "./d.mjs";',
      'type E = import("./e.cjs").E;',
      'import "./side-effect.js";',
      'import type { F } from "./f.d.ts";',
      'import { G } from "pkg/g.ts";',
      'import { H } from "./h.js";',
      'const s = "./not-an-import.ts";',
    ].join("\n"),
  );
});

test("addSelfTypes: inserts the directive once (after a #! line), shifts the source map by one line, and skips JS without declarations", () => {
  const dir = mkdtempSync(join(tmpdir(), "dts-self-types-"));
  try {
    writeFileSync(join(dir, "a.js"), "export const a = 1;\n");
    writeFileSync(join(dir, "a.d.ts"), "export declare const a = 1;\n");
    writeFileSync(join(dir, "a.js.map"), JSON.stringify({ version: 3, mappings: "AAAA;AACA" }));
    writeFileSync(join(dir, "cli.js"), "#!/usr/bin/env node\nrun();\n");
    writeFileSync(join(dir, "cli.d.ts"), "export {};\n");
    writeFileSync(join(dir, "cli.js.map"), JSON.stringify({ version: 3, mappings: ";AAAA;AACA" }));
    writeFileSync(join(dir, "untyped.js"), "1;\n");
    assert.deepEqual((findMissingSelfTypes(dir) as string[]).map((f) => f.slice(dir.length + 1)).sort(), ["a.js", "cli.js"]);
    assert.equal(addSelfTypes(join(dir, "a.js")), true);
    assert.equal(addSelfTypes(join(dir, "a.js")), false, "idempotent");
    assert.equal(readFileSync(join(dir, "a.js"), "utf8"), '// @ts-self-types="./a.d.ts"\nexport const a = 1;\n');
    assert.equal(JSON.parse(readFileSync(join(dir, "a.js.map"), "utf8")).mappings, ";AAAA;AACA");
    assert.equal(addSelfTypes(join(dir, "cli.js")), true);
    assert.equal(readFileSync(join(dir, "cli.js"), "utf8"), '#!/usr/bin/env node\n// @ts-self-types="./cli.d.ts"\nrun();\n');
    assert.equal(JSON.parse(readFileSync(join(dir, "cli.js.map"), "utf8")).mappings, ";;AAAA;AACA");
    assert.deepEqual(findMissingSelfTypes(dir), []);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
