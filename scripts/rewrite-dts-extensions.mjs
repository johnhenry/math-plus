#!/usr/bin/env node
/**
 * Post-build step for every package (issue #157): rewrites relative `.ts`
 * import specifiers in emitted declaration files to `.js`.
 *
 * The sources import siblings as "./x.ts" (Node >= 24, Bun and Deno run them
 * directly), and tsc's `rewriteRelativeImportExtensions` fixes the emitted
 * .js, but tsc leaves the specifiers in .d.ts files untouched. TypeScript
 * consumers happen to resolve "./x.ts" to x.d.ts, but other type checkers do
 * not: Deno's `npm:` type checking fails with TS2307 on every one.
 *
 * The same rewrite as laya-js's `scripts/rewrite-dts-extensions.mjs`, plus a
 * guard that leaves explicit declaration-file specifiers ("./x.d.ts") alone.
 *
 * Second step: every emitted `x.js` with a sibling `x.d.ts` gets a first line
 * `// @ts-self-types="./x.d.ts"` (its source map is shifted by one line to
 * match). TypeScript and npm-aware type checkers map `./x.js` to `x.d.ts`
 * themselves, but Deno does that only inside an `npm:` package. When it loads
 * `dist/` as local files or over a URL (a workspace symlink, a CDN), a `.d.ts`
 * import of "./x.js" reaches the JS file, whose types Deno then infers from
 * the JS, losing every type-only export. The directive points it back at the
 * declarations. It is a comment to everything else.
 *
 * `test/manifest-drift.test.ts` imports {@link findTsSpecifiers} and
 * {@link findMissingSelfTypes} to check the built `dist/` directories, and
 * checks that every package's `build` script runs this step.
 *
 * Usage (from a package directory, after `tsc`):
 *   node ../../scripts/rewrite-dts-extensions.mjs [distDir=dist]
 * Importing this module does not rewrite anything.
 */
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** `from "./x.ts"`, `import("../y.mts")`, `import "./z.cts"` — a relative specifier ending in .ts/.mts/.cts, not .d.ts. */
export const RELATIVE_TS_SPECIFIER = /((?:from|import)\s*\(?\s*["'])(\.{1,2}\/[^"']+?)(?<!\.d)\.(m|c)?ts(["'])/g;

/** Every declaration file (`.d.ts`, `.d.mts`, `.d.cts`) under `dir`. */
export function declarationFiles(dir) {
  const out = [];
  const walk = (d) => {
    for (const e of readdirSync(d, { withFileTypes: true })) {
      const p = join(d, e.name);
      if (e.isDirectory()) walk(p);
      else if (/\.d\.(m|c)?ts$/.test(e.name)) out.push(p);
    }
  };
  walk(dir);
  return out;
}

/** `src` with every relative `.ts`/`.mts`/`.cts` specifier rewritten to `.js`/`.mjs`/`.cjs`, and the number of rewrites. */
export function rewriteSpecifiers(src) {
  let n = 0;
  const out = src.replace(RELATIVE_TS_SPECIFIER, (_m, pre, spec, mc, q) => (n++, `${pre}${spec}.${mc ?? ""}js${q}`));
  return { out, n };
}

/** The `// @ts-self-types` directive line for `file` (an emitted `.js`/`.mjs`/`.cjs`). */
function selfTypesLine(file) {
  return `// @ts-self-types="./${basename(file).replace(/\.(m|c)?js$/, ".d.$1ts")}"`;
}

/** Emitted JS files under `dir` that have a sibling declaration file. */
export function typedJsFiles(dir) {
  return declarationFiles(dir)
    .map((d) => d.replace(/\.d\.(m|c)?ts$/, ".$1js"))
    .filter((js) => existsSync(js));
}

/** Whether `src` has `line` as its first line, or as its second after a `#!` line. */
function hasDirective(src, line) {
  const body = src.startsWith("#!") ? src.slice(src.indexOf("\n") + 1) : src;
  return body.startsWith(line + "\n");
}

/**
 * Inserts the `@ts-self-types` directive as the first line of `js` (after a
 * `#!` line if there is one; idempotent), and shifts its source map by one
 * line to match. Returns whether it changed the file.
 */
export function addSelfTypes(js) {
  const line = selfTypesLine(js);
  const src = readFileSync(js, "utf8");
  if (hasDirective(src, line)) return false;
  const shebang = src.startsWith("#!");
  const cut = shebang ? src.indexOf("\n") + 1 : 0;
  writeFileSync(js, `${src.slice(0, cut)}${line}\n${src.slice(cut)}`);
  const map = `${js}.map`;
  if (existsSync(map)) {
    const m = JSON.parse(readFileSync(map, "utf8"));
    // `mappings` has one `;`-separated group per generated line: add an empty group at the inserted line.
    m.mappings = shebang ? m.mappings.replace(/;|$/, ";;") : `;${m.mappings}`;
    writeFileSync(map, JSON.stringify(m));
  }
  return true;
}

/** Emitted JS files under `dir` with a sibling declaration file but no `@ts-self-types` directive line. */
export function findMissingSelfTypes(dir) {
  return typedJsFiles(dir).filter((js) => !hasDirective(readFileSync(js, "utf8"), selfTypesLine(js)));
}

/** `file:line: specifier` for every relative `.ts` specifier left in the declaration files under `dir`. */
export function findTsSpecifiers(dir) {
  const hits = [];
  for (const f of declarationFiles(dir)) {
    readFileSync(f, "utf8")
      .split("\n")
      .forEach((line, i) => {
        for (const m of line.matchAll(RELATIVE_TS_SPECIFIER)) hits.push(`${f}:${i + 1}: ${m[2]}.${m[3] ?? ""}ts`);
      });
  }
  return hits;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const dir = process.argv[2] ?? "dist";
  let files = 0;
  let edits = 0;
  for (const f of declarationFiles(dir)) {
    const { out, n } = rewriteSpecifiers(readFileSync(f, "utf8"));
    if (n) {
      writeFileSync(f, out);
      files++;
      edits += n;
    }
  }
  let directives = 0;
  for (const js of typedJsFiles(dir)) if (addSelfTypes(js)) directives++;
  if (process.env.DEBUG_DTS) console.log(`rewrote ${edits} specifier(s) in ${files} declaration file(s) and added ${directives} @ts-self-types directive(s) under ${dir}`);
}
