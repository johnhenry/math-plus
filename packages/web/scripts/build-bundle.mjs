#!/usr/bin/env node
/**
 * Builds the browser bundles (math-plus#173, #176) from the compiled entries in dist/:
 * one self-contained ESM file per entry (no import specifiers, so its text can be handed to a Worker or an iframe as a
 * Blob URL and run offline), plus dist/bundle/sizes.json, the per-package size table.
 *
 *   node scripts/build-bundle.mjs            build the bundles and sizes.json
 *   node scripts/build-bundle.mjs --readme   also rewrite the size table in README.md
 *
 * Every standalone bundle carries its own copy of tensor-core, so two of them loaded together hold two `Tensor` classes.
 * `all.js` holds one copy that every member shares; use it when results cross packages.
 */
import { build } from "esbuild";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliCompressSync, gzipSync } from "node:zlib";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "dist");
const out = join(dist, "bundle");

/** entry name -> [compiled module under dist/, the package(s) it carries]. */
export const ENTRIES = {
  fft: ["fft.js", "@johnhenry/math-plus-fft"],
  signal: ["signal.js", "@johnhenry/math-plus-signal/web (includes fft)"],
  unit: ["unit.js", "@johnhenry/math-plus-unit"],
  image: ["image.js", "@johnhenry/math-plus-image"],
  tensor: ["tensor.js", "@johnhenry/math-plus-tensor-core"],
  "tensor-autograd": ["tensor-autograd.js", "@johnhenry/math-plus-tensor-autograd (includes tensor-core)"],
  "adapter-math": ["adapter-math.js", "@johnhenry/math-plus-adapter-math (includes @johnhenry/math, tensor-compile, frame-arrow)"],
  "frame-arrow": ["frame-arrow.js", "@johnhenry/math-plus-frame-arrow (includes apache-arrow)"],
  data: ["data.js", "@johnhenry/math-plus-data (includes @johnhenry/iteration and tensor-core)"],
  all: ["index.js", "all of the above, one shared copy of each"],
};

const repoRoot = join(root, "..", "..");

/** Third-party packages (not this repo's own workspaces) that esbuild put into a bundle, as `name@version`. */
function thirdPartyOf(metafile) {
  const found = new Map();
  for (const input of Object.keys(metafile.inputs)) {
    const m = input.match(/(?:^|\/)node_modules\/((?:@[^/]+\/)?[^/]+)\//);
    if (!m) continue;
    const dir = resolve(root, input.slice(0, input.indexOf(m[0]) + m[0].length));
    const pkg = JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
    if (String(pkg.repository?.url ?? pkg.repository).includes("johnhenry/math-plus")) continue; // a workspace sibling
    found.set(`${pkg.name}@${pkg.version}`, { dir, license: pkg.license });
  }
  return found;
}

function licenseText(dir) {
  const file = readdirSync(dir).find((f) => /^(licen[cs]e|copying)(\.|$)/i.test(f));
  return file ? readFileSync(join(dir, file), "utf8").trim() : null;
}

export async function buildBundles() {
  rmSync(out, { recursive: true, force: true });
  mkdirSync(out, { recursive: true });
  const entries = {};
  const licenses = new Map();
  for (const [name, [src, pkg]] of Object.entries(ENTRIES)) {
    const outfile = join(out, `${name}.js`);
    let thirdParty = new Map();
    await build({
      entryPoints: [join(dist, src)],
      outfile,
      bundle: true,
      format: "esm",
      platform: "browser",
      target: "es2023",
      minify: true,
      legalComments: "none",
      treeShaking: true,
      metafile: true,
      absWorkingDir: root,
      logLevel: "warning",
    }).then((result) => {
      thirdParty = thirdPartyOf(result.metafile);
    });
    const buf = readFileSync(outfile);
    entries[name] = {
      package: pkg,
      file: `${name}.js`,
      bytes: statSync(outfile).size,
      gzip: gzipSync(buf, { level: 9 }).length,
      brotli: brotliCompressSync(buf).length,
      thirdParty: [...thirdParty.keys()].sort(),
    };
    for (const [id, info] of thirdParty) licenses.set(id, info);
  }
  // minify strips every comment, licence headers included, so the licences travel beside the bundles instead.
  const parts = [`== math-plus (every bundle) ==\n${readFileSync(join(repoRoot, "LICENSE"), "utf8").trim()}`];
  for (const [id, info] of [...licenses].sort(([a], [b]) => a.localeCompare(b))) {
    const text = licenseText(info.dir);
    if (!text && !info.license) throw new Error(`${id}: no licence file and no license field in ${info.dir}`);
    parts.push(`== ${id} (${info.license}) ==\n${text ?? `The package declares license "${info.license}" in its package.json and ships no licence file.`}`);
  }
  writeFileSync(join(out, "LICENSES.txt"), `${parts.join("\n\n")}\n`);
  const sizes = { format: "esm", minified: true, entries };
  writeFileSync(join(out, "sizes.json"), `${JSON.stringify(sizes, null, 2)}\n`);
  return sizes;
}

const kb = (n) => `${(n / 1024).toFixed(1)} kB`;

export function sizeTable(sizes) {
  const rows = Object.entries(sizes.entries).map(
    ([name, r]) => `| \`${name}.js\` | ${r.package} | ${kb(r.bytes)} | ${kb(r.gzip)} | ${kb(r.brotli)} |`,
  );
  return ["| Bundle | Carries | Minified | gzip | brotli |", "| --- | --- | ---: | ---: | ---: |", ...rows].join("\n");
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const sizes = await buildBundles();
  if (process.argv.includes("--readme")) {
    const path = join(root, "README.md");
    const text = readFileSync(path, "utf8");
    const next = text.replace(/<!-- sizes:start -->[\s\S]*?<!-- sizes:end -->/, `<!-- sizes:start -->\n${sizeTable(sizes)}\n<!-- sizes:end -->`);
    if (next === text) throw new Error("README.md has no sizes markers");
    writeFileSync(path, next);
  }
  console.log(sizeTable(sizes));
}
