#!/usr/bin/env node
/**
 * Generates/refreshes every publishable workspace package's jsr.json from
 * its package.json (issue #25: dual npm+JSR distribution) — a single
 * source of truth instead of two hand-maintained, driftable manifests.
 *
 * This repo's npm packages are scoped `@johnhenry/math-plus-*`, and each
 * package's JSR name is that same scoped npm name (package.json `name` is
 * already JSR-legal, so no extra scoping/prefixing is applied here).
 *
 * JSR publishes/type-checks the TypeScript SOURCE directly (not the
 * compiled `dist/` npm ships), so `exports` points at `./src/index.ts`.
 * Bare-specifier imports of workspace siblings ("@johnhenry/math-plus-tensor-core")
 * and external npm deps ("apache-arrow") need an explicit import map for
 * JSR's resolver — plain npm `dependencies`/`peerDependencies` entries in
 * package.json are translated into `imports` here:
 *   - a `@johnhenry/math-plus-*` workspace sibling (this monorepo's own
 *     family) maps to `jsr:@johnhenry/math-plus-<suffix>@<version>`.
 *   - `@johnhenry/math` and `@johnhenry/iteration` are a DIFFERENT
 *     monorepo's packages, but are also published to JSR under the same
 *     `@johnhenry` scope, so they map to `jsr:@johnhenry/math@<version>` /
 *     `jsr:@johnhenry/iteration@<version>` rather than `npm:`.
 *   - anything else maps to `npm:<name>@<version>`.
 *
 * Run manually after adding/bumping a dependency, or wire into a
 * pre-publish CI step (see .github/workflows/release.yml's jsr job).
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;

const PACKAGE_DIRS = [
  "packages/fft",
  "packages/data",
  "packages/frame-arrow",
  "packages/frame-parquet",
  "packages/image",
  "packages/mcp",
  "packages/safetensors",
  "packages/scalar-types",
  "packages/signal",
  "packages/special",
  "packages/telemetry",
  "packages/tensor-autograd",
  "packages/tensor-compile",
  "packages/tensor-core",
  "packages/tensor-wasm",
  "packages/tensor-webgpu",
  "adapters/adapter-math",
  "adapters/adapter-onnx",
  "scalars/unit",
];

/**
 * npm workspace packages deliberately NOT published to JSR. Every entry needs
 * a reason; test/manifest-drift.test.ts requires each workspace package to be
 * in exactly one of PACKAGE_DIRS / JSR_EXCLUDED_DIRS.
 */
const JSR_EXCLUDED_DIRS = [
  // Native (mlx-c over koffi/bun:ffi via @johnhenry/backend-mlx), Node + Bun
  // on darwin/arm64 only: no Deno.dlopen path exists, so a JSR listing would
  // advertise a Deno package that cannot load. Revisit if backend-mlx gains
  // a Deno adapter (RFC 0001, open question 4).
  "packages/tensor-mlx",
];
for (const dir of JSR_EXCLUDED_DIRS) {
  if (PACKAGE_DIRS.includes(dir)) throw new Error(`${dir} is in both PACKAGE_DIRS and JSR_EXCLUDED_DIRS`);
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

// This monorepo's own workspace family — mapped to jsr:@johnhenry/math-plus-<suffix>.
const INTERNAL_SCOPE_PREFIX = "@johnhenry/math-plus-";

// Other repos in the @johnhenry family that are ALSO published to JSR
// (not workspace siblings of this repo, but still jsr: specifiers rather
// than npm: ones).
const EXTERNAL_JSR_PACKAGES = new Set(["@johnhenry/math", "@johnhenry/iteration"]);

function buildImports(pkg) {
  const deps = { ...pkg.dependencies, ...pkg.peerDependencies };
  const imports = {};
  for (const [name, version] of Object.entries(deps)) {
    const range = version.replace(/^[\^~]/, "");
    if (name.startsWith(INTERNAL_SCOPE_PREFIX) || EXTERNAL_JSR_PACKAGES.has(name)) {
      // Workspace sibling or sibling-family package, also published to JSR
      // under the same @johnhenry scope.
      imports[name] = `jsr:${name}@^${range}`;
    } else {
      imports[name] = `npm:${name}@${version}`;
    }
  }
  return imports;
}

/**
 * JSR `exports` from package.json's: a lone "." stays the plain
 * `"./src/index.ts"` string; packages with subpath exports (e.g.
 * `@johnhenry/math-plus-safetensors/tensor`,
 * `@johnhenry/math-plus-tensor-autograd/safetensors`) get an exports MAP,
 * each `./dist/<file>.js` target mapped to its `./src/<file>.ts` source —
 * otherwise those subpaths would silently not exist on JSR.
 */
function buildExports(pkg) {
  const exp = pkg.exports;
  if (!exp || typeof exp !== "object" || Object.keys(exp).length <= 1) return "./src/index.ts";
  const out = {};
  for (const [subpath, target] of Object.entries(exp)) {
    const dist = typeof target === "string" ? target : (target.default ?? target.import);
    const match = /^\.\/dist\/(.+)\.js$/.exec(dist ?? "");
    if (!match) throw new Error(`${pkg.name}: cannot map export "${subpath}" (${JSON.stringify(target)}) to a src/*.ts file`);
    out[subpath] = `./src/${match[1]}.ts`;
  }
  return out;
}

for (const dir of PACKAGE_DIRS) {
  const pkgPath = join(ROOT, dir, "package.json");
  const pkg = readJson(pkgPath);
  const imports = buildImports(pkg);

  const jsrConfig = {
    name: pkg.name,
    version: pkg.version,
    // JSR hard-requires a license (error[missing-license] otherwise --
    // found on the first real publish run, 2026-08-14).
    license: pkg.license ?? "MIT",
    exports: buildExports(pkg),
    ...(Object.keys(imports).length > 0 ? { imports } : {}),
  };

  const jsrPath = join(ROOT, dir, "jsr.json");
  writeFileSync(jsrPath, `${JSON.stringify(jsrConfig, null, 2)}\n`);
  console.log(`wrote ${dir}/jsr.json (${pkg.name}@${pkg.version})`);
}
