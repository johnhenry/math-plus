import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { execFileSync } from "node:child_process";
import { build } from "esbuild";
import { existsSync, readFileSync, statSync } from "node:fs";
import { builtinModules } from "node:module";
import { dirname, join } from "node:path";
import { brotliCompressSync, gzipSync } from "node:zlib";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const bundleDir = join(root, "dist", "bundle");
const pkgJson = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));

/** Entry name -> what its bundle must export (a sample, not the whole surface). */
const ENTRIES: Record<string, string[]> = {
  fft: ["fft", "ifft", "rfft", "irfft", "fft2", "ComplexTensor", "Tensor"],
  signal: ["rfft", "welch", "stft", "butter", "sosfiltfilt", "findPeaks", "resamplePoly", "Tensor"],
  unit: ["Unit", "DimensionMismatchError"],
  image: ["resize", "normalize", "Tensor"],
  tensor: ["Tensor"],
  "tensor-autograd": ["Variable", "grad", "nn", "optim", "Tensor", "random"],
  "adapter-math": ["fromMatrix", "toMatrix", "compileExpr", "linalg", "Tensor"],
  "frame-arrow": ["Frame", "Series"],
  data: ["Dataset", "Tensor"],
  all: ["fft", "signal", "unit", "image", "tensor", "autograd", "adapterMath", "frameArrow", "data"],
};

const file = (name: string) => join(bundleDir, `${name}.js`);

test("package.json exposes every bundle and every module entry", () => {
  for (const name of Object.keys(ENTRIES)) {
    const key = name === "all" ? "./bundle" : `./bundle/${name}`;
    assert.equal(pkgJson.exports[key]?.default, `./dist/bundle/${name}.js`, `exports[${key}]`);
    if (name !== "all") assert.ok(pkgJson.exports[`./${name}`]?.default, `exports[./${name}]`);
  }
  assert.ok(pkgJson.exports["."]?.default);
  assert.ok(pkgJson.exports["./bundle/sizes.json"]);
});

test("every bundle was built (run `npm run build -w @johnhenry/math-plus-web`)", () => {
  for (const name of Object.keys(ENTRIES)) assert.ok(existsSync(file(name)), `${file(name)} is missing`);
  assert.ok(existsSync(join(bundleDir, "sizes.json")));
});

test("every bundle is one self-contained file: no import or export-from specifier, no Node built-in, no Node global", async () => {
  const builtins = new Set([...builtinModules, ...builtinModules.map((b) => `node:${b}`)]);
  for (const name of Object.keys(ENTRIES)) {
    const src = readFileSync(file(name), "utf8");
    // esbuild parses the file (no bundling) and lists every import, static or dynamic, that the parse finds.
    const { metafile } = await build({ entryPoints: [file(name)], bundle: false, write: false, metafile: true, format: "esm", logLevel: "silent" });
    const imports: string[] = Object.values(metafile.inputs).flatMap((i) => i.imports.map((x) => x.path));
    assert.equal(imports.length, 0, `${name}: imports ${imports.join(", ")}`);
    for (const b of builtins) assert.ok(!imports.includes(b), `${name}: imports Node built-in ${b}`);
    assert.ok(!/\b(?:process\.(?:env|versions|platform|cwd)|Buffer\.from|require\(|__dirname|__filename)/.test(src), `${name}: uses a Node global`);
    assert.match(src, /\bexport\s*\{/, `${name}: has no export statement`);
  }
});

// Each bundle is loaded as a host that gets a module as text loads it (Viaframe hands frames a Blob URL; Node takes a data: URL), in a process with the Node globals removed.
const LOADER = `
  const { readFileSync } = await import("node:fs"); // before the globals go: Bun's own internals need \`process\` while loading
  const g = globalThis;
  for (const k of ["process", "Buffer", "require", "module", "exports", "__dirname", "__filename"]) {
    try { delete g[k]; } catch {}
    try { Object.defineProperty(g, k, { value: undefined, configurable: true }); } catch {}
  }
  // Node's loader takes no blob: URLs, so the text goes in as a data: URL: the same "module from its text" path.
  // Bun refuses a data: URL that long, so there the file itself is imported.
  const load = async (path) => import(globalThis.Bun ? "file://" + path : "data:text/javascript;base64," + readFileSync(path).toString("base64"));
`;

function runScript(body: string): string {
  return execFileSync(process.execPath, ["--input-type=module", "-e", LOADER + body], { encoding: "utf8" });
}

test("every bundle loads from its text (a data: URL) with the Node globals removed and exports its names", () => {
  const out = runScript(`
    const want = ${JSON.stringify(ENTRIES)};
    const paths = ${JSON.stringify(Object.fromEntries(Object.keys(ENTRIES).map((n) => [n, file(n)])))};
    for (const [name, names] of Object.entries(want)) {
      const m = await load(paths[name]);
      for (const n of names) if (m[n] === undefined) throw new Error(name + " lacks " + n);
    }
    console.log("OK");
  `);
  assert.match(out, /OK/);
});

test("one import of the all-in-one bundle gives fft, signal and unit, working", () => {
  const out = runScript(`
    const { fft, signal, unit, tensor, image, autograd, adapterMath, frameArrow } = await load(${JSON.stringify(file("all"))});
    const N = 64;
    const x = tensor.Tensor.from(Array.from({ length: N }, (_, i) => Math.sin(2 * Math.PI * 8 * i / N)), { dtype: "f64" });
    const spec = fft.rfft(x);
    if (!spec) throw new Error("rfft");
    const { freqs, Pxx } = signal.welch(x, { nperseg: 32 });
    if (freqs.length !== 17 || Pxx.shape[0] !== 17) throw new Error("welch " + freqs.length);
    const sos = signal.butter(2, 0.3, { btype: "lowpass" });
    if (signal.sosfiltfilt(sos, x).shape[0] !== N) throw new Error("sosfiltfilt");
    const d = unit.Unit.of(5, "m").add(unit.Unit.of(20, "cm"));
    if (Math.abs(d.to("m").value - 5.2) > 1e-12) throw new Error("unit add " + d.to("m").value);
    // one copy of Tensor: what signal returns is what tensor and autograd accept
    if (signal.Tensor !== tensor.Tensor) throw new Error("two Tensor classes");
    const { value, grad: dv } = autograd.grad.valueAndGrad((v) => v.mul(v).sum())(tensor.Tensor.from([3, 4], { dtype: "f64" }));
    if (value.item() !== 25 || dv.toArray().join() !== "6,8") throw new Error("autograd across the shared Tensor");
    if (typeof image.resize !== "function" || typeof adapterMath.compileExpr !== "function" || typeof frameArrow.Frame !== "function") throw new Error("missing");
    console.log("OK");
  `);
  assert.match(out, /OK/);
});

test("autograd fits y = 3x + 2 from its own standalone bundle, with the Tensor that bundle exports", () => {
  const out = runScript(`
    const { Tensor, random, nn, optim, trainer } = await load(${JSON.stringify(file("tensor-autograd"))});
    const xs = [0, 1, 2, 3, 4];
    const X = Tensor.from(xs, { dtype: "f32" }).reshape([5, 1]);
    const Y = Tensor.from(xs.map((v) => 3 * v + 2), { dtype: "f32" }).reshape([5, 1]);
    const model = new nn.Linear(1, 1, { rng: random.seed(3) });
    const t = trainer.configure({ model, optimizer: new optim.SGD(model.parameters(), { lr: 0.01 }), lossFn: nn.mseLoss, epochs: 2000 });
    const r = await t.fit({ x: X, y: Y });
    const last = r.lossHistory[r.lossHistory.length - 1];
    if (!(last < 0.01)) throw new Error("loss " + last);
    console.log("OK");
  `);
  assert.match(out, /OK/);
});

test("sizes.json lists every bundle with its true raw, gzip and brotli byte counts", () => {
  const sizes = JSON.parse(readFileSync(join(bundleDir, "sizes.json"), "utf8")) as {
    entries: Record<string, { package: string; file: string; bytes: number; gzip: number; brotli: number }>;
  };
  assert.deepEqual(Object.keys(sizes.entries).sort(), Object.keys(ENTRIES).sort());
  for (const [name, row] of Object.entries(sizes.entries)) {
    const buf = readFileSync(file(name));
    assert.equal(row.file, `${name}.js`);
    assert.equal(row.bytes, statSync(file(name)).size, `${name} raw`);
    // Bun's zlib and brotli are other builds than Node's, so under Bun the counts are checked to 3 %, not to the byte.
    const same = (a: number, b: number) => ((globalThis as { Bun?: unknown }).Bun ? Math.abs(a - b) <= 0.03 * b : a === b);
    assert.ok(same(row.gzip, gzipSync(buf, { level: 9 }).length), `${name} gzip`);
    assert.ok(same(row.brotli, brotliCompressSync(buf).length), `${name} brotli`);
    assert.ok(row.package.length > 0);
  }
});

test("README's size table is the one the build writes (npm run sizes)", () => {
  const readme = readFileSync(join(root, "README.md"), "utf8");
  const m = readme.match(/<!-- sizes:start -->([\s\S]*?)<!-- sizes:end -->/);
  assert.ok(m, "README has no sizes markers");
  const sizes = JSON.parse(readFileSync(join(bundleDir, "sizes.json"), "utf8")) as { entries: Record<string, { bytes: number }> };
  for (const name of Object.keys(ENTRIES)) assert.ok((m as RegExpMatchArray)[1].includes(`\`${name}.js\``), `README table lacks ${name}`);
  void sizes;
});

test("the module entries (for a bundler) load from dist and share one Tensor", async () => {
  const sig = await import(pathToFileURL(join(root, "dist", "signal.js")).href);
  const ten = await import(pathToFileURL(join(root, "dist", "tensor.js")).href);
  assert.equal(sig.Tensor, ten.Tensor);
  const all = await import(pathToFileURL(join(root, "dist", "index.js")).href);
  assert.ok(all.fft && all.signal && all.unit);
});

test("licences: every third-party package inside a bundle is listed in sizes.json and has its licence text in LICENSES.txt", () => {
  const sizes = JSON.parse(readFileSync(join(bundleDir, "sizes.json"), "utf8")) as {
    entries: Record<string, { thirdParty: string[] }>;
  };
  const text = readFileSync(join(bundleDir, "LICENSES.txt"), "utf8");
  assert.match(text, /MIT License/, "the family's own licence");
  assert.deepEqual(sizes.entries.unit?.thirdParty, [], "unit has no third-party code");
  assert.equal(sizes.entries.fft?.thirdParty.length, 1, "fft carries only @johnhenry/math (ComplexNumber, via scalar-types)");
  assert.ok(sizes.entries["frame-arrow"]?.thirdParty.some((p) => p.startsWith("apache-arrow@")), "frame-arrow carries apache-arrow");
  for (const row of Object.values(sizes.entries)) {
    for (const pkg of row.thirdParty) assert.ok(text.includes(`== ${pkg} `), `LICENSES.txt lacks ${pkg}`);
  }
  assert.ok(sizes.entries["adapter-math"]?.thirdParty.some((p) => p.startsWith("@johnhenry/math@")), "adapter-math carries @johnhenry/math");
  assert.match(text, /Apache License/, "apache-arrow's licence text");
});
