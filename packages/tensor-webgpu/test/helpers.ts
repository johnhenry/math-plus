/**
 * Headless-Chrome-with-real-WebGPU test harness, in the same spirit as
 * `packages/frame-parquet/test/helpers.ts`'s pyarrow-oracle resolution
 * (docs/TESTING.md's "oracle unavailable -> skip, never fail" convention),
 * but the "oracle" here is a live `GPUAdapter` reached over the Chrome
 * DevTools Protocol instead of a Python subprocess.
 *
 * Mirrors `~/.local/bin/gl-report`'s pattern on this machine (headless
 * Chrome under Xvfb, driven over a raw CDP WebSocket — no Playwright/
 * Puppeteer, matching this repo's other browser automation): launch Chrome
 * with software/hardware-agnostic ANGLE + WebGPU flags, discover or create a
 * page via `/json`, drive it with `Runtime.evaluate`.
 *
 * Two things gl-report didn't need that WebGPU does:
 *
 * 1. `navigator.gpu` is NOT exposed on `about:blank` or `data:` URLs (found
 *    empirically while building this harness — Chrome 149 reports
 *    `"gpu" in navigator === false` there even with every WebGPU-enabling
 *    flag on, but true on `http://` — likely a secure-context/feature-policy
 *    quirk that only manifests for `data:`/`about:` documents specifically).
 *    So this harness runs a tiny local HTTP server and navigates the page
 *    there instead of `about:blank`.
 * 2. Chrome flags must work BOTH on real hardware (this machine has a real
 *    Intel iGPU reachable via ANGLE's GL backend under Xvfb, confirmed via
 *    `chrome://gpu` — see docs/spikes/webgpu-baseline.md) and with no GPU at
 *    all (CI). `--enable-unsafe-swiftshader` unblocks Dawn's own "CPU
 *    adapters not fully tested" blocklist entry (crbug.com/40057808)
 *    independently of `--use-angle=gl`'s hardware backend choice, so the
 *    same flag set works in both places without branching.
 *
 * Each test FILE launches its own private Chrome + Xvfb (`node --test`'s
 * default file-level process isolation), which means multiple test files
 * running concurrently contend for the same physical GPU render node —
 * observed directly: running the full suite with `node --test`'s default
 * concurrency intermittently starved one file's `requestAdapter()` to
 * `null` (this machine's own `~/CLAUDE.md` documents exactly this failure
 * mode for the shared `gl-chrome.service` instance, via its `~/gpu.lock`
 * convention — same underlying contention, different symptom). Rather than
 * take a filesystem lock (which would only coordinate with other tools that
 * also respect it, not with `node --test`'s own worker pool), this
 * package's `test` script passes `--test-concurrency=1` so its own test
 * files never run their Chrome instances at the same time as each other.
 *
 * ## Two backends: Dawn (in-process) and Chrome (CDP)
 *
 * `getHarness()` hands back one `WebGPUHarness` whose `run(body, bundle)`
 * executes the same flat bundle + test body either
 *
 *  - **in this Node process against Dawn** (the `webgpu` npm package, via
 *    `src/dawn.ts`'s `requestDawnGPU({ unsafe: true })`): the body runs as
 *    an `AsyncFunction` whose `navigator` parameter is `{ gpu: <Dawn> }`, so
 *    test bodies written against `navigator.gpu` run unmodified. No browser,
 *    no display server — this is what makes the kernels testable in plain
 *    local runs (e.g. macOS, where the Xvfb path below doesn't exist). The
 *    instance is created with `allow_unsafe_apis` so Dawn's experimental
 *    subgroup-matrix feature (and hence that GEMM kernel) is exercised too.
 *  - **in headless Chrome over CDP** (everything described above).
 *
 * Selection: `$MATH_PLUS_WEBGPU_HARNESS` = `dawn` | `chrome` forces one;
 * unset/`auto` tries Dawn first and falls back to Chrome, so environments
 * where Dawn has no adapter (a GPU-less CI runner) keep the established
 * Chrome+SwiftShader path. Either way "no adapter" -> skip, never fail.
 */
import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import { createServer, type Server } from "node:http";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import type { AddressInfo } from "node:net";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const CHROME_CANDIDATES = [
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "google-chrome-stable",
  "/opt/google/chrome/chrome",
  "google-chrome",
  "chromium",
  "chromium-browser",
  "chrome", // e.g. browser-actions/setup-chrome's PATH entry in CI
];

function which(cmd: string): string | undefined {
  try {
    if (cmd.startsWith("/")) {
      execFileSync("test", ["-x", cmd]);
      return cmd;
    }
    return execFileSync("which", [cmd], { encoding: "utf8" }).trim();
  } catch {
    return undefined;
  }
}

/**
 * `$MATH_PLUS_CHROME_PATH` (checked first, same override-env-var convention as
 * `$MATH_PLUS_ORACLE_PYTHON` elsewhere in this repo — see docs/TESTING.md) lets
 * CI or a local override pin an exact binary; otherwise the usual PATH/
 * well-known-path candidates are probed in order.
 */
function resolveChrome(): string | undefined {
  if (process.env.MATH_PLUS_CHROME_PATH) return process.env.MATH_PLUS_CHROME_PATH;
  for (const c of CHROME_CANDIDATES) {
    const found = which(c);
    if (found) return found;
  }
  return undefined;
}

function hasXvfb(): boolean {
  return which("Xvfb") !== undefined;
}

/** macOS Chrome needs no X display: it runs `--headless=new` against Metal directly. */
const IS_MAC = process.platform === "darwin";

// ---- TS->JS browser bundler ------------------------------------------------
//
// Bundles this package's `src/*.ts` entry files and their whole dependency
// closure — workspace siblings (tensor-core, tensor-compile, tensor-cpu,
// special) and `@johnhenry/backend-webgpu` + `@johnhenry/tensor-backend`,
// which ship their TypeScript sources under a `source` export condition —
// into ONE import/export-free script injectable via CDP `Runtime.evaluate`
// (or run in-process by the Dawn harness). Every module is transpiled to
// CommonJS and wrapped in its own function scope with a tiny `require`, so
// modules can't collide on top-level names (the flat concatenation this
// replaces could, once backend-webgpu joined the closure). The entries'
// exports are then declared as top-level `const`s, so test bodies call
// `runGemm(...)`, `GPUTensor.fromFloat32Array(...)` etc. directly.
//
// Resolution (not a general bundler; enough for this closure): relative
// specifiers as written (`./x.ts`); package `imports` (`#dawn`) and
// `exports` entries by the `browser` condition, then `source`, else a
// `./dist/X.js` target mapped back to `./src/X.ts`; packages are found by
// walking up `node_modules` from the importing file (so backend-webgpu gets
// its own nested tensor-backend). `import type` is erased by the transpiler.

const HERE = path.dirname(fileURLToPath(import.meta.url));

interface PackageJson {
  exports?: Record<string, unknown> | string;
  imports?: Record<string, unknown>;
}

function pickTarget(entry: unknown): string | undefined {
  if (typeof entry === "string") return entry;
  if (!entry || typeof entry !== "object") return undefined;
  const e = entry as Record<string, unknown>;
  for (const cond of ["browser", "source", "import", "default"]) {
    if (cond in e) {
      const t = pickTarget(e[cond]);
      if (t) return t;
    }
  }
  return undefined;
}

function toSource(pkgDir: string, target: string): string {
  const direct = path.resolve(pkgDir, target);
  if (target.endsWith(".ts")) return direct;
  const m = /^\.\/dist\/(.+)\.js$/.exec(target);
  if (m) return path.resolve(pkgDir, "src", `${m[1]}.ts`);
  throw new Error(`bundleForBrowser: cannot map ${target} in ${pkgDir} to a TypeScript source`);
}

function findPackageDir(name: string, fromFile: string): string {
  let dir = path.dirname(fromFile);
  for (;;) {
    const candidate = path.join(dir, "node_modules", name);
    if (existsSync(path.join(candidate, "package.json"))) return realpathSync(candidate);
    const parent = path.dirname(dir);
    if (parent === dir) throw new Error(`bundleForBrowser: cannot find package ${name} from ${fromFile}`);
    dir = parent;
  }
}

function owningPackageDir(file: string): string {
  let dir = path.dirname(file);
  while (!existsSync(path.join(dir, "package.json"))) dir = path.dirname(dir);
  return dir;
}

function resolveSpecifier(spec: string, fromFile: string): string {
  if (spec.startsWith("./") || spec.startsWith("../")) return path.resolve(path.dirname(fromFile), spec);
  if (spec.startsWith("#")) {
    const dir = owningPackageDir(fromFile);
    const pkg = JSON.parse(readFileSync(path.join(dir, "package.json"), "utf8")) as PackageJson;
    const target = pickTarget(pkg.imports?.[spec]);
    if (!target) throw new Error(`bundleForBrowser: no "imports" entry for ${spec} in ${dir}`);
    return toSource(dir, target);
  }
  const parts = spec.split("/");
  const name = spec.startsWith("@") ? parts.slice(0, 2).join("/") : parts[0]!;
  const sub = `.${spec.slice(name.length)}`;
  const dir = findPackageDir(name, fromFile);
  const pkg = JSON.parse(readFileSync(path.join(dir, "package.json"), "utf8")) as PackageJson;
  const exp = pkg.exports;
  const entry = typeof exp === "string" ? (sub === "." ? exp : undefined) : exp?.[sub];
  const target = pickTarget(entry);
  if (!target) throw new Error(`bundleForBrowser: ${name} does not export "${sub}"`);
  return toSource(dir, target);
}

const REQUIRE_RE = /\brequire\(\s*"([^"]+)"\s*\)/g;

function transpileCjs(absPath: string): string {
  return ts.transpileModule(readFileSync(absPath, "utf8"), {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
      isolatedModules: true,
      esModuleInterop: false,
    },
    fileName: absPath,
  }).outputText;
}

/**
 * Bundle `entryFiles` (absolute paths to this package's own `src/*.ts`) plus
 * their transitive dependency closure into one script (see the section doc)
 * that declares the exports of every bundled module of THIS package (the
 * entries and the `src/` modules they import — as the flat bundle it
 * replaces did) as top-level `const`s. Memoized per entry list.
 */
const bundles = new Map<string, string>();
export function bundleForBrowser(entryFiles: readonly string[]): string {
  const memoKey = entryFiles.join("\n");
  const hit = bundles.get(memoKey);
  if (hit) return hit;
  const ids = new Map<string, number>();
  const modules: string[] = [];
  function visit(absPath: string): number {
    const known = ids.get(absPath);
    if (known !== undefined) return known;
    const id = ids.size;
    ids.set(absPath, id);
    modules.push("");
    const code = transpileCjs(absPath).replace(REQUIRE_RE, (_m, spec: string) => `__req(${visit(resolveSpecifier(spec, absPath))})`);
    modules[id] = `// ${path.relative(path.resolve(HERE, "../../.."), absPath)}\nfunction (exports, __req) {\n${code}\n}`;
    return id;
  }
  entryFiles.forEach(visit);
  const own = [...ids].filter(([file]) => file.startsWith(SRC + path.sep)).map(([, id]) => id);
  const runtime = `const __mpBundle = (() => {
const __defs = [\n${modules.join(",\n")}\n];
const __cache = new Map();
function __req(id) {
  let m = __cache.get(id);
  if (!m) { m = {}; __cache.set(id, m); __defs[id](m, __req); }
  return m;
}
return Object.assign({}, ${own.map((e) => `__req(${e})`).join(", ")});
})();`;
  const names = Object.keys(new Function(`${runtime}\nreturn __mpBundle;`)() as Record<string, unknown>).filter((k) => k !== "__esModule");
  const bundle = `${runtime}\nconst { ${names.join(", ")} } = __mpBundle;`;
  bundles.set(memoKey, bundle);
  return bundle;
}

export const SRC = path.resolve(HERE, "../src");

// ---- Xvfb + Chrome + CDP harness -------------------------------------------

interface ChromeHandle {
  kill(): void;
  cdpBase: string;
}

async function waitFor<T>(fn: () => Promise<T | undefined>, tries: number, delayMs: number): Promise<T | undefined> {
  for (let i = 0; i < tries; i++) {
    const v = await fn().catch(() => undefined);
    if (v !== undefined) return v;
    await new Promise((r) => setTimeout(r, delayMs));
  }
  return undefined;
}

let xvfbProc: ChildProcess | undefined;
let xvfbDisplay: string | undefined;

async function ensureDisplay(): Promise<string> {
  if (IS_MAC) return "";
  if (process.env.DISPLAY) {
    return process.env.DISPLAY;
  }
  if (xvfbDisplay) return xvfbDisplay;
  // A pseudo-random display number (not a fixed one like gl-report's :99)
  // avoids colliding with the shared machine-wide gl-chrome.service instance
  // (per ~/CLAUDE.md's GPU/Chrome lock convention) or a leftover Xvfb from a
  // previous run of this same test suite.
  const display = `:${150 + Math.floor(Math.random() * 800)}`;
  xvfbProc = spawn("Xvfb", [display, "-screen", "0", "1280x1024x24"], { stdio: "ignore" });
  // Without unref(), Node's event loop waits on this child forever (it's a
  // long-lived server, never exits on its own) and `node --test` hangs after
  // the last test completes instead of exiting — `close()` still explicitly
  // SIGKILLs it during cleanup, unref() only stops it from blocking exit.
  xvfbProc.unref();
  xvfbDisplay = display;
  // Give Xvfb a moment to bind before Chrome tries to connect.
  await new Promise((r) => setTimeout(r, 800));
  return display;
}

function launchChrome(chromePath: string, display: string, port: number, userDataDir: string): ChromeHandle {
  const platformArgs = IS_MAC
    ? ["--headless=new", "--use-angle=metal"]
    : ["--use-angle=gl", "--use-gl=angle"];
  const args = [
    ...platformArgs,
    "--ignore-gpu-blocklist",
    "--enable-unsafe-webgpu",
    "--enable-unsafe-swiftshader",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${userDataDir}`,
    "about:blank",
  ];
  const child = spawn(chromePath, args, {
    env: IS_MAC ? process.env : { ...process.env, DISPLAY: display, LIBGL_ALWAYS_SOFTWARE: "1" },
    stdio: "ignore",
    detached: true,
  });
  // Same reasoning as xvfbProc.unref() above: `detached: true` (so `kill()`
  // can SIGKILL the whole process group, since Chrome forks helpers) does
  // NOT by itself stop Node from waiting on this child at exit.
  child.unref();
  return {
    kill: () => {
      try {
        process.kill(-child.pid!, "SIGKILL");
      } catch {
        try {
          child.kill("SIGKILL");
        } catch {
          /* already dead */
        }
      }
    },
    cdpBase: `http://127.0.0.1:${port}`,
  };
}

interface CdpConnection {
  send(method: string, params?: Record<string, unknown>): Promise<{ result?: Record<string, unknown> }>;
  close(): void;
}

async function connectCdp(cdpBase: string, pageUrl: string): Promise<CdpConnection> {
  // NOT `encodeURIComponent(pageUrl)`: Chrome's `/json/new` endpoint treats
  // everything after "?" as the literal URL to open, not a percent-encoded
  // query value — encoding it here (an earlier version of this harness did)
  // made Chrome try to navigate to the literal string
  // "http%3A%2F%2F127.0.0.1%3A.../" and silently fall back to an error page
  // with no real HTTP origin, which is why `navigator.gpu` was missing on
  // every probe despite Chrome, Xvfb, and the GPU itself all being healthy.
  const res = await fetch(`${cdpBase}/json/new?${pageUrl}`, { method: "PUT" });
  const page = (await res.json()) as { webSocketDebuggerUrl: string };
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map<number, (msg: { result?: Record<string, unknown> }) => void>();
  ws.addEventListener("message", (ev: MessageEvent) => {
    const msg = JSON.parse(ev.data as string) as { id?: number; result?: Record<string, unknown> };
    if (msg.id !== undefined && pending.has(msg.id)) {
      pending.get(msg.id)!(msg);
      pending.delete(msg.id);
    }
  });
  await new Promise<void>((resolve, reject) => {
    ws.addEventListener("open", () => resolve());
    ws.addEventListener("error", () => reject(new Error("CDP WebSocket connection failed")));
  });
  return {
    send: (method, params = {}) =>
      new Promise((resolve) => {
        const thisId = ++id;
        pending.set(thisId, resolve);
        ws.send(JSON.stringify({ id: thisId, method, params }));
      }),
    close: () => ws.close(),
  };
}

export interface WebGPUHarness {
  /** Which backend this harness drives (see the module doc). */
  kind: "dawn" | "chrome";
  /** Run an async JS expression (source text of an `async () => {...}` body's *contents*, i.e. what goes between the braces) in the page and return its JSON-serializable result. Throws with the page-side error message on failure. */
  run<T = unknown>(asyncBody: string, extraCode?: string): Promise<T>;
  close(): void;
}

let staticServer: Server | undefined;
let staticServerUrl: string | undefined;

async function ensureStaticServer(): Promise<string> {
  if (staticServerUrl) return staticServerUrl;
  staticServer = createServer((_req, res) => {
    res.writeHead(200, { "content-type": "text/html" });
    res.end("<!doctype html><title>@johnhenry/math-plus-tensor-webgpu test harness</title>");
  });
  await new Promise<void>((resolve) => staticServer!.listen(0, "127.0.0.1", resolve));
  // Same reasoning as the Xvfb/Chrome child processes: an open listening
  // server keeps Node's event loop alive on its own; `close()` still shuts
  // it down explicitly during cleanup, `unref()` just stops it from blocking
  // exit if cleanup is skipped for any reason.
  staticServer.unref();
  const addr = staticServer.address() as AddressInfo;
  staticServerUrl = `http://127.0.0.1:${addr.port}/`;
  return staticServerUrl;
}

let cachedHarness: Promise<WebGPUHarness | { unavailable: true; reason: string }> | undefined;

/**
 * Lazily launch (once per test process — cached) headless Chrome + Xvfb and
 * return either a working harness or `{ unavailable: true, reason }`.
 * Individual tests call this and `t.skip(reason)` when unavailable, per
 * docs/TESTING.md's oracle convention — this is NOT a hard failure, since a
 * genuinely headless-WebGPU-less environment is an expected (if undesirable)
 * state this package must degrade gracefully in.
 */
export function getHarness(): Promise<WebGPUHarness | { unavailable: true; reason: string }> {
  if (!cachedHarness) cachedHarness = selectHarness();
  return cachedHarness;
}

type HarnessResult = WebGPUHarness | { unavailable: true; reason: string };

async function selectHarness(): Promise<HarnessResult> {
  const want = (process.env.MATH_PLUS_WEBGPU_HARNESS ?? "auto").toLowerCase();
  if (want === "chrome") return buildHarness();
  const dawn = await buildDawnHarness();
  if (want === "dawn" || !("unavailable" in dawn)) return dawn;
  const chrome = await buildHarness();
  if (!("unavailable" in chrome)) return chrome;
  return { unavailable: true, reason: `${dawn.reason}; and Chrome: ${chrome.reason}` };
}

// ---- Dawn (in-process) harness ----------------------------------------------

const AsyncFunction = (async () => {}).constructor as new (...args: string[]) => (...a: unknown[]) => Promise<unknown>;

/**
 * Dawn in this process. Devices a test body requests are tracked (by
 * wrapping the adapter's `requestDevice`) and destroyed when that `run()`
 * finishes — in Chrome a page owns its devices, here nothing else would,
 * and leaked Dawn devices keep native resources (and sometimes the event
 * loop) alive.
 */
async function buildDawnHarness(): Promise<HarnessResult> {
  const { requestDawnGPU } = await import("../src/dawn.ts");
  const gpu = await requestDawnGPU({ unsafe: true });
  if (!gpu) return { unavailable: true, reason: "Dawn: the `webgpu` npm package is not installed or its native addon failed to load" };
  const probe = await gpu.requestAdapter().catch(() => null);
  if (!probe) return { unavailable: true, reason: "Dawn: requestAdapter() resolved null (no GPU adapter)" };

  const run = async <T>(asyncBody: string, extraCode = ""): Promise<T> => {
    const devices: GPUDevice[] = [];
    const trackingGpu = {
      requestAdapter: async (options?: GPURequestAdapterOptions): Promise<GPUAdapter | null> => {
        const adapter = await gpu.requestAdapter(options);
        if (!adapter) return adapter;
        const requestDevice = adapter.requestDevice.bind(adapter);
        (adapter as { requestDevice: GPUAdapter["requestDevice"] }).requestDevice = async (desc?: GPUDeviceDescriptor) => {
          const device = await requestDevice(desc);
          devices.push(device);
          return device;
        };
        return adapter;
      },
      getPreferredCanvasFormat: () => gpu.getPreferredCanvasFormat(),
      wgslLanguageFeatures: gpu.wgslLanguageFeatures,
    };
    try {
      const fn = new AsyncFunction("navigator", `${extraCode}\n${asyncBody}`);
      return (await fn({ gpu: trackingGpu })) as T;
    } catch (err) {
      throw new Error(`dawn-side error: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`);
    } finally {
      for (const d of devices) d.destroy();
    }
  };
  return { kind: "dawn", run, close: () => {} };
}

/**
 * Every test file that calls {@link getHarness} MUST register
 * `test.after(closeHarness)` — an open CDP `WebSocket` keeps Node's event
 * loop alive on its own (unlike the Xvfb/Chrome child processes and the
 * static HTTP server, which are `unref()`'d), so without an explicit close
 * `node --test` hangs after the last test passes instead of exiting.
 */
export async function closeHarness(): Promise<void> {
  if (!cachedHarness) return;
  const harness = await cachedHarness;
  if (!("unavailable" in harness)) harness.close();
}

/**
 * Bounded fresh-Chrome retry (issue #49). The single-attempt version of this
 * harness conflated two very different skip causes under one message: an
 * environment that genuinely lacks WebGPU prerequisites, and a healthy
 * environment where Chrome's first cold start raced GPU/X contention and
 * failed a probe that would succeed seconds later (observed repeatedly on
 * this machine as transient `NO_NAVIGATOR_GPU` — always passing on isolated
 * rerun). Retrying with a completely fresh Chrome (new port, new profile)
 * absorbs the transient case, and the final skip message now names which
 * case occurred. Missing prerequisites (no Chrome, no Xvfb) are checked
 * once, before the loop — retrying can't conjure a binary into existence.
 */
const HARNESS_ATTEMPTS = 3;

async function buildHarness(): Promise<WebGPUHarness | { unavailable: true; reason: string }> {
  const chromePath = resolveChrome();
  if (!chromePath) {
    return { unavailable: true, reason: "no Chrome/Chromium binary found on PATH or at /opt/google/chrome/chrome" };
  }
  if (!IS_MAC && !process.env.DISPLAY && !hasXvfb()) {
    return { unavailable: true, reason: "no DISPLAY and no Xvfb on PATH to create one" };
  }

  const attemptFailures: string[] = [];
  for (let attempt = 1; attempt <= HARNESS_ATTEMPTS; attempt++) {
    if (attempt > 1) await new Promise((r) => setTimeout(r, 1000)); // let contention clear
    // A thrown error (e.g. the CDP WebSocket failing mid-connect) is just
    // another way an attempt can fail transiently -- convert rather than
    // letting it escape the retry loop.
    const result = await attemptHarness(chromePath).catch((err: unknown) => ({
      failed: `attempt threw: ${err instanceof Error ? err.message : String(err)}`,
    }));
    if (!("failed" in result)) {
      if (attempt > 1) {
        // Flaky-then-ok deserves a visible trace (issue #49): the machine
        // was contended, the environment is fine.
        console.warn(
          `[webgpu-harness] healthy adapter on attempt ${attempt}/${HARNESS_ATTEMPTS} — earlier fresh-Chrome attempts failed transiently: ${attemptFailures.join("; ")}`,
        );
      }
      return result;
    }
    attemptFailures.push(`attempt ${attempt}: ${result.failed}`);
  }
  return {
    unavailable: true,
    reason:
      `harness failed to start after ${HARNESS_ATTEMPTS} fresh Chrome launches (${attemptFailures.join("; ")}) — ` +
      `this environment HAS Chrome and a display path, so this is a startup/contention failure, ` +
      `not a genuinely WebGPU-less environment; see AGENTS.md's GPU section`,
  };
}

async function attemptHarness(chromePath: string): Promise<WebGPUHarness | { failed: string }> {
  const display = await ensureDisplay();
  // killXvfb (registered on "exit" immediately, BEFORE we know whether the
  // probe below even succeeds) is the fix for a real leak this harness had
  // during development: every early-return path used to kill Chrome but
  // not Xvfb, so a run of N failed/unavailable probes left N orphaned Xvfb
  // processes behind — which, past a few dozen, degraded the whole machine's
  // X/GPU stack enough to make EVERY subsequent probe fail too (observed
  // directly: a healthy adapter one run, then consistent `NO_NAVIGATOR_GPU`
  // a few runs later, cause traced to accumulated Xvfb processes via `ps`).
  // NOTE (issue #49): per-attempt failure paths deliberately do NOT call
  // this — the shared Xvfb is reused by the next fresh-Chrome attempt, and
  // the exit hook alone is what prevents the orphan leak described above.
  const killXvfb = (): void => {
    if (xvfbProc) {
      try {
        xvfbProc.kill("SIGKILL");
      } catch {
        /* already dead */
      }
      xvfbProc = undefined;
    }
  };
  process.once("exit", killXvfb);

  const port = 19222 + Math.floor(Math.random() * 1000);
  const userDataDir = path.join(
    process.env.TMPDIR ?? "/tmp",
    `math-plus-webgpu-test-profile-${process.pid}-${Date.now()}`,
  );
  const chrome = launchChrome(chromePath, display, port, userDataDir);

  const ready = await waitFor(
    async () => {
      const r = await fetch(`${chrome.cdpBase}/json/version`);
      return r.ok ? true : undefined;
    },
    40,
    250,
  );
  if (!ready) {
    chrome.kill();
    return { failed: "headless Chrome did not expose a CDP endpoint in time" };
  }

  const pageUrl = await ensureStaticServer();
  const cdp = await connectCdp(chrome.cdpBase, pageUrl);
  await cdp.send("Runtime.enable");

  // Confirm we actually get a real GPUAdapter before declaring "available" —
  // a Chrome that launched fine can still resolve requestAdapter() to null.
  const probe = await cdp.send("Runtime.evaluate", {
    expression: `(async () => {
      if (!("gpu" in navigator)) return "NO_NAVIGATOR_GPU";
      try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter ? "OK" : "ADAPTER_NULL";
      } catch (e) {
        return "EXCEPTION:" + (e && e.message ? e.message : String(e));
      }
    })()`,
    returnByValue: true,
    awaitPromise: true,
  });
  const probeValue = (probe.result?.result as { value?: string } | undefined)?.value;
  if (probeValue !== "OK") {
    cdp.close();
    chrome.kill();
    return { failed: `WebGPU probe returned ${probeValue ?? "no result"}` };
  }

  const run = async <T>(asyncBody: string, extraCode = ""): Promise<T> => {
    const expression = `(async () => {\n${extraCode}\n${asyncBody}\n})()`;
    const res = await cdp.send("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    const result = res.result as
      | { result?: { value?: unknown }; exceptionDetails?: { text?: string; exception?: { description?: string } } }
      | undefined;
    if (result?.exceptionDetails) {
      const desc = result.exceptionDetails.exception?.description ?? result.exceptionDetails.text;
      throw new Error(`page-side error: ${desc ?? JSON.stringify(result.exceptionDetails)}`);
    }
    return result?.result?.value as T;
  };

  const close = (): void => {
    cdp.close();
    chrome.kill();
    staticServer?.close();
    killXvfb();
  };
  // `killXvfb` alone is already registered on "exit" above; this covers the
  // rest (cdp/chrome/server) for the same "don't rely solely on callers
  // remembering to call close()" reason.
  process.once("exit", close);

  return { kind: "chrome", run, close };
}
