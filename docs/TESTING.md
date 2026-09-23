# Testing

## Unit tests

```bash
npm test            # all workspaces
npm test -w @johnhenry/math-plus-tensor-core
```

Tests are TypeScript run directly by `node --test` (native type stripping). The floor is
`engines.node >=24.0.0` everywhere (root and every package; `test/manifest-drift.test.ts` enforces
that they agree) and CI tests exactly that floor. Nothing needs Node 26: there is no Node-26-only API
in the repo, and `Float16Array`, `Math.f16round` and `node --test` on `.ts` all work on 24.9, where
the full suite passes.

## Running under Bun

```bash
npm run build          # same prerequisite as npm test
npm run test:bun       # every suite + a summary table (scripts/test-bun.mjs)
npm run test:bun -- tensor-core   # only suites whose directory contains "tensor-core"
npm run test:bun -w @johnhenry/math-plus-fft   # one package, raw bun output
```

**Bun 1.2's `node:test` shim registers tests from the first file of a multi-file run only.** The
other files' tests are dropped silently and the run still passes (before this harness, `bun test` in
`packages/fft` ran 8 of 27 tests). A shared module that imports `bun:test` for the files has the same
problem. So every test file imports `bun:test` itself and passes it to the shared
[`test/harness.ts`](../test/harness.ts):

```ts
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
```

Destructure `before`/`after`/`afterEach` from the same call when needed. Under Node, `makeTest(null)`
returns `node:test`'s own functions unchanged. Under Bun it returns a shim for the subset of the API
the suites use: options `skip`/`todo`/`timeout`, and `t.name`/`t.skip`/`t.after`/`t.test`.
`TestContext` is typed as that subset, so `tsc` rejects anything else. For call-counting spies use
the harness's `spyMethod`, not `node:test`'s `mock.method`, which the Bun shim does not count. A
manifest-drift test fails any `*.test.ts` that imports from `node:test` directly or lacks its own
`bun:test` import.

Differences under Bun:
- A runtime `t.skip(reason)` cannot mark a running `bun:test` test as skipped. The test is reported as
  **passed** and a `[skip] <name>: <reason>` line is printed; `scripts/test-bun.mjs` counts these per
  suite. Apply the "0 skipped" rule from the oracle sections below to that count as well.
- Subtests (`t.test`) run in order inside the parent and are not reported on their own, so
  Bun's test totals are lower than Node's for suites that use them (e.g. tensor-core's differential
  suite).
- There is no per-test default timeout, as under `node --test`. Bun's own 5 s default would kill the
  oracle tests.
- `scripts/test-bun.mjs` also fails a suite when Bun ran fewer files than `test/*.test.ts` holds.
  That is the signature of the multi-file bug.
- `*.bench-test.ts` files are not collected by `bun test`, matching `npm test`.

Bun is pinned to 1.2.17 in CI. Recheck whether the harness is still needed when upgrading.

## Benchmarks

GPU and WASM performance numbers follow [`docs/BENCHMARKING.md`](BENCHMARKING.md): a cooldown before
each cell, timing windows of 1 s or less, and backends alternated inside one process.

## Differential tests (NumPy oracle)

`packages/tensor-core/test/differential.test.ts` compares tensor-core ops against a NumPy
subprocess (`packages/tensor-core/scripts/numpy_oracle.py`), exchanging data as `.npy` in both
directions — so every run also validates `.npy` I/O against NumPy's implementation.

Python resolution: `$MATH_PLUS_ORACLE_PYTHON`, else `python3` on PATH. **The suite skips (does not
fail) when no interpreter with numpy is found** — environments that guarantee the oracle should
assert `skipped 0`.

### On NixOS (trycooy)

pip wheels don't work (libstdc++/libz linkage). Use a nix-provided Python:

```bash
ORACLE_PY=$(nix-shell -p "python3.withPackages(ps: with ps; [numpy pyarrow pandas])" --run "which python3")
MATH_PLUS_ORACLE_PYTHON=$ORACLE_PY npm test -w @johnhenry/math-plus-tensor-core
```

The resolved store path stays valid until garbage-collected; re-run the `nix-shell` line to refresh.
The same `$MATH_PLUS_ORACLE_PYTHON` (or a bare `python3` on PATH) is what `@johnhenry/math-plus-frame-arrow`'s and
`@johnhenry/math-plus-frame-parquet`'s pyarrow-round-trip tests look for too (see below) — one env var covers
every Python oracle in the repo.

### Tolerances

Per-op tolerance registry in `differential.test.ts` (`TOLERANCES`): f64 default `rtol 1e-12`,
f32 default `rtol 1e-5`, looser for accumulation-order-sensitive ops (`sum`/`mean` on f32).
Integer dtypes (incl. `i64`) compare exactly.

## pyarrow/pandas round-trip tests (frame-arrow, frame-parquet)

Some `@johnhenry/math-plus-frame-arrow`/`@johnhenry/math-plus-frame-parquet` tests verify a JS-written Arrow IPC/Parquet file
by reading it back with `pyarrow`/`pandas` directly (not just round-tripping through the package's
own reader — self-consistency isn't proof of a valid file). Same resolution and skip-don't-fail
behavior as the NumPy oracle above (`$MATH_PLUS_ORACLE_PYTHON`, else `python3` on PATH,
`packages/frame-parquet/test/helpers.ts`) — the NixOS nix-shell line above already includes
`pyarrow`/`pandas` for exactly this reason.

Most fixtures in these two packages (and in `packages/interop-python/tests/fixtures/`, see below)
are **pre-generated and committed**, not regenerated live on every test run — only the tests that
verify THIS run's own freshly-written output (e.g. `writeParquet`'s pyarrow-round-trip tests) spawn
a live Python subprocess.

## scipy.signal oracle (@johnhenry/math-plus-signal)

`packages/signal/test/helpers.ts` (`runScipyOracle`) compares `@johnhenry/math-plus-signal`'s `butter`/
`sosFilter`/`findPeaks`/`stft` against `scipy.signal` via a subprocess
(`packages/signal/scripts/scipy_oracle.py`). Same skip-don't-fail convention as the NumPy oracle,
resolved via `$MATH_PLUS_SCIPY_ORACLE_PYTHON`, else `$MATH_PLUS_ORACLE_PYTHON`, else `python3` on PATH —
but scipy is NOT part of the NixOS nix-shell line above (numpy/pyarrow/pandas only), since it's a
much heavier dependency (needs a Fortran/BLAS toolchain) only this one package's tests need:

```bash
SCIPY_PY=$(nix-shell -p "python3.withPackages(ps: with ps; [scipy numpy])" --run "which python3")
MATH_PLUS_SCIPY_ORACLE_PYTHON=$SCIPY_PY npm test -w @johnhenry/math-plus-signal
```

`butter`'s own SOS section coefficients are NOT expected to match `scipy.signal.butter`'s
byte-for-byte — scipy's own pole/zero-to-section grouping isn't fixed either (verified empirically:
which zeros pair with which poles varies by filter order). Tests instead verify END-TO-END FILTERING
BEHAVIOR (apply `sosFilter` to `butter`'s output, compare against `scipy.signal.sosfilt` applied to
`scipy.signal.butter`'s own output, on the same input) — invariant to section grouping, and the
property that actually matters.

## SciPy / PyTorch oracle for erf and GELU (tensor-core, issue #122)

`packages/tensor-core/test/special-oracle.test.ts` checks the canonical `erf`/`erfc`
(`packages/tensor-core/src/special.ts`) against `scipy.special.erf`/`erfc` over [-6, 6] plus both
tails, exact GELU against `x·scipy.special.ndtr(x)`, and `Tensor.gelu()` against
`torch.nn.functional.gelu` in both `approximate` modes (f64 and f32), via
`packages/tensor-core/scripts/special_oracle.py`. Same skip-don't-fail convention. SciPy resolves
via `$MATH_PLUS_SCIPY_ORACLE_PYTHON`, else `$MATH_PLUS_ORACLE_PYTHON`, else `python3`. PyTorch
resolves via `$MATH_PLUS_TORCH_ORACLE_PYTHON`, else `$MATH_PLUS_ORACLE_PYTHON`, else `python3`. The
two can be different interpreters. Without a system SciPy, a throwaway venv works:

```bash
uv venv /tmp/scipy-venv && uv pip install --python /tmp/scipy-venv/bin/python scipy numpy
MATH_PLUS_SCIPY_ORACLE_PYTHON=/tmp/scipy-venv/bin/python \
MATH_PLUS_TORCH_ORACLE_PYTHON=$(python3 -c 'import sys; print(sys.executable)') \
  npm test -w @johnhenry/math-plus-tensor-core
```

(On macOS 27, the SciPy 1.15 wheel for Python 3.10 fails to `dlopen`; use Python 3.12 with a current
SciPy.) In the far-left GELU tail SciPy's `ndtr` is the less accurate side (it rounds `x/√2` before
`erfc`, costing ~x²·2^-53 relative), so that test's tolerance grows with x² accordingly. The numpy
oracle's exact-GELU mode uses the C library's `math.erf` and needs only numpy.

## Gradient oracles (autograd)

`adapter-math`'s `@johnhenry/math-plus-adapter-math/test-utils` subpath (`dualGrad`/`dualGradN`) wraps
@johnhenry/math's `DualNumber` forward-mode autodiff — a third gradient oracle, algorithmically
independent of both `tensor-autograd`'s reverse-mode tape and finite differences. Its own tests
(`adapters/adapter-math/test/test-utils.test.ts`) validate it against finite differences AND
against `tensor-autograd`'s `Variable`/`grad.of` on real scalar/multivariate functions.

## PyTorch oracle (tensor-autograd)

`packages/tensor-autograd/test/transformer.test.ts` and `test/safetensors.test.ts` check every
`Variable` view op and transformer layer (`scaledDotProductAttention`, `MultiheadAttention`,
`RotaryEmbedding`, `LayerNorm(bias=False)`, `TransformerEncoderLayer`, `GeGLU`) against PyTorch —
forward output AND the gradient of every input and parameter, for a seeded random upstream
gradient — via one batched subprocess per test file (`packages/tensor-autograd/scripts/torch_oracle.py`).
Layers ship their JS `stateDict()`, which the oracle loads with `load_state_dict(strict=True)`, so
parameter *names* are checked against PyTorch's too. The safetensors tests additionally need
Python's `safetensors` and cover both directions (JS-written file into torch; torch-written f16
file into JS f16/f32 modules).

Python resolution: `$MATH_PLUS_TORCH_ORACLE_PYTHON`, else `$MATH_PLUS_ORACLE_PYTHON`, else `python3`
on PATH; same skip-don't-fail contract (no `import torch` → skip). Tolerances: f64 `rtol 1e-9`,
f32 `rtol 2e-4` (`TOL` in `test/torch-oracle.ts`). On NixOS:

```bash
TORCH_PY=$(nix-shell -p "python3.withPackages(ps: with ps; [torch safetensors numpy])" --run "which python3")
MATH_PLUS_TORCH_ORACLE_PYTHON=$TORCH_PY npm test -w @johnhenry/math-plus-tensor-autograd
```

## Python-side interop tests (`packages/interop-python`)

`johnhenry-math-plus-interop` is a PyPI package outside the npm/Cargo workspaces (see docs/RELEASING.md) — its
`pytest` suite runs separately from `npm test`. Bidirectional conformance (JS writes/Python reads,
and the inverse) is proven with committed fixtures on both sides — see
`packages/interop-python/README.md`'s "Bidirectional conformance fixtures" section for exactly
which test proves which direction, including the two JS-side tests
(`packages/frame-arrow/test/interop-python.test.ts`,
`packages/frame-parquet/test/interop-python.test.ts`) that verify Python-written fixtures read back
correctly — real cross-language verification, not two suites independently trusting their own
output.

```bash
nix-shell -p "python3.withPackages(ps: [ps.pyarrow ps.pandas ps.numpy ps.pytest])" \
  --run "cd packages/interop-python && PYTHONPATH=src python3 -m pytest tests/ -v"
```

## Headless WebGPU oracle (`@johnhenry/math-plus-tensor-webgpu`)

> A **Dawn/Node test path** is being added by the tensor-webgpu tiled-GEMM PR (issue #127, item 4).
> It runs these tests against WebGPU from Node directly, with no Chrome or Xvfb. That PR owns the
> path and its documentation. The Chrome harness below is the current path until it lands.

Same "oracle unavailable -> skip, never fail" convention as the NumPy/pyarrow oracles above, but
the oracle is a live `GPUAdapter` reached over the Chrome DevTools Protocol instead of a Python
subprocess — `packages/tensor-webgpu/test/helpers.ts` launches headless Chrome under Xvfb (mirroring
`~/.local/bin/gl-report`'s pattern on the trycooy dev machine: raw CDP over WebSocket, no
Playwright/Puppeteer) and probes `navigator.gpu.requestAdapter()` once per test file, caching the
result. Individual tests call `getHarness()` and `t.skip(reason)` when unavailable.

Resolution order for the Chrome binary: `$MATH_PLUS_CHROME_PATH` (explicit override), else the usual
PATH/well-known-path candidates (`google-chrome-stable`, `/opt/google/chrome/chrome`, `chromium`,
etc. — see `CHROME_CANDIDATES` in `helpers.ts`). `Xvfb` must also be on `PATH` (or `$DISPLAY` set to
an already-live display) — see `docs/spikes/webgpu-baseline.md` for the exact launch flags and two
non-obvious gotchas found while building this (WebGPU needs a real `http://` origin, not
`about:blank`/`data:`; Chrome's `/json/new?<url>` DevTools endpoint takes a literal, not
percent-encoded, URL).

Every test file that calls `getHarness()` MUST also call `test.after(closeHarness)` — an open CDP
`WebSocket` keeps Node's event loop alive on its own, so without it `node --test` hangs after the
last test passes instead of exiting.

`@johnhenry/math-plus-tensor-webgpu`'s own `test` script passes `node --test --test-concurrency=1` (not
`npm test`'s usual per-workspace default) because each test file launches its own private Chrome
instance, and concurrent files' Chrome instances contend for the same physical GPU render node —
observed directly during development as an intermittent `requestAdapter()` -> `null` in one file
while others succeeded.

**GPU *performance* is never gated by a per-PR test** (per issue #12: no real GPU in a standard
GitHub Actions runner, only WASM has a hardware-verified speedup assertion in CI). GEMM correctness
IS tested against a live adapter; the WASM-vs-WebGPU crossover itself is a manually-run spike
(`packages/tensor-webgpu/scripts/measure-gemm-threshold.ts`), recorded in
`docs/spikes/webgpu-baseline.md`, the same way `docs/spikes/wasm-baseline.md` records the
WASM-vs-pure-JS numbers.
