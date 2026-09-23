# Testing

## Unit tests

```bash
npm test            # all workspaces
npm test -w @johnhenry/math-plus-tensor-core
```

Tests are TypeScript run directly by `node --test` (native type stripping, Node ≥22.12).

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

## Gradient oracles (autograd)

`adapter-math`'s `@johnhenry/math-plus-adapter-math/test-utils` subpath (`dualGrad`/`dualGradN`) wraps
@johnhenry/math's `DualNumber` forward-mode autodiff — a third gradient oracle, algorithmically
independent of both `tensor-autograd`'s reverse-mode tape and finite differences. Its own tests
(`adapters/adapter-math/test/test-utils.test.ts`) validate it against finite differences AND
against `tensor-autograd`'s `Variable`/`grad.of` on real scalar/multivariate functions.

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

Same "oracle unavailable -> skip, never fail" convention as the NumPy/pyarrow oracles above, but
the oracle is a live `GPUAdapter` reached over the Chrome DevTools Protocol instead of a Python
subprocess — `packages/tensor-webgpu/test/helpers.ts` launches headless Chrome under Xvfb (mirroring
`~/.local/bin/gl-report`'s pattern on the trycooy dev machine: raw CDP over WebSocket, no
Playwright/Puppeteer) and probes `navigator.gpu.requestAdapter()` once per test file, caching the
result. Individual tests call `getHarness()` and `t.skip(reason)` when unavailable.

**Two harness backends** (`test/helpers.ts`, selected by `$MATH_PLUS_WEBGPU_HARNESS`):

- `dawn` — in-process Dawn via the `webgpu` npm package (a devDependency of this package; also
  its optional peer for Node users, loaded through `src/dawn.ts`). Test bodies run as an
  `AsyncFunction` with `navigator = { gpu: <Dawn> }`, so the same bodies run unmodified. No
  browser or display server — this is the path that works on macOS (Metal) and any machine with a
  Dawn-supported GPU. Created with `allow_unsafe_apis` so the experimental subgroup-matrix GEMM
  kernel is exercised where the adapter supports it.
- `chrome` — the headless-Chrome-over-CDP harness described below (Linux: Xvfb; macOS:
  `--headless=new --use-angle=metal`, no display needed).
- unset / `auto` — Dawn first, Chrome if Dawn has no adapter (e.g. a GPU-less CI runner, where
  Chrome's SwiftShader path is the one that works).

GEMM correctness (`test/gemm.test.ts`) additionally uses a NumPy float64 oracle
(`packages/tensor-webgpu/scripts/gemm_oracle.py`, same `$MATH_PLUS_ORACLE_PYTHON` / `python3`
resolution and skip-don't-fail rule as above), with error bounds derived from `|A|·|B|` rather than
hand-tuned per case. A healthy local run shows `skipped 0`; the one test that skips on adapters
without f32 8x8x8 subgroup matrices (anything but Apple GPUs today) says so in its skip reason.

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
`docs/spikes/webgpu-tiled-gemm.md` (current kernels; `docs/spikes/webgpu-baseline.md` has the v1
naive-kernel numbers), the same way `docs/spikes/wasm-baseline.md` records the WASM-vs-pure-JS
numbers.
