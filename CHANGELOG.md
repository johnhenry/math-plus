# Changelog

Notable changes to the `math-plus` monorepo as a whole. **Per-package release
history lives in each package's own `CHANGELOG.md`** (Changesets-managed);
this file tracks repo-level changes that no single package owns. Note the
per-package changelogs record pre-rename history — versions were reset to
`0.0.0` when the 17 packages were renamed to `@johnhenry/math-plus-*` (#110).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Per-package READMEs for the 13 packages that lacked one (tensor-core,
  tensor-autograd, tensor-compile, tensor-wasm, tensor-webgpu, fft, signal,
  image, data, frame-arrow, frame-parquet, scalar-types, telemetry) — each
  mined from that package's tests and source doc comments: install, quick
  start, API surface, and the traps the tests actually guard.
- Root `examples/` with ten runnable walkthroughs (at least one per
  cluster), an `npm run examples` loop for the env-independent set, and a
  CI smoke step. The WASM example is written but excluded from the loop
  (needs the gitignored `.wasm` artifact); WebGPU and Python interop are
  documented rather than scripted (browser / Python environment required).
- Root README rewritten as a family map: "Which package do I want?" routing
  table, clustered package tables linking each per-package README, examples
  and dev-workflow sections, docs-site link.
- This changelog.

## [0.0.0] - 2026-08-25

Provenance entry — the state of the repo when this changelog was introduced,
not a release cut on this date.

- 17 npm/JSR packages (`@johnhenry/math-plus-*`) plus one PyPI package
  (`johnhenry-math-plus-interop`), renamed and version-reset in #110;
  prerelease dist-tag routing + provenance in #111.
- Rust workspace (`crates/tensor-wasm-kernels`) building the WASM kernels;
  native cdylib builds in CI (`native-kernels.yml`), not yet published.
- Manifest-drift guard (`test/manifest-drift.test.ts`) keeping the
  hand-maintained root build/test workspace lists honest.
- SIMD benchmark removed from CI in favor of `test:bench` on known hardware
  (mixed-runner wall-clock thresholds can't distinguish a slow machine from
  a real regression).
