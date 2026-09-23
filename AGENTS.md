# Agent playbook

This file exists because a repo largely written by AI agents drifts fast
without one written rulebook everyone (human or agent) actually reads.
The pattern — and several of the rules below — comes from studying
[Woxi](https://github.com/ad-si/Woxi)'s own `AGENTS.md`
(see `docs/spikes/woxi-study.md`); the rest reflects conventions this repo
had already converged on before they were written down anywhere central.

`CLAUDE.md` in this directory is a symlink to this file.

## The verification loop (before every push)

For the package(s) you touched:
1. `npm run typecheck -w <package>`
2. `npm test -w <package>` — differential suites need oracle env vars set
   (below); they must show **0 skipped**, not just 0 failed, or you're not
   actually testing anything.
3. `cargo test --workspace` if any crate changed.
4. Full workspace: `npm run build && npm test` from the repo root, plus
   `npm run test:bun` (every suite under Bun; CI runs both).
5. A genuinely fresh clone: `git clone . /tmp/mallory-verifyN && cd $_ && npm ci && npm run build && npm test`.
   This is the only way to catch "works on my checked-out tree" bugs
   (missing files in `package.json`'s `files`, undeclared deps, etc.).
6. Commit, push, close the issue with a comment naming the commit SHA.

Full detail, exact oracle env vars, and NixOS provisioning commands:
[`docs/TESTING.md`](docs/TESTING.md).

## Oracle discipline

Every new numeric feature gets a differential oracle (NumPy for tensor
ops, scipy.signal for `@johnhenry/math-plus-signal`, pyarrow/pandas for frame
packages) — never hand-typed expected values for anything with a
reference implementation. Oracles resolve via an env var with a
`python3` PATH fallback, and follow a **skip-don't-fail** contract: no
oracle available → the test skips cleanly, it never fails. But "skipped"
in a real run is a red flag, not a pass — see step 2 above.

## The canonical-implementation rule

Never implement the same construct twice. Duplicates drift apart silently
and are the single most common source of divergence in agent-written
code. If two packages need the same logic, factor it into the one place
they both depend on — don't let a second copy exist "for now."

## Regression tests are not optional

Fixing a bug without adding a test that would have caught it means the
bug can come back unnoticed. If you stumble on a pre-existing issue while
working on something else, fix it right there rather than working around
it or leaving a `// TODO` — filing an issue is the fallback for scope
that's genuinely out of the current task, not a substitute for a quick
real fix.

## Disclose scope boundaries loudly

Every new package/feature documents what it does **not** do, in the code
and/or its README, not just in an issue comment — e.g. `rfft` returning
the full spectrum rather than the true half-spectrum optimization,
`butter` supporting lowpass/highpass only. A future reader (agent or
human) should never have to reverse-engineer a limitation from a test
failure.

## New-package definition of done

See the README's [`## Adding a new package`](README.md#adding-a-new-package)
section for the narrative version (the real worked example, and the test
for whether a new package is warranted at all). Mechanically, adding a
package under `packages/`, `adapters/`, or `scalars/` means all of the
following, not just `npm init`:
- `tsconfig.json` + `tsconfig.typecheck.json` pair matching an existing
  package's shape.
- The package name added to root `package.json`'s `build` **and** `test`
  script strings.
- The package directory added to `scripts/sync-jsr-configs.mjs`'s
  `PACKAGE_DIRS`.
- A `"test:bun": "bun test ./test/"` script, and every test file registering
  through `test/harness.ts` with its own `bun:test` import (never
  `import { test } from "node:test"`; see docs/TESTING.md "Running under
  Bun" for why). `test/manifest-drift.test.ts` enforces both.
- `engines.node` identical to the root's (currently `>=24.0.0`).
- Once [#47](https://github.com/johnhenry/math-plus/issues/47) lands: run its manifest-drift check — it verifies the two
  points above for you and fails loudly if either is missed.

## GPU / WebGPU work

Any `@johnhenry/math-plus-tensor-webgpu` test run drives a real headless Chrome+Xvfb
instance sharing this machine's one GPU. Follow the `~/gpu.lock`
convention (see the user-level `~/CLAUDE.md`) before running those tests:
check for an existing lock, acquire your own, always release it
(including on error paths). Treat `NO_NAVIGATOR_GPU` failures as probable
transient contention — rerun the affected file in isolation before
concluding it's a real regression; see
[`docs/spikes/woxi-study.md`](docs/spikes/woxi-study.md#test-methodology-the-best-material-in-the-repo)
for why this class of flake is expected and how Woxi's harness handles
the analogous case.

## Benchmarks

Performance numbers follow [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md):
5 s cooldown before each cell, timing windows of 1 s or less, backends
alternated in one process, and the machine and thermal state recorded
(`scripts/bench/thermal.ts` does all four). A fanless laptop drops to about
35 % of its cold GPU speed after about 10 s of load, so a long loop measures
the throttled machine.

## Non-goals

Before adding scope beyond what a task asks for, check
[`docs/PLAN.md` §2](docs/PLAN.md#2-non-goals-v1-whole-project) — several
things that look like obvious improvements (e.g. a general symbolic
rewrite engine, byte-exact cross-platform snapshots) are deliberately out
of scope, with the reasoning recorded there.
