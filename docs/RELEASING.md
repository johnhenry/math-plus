# Releasing

Published from GitHub Actions using **npm trusted publishing** (OIDC), driven by
[Changesets](https://github.com/changesets/changesets) because this repo publishes many
independently versioned packages instead of one. There is no `NPM_TOKEN` secret: npm trusts
this repository's `.github/workflows/release.yml` directly, and every publish carries a
provenance attestation.

## Prerequisite: each package trusts the release workflow

Every published package was linked once (2026-09-24) with:

```bash
npm trust github @johnhenry/math-plus-<name> --file release.yml --repo johnhenry/math-plus --allow-publish
```

`npm trust` must run from a 2FA-enabled account session (`npm login --auth-type=web`); granular
tokens that bypass 2FA are refused. **A brand-new package** can't be trusted before it exists on
npm: publish its first version by hand (`npm publish --access public` from a 2FA session), then run
`npm trust` for it; later releases then flow through CI. The workflow needs npm >= 11.5.1 (it
upgrades npm itself) and `id-token: write` (already set).

## Normal flow

1. **Describe the change** in the PR that makes it:
   ```bash
   npm run changeset      # pick packages, pick bump type, write the summary
   git add .changeset && git commit -m "Add changeset"
   ```
2. **Merge to `main`.** The Release workflow opens (or updates) a **"chore: version packages"** PR
   applying the bumps and changelogs.
3. **Merge the version PR.** The workflow runs again, builds (WASM kernels + TypeScript), tests,
   and publishes every package whose version isn't yet on the registry.

`workflow_dispatch` re-runs the publish step without a new changeset — useful if a publish failed
partway through a multi-package release.

## What CI verifies before publishing

`npm run build` (Rust → `wasm32-unknown-unknown` via cargo + lld, then `tsc` across workspaces) and
`npm test` (unit tests plus the NumPy differential suite — CI installs numpy and asserts it imports,
so those tests can't silently skip; see [TESTING.md](./TESTING.md)).

## Package-name notes

Packages are scoped `@johnhenry/math-plus-*` names. Originally this repo shipped unscoped
`mallory-*` names, but the `@mallory` npm scope belongs to another user — the move to `@johnhenry`
(alongside the sibling `math` and `mallory` repos, see `docs/FAMILY.md`) put every package in this
family under one scope the project actually controls, with room for future family members. The
root package is `private: true` and never publishes. `packages/interop-python` is a PyPI package
outside both the npm and Cargo workspaces, released separately.

## PyPI (`interop-python`, issue #21)

`packages/interop-python` ships to PyPI as **`johnhenry-math-plus-interop`**, on its own
tag-triggered release cadence (`.github/workflows/release-interop-python.yml`) — independent of the
npm/JSR Changesets flow above, since it's a different package manager, versioning scheme, and
release cadence entirely.

### Prerequisite: PyPI Trusted Publishing

Like the JSR job below, this uses OIDC — no token to store — but needs a one-time manual step:

1. Create the `johnhenry-math-plus-interop` project on [pypi.org](https://pypi.org) (or publish it
   manually once first via `twine` to reserve the name, if pypi.org requires the project to already
   exist before configuring trusted publishing for it — check current PyPI UI).
2. In that project's **Settings → Publishing**, add a trusted publisher: repo `johnhenry/math-plus`,
   workflow `release-interop-python.yml`, environment (leave blank unless you've configured one).

Until that's done, `release-interop-python.yml` fails at the publish step (no token, no configured
trusted publisher) — it isn't `continue-on-error` like the JSR job, since a failed *tag push*
release should be visibly red, not silently swallowed.

### Publishing

```bash
# bump the version in packages/interop-python/pyproject.toml first
git tag johnhenry-math-plus-interop-v0.0.1
git push origin johnhenry-math-plus-interop-v0.0.1
```

`workflow_dispatch` re-runs the publish without a new tag (e.g. after fixing a Trusted Publishing
misconfiguration).

## JSR (dual publish, issue #25)

npm is the baseline distribution channel; every publishable package is **also** published to
[JSR](https://jsr.io) under the `@johnhenry` scope (`@johnhenry/math-plus-tensor-core`, etc.) — decided
in favor of dual publishing over npm-only, for first-class Deno reach (Deno consumes npm packages
fine already, so this is about ergonomics/discoverability, not capability).

### Prerequisite: claim the scope and enable Trusted Publishing

JSR publishing in CI uses GitHub's OIDC **Trusted Publishing** — no secret to store, but it needs a
one-time manual setup on jsr.io before the `jsr-release` workflow job does anything:

1. Sign in at [jsr.io](https://jsr.io) and claim the `@johnhenry` scope, if not already claimed.
2. For **each** package (see the loop in `.github/workflows/release.yml`'s `jsr-release` job, or
   `scripts/sync-jsr-configs.mjs`'s `PACKAGE_DIRS` list), create the package on jsr.io under that
   scope and link `johnhenry/math-plus` as a trusted GitHub Actions publisher (jsr.io's package
   settings → "Publishing" → add a trusted publisher, workflow file `release.yml`).

Until every package is linked, `jsr-release` fails harmlessly (`continue-on-error: true`) — it's
fully independent of npm publishing, which is unaffected either way.

### How `jsr.json` stays in sync

Each publishable package has a `jsr.json` **generated from its `package.json`** by
`scripts/sync-jsr-configs.mjs` — never hand-edited, so the two registries' versions and dependency
ranges can't drift apart. Run it after adding/bumping a dependency:

```bash
node scripts/sync-jsr-configs.mjs
```

The release workflow re-runs it right before publishing, so a Changesets version bump (which only
touches `package.json`) is picked up automatically.

### What's different about the JSR side

- JSR publishes and type-checks the **TypeScript source** directly (`./src/index.ts`), not the
  compiled `dist/` npm ships — there's no separate JSR build step.
- Workspace-sibling dependencies (`@johnhenry/math-plus-tensor-core`, etc.) are mapped to
  `jsr:@johnhenry/...` specifiers in each package's `imports` map; external npm dependencies map to
  `npm:...` specifiers. Both are generated by `scripts/sync-jsr-configs.mjs` from the package's own
  `dependencies`/`peerDependencies` — see that script's own doc comment for how it tells this
  repo's own `@johnhenry/math-plus-*` family apart from the sibling `@johnhenry/math`/
  `@johnhenry/iteration` packages (both still map to `jsr:`, just not treated as "internal").
- `@johnhenry/math-plus-scalar-types` and `@johnhenry/math-plus-adapter-math` depend on
  `@johnhenry/math`, which lives in the **sibling**
  [`johnhenry/math`](https://github.com/johnhenry/math) repo. Their JSR `imports`
  map points at `jsr:@johnhenry/math`, which only resolves once that repo *also* publishes to
  JSR under the same scope — tracked as a follow-up there, not something this repo can complete
  alone.
- `--allow-slow-types` is passed to every `jsr publish` call — JSR's fast type-checker is stricter
  than `tsc` about some patterns already in use here (private fields, complex generics); revisit
  removing the flag package-by-package once each one is verified clean under JSR's own checker.
