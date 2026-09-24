# @johnhenry/math-plus-tensor-wasm

## 0.1.2

### Patch Changes

- d122aef: Published declarations no longer import `./x.ts` (closes #157). tsc's `rewriteRelativeImportExtensions` rewrites `.ts` specifiers to `.js` in emitted JS but not in emitted `.d.ts`, so `dist/*.d.ts` referenced files that aren't in the package, and Deno's type check failed on them. Every package's build now runs `scripts/rewrite-dts-extensions.mjs` after `tsc`:

  - relative `.ts` / `.mts` / `.cts` specifiers in `dist/**/*.d.ts` become `.js` / `.mjs` / `.cjs` (as in laya-js),
  - each `dist/*.js` with a declaration file starts with `// @ts-self-types="./x.d.ts"` (after the `#!` line of a bin), so Deno finds the types when it loads `dist/` as plain files or from a URL instead of through `npm:`. Source maps are shifted by the inserted line.

  No runtime change. The manifest drift test checks every built `dist/` for both.

- Updated dependencies [d122aef]
  - @johnhenry/math-plus-telemetry@0.0.3

## 0.1.1

### Patch Changes

- 80123df: Republish with an npm provenance attestation. The previous versions were published from a local machine on 2026-09-23 without provenance (the CI `NPM_TOKEN` secret had expired). No code changes.
- Updated dependencies [80123df]
  - @johnhenry/math-plus-telemetry@0.0.2

## 0.1.0

### Minor Changes

- 46e43cb: `matmulInto` now runs a cache-blocked, register-tiled GEMM (issue #121): the scalar module's `gemm_f32` uses a portable 4x8 micro-kernel and the SIMD128 module gains `gemm_f32_simd128` (f32x4, used for every `matmulInto` call when SIMD is available, any strides). ~37 GFLOP/s at 1024³ vs ~1.7 for the old naive loop on the same machine (21.6x; docs/spikes/wasm-simd.md); SIMD and scalar results are bit-identical. `matmulInto` now rejects a wrongly-shaped `out` with a RangeError instead of writing past its buffer, and `Kernels.load()` verifies that instantiating the SIMD module leaves the shared linear memory untouched (falling back to scalar-only, memory restored, if not). The native cdylib gets the same blocked kernel plus an opt-in `accelerate` cargo feature (macOS, `cblas_sgemm`). Same JS API.

### Patch Changes

- 5d7172b: Lower `engines.node` from `>=26.0.0` to `>=24.0.0`. Nothing in these packages needs Node 26: the full test suite passes on Node 24.9, and CI now tests Node 24. Every suite also runs under Bun 1.2.17 (`npm run test:bun`).

  `@johnhenry/math-plus-frame-parquet`: `scanParquet`/`scanParquetLazy` now accept an absolute file path without wildcards under Bun. Bun 1.2's `fs.promises.glob` returns no matches for such a path, so a pattern with no glob metacharacters is now resolved with `stat` on every runtime.

- Updated dependencies [5d7172b]
  - @johnhenry/math-plus-telemetry@0.0.1

## 0.1.0

### Minor Changes

- aeeeb35: Gap-analysis backlog (issues #64-#72): additive new API surface across six packages, all backward-compatible.

  - **@johnhenry/math-plus-tensor-core**: eager `Tensor` unary op-table parity with the compiled IR (`exp`/`pow`/`abs`/`neg`/full trig+hyperbolic families/`cbrt`/`log10`/`log2`/`expm1`/`log1p`/`floor`/`ceil`/`round`/`trunc`), plus structural ops `clip`/`prod`/`pad`/`split`/`repeat`/`flip`/`roll`/`nonzero`.
  - **@johnhenry/math-plus-tensor-wasm**: `subInto`/`divInto` WASM kernels, parity with the existing `addInto`/`mulInto`.
  - **@johnhenry/math-plus-adapter-math**: `det`/`inv` (derived from the existing `lu`/`solve`), and `eigGeneral` — eigenvalues of a general non-symmetric real matrix via Hessenberg reduction + shifted QR, returned as `ComplexNumber[]` to support genuine complex-conjugate pairs.
  - **@johnhenry/math-plus-fft**: `fft2`/`ifft2` and `fftshift`/`ifftshift`.
  - **@johnhenry/math-plus-signal**: `correlate`/`correlate1D` (convolution's cross-correlation dual), `freqz` (SOS filter frequency response), `welch` (power spectral density).
  - **@johnhenry/math-plus-tensor-autograd**: `nn.Sequential`, `nn.Dropout`, `nn.huberLoss`, `nn.binaryCrossEntropy`; `optim.Adam`, `optim.RMSprop`, `optim.StepLR` (a learning-rate scheduler — `optim.SGD`/`optim.AdamW`'s `lr` field is now mutable rather than `readonly` to support this, a backward-compatible widening).
