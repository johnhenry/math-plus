# Spike: WASM SIMD128 for the contiguous elementwise fast path (2026-08-12)

**Status: measured, shipped.** Per issue #13's own instruction ("do not merge SIMD without an
attached benchmark showing a real speedup — if unproven, ship scalar-only and say so"), this
document is that benchmark. `docs/spikes/wasm-baseline.md` measured the *scalar* `...Into` path's
1.78x win over pure JS at N=1e6 and explicitly deferred SIMD ("a 1.78x scalar win already exists
untapped behind the copy overhead — measure SIMD only after the Into path lands," issue #3). #3
shipped; this is that follow-up measurement.

## Method

`crates/tensor-wasm-kernels/src/lib.rs` gained a `simd` Cargo feature gating a new `simd` module:
`add_f32_contiguous_simd128`/`mul_f32_contiguous_simd128`, hand-written `std::arch::wasm32`
SIMD128 (`f32x4_add`/`f32x4_mul`, 4 lanes/instruction, scalar tail loop for `len % 4 != 0`).
Deliberately **contiguous-only, no offset/stride params** — SIMD loads need contiguous memory,
and there's no benefit to a strided variant; the existing general `add_f32_strided`/
`mul_f32_strided` kernels stay as the fallback for non-contiguous views.

Benchmarked three variants at N=1,000,000 (Node, `process.hrtime.bigint()`, 3-iteration warmup,
50-iteration mean, `/tmp/simd_bench.mjs` — not committed, throwaway measurement script):

1. **strided-scalar** — the kernel that ships today (`add_f32_strided`, called with `stride=1`).
2. **contiguous-scalar** — a new scalar-only kernel taking plain base pointers (no offset/stride
   params at all), isolating how much of any speedup is just from removing the runtime
   offset/stride multiply the compiler can't prove is trivial (a `stride=1` call to
   `add_f32_strided` still can't auto-vectorize, since `stride` is a runtime `isize` parameter).
3. **SIMD128** — `add_f32_contiguous_simd128`.

## Results (representative run; stable across repeats, see below)

| Variant | Time (ms/call) | vs. strided-scalar | vs. contiguous-scalar |
|---|---|---|---|
| strided-scalar (shipped today) | 1.29 ms | baseline | — |
| contiguous-scalar (no SIMD) | 0.95 ms | 1.36x | baseline |
| SIMD128 | 0.33 ms | **3.88x** | **2.84x** |

Repeated (3 runs): contiguous-scalar-vs-strided-scalar ranged **1.21x–1.42x**; SIMD-vs-contiguous-
scalar (SIMD's *own* marginal contribution, apples-to-apples) ranged **2.63x–3.03x**; total
SIMD-vs-strided-scalar (what actually ships today) ranged **3.17x–4.27x**. Correctness verified
every run: 0 mismatches between the SIMD output and the scalar `add_f32_strided` output over all
1,000,000 elements.

**Attribution matters here**: roughly 1.2–1.4x of the total win is just from having a dedicated
contiguous fast path at all (no runtime stride multiply) — SIMD's *own* contribution on top of
that is a separate, real 2.6–3x. Both numbers are reported so a future reader can tell how much
of the observed total speedup to credit to "SIMD" specifically vs. "a contiguous fast path,
which could in principle have been scalar."

## Decision: ship it

2.6–3x is well above any reasonable bar for "a real speedup" — this isn't a wash or a marginal
win. Shipped as a **separate `.wasm` build artifact** (`packages/tensor-wasm/wasm/
tensor_wasm_kernels_simd128.wasm`, `npm run build:wasm:simd`), never merged into the default
scalar build:

- **Why a separate artifact, not a single module with both paths**: a WASM module containing ANY
  v128 instruction fails `WebAssembly` validation **in its entirety** on a runtime without SIMD
  support — module loading is all-or-nothing (unlike native code's per-call feature detection),
  so there is no way to ship one `.wasm` file that both uses SIMD and is guaranteed loadable
  everywhere.
- **Feature detection**: `Kernels.load()` reads the SIMD module's bytes and calls
  `WebAssembly.validate()` on them directly (not a separate hand-crafted minimal probe module —
  simpler, and it's checking the exact bytes about to be instantiated) before attempting
  `WebAssembly.instantiate()`. Any failure at any step (unsupported runtime, the artifact wasn't
  built, a bad import) is caught and leaves the SIMD path unavailable — `addInto`/`mulInto` fall
  back to the always-present scalar/strided kernels transparently, never throwing.
- **Shared memory, not a second buffer**: the SIMD build is compiled with
  `RUSTFLAGS="-C link-args=--import-memory"`, so it *imports* `env.memory` instead of allocating
  its own. `Kernels.load()` instantiates it passing the scalar module's own `memory` export as
  that import — both modules genuinely share one linear memory / one `ArrayBuffer`, so the SIMD
  kernels operate on the *exact same* resident `WasmTensor` data with zero copying. Without this,
  the SIMD module would have its own separate memory and `WasmTensor` data (allocated via the
  scalar module's `alloc`) would be invisible to it — defeating the entire point.
- **Eligibility**: `addInto`/`mulInto` use the SIMD path only when it's available AND every
  operand (`a`, `b`, `out`) is contiguous (`stride === 1`, per `flatSpec`'s check) — a
  non-contiguous view (e.g. a strided slice) always falls back to the general strided kernel,
  which is unaffected and unchanged.

## Known cost (risk #7 in docs/PLAN.md, paid deliberately here)

Two implementations of each accelerated kernel (add, mul) that must agree bit-for-bit — verified
by a dedicated test (`addInto/mulInto: SIMD-accelerated result is bit-for-bit identical to the
scalar fallback`) comparing SIMD output against the scalar `add_f32_strided`/`mul_f32_strided`
kernels element-by-element, including a length not a multiple of 4 to exercise the SIMD kernel's
scalar tail loop. `gemm_f32` (matmul) does **not** get a SIMD variant in this pass — it's
compute-bound rather than memory-bandwidth-bound like elementwise add/mul, has a fundamentally
different (blocked/tiled) vectorization shape, and is explicitly out of scope for issue #13
(which named "the Into path," i.e. the elementwise kernels this measured). *(Superseded by issue #121 —
see "GEMM" below: blocked + register-tiled, then SIMD128.)*

## Reproduction

Scratch benchmark script was `/tmp/simd_bench.mjs` (not committed — throwaway, superseded by the
package's own committed tests, which cover the same ground with proper CI-safe thresholds). To
re-measure by hand: build both artifacts (`npm run build:wasm -w @johnhenry/math-plus-tensor-wasm`), then
compare `add_f32_strided`/`add_f32_contiguous_simd128` directly via `WebAssembly.instantiate()`
on the two `.wasm` files in `packages/tensor-wasm/wasm/`, matching this doc's three-variant
methodology.

## GEMM: cache-blocked, register-tiled, SIMD128 (issue #121, 2026-09-23)

**Status: measured, shipped.** Issue #121 (found while building laya-js): `gemm_f32` was a naive
`i/j/p` triple loop over strided reads — ~0.4 GFLOP/s at 1024³ on trycooy
(`deno-ffi-baseline.md`: 6072 ms), and the reason `tensor-webgpu`'s `threshold.ts` could never
find a crossover: both sides were slow. Acceptance bar: ≥10x GFLOP/s at 1024³ in the WASM build,
NumPy oracle green, threshold spike re-run.

### What shipped

- `crates/tensor-wasm-kernels/src/gemm.rs` — ONE Goto/BLIS-style driver (NC=256 column blocks of
  B, KC=256 shared-dim blocks, MC=64 row blocks of A; both operands packed into contiguous,
  zero-padded MR/NR panels; MR x NR register tile; alpha/beta + edge clipping at write-back).
  Packing is what keeps the ABI's arbitrary strides: a `.transposed()` view costs an O(n²)
  gather per block, not a strided load in the O(n³) loop.
- Two micro-kernels plugged into that one driver: the portable scalar 4x8 (32 accumulators;
  `gemm_f32`, scalar wasm module + native cdylib) and an f32x4 4x8 (8 v128 accumulators + 2 B
  vectors + 1 broadcast = 11 live vectors, inside x86-64's 16 XMM registers;
  `gemm_f32_simd128`, SIMD module only). `matmulInto` routes **every** call to the SIMD export
  when the SIMD module loaded — no contiguity requirement, unlike add/mul.
- Same arithmetic per output element on both paths (mul-then-add, no FMA, identical packing and
  K-block order) → SIMD and scalar results are **bit-for-bit identical**, asserted in
  `kernels.test.ts`. vs NumPy: `test/gemm-differential.test.ts` (both paths, shapes straddling
  every block/tile edge, transposed operands; tensor-core's `matmul:f32` tolerance).
- Native cdylib: the portable blocked kernel (LLVM autovectorizes the 4x8 tile on aarch64/
  x86-64), plus an opt-in `accelerate` cargo feature (macOS only) that hands BLAS-expressible
  layouts to `cblas_sgemm` and falls back to the portable kernel otherwise.

### Shared-memory hazard found (and now guarded)

The SIMD module is linked with `--import-memory` and shares the scalar module's linear memory —
but it is a *complete* second build of the crate, with its own `.rodata` at the same address
(1 MiB) and a start function (`__wasm_init_memory`) that `memory.fill`s its `.bss` — which is
where the scalar module's allocator (dlmalloc) state lives. Two consequences:

1. Instantiation rewrites the scalar module's static data. Harmless only while both builds'
   data segments are byte-identical — true today (verified), but any SIMD-only code that adds
   a panic location or constant table would silently corrupt the scalar module's constants.
   `Kernels.load()` now snapshots the memory, instantiates, compares, and on any difference
   restores the snapshot and runs scalar-only (tested with a deliberately patched module).
   The GEMM driver is written panic-free (raw pointers, no bounds-checked indexing) to keep
   the segments identical.
2. The SIMD module's copy of the allocator knows nothing about the scalar allocator's live
   blocks, so SIMD code must never heap-allocate. `gemm_f32_simd128`'s packing scratch
   (MC·KC + KC·NC f32 = 320 KiB) lives on the wasm shadow stack (1 MiB, below the data); the
   scalar wasm entry does the same (zero allocator traffic, so `matmulInto`'s
   `allocCallCount` stays flat — tested). Native uses a problem-sized heap `Vec` instead.

### Results

Apple M2 (8 GB), macOS 27, Node 24.9.0, rustc stable, `--release` (lto, codegen-units=1).
`node packages/tensor-wasm/scripts/gemm-bench.ts` — square n³ f32, resident buffers, kernel call
only, best of N after warm-up. The "naive" column is origin/main's artifacts built from
`git archive` and passed as the script's `wasmDir` argument — same machine, same harness.

| n | naive (pre-#121) | scalar blocked | SIMD128 blocked | SIMD vs naive |
|---:|---:|---:|---:|---:|
| 64 | 4.52 GFLOP/s | 10.24 | 11.35 | 2.5x |
| 256 | 2.77 | 11.01 | 39.74 | 14.3x |
| 512 | 2.18 | 10.74 | 38.71 | 17.8x |
| 1024 | 1.73 (1239 ms) | 10.35 (208 ms) | **37.41 (57 ms)** | **21.6x** |

- **Acceptance: met.** 21.6x at 1024³ on the same machine; against the issue's own ~0.4 GFLOP/s
  (trycooy, x86-64) the M2 SIMD number is ~90x, but that cross-machine ratio is not a fair
  claim — the same-machine 21.6x is the number to quote. Not re-measured on trycooy (x86-64)
  from this session; expect a lower absolute number there (16 XMM registers, older core).
- Attribution, like the elementwise section above: blocking + packing + register tiling alone
  (scalar blocked vs naive) is ~6x at 1024³; SIMD128's own contribution on top is ~3.6x.
- Small n (64) is overhead-bound (packing + tile write-back dominate a 0.05 ms call).
- Not attempted (possible follow-ups, none needed for the bar): relaxed-SIMD FMA (see README —
  engine-dependent results, and a Safari validation failure would disable the whole SIMD
  module), multithreading (needs SharedArrayBuffer + COOP/COEP), autotuned block sizes.

### Native (Deno FFI) re-measure

See `deno-ffi-baseline.md` → "Re-measured after issue #121".

### WebGPU/WASM threshold

Not re-run in this session: `packages/tensor-webgpu/scripts/measure-gemm-threshold.ts` drives
headless Chrome under Xvfb (Linux harness; not available on the macOS machine used here), and
`threshold.ts` is being re-derived separately. What changed for that spike: the WASM side of the
crossover got ~20x faster (e.g. 1024³ kernel time 1239 ms → 57 ms here), so the naive WGSL
shader must now beat ~37 GFLOP/s + copy overhead rather than ~2 — `GEMM_ELEMENT_THRESHOLD =
Infinity` is, if anything, more clearly right until the WGSL kernel is tiled too.

(Later, 2026-09-24: the WGSL kernel was tiled, and the crossover was re-measured against this SIMD
GEMM — [`webgpu-tiled-gemm.md`](webgpu-tiled-gemm.md#re-measured-against-the-simd-wasm-gemm-2026-09-24).
It moved from m·n = 128² to m·n >= 192² plus a k-aware work floor.)

### Reproduction

```bash
npm run build:wasm -w @johnhenry/math-plus-tensor-wasm
node packages/tensor-wasm/scripts/gemm-bench.ts 64,256,512,1024
# naive baseline: build origin/main's crate into a scratch dir the same two ways
# (scalar, then --features simd with --import-memory), copy both .wasm files into one
# directory, and pass it as the second argument.
```
