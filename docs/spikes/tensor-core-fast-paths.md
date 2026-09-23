# tensor-core contiguous fast paths + blocked GEMM (2026-09-23, issue #120)

`tensor-core` was correct but slow on the CPU. Every elementwise op walked its
elements through the `elementOffsets()` generator. `softmax` built about 6
temporaries. `matmul` was a naive strided triple loop. This spike records what
changed and what it bought. The API did not change.

## What changed

| Area | Before | After |
|---|---|---|
| `add/sub/mul/div` | Per-element odometer and a closure call | Flat loops, one per op. Covers both sides full-shape, a scalar side, trailing-block "bias" broadcast (`[B,T,C] + [C]`) and per-row broadcast (`[B,T,C] - [B,T,1]`), in both operand orders |
| Unary ops (`exp`, `tanh`, `gelu`, …), `sqrt`/`log`, `abs` | Generator walk | Flat loop when the input is contiguous. `sqrt`/`log` now share `#unaryFloat` instead of keeping two private copies |
| `cast`, `contiguous()` | Generator walk | Bulk `TypedArray.set` / `slice` when the input is contiguous |
| Comparisons | Odometer | Flat loop for same-shape and scalar operands |
| `sum`/`mean`/`min`/`max` | Strided loop, generator for the full reduction | `[outer, dim, inner]` kernel. For `inner > 1` it walks memory row by row into an f64 accumulator row |
| `softmax` | max, sub, broadcast copy, exp, sum, broadcast copy, div | One fused kernel (max, exp+sum, scale per lane), no temporaries |
| `variance`/`std` | mean, sub, mul, sum, div | One fused two-pass kernel, no temporaries |
| `matmul` (all Number dtypes) | Naive strided triple loop | Operands packed into dense f64 panels (A as `[m,k]`, Bᵀ as `[n,k]`), then a 4×4 register-blocked NT GEMM with 64-column j-blocking. Adapted from laya-js `backend-cpu/src/gemm.ts` (Apache-2.0, same author) |

The kernels live in `packages/tensor-core/src/kernels.ts`, an internal
module that is not exported.

### Bit-identical by design

Each fast path does the same per-element operation as the general strided
path, accumulates in the same order (ascending along the reduced axis, in an
f64 JS number), and rounds to the storage dtype at the same points. Where the
old composed path stored an f32 temporary, the fused kernels call
`Math.fround` at that point. For example, softmax rounds `x - max`, `exp(.)`
and the row sum. `matmul` accumulates each output over `p = 0..k-1` from 0 in
f64 and rounds once on store. That is the same sequence of f64 operations as
the naive loop, so blocking reorders memory traffic but not the arithmetic.

As a result, `test/fast-paths.test.ts` checks equivalence with `Object.is`
per element, not a tolerance. It compares a contiguous tensor (fast path)
against a strided twin with the same values (general path). `matmul` is also
checked against an independent naive loop. I mutation-checked these tests by
dropping one `Math.fround` from softmax, dropping one from variance, and
breaking the row-broadcast index. Each mutation failed the suite. The NumPy
differential suite passes unchanged.

## Numbers

Apple M2 (fanless MacBook Air), Node v26.9.0, single thread, f32, uniform
[-1, 1] inputs. The script is `packages/tensor-core/scripts/bench-fast-paths.ts`.
It makes one warmup call, then reports the median of N timed calls.

**Caveat on the environment:** other agents' test suites were running on this
machine at the same time (load average 10–28), and the M2 throttles when hot.
To make before and after comparable under the same conditions, each case ran
**interleaved**: before, idle 3 s, after, idle 3 s, repeated 3×. "Before" is
`git archive` of `origin/main`'s `src/`. The table shows the **best of the 3
medians** per side. Absolute times are pessimistic, and the ratios are the
meaningful part. The last column is from an earlier sequential run at lower
load, and it agrees within noise.

| Case | Before (ms) | After (ms) | Speedup | Speedup, quieter sequential run |
|---|---:|---:|---:|---:|
| matmul 256×256 @ 256×256 | 73.13 | 12.39 | **5.9×** | 7.4× (39.6 → 5.4 ms; 0.85 → 6.2 GFLOP/s) |
| matmul 1024×1024 @ 1024×1024 | 4237.70 | 603.90 | **7.0×** | 7.9× (2482 → 313 ms; 0.87 → 6.9 GFLOP/s) |
| matmul [8,128,64] @ [8,64,128] (batched) | 26.39 | 4.74 | **5.6×** | 5.4× |
| add [16,128,1024] + same shape | 45.57 | 6.64 | **6.9×** | 5.3× |
| add [16,128,1024] + [1024] (bias) | 27.91 | 6.02 | **4.6×** | 4.6× |
| mul [16,128,1024] × scalar | 23.70 | 4.56 | **5.2×** | 14.5× |
| exp [16,128,1024] | 104.29 | 24.55 | **4.2×** | 4.1× |
| softmax [16,128,1024], axis −1 | 267.04 | 35.11 | **7.6×** | 11.5× |
| variance [16,128,1024], axis −1 | 68.67 | 5.94 | **11.6×** | 12.9× |
| max [16,128,1024], axis −1 | 4.81 | 1.77 | **2.7×** | 1.3× |
| sum [16,128,1024], axis −1 | 3.19 | 2.85 | 1.1× | 1.1× |
| sum [16,128,1024], axis 1 | 2.25 | 2.03 | 1.1× | 1.6× |
| add, transposed view (strided, general path) | 26.09 | 24.54 | 1.1× (no regression) | n/a |

At its best, the GEMM reached **6.9 GFLOP/s** on 1024². That is in line with
laya-js's reported 5–8 GFLOP/s for the same kernel shape. The issue title
guessed 10–50×. The measured speedup is about 4–12× for elementwise ops and
fused kernels, and about 6–8× for matmul.

## Reading the numbers honestly

- **Axis reductions barely moved.** The old axis `sum` was already a tight
  strided loop, not a generator. The win only shows up for full reductions,
  which used to go through the generator, and for middle-axis reductions,
  where the fast path now streams rows.
- **The strided path is unchanged, not faster.** Transposed or stepped
  views, stride-0 `broadcastTo` views, and uncovered broadcast patterns
  (e.g. `[B,T,C] + [1,T,1]`) still use the general odometer loop.
- **`matmul` packs every time.** Even an already-contiguous operand is
  copied into an f64 panel, which costs `(m+n)·k` f64 of scratch plus an
  `m·n` f64 accumulator when the output is not f64. This is O(n²) against
  O(n³) compute, so it is negligible at the tested sizes. It does add
  allocation overhead for many tiny batched matmuls. A broadcast operand
  (e.g. a shared weight across the batch) is packed once, not once per batch.
- **No SIMD and no threads.** Going further means tensor-wasm or
  tensor-webgpu, which this change does not touch.
- **i64/u64 keep the BigInt paths** everywhere, including naive `matmul`.

## Found along the way

`broadcastShapes` used `Math.max(da, db)`, so a zero-size dim against a
size-1 dim produced 1 instead of 0: `[0, 4]` broadcast with `[4]` gave
`[1, 4]`, while NumPy gives `[0, 4]`. I fixed it in this change and added a
regression test in `tensor.test.ts`. The new fast-path test for empty tensors
is what surfaced it.
