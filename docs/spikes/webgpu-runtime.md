# WebGPU runtime wins ported from laya-js (2026-09-23)

Issue #126. laya-js's WebGPU backend (same author) found several wins beyond its GEMM kernels.
This spike ports the ones that fit `@johnhenry/math-plus-tensor-webgpu` and measures each one on
this package's own code paths. The laya-js numbers don't carry over directly. laya-js batches
every dispatch of a forward pass into one command encoder, but this package submits once per op.
That difference explains most of the results below.

| # | laya-js win | Here | Result on an Apple M2 |
|---|---|---|---|
| 1 | Bind-group cache + packed uniforms with dynamic offsets | Ported (`dispatchKernel`) | Saves the ~7 µs `createBindGroup` per repeated dispatch, but the per-op `queue.submit` costs 50-80 µs, so op-level host time is unchanged within noise. Batching submits is the real win (≈6 µs/dispatch) and is left open |
| 2 | Strided copies (drop unit axes, merge contiguous axes, 4 elements/thread) | **Not applicable** | This package has no strided GPU copies: `GPUTensor` is contiguous-only, and `toWebGPU` rejects views |
| 3 | Flash attention that skips fully masked key tiles | Ported, plus mask support (`runAttention`) | Sliding window ±64 at B=16, L=512, D=64: 1.7-2.4 → 1.1-1.25 ms of GPU time. Fused vs the 3-primitive chain: ≈5x |
| 4 | No busy-polling on `mapAsync` under Dawn | Ported (on by default without `navigator.gpu`, 15 ms threshold) | 35 ms waits: process CPU cut 3.0x under Bun and 1.8x under Node, for about 1-2% more latency |
| 5 | `queue.writeBuffer` ignores a view's `byteOffset` under Bun | All uploads now go through `writeBytes` | Reproduced under Bun 1.2.17 + `webgpu@0.6.1`. The raw call uploads `[0,1,2,3]` for a view of `[4,5,6,7]`. The package's own paths are correct, and tests cover them |
| 6 | GPU timestamp profiler | Ported (`startProfiling` / `stopProfiling`) | Used for the attention numbers below |

## Setup

| | |
|---|---|
| Machine | MacBook Air, Apple M2 (10-core GPU), macOS 27.0, **fanless**: every cell idles 4 s first, and attention cells then run ~150 ms of the same work to ramp the GPU clock before timing |
| Dawn | `webgpu@0.6.1`, Node 26.9 and Bun 1.2.17, `allow_unsafe_apis` |
| Script | `packages/tensor-webgpu/scripts/measure-runtime.ts [dispatch\|attention\|readback]` (runs under Node or Bun; follow the `~/gpu.lock` convention) |

GPU timings on this machine move by about ±0.3 ms from run to run, depending on the GPU's clock
state. The attention tables therefore give the range across three full runs, not one number.

## 1. Bind-group cache and uniform ring

**What changed.** Every kernel's `@group(0)` bindings are parsed from its WGSL
(`parseWGSLBindings`). They become an explicit bind-group layout, shared by kernels with the same
signature, such as `rrwu`, and the uniform binding uses `hasDynamicOffset`. Uniforms go into a
64 KiB per-device ring buffer, and the dispatch binds them with a dynamic offset. The cached bind
group is then keyed only by (layout, storage buffers, ring, **uniform size**). Kernels that
declare other resources fall back to `layout: "auto"` without a uniform.

**Why the uniform size is part of the key.** Two kernels can have the same binding signature and
bind the same buffers while declaring different uniform struct sizes. If the size is left out of
the key, the second kernel reuses a bind group whose uniform range was sized for the first
kernel. The result is a validation error ("binding is too small") and a silently skipped
dispatch. laya-js hit exactly this bug. `test/runtime.test.ts` pins it with a 16-byte and a
32-byte kernel over one buffer, and the test fails when `:${size}` is removed from the key
(verified).

**Measured** (`measure-runtime.ts dispatch`, Node, 2000 dispatches of the 64³ tiled GEMM kernel
over persistent buffers; median / min of 6 interleaved rounds):

| Path | µs per dispatch (host) |
|---|---:|
| Old path (pooled 16-byte uniform buffer + `createBindGroup` + encode + submit) | 120 / 92 |
| `dispatchKernel` (ring + cached bind group + encode + submit) | 110 / 86 |
| Reference: the same 2000 dispatches in **one** pass and one submit | 5.9 |

Component micro-benchmarks from the same run: `createBindGroup` costs 6-8 µs, `writeBuffer` of
16 bytes about 0.8 µs, and encode + `submit` of one dispatch about 50 µs, which rises to 80 µs
when a `writeBuffer` is pending. At the op level, which uses a fresh output buffer per call and
so misses the cache every time, host time per call barely moves:

| Op (GPU-resident operands) | before (base branch `src/`) | after |
|---|---:|---:|
| `runGemm` 64x64x64 | 83 µs | 80 µs |
| `runQKT` 4x64x64x32 | 89 µs | 89-91 µs |

**Conclusion.** The port is correct and removes all per-dispatch allocation. It also avoids
`createBindGroup` for repeated dispatches over persistent or pooled buffers: host-array
`runGemmWGSL` hits the cache on every same-shape call, and so does the elementwise runner. It
cannot deliver laya-js's 13 → 8 µs, because that number was measured inside a batched encoder.
Here, one `queue.submit` per op costs 50-80 µs, which is 10x the whole bind-group budget.
**Open item:** add an opt-in batching scope that records many dispatches into one pass and
submits once, for ≈15x lower per-dispatch host cost. It needs laya-js's hazard rules: no
`writeBuffer` to a buffer that a pending dispatch reads, deferred `destroy()` for buffers that
pending commands still reference, and flushing before any readback or user-visible queue
operation. Those rules change the package's "every op is submitted when it returns" contract, so
the work is left for a follow-up.

## 3. Fused attention with masked-key-tile skipping

math-plus's attention consisted of three unmasked primitives (`runQKT` → `runSoftmax` →
`runWeightedSum`). The port adds `runAttention(device, q, k, v, { mask?, scale?,
skipMaskedTiles?, kernel? })`. It runs as one flash-style dispatch: an online softmax over key
tiles in workgroup memory, with no `(seqQ, seqK)` scores tensor in global memory. It supports
an optional f32 mask (nonzero means attend), broadcast against `(batch, seqQ, seqK)`. Masks can
be `[seqQ, seqK]` (causal or sliding window), `[batch, 1, seqK]` (key padding), `[seqK]`, or a
full per-element mask. Query rows where every key is masked produce 0.

The two kernels come from laya-js and are adapted to f32 and to `(batch, seq, dim)`:

- `fast` handles head dim 32 or 64: 32×16 tiles, 128 threads, vec4 loads. For D=64 it needs
  ~20 KiB of workgroup memory, which is above the 16 KiB WebGPU default. `detectWebGPU()` now
  requests the adapter's maximum `maxComputeWorkgroupStorageSize`.
- `generic` handles any head dim. Its square tiles shrink until they fit the device's limit.

With a mask, each workgroup first reduces the [lo, hi) range of keys that any of its queries can
see. It then walks only the tiles covering that range. laya-js had tile skipping only in its fast
kernel; here both kernels have it.

**Correctness.** `test/flash-attention.test.ts` checks both kernels, D ∈ {5, 32, 48, 64, 128},
and every mask shape, with skipping on and off, against a NumPy float64 oracle
(`scripts/attention_oracle.py`). The oracle includes a fully masked row, the 16 KiB default limit
(D=64 falling back to `generic`), and a forced `generic` at D=32. A second test proves that
tiles are actually skipped. It puts NaN in the V rows of tiles that no query can see. With
skipping on, the output matches the oracle. With skipping off, the NaN reaches the output,
because the zero weights multiply NaN.

**Measured** (B=16, L=512, D=64, `fast` kernel; GPU ms per forward from the timestamp
profiler, median of 15; range over 3 runs):

| Mask | skip off | skip on | speedup |
|---|---:|---:|---:|
| none (fused) | 2.2-2.6 | n/a | n/a |
| sliding window ±64 (`[L, L]`) | 1.7-2.4 | **1.11-1.25** | 1.4-2.1x |
| key padding, lengths 128..512 (`[B, 1, L]`) | 2.4 | 1.7-2.0 | 1.2-1.4x |
| causal (`[L, L]`) | 1.7-2.1 | 1.6-1.9 | ≈1.0-1.3x, not reliable |
| unmasked 3-primitive chain, for comparison | 11.1-12.9 | n/a | fused ≈5x faster |

The speedup is largest where every workgroup has little visible work, as with a sliding window.
Skipping helps less where the work is unbalanced. Under a causal mask, the last query blocks
still see every key; under padding, the full-length batch entries do. Those workgroups bound the
dispatch, and the extra mask scan is not free. laya-js's 793 → 516 ms at B=16, L=512 was for a
whole ModernBERT forward whose local layers use a sliding window, which is the favorable case.

## 4. Readback without busy-polling

`webgpu` (Dawn's Node binding) resolves `mapAsync` by polling. Under Bun, polling uses a full
core for the whole GPU wait (108% process CPU/wall below). When sleeping is enabled,
`readBackBytes` first sleeps (`setTimeout`) for 80% of the last measured wait for the same
work, then polls for the rest. "The same work" is a hash of every dispatch since the previous
readback: kernel, grid and uniforms. Work that hasn't been seen before is polled as before.

The port fixes one thing in laya-js's heuristic. laya-js recorded the measured wait even after
oversleeping, so timer lateness fed back into the estimate and pushed latency up: +60% on a 9 ms
readback under Node in the first measurement here. Now, when the map is already done on waking,
the sleep target is recorded instead. An overestimate therefore shrinks 20% per readback, as
laya-js intended.

**Measured** (`measure-runtime.ts readback`, 30 iterations of N × `runGemm` 2048³ + one 16-byte
readback, threshold lowered to 3 ms so that both sizes sleep; two rounds each):

| Runtime | Work per readback | polling: latency / CPU per iter / CPU÷wall | sleeping: latency / CPU per iter / CPU÷wall |
|---|---|---|---|
| Bun | 4 GEMMs (~35 ms) | 34.9 ms / 37.7 ms / 108% | 35.3-35.4 ms / **12.5-12.8 ms** / 35% |
| Node | 4 GEMMs (~35 ms) | 34.7-34.9 ms / 12.5-13.0 ms / 36% | 35.3-35.6 ms / **6.9-7.1 ms** / 19% |
| Bun | 1 GEMM (~9 ms) | 9.5 ms / 10.1 ms / 108% | 11.7-17.2 ms / 4.5-6.8 ms / 38% |
| Node | 1 GEMM (~9 ms) | 9.4 ms / 3.6-3.8 ms / 38% | 13.8-14.4 ms / 3.8-4.5 ms / 28% |

For waits of ~35 ms, sleeping cuts process CPU 3.0x under Bun and 1.8x under Node, for
+0.4-0.9 ms of latency. For waits of ~9 ms, the poll after waking resolves later than a
continuous poll would, so latency rises 25-80% for a smaller saving. **Defaults:** sleeping is
on only when there is no `navigator.gpu` (a native binding such as Dawn), and only when the
expected wait is over **15 ms**. `configureGPURuntime(device, { sleepWhileWaiting,
sleepThresholdMs })` overrides either default. Browsers don't busy-poll, so sleeping there would
only add latency.

## 5. The `writeBuffer` byteOffset bug

`test/bun/write-buffer-offset.bun.ts` reproduces the bug. For a `Float32Array` view at byte
offset 16, `queue.writeBuffer(buf, 0, view)` uploads the start of the underlying `ArrayBuffer`
under Bun (`[0,1,2,3]`) and the right bytes under Node (`[4,5,6,7]`). This package already
passed `(buffer, byteOffset, byteLength)` in `writePadded`. Every upload, uniforms included, now
goes through a single helper, `writeBytes`, and `test/write-buffer.test.ts` enforces that:

- a static check that `writeBytes` contains the only `queue.writeBuffer(` call in `src/`,
- round trips of offset views on the harness's adapter: f32 subarray, f16 subarray at a 2-byte
  offset, odd-length f16, and `writeBytes` at a non-zero buffer offset,
- a run of the Bun repro under `bun`. The node:test suite runs under Node, so the repro is a
  standalone script. It reports whether the binding bug is still present.

## 6. Timestamp profiler

`startProfiling(device)` brackets each dispatch's compute pass with `timestampWrites`, resolving
in batches of `capacity` dispatches (default 256). `stopProfiling(device)` returns
`{ kernel, ms, count }[]` per label, slowest first. Dispatch labels are `gemm:<kernel>`,
`attention:qkt`, `attention:fast:masked`, and so on. The device needs `timestamp-query`
(`detectWebGPU({ timestampQuery: true })`). Chrome quantizes timestamps unless it runs with
`--enable-unsafe-webgpu`. Both test harnesses on this machine expose the feature, and
`test/runtime.test.ts` checks labels and counts across resolve batches.
