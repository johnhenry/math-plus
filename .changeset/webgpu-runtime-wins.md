---
"@johnhenry/math-plus-tensor-webgpu": minor
---

WebGPU runtime improvements ported from laya-js (issue #126); measurements are in `docs/spikes/webgpu-runtime.md`.

- New `runAttention(device, q, k, v, { mask, scale, skipMaskedTiles, kernel })`: fused flash-style attention in one dispatch, with an optional f32 mask (nonzero = attend) that broadcasts against `(batch, seqQ, seqK)`. Key tiles that no query can see are skipped; a ±64 sliding window at B=16, L=512, D=64 runs about 1.5-2x faster on an Apple M2. Two kernels: `fast` for head dim 32/64 and `generic` for any head dim. `planAttention` and the WGSL generators are exported. `scale` defaults to `1/sqrt(dim)`. Query rows where every key is masked return 0.
- Dispatches now go through `dispatchKernel`. Bind-group layouts are parsed from each kernel's WGSL, bind groups are cached, and uniforms are written to a per-device ring buffer and bound with dynamic offsets. The cache key includes the uniform size (regression-tested).
- Readbacks no longer busy-poll under Dawn. When there is no `navigator.gpu` and the expected wait is over 15 ms, `readBackBytes` sleeps for most of the wait before polling. Configure this with `configureGPURuntime(device, { sleepWhileWaiting, sleepThresholdMs })`.
- Every upload goes through the new `writeBytes`, which always passes `(arrayBuffer, byteOffset, byteLength)`. Bun's Dawn binding ignores a view's `byteOffset` if you pass the view itself.
- New GPU timestamp profiler, `startProfiling` / `stopProfiling`. It needs `timestamp-query`, which the new `detectWebGPU({ timestampQuery: true })` option requests.
- `detectWebGPU()` also requests the adapter's maximum `maxComputeWorkgroupStorageSize`.
- New `gpuRuntimeStats`, `getKernel`, `getKernelChecked` and `parseWGSLBindings`. `getOrCreateComputePipeline` now returns pipelines with explicit layouts for kernels whose bindings it can parse.
