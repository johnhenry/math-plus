---
"@johnhenry/math-plus-tensor-webgpu": patch
---

Moves to `@johnhenry/backend-webgpu` 0.3.1's documented runtime hooks (dependency `^0.3.1`).

- **Fusion broadcasts.** `gpu.fuse` / `gpu.compile` run on the backend's `elementwise` hook, so inputs broadcast with NumPy's rules, as in tensor-compile's CPU `forward` (for example `[B, N]` with `[N]` or `[B, 1]`). Before, every input had to have the same shape. It is still one dispatch per expression, and inputs are still f32. New export: `compileIRToElementwise(node, n)`, the lowering as the hook's expression and helpers. `compileIRToKernel` stays but `fuse` no longer uses it.
- **Attention is fused on every device.** backend-webgpu 0.3.1 fits `sdpa` to the device's workgroup-memory limit, so `runAttention` no longer composes attention from matmul and softmax on devices below 32 KiB, such as one with the 16 KiB WebGPU default.
- **Readback sleep defaults to a 15 ms threshold.** It was off. Backends created here now follow backend-webgpu's default (sleep under Dawn, not for `navigator.gpu`) with `sleepThresholdMs: 15`. Readbacks of a few milliseconds keep polling at full speed, and waits over 15 ms use about 3× less CPU (measured in `docs/spikes/webgpu-runtime.md`). `configureGPURuntime` takes `sleepThresholdMs` again.
- `createWebGpuDevice({ device, adapter })`: a device you pass in goes through `createWebGpuBackend`, so subgroup-matrix GEMM is detected from `adapter`, or from the adapter `detectWebGPU()` used.
- `GPUTensor.fromBuffer` wraps the buffer with the backend's `wrapBuffer`, so it is never pooled. f16 now needs a device with `shader-f16`, because without it backend-webgpu stores f16 as f32.
- `src/bridge.ts` no longer uses undocumented backend internals, except for one call: the synchronous `backendFor(device)` of the deprecated API still calls the `WebGpuBackend` constructor.

This is a patch release because the deprecated 0.1 API is scheduled for removal in the first minor release after these hooks shipped (see 0.2.0). That removal is a separate change.
