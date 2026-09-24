---
"@johnhenry/math-plus-tensor-cpu": patch
"@johnhenry/math-plus-tensor-mlx": patch
"@johnhenry/math-plus-tensor-webgpu": minor
---

One chainable device-array API for every math-plus device (the open follow-up of #159/#162).

- **tensor-cpu** now hosts it, next to the shared `Tensor` <-> `HostTensor` bridge: `ArrayDevice<B>` wraps any `@johnhenry/tensor-backend` `Backend`, and `DeviceArray` is the chainable array (`add` … `pow`, the unary math, comparisons and logic, `sum` … `cumsum`, `softmax`, `matmul`, `layerNorm`, `cast`, `reshape`, `transpose`, explicit async `toTensor()`/`toHost()`, `eval`, `dispose`, plus `handle` and `device.wrap(handle)` to reach the backend ops it does not wrap). The optional contract ops go through tensor-backend's compose helpers, so any backend works. New CPU device facade: `createCpuDevice()` → `CpuDevice` / `CpuArray` (f32/i32/bool; f16/bf16 are refused instead of widened silently). Additive, so a patch: `^0.2` ranges (including laya-js's `@johnhenry/backend-cpu@0.3.2`) keep matching.
- **tensor-mlx**: `MlxDevice` extends `ArrayDevice` and `MlxArray` is a `DeviceArray` subclass that adds nothing — the public API and its tests are unchanged; `supports()`, `wrap()` and `handle` are new.
- **tensor-webgpu (breaking, hence 0.4.0)**: `gpu.fromTensor()` / `gpu.fromHost()` now resolve to a chainable `WebGpuArray` (the same `DeviceArray`) instead of a raw `WebGpuTensor`, and their validation errors (non-contiguous tensor, a dtype the device does not support — f16 without `shader-f16` is now refused, not widened) throw synchronously instead of rejecting. `gpu.backend` is unchanged. Migration: raw-tensor code calls `gpu.backend.fromHost(h)`, or passes `x.handle` to `gpu.backend.*` and `gpu.wrap()`s the result. `gpu.toTensor`/`toHost`/`dispose`/`scope` and `fuse`/`compile` take arrays or raw tensors (arrays in → array out). See the README's "Changed in 0.4.0".

One behavioural suite (`packages/tensor-cpu/test/device-array-suite.ts`, NumPy oracle `scripts/device_array_oracle.py`, moved from tensor-mlx) runs over the CPU, MLX (Metal) and WebGPU (Dawn) devices in every dtype each supports.
