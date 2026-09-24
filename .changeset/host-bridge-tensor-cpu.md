---
"@johnhenry/math-plus-tensor-cpu": minor
"@johnhenry/math-plus-tensor-mlx": patch
---

The tensor-core `Tensor` <-> tensor-backend `HostTensor` bridge (`hostFromTensor(t, label?)`, `tensorFromHost(h)`, `DEVICE_DTYPES`, `isDeviceDType`) moved from tensor-mlx into `@johnhenry/math-plus-tensor-cpu`. tensor-mlx and tensor-webgpu both use it, so the mapping has one implementation (#146). tensor-mlx re-exports it unchanged, with its errors still labelled `tensor-mlx:`. The f64 hint now reads "the device has no float64" instead of "MLX on Metal has no float64".
