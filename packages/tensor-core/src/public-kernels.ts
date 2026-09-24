/**
 * `@johnhenry/math-plus-tensor-core/kernels` — the flat typed-array kernels
 * under tensor-core's fast paths (kernels.ts) and the fused NN kernels
 * (nn-kernels.ts), for device packages that store tensors as flat typed
 * arrays (issue #144: the CPU `Backend` in @johnhenry/math-plus-tensor-cpu).
 *
 * Low-level: no shape or dtype validation, contiguous row-major inputs
 * unless a kernel says otherwise. Most code wants the `Tensor` API from the
 * package entry instead.
 */
export * from "./kernels.ts";
export * from "./nn-kernels.ts";
