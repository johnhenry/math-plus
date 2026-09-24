/**
 * The host boundary: tensor-core `Tensor` <-> tensor-backend `HostTensor`.
 *
 * The implementation lives in `@johnhenry/math-plus-tensor-cpu` (host.ts),
 * shared with `@johnhenry/math-plus-tensor-webgpu` (AGENTS.md's
 * canonical-implementation rule). This module only labels its errors
 * `tensor-mlx:` and keeps this package's exports unchanged.
 *
 * Neither direction copies element data; non-contiguous tensors are
 * rejected instead of being packed silently — call `.contiguous()` first.
 */
import type { Tensor } from "@johnhenry/math-plus-tensor-core";
import { hostFromTensor as sharedHostFromTensor } from "@johnhenry/math-plus-tensor-cpu";
import type { HostTensor } from "@johnhenry/tensor-backend";

export { DEVICE_DTYPES, isDeviceDType, tensorFromHost, type DeviceDType } from "@johnhenry/math-plus-tensor-cpu";

/**
 * A zero-copy `HostTensor` view of a tensor-core `Tensor`'s storage.
 * f16/bf16 storage is the raw IEEE-754 bit pattern in a `Uint16Array`
 * (tensor-core's layout); f16 is re-viewed as a `Float16Array`.
 */
export function hostFromTensor(t: Tensor): HostTensor {
  return sharedHostFromTensor(t, "tensor-mlx");
}
