/**
 * @johnhenry/math-plus-tensor-mlx — EXPERIMENTAL native Apple Silicon arrays
 * for Math Plus (issue #125), the prototype of RFC 0001's recommended
 * direction (docs/rfcs/0001-device-backends.md): a device package built on
 * the `@johnhenry/tensor-backend` contract, with tensor-core's `Tensor`
 * staying the host type and every transfer explicit.
 *
 *   const mlx = createMlxDevice();                  // no global default device
 *   const x = mlx.fromTensor(Tensor.from([1, 2, 3])); // explicit upload (one copy)
 *   const y = x.mul(2).softmax();                    // lazy MLX graph
 *   const t = await y.toTensor();                    // explicit download (evaluates)
 *
 * See README.md for the op list, dtype rules and limitations.
 */
export { createMlxDevice, mlxUnavailableReason, MlxArray, MlxDevice, type MlxDeviceOptions } from "./device.ts";
export { DEVICE_DTYPES, hostFromTensor, isDeviceDType, tensorFromHost, type DeviceDType } from "./convert.ts";
export type { HostTensor } from "@johnhenry/tensor-backend";
