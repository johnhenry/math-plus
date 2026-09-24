/**
 * Test plumbing: describe/it/itUnless over the repo's runtime-neutral test
 * harness (test/harness.ts) — tensor-cpu's `testFns`, shared with the
 * DeviceArray suite this package runs (test/device-array.test.ts) — plus the
 * skip-don't-fail MLX gate and an f16 tensor builder. The NumPy oracle moved
 * to tensor-cpu with the shared suite (scripts/device_array_oracle.py).
 */
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { mlxUnavailableReason } from "../src/index.ts";

export { testFns, type TestFns } from "../../tensor-cpu/test/helpers.ts";

/** Why MLX tests cannot run here (non-darwin-arm64 or no libmlxc), or null. */
export const mlxSkip: string | null = mlxUnavailableReason();

/** A tensor-core f16 Tensor (raw IEEE bits in a Uint16Array) holding `values` rounded to f16. */
export function f16Tensor(values: readonly number[], shape: readonly number[]): Tensor {
  const half = Float16Array.from(values);
  return Tensor.fromTypedArray(new Uint16Array(half.buffer), shape, { dtype: "f16" });
}
