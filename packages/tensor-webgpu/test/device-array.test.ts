/**
 * The shared DeviceArray suite (tensor-cpu's test/device-array-suite.ts,
 * the one behavioural suite of the chainable array API every math-plus
 * device shares) over `createWebGpuDevice()`, in-process on Dawn (Node and
 * Bun): f32/bf16/i32/bool, and f16 where the adapter has `shader-f16`,
 * against the NumPy oracle. Arrays of a CPU device are the "other device"
 * it must refuse. Skips (never fails) without a WebGPU adapter or a python3
 * that imports numpy.
 */
import { createCpuDevice } from "@johnhenry/math-plus-tensor-cpu";
import { createWebGpuDevice, webGpuUnavailableReason, type WebGpuDevice } from "../src/index.ts";
import { deviceArraySuite } from "../../tensor-cpu/test/device-array-suite.ts";
import { testFns } from "../../tensor-cpu/test/helpers.ts";
import { makeTest } from "../../../test/harness.ts";

// Each file imports bun:test itself (see test/harness.ts).
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const harness = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);

const skip = await webGpuUnavailableReason();
const gpu: WebGpuDevice | null = skip ? null : await createWebGpuDevice();
harness.after(() => gpu?.destroy());

deviceArraySuite(testFns(harness), { label: "tensor-webgpu", skip, device: gpu, other: () => createCpuDevice() });
