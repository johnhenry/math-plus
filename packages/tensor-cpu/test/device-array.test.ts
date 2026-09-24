/**
 * The shared DeviceArray suite (test/device-array-suite.ts) over this
 * package's CPU device facade, `createCpuDevice()`: f32/i32/bool (the CPU
 * backend refuses f16/bf16 instead of widening them), NumPy oracle. Pure
 * TypeScript: only the oracle can skip.
 */
import assert from "node:assert/strict";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { createCpuDevice, CpuDevice, DeviceArray } from "../src/index.ts";
import { deviceArraySuite } from "./device-array-suite.ts";
import { testFns } from "./helpers.ts";
import { makeTest } from "../../../test/harness.ts";

// Each file imports bun:test itself (see test/harness.ts).
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const harness = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
const fns = testFns(harness);

deviceArraySuite(fns, { label: "tensor-cpu", skip: null, device: createCpuDevice(), other: () => createCpuDevice() });

fns.describe("tensor-cpu CpuDevice", () => {
  fns.it("is a DeviceArray facade over its own CpuBackend, with tensor-cpu-labelled errors", async () => {
    const cpu = createCpuDevice();
    assert.ok(cpu instanceof CpuDevice);
    assert.equal(cpu.name, "cpu");
    assert.equal(cpu.backend.name, "cpu");
    assert.notEqual(createCpuDevice().backend, cpu.backend, "no shared global backend");
    const x = await cpu.fromTensor(Tensor.from([1, 2, 3]));
    assert.ok(x instanceof DeviceArray);
    assert.equal(x.device, cpu);
    assert.throws(() => x.add(Tensor.from([1]) as never), /^TypeError: tensor-cpu add: expected a CpuArray, got a tensor-core Tensor/);
    const y = await createCpuDevice().fromTensor(Tensor.from([1, 2, 3]));
    assert.throws(() => x.add(y), /tensor-cpu add: array belongs to a different CpuDevice/);
    assert.throws(() => cpu.fromTensor(Tensor.from([1], { dtype: "f16" })), /tensor-cpu fromHost: this cpu device does not support f16; cast .* explicitly first/);
    cpu.destroy();
  });
});
