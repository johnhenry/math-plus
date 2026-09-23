/**
 * `@johnhenry/math-plus-tensor-autograd/safetensors` (issue #123): Module
 * state dicts to/from safetensors, including both directions of PyTorch
 * interop (JS-written file -> torch `load_state_dict(strict=True)`, and a
 * torch-written f16 file -> JS f16 and f32 modules).
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test, after } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { random } from "@johnhenry/math-plus-tensor-core";
import { readSafetensors } from "@johnhenry/math-plus-safetensors";
import { Variable, nn } from "../src/index.ts";
import { loadSafetensors, loadSafetensorsInto, saveSafetensors, stateDictFromSafetensors } from "../src/safetensors.ts";
import {
  TOL,
  assertClose,
  checkCase,
  prepareCases,
  rand,
  randomizeParams,
  runTorchOracle,
  toOracle,
  torchSafetensorsSkip,
  TORCH_ST_PYTHON,
  type OracleTensor,
} from "./torch-oracle.ts";

const dir = mkdtempSync(join(tmpdir(), "autograd-st-"));
after(() => rmSync(dir, { recursive: true, force: true }));

function encoder(dtype?: nn.ParamDType): nn.TransformerEncoderLayer {
  return new nn.TransformerEncoderLayer(8, 2, { dimFeedforward: 16, normFirst: true, batchFirst: true, dtype });
}
const ENCODER_CFG = {
  dModel: 8, nhead: 2, dimFeedforward: 16, eps: 1e-5, activation: "relu", batchFirst: true, normFirst: true, bias: true,
};

test("saveSafetensors -> stateDictFromSafetensors -> loadStateDict reproduces the module exactly", () => {
  const src = encoder();
  randomizeParams(src, 1);
  const bytes = saveSafetensors(src);
  assert.equal(readSafetensors(bytes).metadata.format, "pt");
  const dst = encoder();
  dst.loadStateDict(stateDictFromSafetensors(bytes));
  const x = Variable.constant(rand([2, 3, 8], "f32", 2));
  assert.deepEqual(dst.forward(x).value.toArray(), src.forward(x).value.toArray());
});

test("f16 / bf16 parameters are stored as F16 / BF16 and load back bit-exactly", () => {
  for (const [dtype, st] of [["f16", "F16"], ["bf16", "BF16"]] as const) {
    const m = new nn.Linear(4, 3, { dtype, rng: random.seed(3) });
    const bytes = saveSafetensors(m);
    assert.equal(readSafetensors(bytes).info("weight").dtype, st);
    const back = stateDictFromSafetensors(bytes);
    assert.equal(back.weight?.dtype, dtype);
    assert.deepEqual([...(back.weight?.contiguous().data ?? [])], [...m.weight.value.data]);
  }
});

test("prefix selects and strips a sub-model's keys", () => {
  const bytes = saveSafetensors({ ...Object.fromEntries(Object.entries(new nn.Linear(2, 2).stateDict()).map(([k, v]) => [`enc.${k}`, v])), "head.weight": new nn.Linear(2, 1).weight.value });
  assert.deepEqual(Object.keys(stateDictFromSafetensors(bytes, { prefix: "enc." })).sort(), ["bias", "weight"]);
});

test("loadSafetensors / loadSafetensorsInto read lazily from a file path", async () => {
  const src = encoder();
  randomizeParams(src, 4);
  const path = join(dir, "enc.safetensors");
  writeFileSync(path, saveSafetensors(src));
  assert.deepEqual(Object.keys(await loadSafetensors(path, { prefix: "self_attn." })).sort(), [
    "in_proj_bias", "in_proj_weight", "out_proj.bias", "out_proj.weight",
  ]);
  const dst = encoder("f16"); // f32 file into f16 storage: cast on load
  await loadSafetensorsInto(dst, path);
  assert.equal(dst.linear1.weight.dtype, "f16");
  assert.deepEqual(dst.linear1.weight.value.toArray(), src.linear1.weight.value.cast("f16").toArray());
});

test("PyTorch parity: a JS-written safetensors file loads into torch's TransformerEncoderLayer (strict) and matches forward + backward", { skip: torchSafetensorsSkip }, () => {
  const m = encoder("f64");
  randomizeParams(m, 5);
  const path = join(dir, "js-written.safetensors");
  writeFileSync(path, saveSafetensors(m));
  const { prepared, requests } = prepareCases([
    { id: "st-js-to-torch", kind: "encoderLayer", dtype: "f64", config: ENCODER_CFG, module: m,
      inputs: { x: rand([2, 3, 8], "f64", 6) }, forward: ({ x }) => m.forward(x!) },
  ]);
  const req = requests[0] as Record<string, unknown>;
  delete req.state; // torch must get the weights from the FILE, not from the JSON side channel
  req.stateFile = path;
  const results = runTorchOracle(requests, TORCH_ST_PYTHON);
  checkCase(prepared.get("st-js-to-torch")!, results["st-js-to-torch"]);
});

test("PyTorch parity: a torch-written f16 safetensors checkpoint drives JS f16-storage and f32 modules to torch's output", { skip: torchSafetensorsSkip }, async () => {
  const path = join(dir, "torch-written.safetensors");
  const x = rand([2, 4, 8], "f32", 7);
  const gradOut = random.normal([2, 4, 8], { dtype: "f32", rng: random.seed(8) });
  const results = runTorchOracle(
    [{ id: "st-torch-to-js", kind: "encoderLayer", dtype: "f32", config: ENCODER_CFG, inputs: { x: toOracle(x) }, masks: {},
       gradOut: toOracle(gradOut), exportState: path }],
    TORCH_ST_PYTHON,
  );
  const r = results["st-torch-to-js"]!;
  assert.equal(r.error, undefined, r.error);
  assert.equal(readSafetensors(new Uint8Array(await (await import("node:fs/promises")).readFile(path))).info("linear1.weight").dtype, "F16");

  for (const dtype of ["f16", "f32"] as const) {
    const m = encoder(dtype);
    await loadSafetensorsInto(m, path);
    const xv = Variable.variable(x);
    const out = m.forward(xv);
    assertClose(out.value, r.out as OracleTensor, TOL.f32, `${dtype} module forward`);
    if (dtype === "f32") {
      out.backward(gradOut);
      assertClose(xv.grad!, r.grads!["input:x"]!, TOL.f32, "input grad");
      for (const [name, p] of Object.entries(m.namedParameters())) {
        assertClose(p.grad!, r.grads![name]!, TOL.f32, `grad ${name}`);
      }
    }
  }
});
