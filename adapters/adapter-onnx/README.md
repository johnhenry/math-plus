# @johnhenry/math-plus-adapter-onnx

Run ONNX models with math-plus tensors: `load(modelSource)` /
`model.run(inputs)`, marshalling between `@johnhenry/math-plus-tensor-core`'s
`Tensor` and ONNX Runtime Web's tensor on the way in and out. That
marshalling is the entire job — inference itself is ONNX Runtime Web,
untouched.

## Install

```bash
npm install @johnhenry/math-plus-adapter-onnx
```

Depends on `onnxruntime-web` and `@johnhenry/math-plus-tensor-core`. Works in
Node and the browser (wherever ONNX Runtime Web runs).

## Quick start

```js
import { load } from "@johnhenry/math-plus-adapter-onnx";
import { Tensor } from "@johnhenry/math-plus-tensor-core";

// modelSource: a file path/URL string, or raw bytes (Uint8Array/ArrayBuffer)
const model = await load("model.onnx");
model.inputNames;  // ["a", "b"]
model.outputNames; // ["sum"]

const outputs = await model.run({
  a: Tensor.from([1, 2, 3], { dtype: "f32" }),
  b: Tensor.from([10, 20, 30], { dtype: "f32" }),
});
outputs.sum.toArray(); // [11, 22, 33] — a math-plus Tensor

await model.release(); // frees the ORT session; the model is dead after this
```

`i64` works too — `input_ids`-style tensors marshal as `BigInt64Array`
without conversion. An `onnx.load(...)` namespace form is also exported.

## API surface

- **`load(modelSource, options?)`** → `Promise<OnnxModel>`. `options` is a
  pass-through of ONNX Runtime Web's own `SessionOptions` — execution
  providers (wasm/webgpu/…), graph optimization level, everything. v1 adds
  nothing on top and deliberately does *not* route ORT through math-plus's
  own device abstraction.
- **`OnnxModel`:** `inputNames`, `outputNames`,
  `run(inputs: Record<string, Tensor>)` → `Record<string, Tensor>`,
  `release()`.
- **`UnsupportedDTypeError`** — thrown when a dtype has no ORT equivalent.

Every supported dtype (`bool`, `u8`–`u64`, `i8`–`i64`, `f16`, `f32`, `f64`)
is backed by the *identical* TypedArray class on both sides, so marshalling
is a metadata reshape, never an element-by-element copy. Non-contiguous
inputs (e.g. a transposed view) are packed via `.contiguous()` first;
outputs wrap ORT's buffer with no copy at all.

## Traps

- **`bf16` throws.** ONNX Runtime Web has no `bfloat16` tensor type, and
  `bf16` is *not* the same bit layout as `f16` — the adapter raises
  `UnsupportedDTypeError` rather than silently mis-mapping. (ORT's
  `string`/`uint4`/`int4` outputs are likewise unrepresentable and throw on
  the way back.)
- **`f16` is bit storage, not numbers.** Both sides store `float16` as raw
  `Uint16Array` bits; neither converts to JS numbers for you.
- **Backend selection is ORT's, not ours.** If you need a specific
  execution provider, set it in `options.executionProviders` — nothing in
  math-plus's device/backend machinery applies here.
- **Modes 2 and 3 of the design are deliberately unbuilt** — no shared
  storage with ORT, no importing ONNX graphs into math-plus's own runtime.
  The scope is `load`/`run` marshalling, by design.

Part of the [math-plus](https://github.com/johnhenry/math-plus) family —
docs at [opensource.johnhenry.me/math/](https://opensource.johnhenry.me/math/).
