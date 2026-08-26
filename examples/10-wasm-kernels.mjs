// tensor-wasm: Rust->WASM kernels over resident buffers with the
// zero-allocation ...Into interface.
//
// ENV-DEPENDENT — excluded from the `npm run examples` CI loop: it needs
// the built .wasm artifact, which is gitignored. In a clone:
//   rustup target add wasm32-unknown-unknown   (plus lld)
//   npm run build:wasm
//   node examples/10-wasm-kernels.mjs
// (The published npm package ships the .wasm prebuilt.)
import assert from "node:assert/strict";
import { Kernels } from "@johnhenry/math-plus-tensor-wasm";

const kernels = await Kernels.load(); // feature-detects SIMD via WebAssembly.validate()
console.log("SIMD available:", kernels.simdAvailable);

const a = kernels.fromArray(new Float32Array([1, 2, 3, 4]), [4]);
const b = kernels.fromArray(new Float32Array([10, 20, 30, 40]), [4]);
const out = kernels.zeros([4]);

// The whole point of ...Into: repeated calls over resident buffers
// allocate NOTHING (the alloc-and-copy-per-call wrapper design measured as
// a 2.27x REGRESSION vs pure JS; residency is what wins the 1.78x).
const before = kernels.allocCallCount;
for (let i = 0; i < 500; i++) kernels.addInto(out, a, b);
assert.equal(kernels.allocCallCount, before);
console.log([...out.toFloat32Array()]); // [11, 22, 33, 44]

// Stride-aware matmul: a transposed view multiplies without a copy
const m = kernels.fromArray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
const n = kernels.fromArray(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]);
const p = kernels.zeros([2, 2]);
kernels.matmulInto(p, m, n);
console.log([...p.toFloat32Array()]); // [58, 64, 139, 154]

// Manual memory management — WasmTensors are not GC'd
for (const t of [a, b, out, m, n, p]) t.free();
