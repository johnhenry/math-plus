// tensor-core basics: creation, dtypes, views vs copies, broadcasting,
// matmul, and .npy round-tripping — all pure JS, no WASM/GPU involved.
//
// Run: npm install && npm run build && node examples/01-tensor-basics.mjs
import assert from "node:assert/strict";
import { Tensor, broadcastShapes, random } from "@johnhenry/math-plus-tensor-core";

// Default dtype is f32 — most numeric work here wants an explicit f64.
const a = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
const b = Tensor.from([7, 8, 9, 10, 11, 12], { dtype: "f64" }).reshape([3, 2]);
console.log(a.matmul(b).toArray()); // [[58, 64], [139, 154]]

// Slices/permutes/reshapes are VIEWS — shared storage, never copied.
const t = Tensor.arange(12).reshape([3, 4]);
const s = t.slice({ start: 1, end: 3 }, { start: 1, end: 3 });
console.log(s.toArray()); // [[5, 6], [9, 10]]
assert.equal(t.data, s.data); // identity check is how you detect a view

// Broadcasting follows NumPy's trailing-axis rules.
console.log(broadcastShapes([2, 1, 4], [3, 1])); // [2, 3, 4]
const row = Tensor.from([10, 20, 30, 40], { dtype: "f64" });
console.log(t.cast("f64").add(row).toArray()[0]); // [10, 21, 32, 43]

// No implicit dtype promotion — mixing dtypes throws; cast() first.
assert.throws(() => t.add(row), TypeError);

// .npy round trip (little-endian, C-order, NPY v1.0)
const bytes = a.toNpy();
console.log(Tensor.fromNpy(bytes).toArray()); // [[1, 2, 3], [4, 5, 6]]

// Seeded RNG: reproducible sequences
const r1 = random.uniform([3], { rng: random.seed(42) });
const r2 = random.uniform([3], { rng: random.seed(42) });
assert.deepEqual(r1.toArray(), r2.toArray());
console.log(r1.toArray());
