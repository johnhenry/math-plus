# @johnhenry/math-plus-tensor-core

Typed n-dimensional arrays for JS/TypeScript: dtypes, strides and views,
NumPy-style broadcasting, `.npy` I/O, seeded RNG. Pure JS/TypedArray
execution, zero dependencies — the root of the math-plus dependency graph.
Start here.

Architectural rules, stated up front: no Proxy-based indexing; views and
contiguous tensors are semantically distinct (`permute`/`transpose`/`reshape`
never copy, `contiguous()` copies iff needed); no implicit copies; **no
implicit dtype promotion**.

## Install

```bash
npm install @johnhenry/math-plus-tensor-core
```

## Quick start

```js
import { Tensor, random } from "@johnhenry/math-plus-tensor-core";

const a = Tensor.from([1, 2, 3, 4, 5, 6], { dtype: "f64" }).reshape([2, 3]);
const b = Tensor.from([7, 8, 9, 10, 11, 12], { dtype: "f64" }).reshape([3, 2]);
a.matmul(b); // [2,2], NumPy semantics (batched broadcast, 1-D squeeze rules)

// Slices are VIEWS — shared storage, NumPy basic-slicing semantics
const t = Tensor.arange(12).reshape([3, 4]);
const s = t.slice({ start: 1, end: 3 }, { start: 1, end: 3 });
t.data === s.data; // true — compare identity to detect views

// Seeded, reproducible RNG
const r1 = random.uniform([5], { rng: random.seed(42) });
const r2 = random.uniform([5], { rng: random.seed(42) }); // identical
```

## API surface

- **Creation:** `zeros`, `ones`, `full`, `arange`, `from`, `fromTypedArray`
  (wraps **without copying**), `concat`, `stack`, `where`, `fromNpy`.
- **Views (never copy):** `reshape` (-1 inference), `permute`, `transpose`,
  `squeeze`/`unsqueeze`, `broadcastTo` (stride-0), `slice`, `select`,
  `unfold` (sliding-window patches).
- **Copies:** `contiguous`, `take`, `gather`, `mask`, `cast`, `pad`,
  `split`, `repeat`, `flip`, `roll`, `nonzero`, `clip`, `flatten`.
- **Math:** `add/sub/mul/div` (tensor or scalar), full unary set (`sqrt`,
  `exp`, `log*`, trig, hyperbolic, `relu`/`sigmoid`/`gelu`/`softmax`),
  `matmul`, `dot`, comparisons/logic, reductions (`sum mean min max
  argmin argmax variance std prod cumsum cumprod sort argsort topK`).
- **I/O:** `toNpy()` / `Tensor.fromNpy(bytes)` — NPY v1.0.
- **Random:** `random.seed`, `random.uniform`, `random.normal`,
  `random.randint`; plus `broadcastShapes`, `allocate`, `BYTES_PER_ELEMENT`,
  `isBigIntDType`.

## Traps

- **Default dtype is `f32`** for `zeros`/`ones`/`full`/`arange`/`from`
  (`random.randint` defaults to `i32`). Most numerical work here wants an
  explicit `{ dtype: "f64" }`.
- **No implicit promotion:** mixing dtypes in any binary op, `matmul`, or a
  comparison throws `TypeError` — `cast()` first. `div` on `i64` throws too
  (NumPy true-division returns f64; cast first). Exception: `mean`/
  `variance`/`std` of integer dtypes return `f64` (NumPy semantics); `sum`
  keeps the dtype.
- **`cast()` always copies** (never aliases, even for a same-dtype cast) and
  integer conversion **truncates toward zero**, not rounds.
- **`fromTypedArray` does not copy** — aliasing is your problem.
- **`.npy` scope:** little-endian, C-order only (`fortran_order: True`
  throws); `f16`/`bf16` have no `.npy` representation and throw. A
  non-contiguous view serializes packed.
- `f16`/`bf16` are declared in the `DType` union but stored as `Uint16Array`
  until `Float16Array` is universal.
- `shape` is frozen; an `Rng`'s state advances between calls (not reset).

## Tests

`npm test`. Differential tests against a NumPy oracle skip unless a Python
with NumPy is found (`MATH_PLUS_ORACLE_PYTHON`); CI verifies the oracle is
importable so they can't silently skip.

## Provenance

Built across issues #1 (indexing/slicing), #2 (matmul), #4 (concat/stack/
where), #5 (random), #64/#65 (op-table parity with the compiled IR), #84
(unfold). Part of the [math-plus](https://github.com/johnhenry/math-plus)
monorepo; family docs at <https://opensource.johnhenry.me/math/>.
