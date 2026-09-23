# @johnhenry/math-plus-tensor-core

[![npm version](https://img.shields.io/npm/v/%40johnhenry%2Fmath-plus-tensor-core.svg)](https://www.npmjs.com/package/@johnhenry/math-plus-tensor-core)
[![license](https://img.shields.io/npm/l/%40johnhenry%2Fmath-plus-tensor-core.svg)](../../LICENSE)

Typed n-dimensional arrays for JS/TypeScript: dtypes, strides and views,
NumPy-style broadcasting, `.npy` I/O, seeded RNG. Pure JS/TypedArray
execution, zero dependencies — the root of the math-plus dependency graph.
Start here.

**Need more speed?** This package is pure-JS `Tensor`, not accelerated, and
there's no `setBackend` switch here -- reaching for acceleration means
explicitly bringing in a sibling package and its own API surface:
[`@johnhenry/math-plus-tensor-wasm`](https://github.com/johnhenry/math-plus/tree/main/packages/tensor-wasm)
(Rust→WASM kernels over a separate `WasmTensor` storage type -- f32,
1-D/2-D ops, manual `free()` -- measured 1.78x faster than JS at N=1e6 for
resident buffers), or
[`@johnhenry/math-plus-tensor-webgpu`](https://github.com/johnhenry/math-plus/tree/main/packages/tensor-webgpu)
(WebGPU GEMM/attention primitives, Chromium-only in v1 -- as of writing its
own measurements say to reach for tensor-wasm instead at every size tested,
see its "honest threshold" section).

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
  `exp`, `log*`, trig, hyperbolic, `erf`/`erfc`, `relu`/`sigmoid`/`gelu`/`softmax`),
  `matmul`, `dot`, comparisons/logic, reductions (`sum mean min max
  argmin argmax variance std prod cumsum cumprod sort argsort topK`).
- **I/O:** `toNpy()` / `Tensor.fromNpy(bytes)` — NPY v1.0.
- **Special functions (scalar):** `erf`, `erfc`, `gelu(x, approximate)`,
  `geluErf`, `geluTanh`, `geluDerivative` — the monorepo's ONE canonical
  double-precision erf (`src/special.ts`, ~1e-15 relative, SciPy-verified).
  tensor-compile's IR and tensor-webgpu's WGSL `erf` derive from it; don't
  add another.
- **Random:** `random.seed`, `random.uniform`, `random.normal`,
  `random.randint`; plus `broadcastShapes`, `allocate`, `BYTES_PER_ELEMENT`,
  `isBigIntDType`.

## Traps

- **`gelu()` defaults to EXACT erf-GELU** (`approximate: "none"`, like
  PyTorch's `nn.GELU()`), since issue #122. Earlier versions always used the
  tanh approximation; pass `gelu({ approximate: "tanh" })` for those numbers
  (they differ by up to ~4.7e-4).
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
  throws); `f16` round-trips as `<f2`. A non-contiguous view serializes packed.
- **`.npy` bf16 follows the `ml_dtypes` convention.** NumPy has no bfloat16.
  `np.save` of an `ml_dtypes.bfloat16` array writes descr `'<V2'` (an untyped
  2-byte void) plus the raw bits, and Python reads it back with
  `np.load(f).view(ml_dtypes.bfloat16)`. `toNpy()` on a bf16 tensor writes
  exactly those bytes; they are byte-identical to `np.save` (oracle-tested).
  Reading one needs `Tensor.fromNpy(bytes, { voidAs: "bf16" })`, the JS
  equivalent of that `.view()`. Without the option a `'<V2'`/`'|V2'` file
  throws, because the file itself does not say it is bfloat16.
- **`f16`/`bf16` are storage dtypes.** Elements are raw IEEE binary16 /
  bfloat16 bit patterns in a `Uint16Array` (the layout safetensors, ONNX
  Runtime and WebGPU use — zero-copy across those boundaries). Values
  cross the boundary correctly: `from`/`full`/`arange`/`random.*` encode
  (round-to-nearest-even, direct from the double), `at`/`item`/`toArray`
  decode, and `cast()` converts values both ways (bit-for-bit equal to
  NumPy's `astype(float16)`; bf16 equal to the standard f32→bf16 RNE).
  Structural ops (views, `contiguous`, `concat`/`stack`/`take`/`flip`/
  `pad`/`where`/`mask`…) just move bits and work. **Arithmetic, comparison,
  reduction, sort and matmul kernels throw** `TypeError` on half dtypes.
  Computing internally in f32 would be implicit promotion, which this
  package forbids; a fused f16 kernel is a WASM/WebGPU concern. `.data` of a
  half tensor is bits, not values. There are two explicit ways to compute:
  - `cast("f32")`, compute, then `cast("f16")` back, by hand.
  - `withCompute("f32", [a, b], (a, b) => a.matmul(b).relu())`, which is
    the opt-in form of the same thing. It casts the f16/bf16 inputs to the
    named compute dtype (`"f32"` or `"f64"`) and passes other inputs through
    unchanged. It runs `fn`, then casts every compute-dtype result back to
    the inputs' half dtype; a bool or index result is returned as it is.
    **Rounding happens once, when the region exits.** A single op is
    bit-identical to NumPy's own float16 ufunc. A chain equals NumPy's
    `f(a.astype(float32)).astype(float16)`, not a per-op f16 chain. Mixed
    f16/bf16 inputs, or no half input at all, throw. All of this is
    oracle-tested against NumPy and, for bf16, `ml_dtypes`.
  The codec is exported as `encodeHalf`/`decodeHalf`/`isHalfDType`
  — the one f16/bf16 implementation in Math Plus.
- `shape` is frozen; an `Rng`'s state advances between calls (not reset).

## Performance: fast paths and their limits

Contiguous inputs (C-order, any storage offset) take flat typed-array
kernels; everything else falls back to the general strided path. Both
produce **bit-identical** results (`test/fast-paths.test.ts` asserts it),
so which path ran is never observable except in time. Numbers:
[`docs/spikes/tensor-core-fast-paths.md`](../../docs/spikes/tensor-core-fast-paths.md).

- **Fast:** `add/sub/mul/div` when both sides are full-shape, one side is
  a single element, or one side is broadcast as a trailing block
  (`[B,T,C] + [C]`) or per row (`[B,T,C] - [B,T,1]`); unary math ops;
  `cast`; `contiguous`; comparisons (same shape or scalar); `sum`/`mean`/
  `min`/`max` (full or any axis); fused `softmax` and `variance`/`std`
  (f32/f64). `matmul` always packs operands into f64 panels and runs a
  4×4 register-blocked GEMM (strided operands included).
- **Not fast-pathed:** strided views (transposes, stepped/negative slices,
  stride-0 `broadcastTo` views, other broadcast patterns such as
  `[B,T,C] + [1,T,1]`), `i64`/`u64` (BigInt), `argmin`/`argmax`,
  cumulative scans, sorting, `where`, logical ops. These are correct, just
  slower.
- Single-threaded, no SIMD. `matmul` allocates up to `(m+n)·k + m·n` f64
  scratch per call. For more, see tensor-wasm above.

## Tests

`npm test`. Differential tests against a NumPy oracle skip unless a Python
with NumPy is found (`MATH_PLUS_ORACLE_PYTHON`); CI verifies the oracle is
importable so they can't silently skip. The bf16 oracle tests in
`test/half-compute.test.ts` also need `ml_dtypes`. They use the first
interpreter that can import it, falling back to
`uv run --with numpy --with ml_dtypes python` when `uv` is installed, and
skip otherwise.

## Provenance

Built across issues #1 (indexing/slicing), #2 (matmul), #4 (concat/stack/
where), #5 (random), #64/#65 (op-table parity with the compiled IR), #84
(unfold), #120 (contiguous fast paths, blocked GEMM). Part of the [math-plus](https://github.com/johnhenry/math-plus)
monorepo; family docs at <https://opensource.johnhenry.me/math/>.
