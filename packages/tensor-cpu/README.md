# @johnhenry/math-plus-tensor-cpu

The CPU reference implementation of the
[`@johnhenry/tensor-backend`](https://www.npmjs.com/package/@johnhenry/tensor-backend)
`Backend` contract: the op interface that transformer inference code (for
example laya-js's ModernBERT encoder) is written against, with MLX and
WebGPU backends alongside it. Pure TypeScript, eager, f32.

[RFC 0001 §12 Q3](../../docs/rfcs/0001-device-backends.md) decided that
math-plus owns this backend (issue
[#144](https://github.com/johnhenry/math-plus/issues/144)). It is built on
tensor-core's kernels, so GEMM, softmax, LayerNorm, RoPE, attention,
GELU/erf and the reductions each have **one** implementation in math-plus.
laya-js's `@johnhenry/backend-cpu` is to become a thin re-export of this
package, then be deprecated.

## Install

```bash
npm install @johnhenry/math-plus-tensor-cpu @johnhenry/tensor-backend
bun add @johnhenry/math-plus-tensor-cpu @johnhenry/tensor-backend
```

It uses no Node-only APIs, and its only dependencies are
`@johnhenry/math-plus-tensor-core` and `@johnhenry/tensor-backend`.

## Quick start

```ts
import { host, toF32 } from "@johnhenry/tensor-backend";
import { createCpuBackend } from "@johnhenry/math-plus-tensor-cpu";

const cpu = createCpuBackend();                       // no global default backend
const x = await cpu.fromHost(host("f32", [2, 3], [1, 2, 3, 4, 5, 6]));
const w = await cpu.fromHost(host("f32", [4, 3], [/* 12 values, PyTorch [out, in] */ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]));

const y = cpu.scope(() => cpu.softmax(cpu.linear(x, w), -1)); // intermediates are freed
console.log(toF32(await cpu.read(y)));
cpu.dispose(y);
```

To move data to or from a tensor-core `Tensor`, use the contract's
`toMathPlusArgs(host)` with `Tensor.fromTypedArray`, or build a `HostTensor`
from `tensor.data` (contiguous f32/i32/bool tensors).

## API

- `createCpuBackend(): CpuBackend` creates an independent backend. There is
  no shared global state.
- `CpuBackend` is `Backend<CpuTensor>` with **every** optional op present:
  `geglu`, `meanPool`, `flush`, `destroy`, and all the general-numerics ops
  (`equal` … `greaterEqual`, `logicalAnd/Or/Not`, `sqrt`, `rsqrt`, `pow`,
  `neg`, `abs`, `tanh`, `sigmoid`, `erf`, `argmax`, `argmin`, `mean`, `min`,
  `cumsum`). Only `compile` is absent (it is an MLX graph feature).
- `CpuTensor` has `shape`, `dtype`, `data` (row-major storage, possibly
  shared with other tensors, so never mutate it; it throws after dispose)
  and `disposed`.
- `erf`, `erfc` and `geluScalar` are the canonical double-precision scalar
  functions from [`@johnhenry/math-plus-special`](../special), which the
  backend's `erf`/`gelu`/`geglu` use.

Where each op's computation lives:

| Backend ops | tensor-core |
|---|---|
| `linear` | `linearNT`: packed f64 panels + `gemmNT`, the GEMM `Tensor.matmul` uses |
| `matmul` | `Tensor.matmul` itself (batched, broadcasting) |
| `sdpa` | `attention`: two `gemmNT` products per head, grouped-query heads, f64 softmax |
| `softmax` / `sum` `mean` / `max` `min` / `argmax` `argmin` / `cumsum` / `sort` | `softmaxAxis` / `sumAxis` / `extremumAxis` / `argExtremumAxis` / `cumsumAxis` / `sortAxis` |
| `layerNorm`, `rope`, `geglu`, `meanPool`, `embedding` + `gatherRows` | `layerNormRows`, `ropeHalf`, `gegluRows`, `maskedMeanPool`, `takeRows` |
| elementwise, comparisons, `where` (numpy broadcasting) | `binaryFlat` / `binaryStrided` / `compareStrided` / `whereStrided` over `rowOffsets` |
| `exp` `log` `relu` `gelu` `sqrt` `rsqrt` `tanh` `sigmoid` `erf` `neg` `abs` | `unaryFlat` (erf/GELU from `@johnhenry/math-plus-special`) |
| `transpose`, `slice`/`split` | `stridedCopy` |

All of these come from the `@johnhenry/math-plus-tensor-core/kernels`
subpath.

### Host bridge (shared by the device packages)

`hostFromTensor(t, label?)` and `tensorFromHost(h)` convert between a
tensor-core `Tensor` and a tensor-backend `HostTensor` without copying
element data (f16 is re-viewed as a `Float16Array`; non-contiguous tensors
and dtypes outside `DEVICE_DTYPES` throw instead of being converted
implicitly). `@johnhenry/math-plus-tensor-mlx` and
`@johnhenry/math-plus-tensor-webgpu` use these for their explicit
`fromTensor`/`toTensor` transfers, so the mapping has one implementation.
Also exported: `DEVICE_DTYPES`, `isDeviceDType`.

## Dtypes

- **Storage.** Float tensors are `Float32Array`, `i32` is `Int32Array`, and
  `bool` is `Uint8Array` (0/1). `supports()` is true for `f32`, `i32` and
  `bool`.
- **f16/bf16 are widened, not supported.** `fromHost` accepts them and
  widens to f32, and the result's `.dtype` is `"f32"`. This is the
  contract's documented widening rule. `supports("f16" | "bf16")` is false,
  and `cast` to either throws. The conformance suite therefore runs this
  backend in f32 only.
- **Result dtypes** are the same as `@johnhenry/backend-cpu@0.2.0` (a test
  checks this against the real package):
  - Arithmetic gives f32 if either side is f32, and always for `div`/`pow`.
    Otherwise it gives i32, so `bool + bool` is i32.
  - `where` follows its two value operands.
  - `sum` and `cumsum` keep f32 and give i32 for everything else.
  - `max`, `min` and `sort` keep the input dtype.
  - `argmax` and `argmin` give i32.
  - Comparisons and logical ops give bool.
  - Every other op gives f32. Integer inputs to float ops compute in f32.
- **Precision.** Results round once to f32. Reductions, softmax,
  LayerNorm, attention and matmul accumulate in f64.

## Limitations

- **Single-threaded, scalar JS.** There are no workers and no SIMD.
  `linear` reaches about 7 GFLOP/s on an M-series core. For large models,
  use an MLX or WebGPU backend. This one is the reference and the
  fallback.
- **Eager.** Every op allocates its result right away. `reshape` and
  same-dtype `cast` share storage. `scope` frees intermediates, but there is
  no buffer pool.
- **Scratch memory.** `linear` packs its weight to f64 (8·out·in bytes)
  for each call, and its input in 256-row blocks. `matmul` allocates up to
  `(m+n)·k + m·n` f64s. `sdpa` allocates per-head f64 panels.
- **NaN handling is backend-defined** (the contract allows this):
  - `max`, `min`, `argmax` and `argmin` use strict comparisons from
    position 0, so a NaN wins only when it comes first. This is
    tensor-core's rule, not NumPy's NaN propagation.
  - `maximum` propagates NaN.
  - `relu(NaN)` is 0.
  - `sort` puts NaNs last.
- **Fully masked attention rows** give zeros. The contract leaves them
  undefined.
- `sdpa` also accepts an additive float mask, which the contract does not
  require. It is kept for compatibility with `backend-cpu@0.2`.
- `rope` needs an even head dim. Empty-axis `max`/`min`/`argmax`/`argmin`
  throw `RangeError`.
- **Differences from `backend-cpu@0.2.0`:**
  - The raw `gemmNT(f32 …)` export is gone. The GEMM lives in tensor-core.
  - `erf`, `erfc` and `geluScalar` are math-plus's canonical versions
    (SciPy-verified, ~1e-15 relative).
  - `where` with one bool and one i32 value operand returns i32, where
    0.2.0 returned the first operand's dtype.
  - The NaN rules above differ from 0.2.0's NaN-propagating `max`/`min`/
    `argmax`.

## Performance

Measured against laya-js's `@johnhenry/backend-cpu@0.2.0`, the
implementation this package replaces, with `npm run bench -w
@johnhenry/math-plus-tensor-cpu` (source in `bench/vs-laya.ts`). The run
follows [`docs/BENCHMARKING.md`](../../docs/BENCHMARKING.md): 5 s cooldown
before each cell, timing windows of 1 s or less, both backends alternated in
one process, and the median reported. Each timing covers compute only
(inputs uploaded once, the op run inside `scope`, the result disposed).
Recorded 2026-09-24 (the raw JSON, including run order, is printed by the script).

Median ms per call (lower is better):

| cell | Node 24.9: tensor-cpu | Node: backend-cpu 0.2.0 | Bun 1.2.17: tensor-cpu | Bun: backend-cpu 0.2.0 |
|---|---:|---:|---:|---:|
| `linear` [16·128, 1024] × [1024, 1024]ᵀ + bias | **626** | 704 | **487** | 501 |
| `sdpa` B=1 H=16 L=128 D=64, bool key mask | **13.0** | 13.8 | **9.0** | 12.3 |
| `matmul` [16,128,64] @ [16,64,128] | **5.3** | 5.8 | 4.4 | 4.2 |
| encoder layer (ModernBERT-base-shaped: LN → QKV → RoPE → SDPA → out-proj → LN → GEGLU MLP; 4×128 tokens, d=768) | **782** | 946 | **634** | 700 |

A real model gives a similar result. The tiny ModernBERT checkpoint from
laya-js (`laya-fixtures`: 4 layers, hidden size 64, B=3, L=40, run with
laya-js's `@johnhenry/modernbert` and the same harness) takes 9.1 ms on
Node and 7.9 ms on Bun with this package, against 10.9 ms and 11.7 ms with
backend-cpu 0.2.0. That run is not reproducible from this repo alone,
because the model code lives in laya-js. Its stage-by-stage parity test
against MLX (within 1e-5) passes with this backend.

Machine: MacBook Air (Mac14,15), Apple M2, 8 cores, 24 GiB, macOS
(Darwin 27.0), on AC power, no thermal warnings recorded. The load average
was about 10 from other work, so single cells vary by about ±5% between
runs. A rerun of the Bun `matmul` cell gave 4.16 vs 4.55 ms: it is at
parity, within noise.

The GEMM-bound cells are close, because both backends run the same 4×4
register-blocked kernel. `linear` here packs its operands into f64 panels,
which measured as fast as reading f32 directly or faster. The whole encoder
layer is the largest gain. It has not been profiled op by op.

## Tests

```bash
npm test -w @johnhenry/math-plus-tensor-cpu        # Node
npm run test:bun -w @johnhenry/math-plus-tensor-cpu
```

- `test/conformance.test.ts` runs tensor-backend's conformance suite: 103
  MLX/NumPy-generated cases, core plus general numerics. It runs twice:
  once with every op native, and once with the numerics ops hidden, so the
  `compose.ts` default compositions are checked on this backend too.
- `test/differential.test.ts` checks the backend against a NumPy float64
  oracle (`scripts/numpy_oracle.py`, one Python process for every case).
  It covers what the fixtures do not reach: shapes that straddle the GEMM
  blocks, grouped-query and causal/padding/additive-mask attention, RoPE at
  L=70, two-sided broadcasting, middle-axis reductions, i32 and bool result
  dtypes, and negative gather indices. Without a python3 that has numpy
  (`MATH_PLUS_ORACLE_PYTHON`), these tests skip rather than fail. A real
  run must show 0 skipped.
- `test/backend.test.ts` covers widening, scopes, error paths and
  drop-in compatibility with `backend-cpu@0.2.0`.
