# @johnhenry/math-plus-tensor-wasm

Rust→WASM CPU kernels for math-plus tensors — a flat-numeric extern-C ABI
with no wasm-bindgen marshalling on hot paths, an arena allocator, and
zero-allocation `...Into` ops over resident buffers. Plus an opt-in native
FFI path for Deno.

The measured numbers that shaped the design: a WASM kernel over *resident*
buffers is 1.78x faster than pure JS at N=1e6 — but a wrapper that allocates
and copies in/out per call turns that into a 2.27x **regression**. Hence the
`...Into` interface and manual buffer residency.

## Install

```bash
npm install @johnhenry/math-plus-tensor-wasm
```

The published package ships prebuilt `.wasm` (scalar + SIMD variants). In a
git clone, `wasm/` is gitignored — run `npm run build:wasm` first, which
needs `rustup target add wasm32-unknown-unknown` and `lld` (no wasm-pack).

## Quick start

```js
import { Kernels } from "@johnhenry/math-plus-tensor-wasm";

const kernels = await Kernels.load(); // feature-detects SIMD via WebAssembly.validate()

const a = kernels.fromArray(new Float32Array([1, 2, 3, 4]), [4]);
const b = kernels.fromArray(new Float32Array([10, 20, 30, 40]), [4]);
const out = kernels.zeros([4]);

kernels.addInto(out, a, b); // writes into out; allocates NOTHING
out.toFloat32Array();       // [11, 22, 33, 44]

a.free(); b.free(); out.free(); // manual memory management — not GC'd
```

Opt-in native (Deno FFI) path:

```js
import { Kernels, NativeKernels } from "@johnhenry/math-plus-tensor-wasm";
const native = NativeKernels.load();        // undefined outside Deno / without a binary
const kernels = native ?? await Kernels.load(); // WASM stays the zero-install default
```

## API surface

- `Kernels.load(wasmBytes?, simdWasmBytes?)`; ops `addInto`/`subInto`/
  `mulInto`/`divInto` (1-D), `matmulInto` (2-D, stride-aware — a transposed
  view multiplies without a copy), `solveInto`; `allocCallCount`,
  `simdAvailable`, `poisoned`/`poisonedBy`.
- `WasmTensor`: `allocate`, `fromArray`, zero-copy `view1D`/`transposed`,
  `toFloat32Array`, `free`. `bufferPtr` is a **byte** address;
  offsets/strides are in **elements**.
- `NativeKernels.load(options?)` (never throws — returns `undefined` when
  unavailable), `matrix(data, rows, cols)`. Binary resolution:
  explicit path → `$MATH_PLUS_NATIVE_KERNELS_PATH` → bundled platform dir →
  repo `target/release/`.

## Traps

- **There is no `setBackend` API anywhere in math-plus.** `WasmTensor` is a
  *separate storage type*, not a plug-in backend for tensor-core's `Tensor`
  — that storage-model merge is tracked separately. You choose WASM by
  explicitly importing `Kernels` and writing against this API (f32 only,
  1-D/2-D ops, manual `free()`).
- **Trap poisoning (issue #46):** a Rust panic becomes a WASM trap, and the
  first trap **permanently poisons the whole `Kernels` instance** — every
  later call throws, `toFloat32Array` refuses to read possibly-corrupt
  memory, `free()` becomes a silent no-op. Recovery is a fresh
  `Kernels.load()`. Ordinary JS validation errors and IEEE division-by-zero
  (±Infinity/NaN, same as JS) do *not* poison.
- **SIMD:** used only when the SIMD module validated *and* all operands are
  stride-1; only `addInto`/`mulInto` have SIMD kernels. A wasm32 module
  containing any v128 instruction fails validation *in its entirety* on a
  non-SIMD runtime — that's why two modules ship. Results are bit-identical
  to scalar.
- Missing scalar `.wasm` artifact → `Kernels.load()` throws `ENOENT` (the
  SIMD artifact degrades gracefully instead).
- The SIMD **benchmark** is deliberately not in `npm test`: on GitHub's
  mixed runner fleet the same commit measured 1.12x on one machine and
  1.01x-with-zero-variance on another; any threshold low enough for the
  slow runner would also pass a real regression to parity. `npm run
  test:bench` on known hardware instead.
- The `deno` export condition does **not** auto-switch to native — native is
  opt-in; a native panic is a process abort (fail-fast), not WASM-style
  poisoning.

## Tests

`npm test` — differential legs against tensor-core's `Tensor` ops (dev
dependency, oracle only) and adapter-math's `linalg.solve` (issue #39),
poisoning-state coverage, SIMD tail-loop and graceful-degradation tests.

## Provenance

Built across issues #3 (`...Into` zero-alloc), #13 (SIMD), #46 (trap
poisoning), #55 (defined alloc failure), #66 (`subInto`/`divInto`). Measured
baselines in `docs/spikes/wasm-baseline.md` / `wasm-simd.md` /
`deno-ffi-baseline.md`. Part of the
[math-plus](https://github.com/johnhenry/math-plus) monorepo; family docs at
<https://opensource.johnhenry.me/math/>.
