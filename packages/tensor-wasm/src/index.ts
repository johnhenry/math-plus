/**
 * @johnhenry/math-plus-tensor-wasm — persistent WASM-resident tensor storage + the
 * `...Into` performance interface (issue #3, the M2 gate).
 *
 * docs/spikes/wasm-baseline.md measured the actual problem this solves: a
 * WASM kernel over RESIDENT buffers is 1.78x faster than pure JS at N=1e6
 * (0.864ms vs 1.539ms, scalar, no SIMD) — but a wrapper that allocates and
 * copies in/out per call turns that into a 2.27x REGRESSION (4.046ms). The
 * fix is this file: `WasmTensor` storage lives in linear memory for its
 * whole lifetime, and `addInto`/`mulInto`/`matmulInto` write into an
 * already-allocated `out` with zero WASM allocation per call.
 *
 * v1 scope: f32 only, matching the current kernel set in
 * crates/tensor-wasm-kernels. `WasmTensor` is NOT yet wired into
 * @johnhenry/math-plus-tensor-core's `Tensor` (that storage-model merge is bigger scope,
 * tracked separately) — this package proves the seam and its performance
 * in isolation first.
 *
 * SIMD128 fast path (issue #13): docs/spikes/wasm-simd.md measured a real,
 * stable ~2.6-3x speedup for contiguous elementwise add/mul, so `addInto`/
 * `mulInto` use it automatically when the runtime supports WASM SIMD AND
 * every operand is contiguous (stride 1) — falling back to the always-
 * present scalar/strided kernels otherwise (non-contiguous views, or a
 * runtime without SIMD support). `matmulInto` (issue #121) uses the SIMD
 * module's blocked f32x4 GEMM for EVERY call when it loaded (any strides —
 * the kernel packs its operands), ~37 GFLOP/s at 1024^3 vs ~10 for the
 * scalar blocked kernel and ~1.7 for the pre-#121 naive loop (Apple M2,
 * Node 24; same doc). This is a SEPARATE .wasm artifact
 * (`tensor_wasm_kernels_simd128.wasm`, built by `npm run build:wasm:simd`),
 * never merged into the default build: a wasm32 module containing ANY v128
 * instruction fails WebAssembly validation in its ENTIRETY on a runtime
 * without SIMD support (module loading is all-or-nothing), so shipping one
 * module covering both cases isn't possible — `Kernels.load()` feature-
 * detects via `WebAssembly.validate()` and loads the SIMD module only when
 * it'll actually instantiate.
 */
import { readFile } from "node:fs/promises";
import { hasSink, metric } from "@johnhenry/math-plus-telemetry";

interface KernelExports {
  memory: WebAssembly.Memory;
  alloc(len: number, align: number): number;
  dealloc(ptr: number, len: number, align: number): void;
  add_f32_strided(
    aPtr: number,
    aOffset: number,
    aStride: number,
    bPtr: number,
    bOffset: number,
    bStride: number,
    outPtr: number,
    outOffset: number,
    outStride: number,
    len: number,
  ): void;
  mul_f32_strided(
    aPtr: number,
    aOffset: number,
    aStride: number,
    bPtr: number,
    bOffset: number,
    bStride: number,
    outPtr: number,
    outOffset: number,
    outStride: number,
    len: number,
  ): void;
  sub_f32_strided(
    aPtr: number,
    aOffset: number,
    aStride: number,
    bPtr: number,
    bOffset: number,
    bStride: number,
    outPtr: number,
    outOffset: number,
    outStride: number,
    len: number,
  ): void;
  div_f32_strided(
    aPtr: number,
    aOffset: number,
    aStride: number,
    bPtr: number,
    bOffset: number,
    bStride: number,
    outPtr: number,
    outOffset: number,
    outStride: number,
    len: number,
  ): void;
  gemm_f32(
    aPtr: number,
    aOffset: number,
    aRowStride: number,
    aColStride: number,
    bPtr: number,
    bOffset: number,
    bRowStride: number,
    bColStride: number,
    outPtr: number,
    outOffset: number,
    outRowStride: number,
    outColStride: number,
    m: number,
    n: number,
    k: number,
    alpha: number,
    beta: number,
  ): void;
  solve_f32(
    aPtr: number,
    aOffset: number,
    aRowStride: number,
    aColStride: number,
    bPtr: number,
    bOffset: number,
    bStride: number,
    outPtr: number,
    outOffset: number,
    outStride: number,
    n: number,
  ): void;
}

/** The SIMD128 module's export surface used here — the two contiguous elementwise fast-path kernels (issue #13) and the blocked f32x4 GEMM (issue #121, same strided signature as `gemm_f32`); everything else still goes through `KernelExports`' scalar/strided kernels, which stay resident in a separate always-loaded module. */
interface SimdKernelExports {
  memory: WebAssembly.Memory;
  add_f32_contiguous_simd128(aPtr: number, bPtr: number, outPtr: number, len: number): void;
  mul_f32_contiguous_simd128(aPtr: number, bPtr: number, outPtr: number, len: number): void;
  gemm_f32_simd128: KernelExports["gemm_f32"];
}


const F32_ALIGN = 4;

/**
 * A tensor whose f32 storage lives in WASM linear memory for its whole
 * lifetime. Call `.free()` when done — this is manual memory management,
 * not GC'd JS storage.
 */
export class WasmTensor {
  readonly kernels: Kernels;
  /** Byte pointer returned by `alloc` — the buffer's true start, for `dealloc`. */
  readonly bufferPtr: number;
  readonly byteLength: number;
  readonly shape: readonly number[];
  /** Strides in ELEMENTS (not bytes), matching @johnhenry/math-plus-tensor-core's convention. */
  readonly strides: readonly number[];
  /** Element offset from `bufferPtr` — always 0 for tensors created here; view() can differ. */
  readonly elementOffset: number;
  #freed = false;

  private constructor(
    kernels: Kernels,
    bufferPtr: number,
    byteLength: number,
    shape: readonly number[],
    strides: readonly number[],
    elementOffset: number,
  ) {
    this.kernels = kernels;
    this.bufferPtr = bufferPtr;
    this.byteLength = byteLength;
    this.shape = shape;
    this.strides = strides;
    this.elementOffset = elementOffset;
  }

  static allocate(kernels: Kernels, shape: readonly number[]): WasmTensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const byteLength = size * 4;
    const ptr = kernels.exports.alloc(byteLength, F32_ALIGN);
    if (ptr === 0) throw new Error("WASM allocation failed");
    return new WasmTensor(kernels, ptr, byteLength, shape, contiguousStrides(shape), 0);
  }

  /** Allocate and copy `data` in once. */
  static fromArray(kernels: Kernels, data: Float32Array, shape: readonly number[]): WasmTensor {
    const t = WasmTensor.allocate(kernels, shape);
    new Float32Array(kernels.exports.memory.buffer, t.bufferPtr, data.length).set(data);
    return t;
  }

  /** A strided 1-D VIEW into this tensor's own buffer — zero copy. `elementOffset`/`stride` are relative to THIS tensor's own element 0 (composes with an existing view's own offset). Mainly for tests exercising the non-contiguous fallback path (issue #13's SIMD fast path only applies to `stride === 1`); real strided access normally comes from `@johnhenry/math-plus-tensor-core`. */
  view1D(elementOffset: number, length: number, stride: number): WasmTensor {
    if (this.shape.length !== 1) {
      throw new RangeError("view1D is only defined on an already-1-D tensor in v1");
    }
    return new WasmTensor(
      this.kernels,
      this.bufferPtr,
      this.byteLength,
      [length],
      [stride],
      this.elementOffset + elementOffset,
    );
  }

  /** A transposed 2-D VIEW — swaps strides, shares the same WASM buffer, zero copy. */
  transposed(): WasmTensor {
    if (this.shape.length !== 2) {
      throw new RangeError("transposed() only supports 2-D tensors in v1");
    }
    return new WasmTensor(
      this.kernels,
      this.bufferPtr,
      this.byteLength,
      [this.shape[1] as number, this.shape[0] as number],
      [this.strides[1] as number, this.strides[0] as number],
      this.elementOffset,
    );
  }

  /** Copy the buffer out as a plain Float32Array (for inspection/testing only). */
  toFloat32Array(): Float32Array {
    if (this.#freed) throw new Error("WasmTensor: use after free");
    // A direct memory read, not an export call -- but a trapped instance's
    // memory contents are untrustworthy too (issue #46), so refuse loudly
    // rather than hand back possibly-corrupt data.
    if (this.kernels.poisoned) {
      throw new Error(
        `WasmTensor: this tensor's Kernels instance is poisoned (WASM export "${this.kernels.poisonedBy}" ` +
          `trapped) -- its memory contents are untrustworthy. Create a fresh instance with Kernels.load().`,
      );
    }
    const size = this.shape.reduce((a, b) => a * b, 1);
    if (this.strides.length === 1 && this.strides[0] === 1 && this.elementOffset === 0) {
      return new Float32Array(this.kernels.exports.memory.buffer, this.bufferPtr, size).slice();
    }
    // Non-contiguous or offset view: read element-by-element via strides.
    const out = new Float32Array(size);
    const view = new Float32Array(this.kernels.exports.memory.buffer);
    const base = this.bufferPtr / 4 + this.elementOffset;
    let flatIndex = 0;
    const walk = (axis: number, offset: number): void => {
      const dim = this.shape[axis] as number;
      const stride = this.strides[axis] as number;
      for (let i = 0; i < dim; i++) {
        if (axis === this.shape.length - 1) {
          out[flatIndex++] = view[offset + i * stride] as number;
        } else {
          walk(axis + 1, offset + i * stride);
        }
      }
    };
    walk(0, base);
    return out;
  }

  free(): void {
    if (this.#freed) return;
    // On a poisoned instance (issue #46), calling dealloc would re-enter the
    // corrupted allocator (and throw the poisoned error into what is usually
    // a cleanup path). The whole instance's memory is unrecoverable and gets
    // garbage-collected wholesale when the Kernels instance is dropped, so
    // just mark this tensor freed and skip the call.
    if (!this.kernels.poisoned) {
      this.kernels.exports.dealloc(this.bufferPtr, this.byteLength, F32_ALIGN);
    }
    this.#freed = true;
  }
}

function contiguousStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i] as number;
  }
  return strides;
}

function flatSpec(t: WasmTensor): { bufferPtr: number; offset: number; stride: number } {
  if (t.strides.length !== 1) {
    throw new RangeError(
      `addInto/mulInto require 1-D tensors in v1 (got ${t.strides.length}-D); flatten first`,
    );
  }
  // bufferPtr stays a raw BYTE address: the Rust `*const f32` parameter is a
  // byte pointer at the WASM ABI level, and its `.offset(n)` already scales
  // `n` by sizeof(f32) internally. Only `offset`/`stride` are element units.
  return {
    bufferPtr: t.bufferPtr,
    offset: t.elementOffset,
    stride: t.strides[0] as number,
  };
}

/** Per-instance trap-poisoning state (issue #46), shared by every guarded export wrapper (scalar and SIMD modules alike — they share one linear memory, so a trap in either corrupts both). */
interface PoisonState {
  poisonedBy: string | undefined;
}

/**
 * Wrap one WASM export with the trap guard (issue #46). A Rust panic on
 * wasm32-unknown-unknown becomes an `unreachable` trap, surfacing in JS as
 * `WebAssembly.RuntimeError` — and after a trap, the instance's allocator/
 * memory state is undefined (the same failure mode the Woxi study's
 * playground works around by re-instantiating the whole module; see
 * docs/spikes/woxi-study.md, "WASM packaging"). Re-entering a trapped
 * instance can silently compute garbage, the worst failure class for a
 * numerics library — so the first trap permanently poisons this instance,
 * and every later call fails loudly instead. Resident WasmTensors point
 * into the (now-untrustworthy) linear memory and cannot survive; the
 * recovery path is a fresh `Kernels.load()`.
 *
 * Only genuine traps poison — ordinary JS errors (e.g. `solveInto`'s own
 * RangeError validation) pass through unchanged. Cost: one flag check per
 * call, negligible next to any kernel's actual work; the `...Into` path's
 * zero-allocation property is unaffected.
 */
function guardExport<A extends unknown[], R>(
  poison: PoisonState,
  name: string,
  fn: (...args: A) => R,
): (...args: A) => R {
  return (...args: A): R => {
    if (poison.poisonedBy !== undefined) {
      throw new Error(
        `Kernels instance is poisoned: WASM export "${poison.poisonedBy}" trapped earlier, leaving this ` +
          `instance's memory in an undefined state. Resident WasmTensors are invalid; create a fresh ` +
          `instance with Kernels.load().`,
      );
    }
    try {
      return fn(...args);
    } catch (err) {
      if (err instanceof WebAssembly.RuntimeError) {
        poison.poisonedBy = name;
        throw new Error(
          `WASM export "${name}" trapped (${err.message}) — a Rust panic aborted mid-operation. This ` +
            `Kernels instance is now poisoned (its memory state is undefined); resident WasmTensors are ` +
            `invalid. Create a fresh instance with Kernels.load().`,
          { cause: err },
        );
      }
      throw err;
    }
  };
}

export class Kernels {
  readonly exports: KernelExports;
  readonly #allocCounter: { count: number };
  readonly #simd: SimdKernelExports | undefined;
  readonly #poison: PoisonState;

  /** Calls to the WASM `alloc` export since load() — proves the `...Into` path allocates zero times. */
  get allocCallCount(): number {
    return this.#allocCounter.count;
  }

  /** Whether the SIMD128 fast path (issue #13) is active for this instance — false if the runtime doesn't support WASM SIMD, or the SIMD .wasm artifact wasn't built/found. `addInto`/`mulInto` fall back to the scalar/strided kernels transparently either way; this is exposed for tests/diagnostics. */
  get simdAvailable(): boolean {
    return this.#simd !== undefined;
  }

  /** Whether a WASM trap has poisoned this instance (issue #46) — see {@link guardExport}. Once true, never resets; recover with a fresh `Kernels.load()`. */
  get poisoned(): boolean {
    return this.#poison.poisonedBy !== undefined;
  }

  /** The export whose trap poisoned this instance, or undefined if healthy. */
  get poisonedBy(): string | undefined {
    return this.#poison.poisonedBy;
  }

  private constructor(
    exports: KernelExports,
    allocCounter: { count: number },
    simd: SimdKernelExports | undefined,
    poison: PoisonState,
  ) {
    this.exports = exports;
    this.#allocCounter = allocCounter;
    this.#simd = simd;
    this.#poison = poison;
  }

  static async load(wasmBytes?: Uint8Array, simdWasmBytes?: Uint8Array): Promise<Kernels> {
    const bytes =
      wasmBytes ??
      (await readFile(new URL("../wasm/tensor_wasm_kernels.wasm", import.meta.url)));
    const { instance } = await WebAssembly.instantiate(bytes as BufferSource, {});
    const rawExports = instance.exports as unknown as KernelExports;
    const rawAlloc = rawExports.alloc.bind(rawExports);
    const counter = { count: 0 };
    const poison: PoisonState = { poisonedBy: undefined };
    // Build a fresh exports object rather than mutating properties on the
    // WASM-provided one (whether those are writable is spec-uncertain and
    // varies by engine) -- every export is rebound through the trap guard
    // (issue #46, see guardExport above).
    const wrapped: KernelExports = {
      memory: rawExports.memory,
      alloc: guardExport(poison, "alloc", (len: number, align: number) => {
        counter.count++;
        // Opt-in telemetry (issue #10): the differentiated panel this
        // enables isn't loss curves, it's JS<->WASM memory residency --
        // which tensors live in linear memory and how many bytes cross
        // the boundary. Guarded by hasSink() so the zero-allocation
        // ...Into path (issue #3) stays exactly zero-cost when unused.
        if (hasSink()) {
          metric("wasm", counter.count, "wasm/alloc.bytes", len);
          metric("wasm", counter.count, "wasm/alloc.calls", counter.count);
        }
        return rawAlloc(len, align);
      }),
      dealloc: guardExport(poison, "dealloc", rawExports.dealloc.bind(rawExports)),
      add_f32_strided: guardExport(poison, "add_f32_strided", rawExports.add_f32_strided.bind(rawExports)),
      mul_f32_strided: guardExport(poison, "mul_f32_strided", rawExports.mul_f32_strided.bind(rawExports)),
      sub_f32_strided: guardExport(poison, "sub_f32_strided", rawExports.sub_f32_strided.bind(rawExports)),
      div_f32_strided: guardExport(poison, "div_f32_strided", rawExports.div_f32_strided.bind(rawExports)),
      gemm_f32: guardExport(poison, "gemm_f32", rawExports.gemm_f32.bind(rawExports)),
      solve_f32: guardExport(poison, "solve_f32", rawExports.solve_f32.bind(rawExports)),
    };

    // SIMD128 fast path (issue #13) — best-effort, never fatal. Any failure
    // (unsupported runtime, missing/not-yet-built artifact, a bad import
    // object) just leaves simdExports undefined and addInto/mulInto fall
    // back to the always-present scalar/strided kernels above.
    //
    // Feature-detected by validating the REAL simd .wasm module's bytes
    // directly (`WebAssembly.validate()`, which never throws — it returns
    // `false` on a runtime that doesn't recognize the v128 opcodes actually
    // present in this specific module) rather than a separate hand-crafted
    // minimal probe module: one less thing to get the bytes wrong on, and
    // it's checking the exact module that's about to be instantiated.
    let simdExports: SimdKernelExports | undefined;
    try {
      const simdBytes =
        simdWasmBytes ??
        (await readFile(new URL("../wasm/tensor_wasm_kernels_simd128.wasm", import.meta.url)));
      if (WebAssembly.validate(simdBytes as BufferSource)) {
        // Imports the SCALAR module's own memory (see lib.rs's simd module
        // doc comment for why --import-memory is required at build time) --
        // both modules genuinely share one linear memory / one ArrayBuffer,
        // so the SIMD kernels operate on the exact same resident WasmTensor
        // data, zero-copy.
        //
        // Instantiating the SIMD module also runs its data-segment
        // initialization against that SHARED memory: it rewrites its own
        // .rodata over the scalar module's at the same addresses and zeroes
        // its .bss (where the scalar allocator's state lives). That is a
        // no-op only while both builds' static data is byte-identical
        // (crates/tensor-wasm-kernels/src/gemm.rs's module doc explains how
        // the kernels keep it so). Verify instead of assuming: snapshot the
        // memory, instantiate, compare -- on any difference, restore the
        // snapshot and run scalar-only rather than on silently corrupted
        // constants.
        const snapshot = new Uint8Array(rawExports.memory.buffer).slice();
        const { instance: simdInstance } = await WebAssembly.instantiate(simdBytes as BufferSource, {
          env: { memory: rawExports.memory },
        });
        const live = new Uint8Array(rawExports.memory.buffer);
        let clobbered = false;
        for (let i = 0; i < snapshot.length; i++) {
          if (live[i] !== snapshot[i]) {
            clobbered = true;
            break;
          }
        }
        if (clobbered) {
          live.set(snapshot);
          throw new Error("SIMD module's static data differs from the scalar module's; SIMD disabled");
        }
        const rawSimd = simdInstance.exports as unknown as SimdKernelExports;
        // Same poison state as the scalar module: the two share one linear
        // memory, so a trap in either corrupts both (issue #46).
        simdExports = {
          memory: rawSimd.memory,
          add_f32_contiguous_simd128: guardExport(
            poison,
            "add_f32_contiguous_simd128",
            rawSimd.add_f32_contiguous_simd128.bind(rawSimd),
          ),
          mul_f32_contiguous_simd128: guardExport(
            poison,
            "mul_f32_contiguous_simd128",
            rawSimd.mul_f32_contiguous_simd128.bind(rawSimd),
          ),
          gemm_f32_simd128: guardExport(poison, "gemm_f32_simd128", rawSimd.gemm_f32_simd128.bind(rawSimd)),
        };
      }
    } catch {
      simdExports = undefined;
    }

    return new Kernels(wrapped, counter, simdExports, poison);
  }

  zeros(shape: readonly number[]): WasmTensor {
    return WasmTensor.allocate(this, shape);
  }

  fromArray(data: Float32Array, shape: readonly number[]): WasmTensor {
    return WasmTensor.fromArray(this, data, shape);
  }

  /** out[i] = a[i] + b[i], writing directly into `out`'s WASM buffer. Zero allocation. Uses the SIMD128 fast path (issue #13) when available and every operand is contiguous (stride 1); falls back to the general strided kernel otherwise. */
  addInto(out: WasmTensor, a: WasmTensor, b: WasmTensor): WasmTensor {
    const A = flatSpec(a);
    const B = flatSpec(b);
    const O = flatSpec(out);
    const len = out.shape.reduce((x, y) => x * y, 1);
    if (this.#simd && A.stride === 1 && B.stride === 1 && O.stride === 1) {
      this.#simd.add_f32_contiguous_simd128(
        A.bufferPtr + A.offset * 4,
        B.bufferPtr + B.offset * 4,
        O.bufferPtr + O.offset * 4,
        len,
      );
      return out;
    }
    this.exports.add_f32_strided(
      A.bufferPtr,
      A.offset,
      A.stride,
      B.bufferPtr,
      B.offset,
      B.stride,
      O.bufferPtr,
      O.offset,
      O.stride,
      len,
    );
    return out;
  }

  /** out[i] = a[i] * b[i], writing directly into `out`'s WASM buffer. Zero allocation. Uses the SIMD128 fast path (issue #13) when available and every operand is contiguous (stride 1); falls back to the general strided kernel otherwise. */
  mulInto(out: WasmTensor, a: WasmTensor, b: WasmTensor): WasmTensor {
    const A = flatSpec(a);
    const B = flatSpec(b);
    const O = flatSpec(out);
    const len = out.shape.reduce((x, y) => x * y, 1);
    if (this.#simd && A.stride === 1 && B.stride === 1 && O.stride === 1) {
      this.#simd.mul_f32_contiguous_simd128(
        A.bufferPtr + A.offset * 4,
        B.bufferPtr + B.offset * 4,
        O.bufferPtr + O.offset * 4,
        len,
      );
      return out;
    }
    this.exports.mul_f32_strided(
      A.bufferPtr,
      A.offset,
      A.stride,
      B.bufferPtr,
      B.offset,
      B.stride,
      O.bufferPtr,
      O.offset,
      O.stride,
      len,
    );
    return out;
  }

  /** out[i] = a[i] - b[i], writing directly into `out`'s WASM buffer. Zero allocation. No SIMD128 fast path yet (issue #66 — deferred pending a measured win, same discipline `add`/`mul`'s SIMD kernels followed before being added). */
  subInto(out: WasmTensor, a: WasmTensor, b: WasmTensor): WasmTensor {
    const A = flatSpec(a);
    const B = flatSpec(b);
    const O = flatSpec(out);
    const len = out.shape.reduce((x, y) => x * y, 1);
    this.exports.sub_f32_strided(
      A.bufferPtr,
      A.offset,
      A.stride,
      B.bufferPtr,
      B.offset,
      B.stride,
      O.bufferPtr,
      O.offset,
      O.stride,
      len,
    );
    return out;
  }

  /** out[i] = a[i] / b[i], writing directly into `out`'s WASM buffer. Zero allocation. IEEE 754 semantics on division by zero (±Infinity/NaN), matching `Tensor.div`. No SIMD128 fast path yet (issue #66 — see {@link subInto}). */
  divInto(out: WasmTensor, a: WasmTensor, b: WasmTensor): WasmTensor {
    const A = flatSpec(a);
    const B = flatSpec(b);
    const O = flatSpec(out);
    const len = out.shape.reduce((x, y) => x * y, 1);
    this.exports.div_f32_strided(
      A.bufferPtr,
      A.offset,
      A.stride,
      B.bufferPtr,
      B.offset,
      B.stride,
      O.bufferPtr,
      O.offset,
      O.stride,
      len,
    );
    return out;
  }

  /**
   * out = a @ b (2-D only in v1), writing directly into `out`'s WASM buffer.
   * `a`/`b` may be `.transposed()` views — read via strides, never copied.
   *
   * Both paths run the same cache-blocked, register-tiled GEMM (issue
   * #121); when the SIMD128 module loaded (`simdAvailable`), EVERY call —
   * any strides, not just contiguous operands — uses its f32x4
   * micro-kernel (packing absorbs the layout), otherwise the scalar one.
   * The two are bit-for-bit identical. `out` must not alias `a` or `b`.
   */
  matmulInto(out: WasmTensor, a: WasmTensor, b: WasmTensor): WasmTensor {
    if (a.shape.length !== 2 || b.shape.length !== 2 || out.shape.length !== 2) {
      throw new RangeError("matmulInto supports 2-D tensors only in v1");
    }
    const [m, k] = a.shape as [number, number];
    const [k2, n] = b.shape as [number, number];
    if (k !== k2) {
      throw new RangeError(`matmulInto: inner dims ${k} and ${k2} don't match`);
    }
    if (out.shape[0] !== m || out.shape[1] !== n) {
      throw new RangeError(`matmulInto: out is ${out.shape[0]}x${out.shape[1]}, expected ${m}x${n}`);
    }
    const gemm = this.#simd ? this.#simd.gemm_f32_simd128 : this.exports.gemm_f32;
    gemm(
      a.bufferPtr,
      a.elementOffset,
      a.strides[0] as number,
      a.strides[1] as number,
      b.bufferPtr,
      b.elementOffset,
      b.strides[0] as number,
      b.strides[1] as number,
      out.bufferPtr,
      out.elementOffset,
      out.strides[0] as number,
      out.strides[1] as number,
      m,
      n,
      k,
      1.0,
      0.0,
    );
    return out;
  }

  /**
   * Solve `A @ x = b` for a square `n x n` system (issue #39), writing
   * `x` directly into `out`'s WASM buffer. `a` may be a `.transposed()`
   * view — read via strides, never copied (the kernel copies `A` into its
   * OWN internal scratch buffer for the LU pivoting, but never requires the
   * CALLER to pre-pack a transposed/strided operand). `b`/`out` must be
   * 1-D, `a` must be 2-D square, all matching size `n`. See `linalg.solve`
   * in `adapter-math` for the reference-speed fallback/correctness oracle
   * this kernel is meant to sit alongside, not replace.
   */
  solveInto(out: WasmTensor, a: WasmTensor, b: WasmTensor): WasmTensor {
    if (a.shape.length !== 2 || a.shape[0] !== a.shape[1]) {
      throw new RangeError("solveInto requires a square 2-D matrix for `a`");
    }
    const n = a.shape[0] as number;
    const B = flatSpec(b);
    const O = flatSpec(out);
    if (b.shape[0] !== n || out.shape[0] !== n) {
      throw new RangeError(`solveInto: size mismatch (a is ${n}x${n}, b has ${b.shape[0]}, out has ${out.shape[0]})`);
    }
    this.exports.solve_f32(
      a.bufferPtr,
      a.elementOffset,
      a.strides[0] as number,
      a.strides[1] as number,
      B.bufferPtr,
      B.offset,
      B.stride,
      O.bufferPtr,
      O.offset,
      O.stride,
      n,
    );
    return out;
  }
}
