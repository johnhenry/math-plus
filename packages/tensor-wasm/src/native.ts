/**
 * Native (Deno FFI) sibling of the WASM kernels (issue #55 Phase 2) — the
 * same Rust source (`crates/tensor-wasm-kernels`) compiled as a platform
 * cdylib instead of to wasm32, called through `Deno.dlopen`. Measured on
 * this family's baseline machine (docs/spikes/deno-ffi-baseline.md):
 * `solve` 2.97–5.27x, elementwise 1.75–4.65x over the WASM path (gemm:
 * after issue #121's blocked kernels, native-portable is ~1.4–1.8x the
 * SIMD128 WASM path, and the opt-in `accelerate` cargo feature on macOS
 * ~34x — see that doc's re-measurement section) — plus the structural win that kernels operate on host
 * `Float32Array`s directly, so there is no copy-in/copy-out residency tax
 * and no resident-tensor lifecycle at all.
 *
 * Availability contract (graceful, never fatal): {@link loadNative} returns
 * `undefined` — never throws — when any prerequisite is missing:
 * not running under Deno, `--allow-ffi` not granted, or no platform binary
 * found. Callers fall back to {@link Kernels} (the WASM path), which works
 * everywhere. Binary resolution order:
 *   1. explicit `libraryPath` option
 *   2. `$MATH_PLUS_NATIVE_KERNELS_PATH`
 *   3. `<package>/native/<os>-<arch>/` (where the CI matrix artifacts land;
 *      see .github/workflows/native-kernels.yml)
 *   4. a repo-checkout `target/release/` build (development convenience)
 *
 * Panic boundary (the issue's hardening item): the crate's one
 * input-validation panic (`alloc` on an invalid layout) is now a defined
 * null return on both build targets. Any *remaining* panic (a genuine bug)
 * escaping an `extern "C"` fn is a guaranteed process abort on Rust ≥1.81
 * — never undefined behavior — which is the documented trade: the WASM
 * path's trap poisoning (#46) degrades to fail-fast process death here.
 * There is no transparent recovery to hide it, matching this repo's
 * loud-failure convention.
 */

/** The minimal slice of Deno's FFI surface this module touches — declared
 * locally so the package compiles under plain Node tooling without Deno's
 * type definitions. Checked at runtime before any use. */
interface DenoFFI {
  build: { os: string; arch: string };
  env: { get(name: string): string | undefined };
  dlopen(
    path: string,
    symbols: Record<string, { parameters: readonly string[]; result: string }>,
  ): { symbols: Record<string, (...args: unknown[]) => unknown>; close(): void };
}

declare const Deno: DenoFFI | undefined;

const SYMBOLS = {
  add_f32_strided: {
    parameters: ["buffer", "isize", "isize", "buffer", "isize", "isize", "buffer", "isize", "isize", "usize"],
    result: "void",
  },
  mul_f32_strided: {
    parameters: ["buffer", "isize", "isize", "buffer", "isize", "isize", "buffer", "isize", "isize", "usize"],
    result: "void",
  },
  gemm_f32: {
    parameters: [
      "buffer", "isize", "isize", "isize",
      "buffer", "isize", "isize", "isize",
      "buffer", "isize", "isize", "isize",
      "usize", "usize", "usize", "f32", "f32",
    ],
    result: "void",
  },
  solve_f32: {
    parameters: ["buffer", "isize", "isize", "isize", "buffer", "isize", "isize", "buffer", "isize", "isize", "usize"],
    result: "void",
  },
} as const;

function libraryFileName(os: string): string {
  if (os === "windows") return "tensor_wasm_kernels.dll";
  if (os === "darwin") return "libtensor_wasm_kernels.dylib";
  return "libtensor_wasm_kernels.so";
}

function candidatePaths(explicit: string | undefined, deno: DenoFFI): string[] {
  const paths: string[] = [];
  if (explicit) paths.push(explicit);
  const env = deno.env.get("MATH_PLUS_NATIVE_KERNELS_PATH");
  if (env) paths.push(env);
  const file = libraryFileName(deno.build.os);
  const pkgRoot = new URL("..", import.meta.url).pathname;
  paths.push(`${pkgRoot}native/${deno.build.os}-${deno.build.arch}/${file}`);
  // Development convenience: a repo checkout's own release build.
  paths.push(new URL(`../../../target/release/${file}`, import.meta.url).pathname);
  return paths;
}

export interface LoadNativeOptions {
  /** Absolute path to the cdylib; checked before the env var / bundled locations. */
  libraryPath?: string;
}

/** 2-D matrix descriptor over a host Float32Array — offset/strides in ELEMENTS, mirroring the WASM path's view semantics. */
export interface MatrixRef {
  data: Float32Array;
  offset: number;
  rowStride: number;
  colStride: number;
  rows: number;
  cols: number;
}

/** Row-major contiguous MatrixRef over a plain array. */
export function matrix(data: Float32Array, rows: number, cols: number): MatrixRef {
  if (data.length !== rows * cols) throw new RangeError(`matrix: data.length ${data.length} != ${rows}*${cols}`);
  return { data, offset: 0, rowStride: cols, colStride: 1, rows, cols };
}

/**
 * The native kernel surface: same operations, same offset/stride ABI as the
 * WASM `Kernels`, but over host `Float32Array`s — no residency, no
 * alloc/free lifecycle, no copies.
 */
export class NativeKernels {
  readonly #lib: ReturnType<DenoFFI["dlopen"]>;
  /** Where the cdylib was actually loaded from (diagnostics). */
  readonly libraryPath: string;

  private constructor(lib: ReturnType<DenoFFI["dlopen"]>, libraryPath: string) {
    this.#lib = lib;
    this.libraryPath = libraryPath;
  }

  /** See the module doc: returns undefined (never throws) when native isn't available here. */
  static load(options: LoadNativeOptions = {}): NativeKernels | undefined {
    if (typeof Deno === "undefined" || typeof Deno.dlopen !== "function") return undefined;
    for (const path of candidatePaths(options.libraryPath, Deno)) {
      try {
        return new NativeKernels(Deno.dlopen(path, SYMBOLS), path);
      } catch {
        // missing file, wrong architecture, or --allow-ffi denied for this
        // path — try the next candidate; undefined overall means "use WASM".
      }
    }
    return undefined;
  }

  close(): void {
    this.#lib.close();
  }

  /** out[i] = a[i] + b[i] over host arrays (contiguous, stride 1). */
  addInto(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array {
    this.#strided("add_f32_strided", out, a, b);
    return out;
  }

  /** out[i] = a[i] * b[i] over host arrays (contiguous, stride 1). */
  mulInto(out: Float32Array, a: Float32Array, b: Float32Array): Float32Array {
    this.#strided("mul_f32_strided", out, a, b);
    return out;
  }

  #strided(symbol: "add_f32_strided" | "mul_f32_strided", out: Float32Array, a: Float32Array, b: Float32Array): void {
    if (a.length !== out.length || b.length !== out.length) {
      throw new RangeError(`${symbol}: length mismatch (a=${a.length}, b=${b.length}, out=${out.length})`);
    }
    this.#lib.symbols[symbol]!(a, 0n, 1n, b, 0n, 1n, out, 0n, 1n, BigInt(out.length));
  }

  /** out = A @ B (alpha=1, beta=0), strided operands — a transposed view is just swapped strides, no copy. */
  matmulInto(out: MatrixRef, a: MatrixRef, b: MatrixRef): MatrixRef {
    if (a.cols !== b.rows || out.rows !== a.rows || out.cols !== b.cols) {
      throw new RangeError(`matmulInto: shape mismatch (${a.rows}x${a.cols}) @ (${b.rows}x${b.cols}) -> (${out.rows}x${out.cols})`);
    }
    this.#lib.symbols.gemm_f32!(
      a.data, BigInt(a.offset), BigInt(a.rowStride), BigInt(a.colStride),
      b.data, BigInt(b.offset), BigInt(b.rowStride), BigInt(b.colStride),
      out.data, BigInt(out.offset), BigInt(out.rowStride), BigInt(out.colStride),
      BigInt(a.rows), BigInt(b.cols), BigInt(a.cols), 1, 0,
    );
    return out;
  }

  /** Solve A·x = b (LU, partial pivoting) into `out`; A must be square. */
  solveInto(out: Float32Array, a: MatrixRef, b: Float32Array): Float32Array {
    if (a.rows !== a.cols) throw new RangeError(`solveInto: A must be square, got ${a.rows}x${a.cols}`);
    if (b.length !== a.rows || out.length !== a.rows) {
      throw new RangeError(`solveInto: size mismatch (A is ${a.rows}x${a.cols}, b=${b.length}, out=${out.length})`);
    }
    this.#lib.symbols.solve_f32!(
      a.data, BigInt(a.offset), BigInt(a.rowStride), BigInt(a.colStride),
      b, 0n, 1n, out, 0n, 1n, BigInt(a.rows),
    );
    return out;
  }
}

/** Convenience: native when available, `undefined` otherwise — spell the fallback explicitly at the call site: `NativeKernels.load() ?? await Kernels.load()`. */
export const loadNative = NativeKernels.load;
