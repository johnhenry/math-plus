/**
 * Issue #55 Phase 1: measure a native Deno-FFI path against the WASM path,
 * same Rust source, same machine — the benchmark gate that decides whether
 * Phase 2 (distribution matrix, panic-boundary hardening, conditional
 * exports) is worth building. Run with:
 *
 *   cargo build --release -p tensor-wasm-kernels   # produces the cdylib
 *   npm run build:wasm -w @johnhenry/math-plus-tensor-wasm
 *   deno run --allow-ffi --allow-read --allow-env packages/tensor-wasm/scripts/deno-ffi-bench.ts
 *
 * Fairness notes:
 * - Both sides call the IDENTICAL Rust functions (the crate's crate-type
 *   already included cdylib, so the .so needed zero source changes).
 * - WASM timings are kernel-call-only on RESIDENT buffers (its best case —
 *   the copy-in cost that the FFI path structurally avoids is measured
 *   separately, not folded in).
 * - Correctness is cross-checked (native vs wasm results bit-compared)
 *   before anything is timed.
 */

// deno-lint-ignore-file no-explicit-any

const ROOT = new URL("../../..", import.meta.url).pathname;
const LIB_FILE = Deno.build.os === "darwin"
  ? "libtensor_wasm_kernels.dylib"
  : Deno.build.os === "windows"
  ? "tensor_wasm_kernels.dll"
  : "libtensor_wasm_kernels.so";
// $MATH_PLUS_NATIVE_KERNELS_PATH (same variable native.ts honors) lets a
// differently-featured build be benchmarked, e.g. `--features accelerate`.
const SO_PATH = Deno.env.get("MATH_PLUS_NATIVE_KERNELS_PATH") ?? `${ROOT}target/release/${LIB_FILE}`;
const WASM_PATH = `${ROOT}packages/tensor-wasm/wasm/tensor_wasm_kernels.wasm`;
const SIMD_WASM_PATH = `${ROOT}packages/tensor-wasm/wasm/tensor_wasm_kernels_simd128.wasm`;

// ---- native (FFI) side -------------------------------------------------------

const lib = Deno.dlopen(SO_PATH, {
  add_f32_strided: {
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
} as const);

// ---- wasm side ---------------------------------------------------------------

const wasmBytes = Deno.readFileSync(WASM_PATH);
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const w = instance.exports as any;
const wasmMem = (): ArrayBuffer => (w.memory as WebAssembly.Memory).buffer;
// The SIMD128 module shares the scalar module's memory (see index.ts's
// Kernels.load) -- its gemm_f32_simd128 (issue #121) is what matmulInto
// actually runs on any SIMD-capable runtime, so gemm is compared against both.
const { instance: simdInstance } = await WebAssembly.instantiate(Deno.readFileSync(SIMD_WASM_PATH), {
  env: { memory: w.memory },
});
const ws = simdInstance.exports as any;

function wasmAllocF32(n: number): number {
  return w.alloc(n * 4, 4) as number;
}
function wasmView(ptr: number, n: number): Float32Array {
  return new Float32Array(wasmMem(), ptr, n);
}

// ---- harness -----------------------------------------------------------------

function fill(rng: () => number, n: number): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = rng() * 2 - 1;
  return a;
}
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function bench(fn: () => void, minIters: number, minMs: number): number {
  fn(); fn(); // warm-up
  let iters = 0;
  const start = performance.now();
  do {
    fn();
    iters++;
  } while (iters < minIters || performance.now() - start < minMs);
  return (performance.now() - start) / iters;
}

/** `atol` defaults to 1e-5; gemm passes one scaled by sqrt(k), since kernels with different summation orders (and Accelerate's FMA) legitimately differ by ~sqrt(k) ulps on cancellation-heavy outputs. */
function assertClose(a: Float32Array, b: Float32Array, label: string, atol = 1e-5): void {
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as number;
    const y = b[i] as number;
    if (Math.abs(x - y) > atol + 1e-4 * Math.max(Math.abs(x), Math.abs(y))) {
      throw new Error(`${label}: native/wasm divergence at [${i}]: native=${x} wasm=${y}`);
    }
  }
}

interface Row {
  op: string;
  size: string;
  nativeMs: number;
  wasmMs: number;
  speedup: number;
}
const rows: Row[] = [];
const rng = mulberry32(20260813);

// ---- addInto -----------------------------------------------------------------

for (const N of [10_000, 1_000_000, 4_000_000]) {
  const a = fill(rng, N);
  const b = fill(rng, N);
  const out = new Float32Array(N);

  const aPtr = wasmAllocF32(N);
  const bPtr = wasmAllocF32(N);
  const outPtr = wasmAllocF32(N);
  wasmView(aPtr, N).set(a);
  wasmView(bPtr, N).set(b);

  lib.symbols.add_f32_strided(a, 0n, 1n, b, 0n, 1n, out, 0n, 1n, BigInt(N));
  w.add_f32_strided(aPtr, 0, 1, bPtr, 0, 1, outPtr, 0, 1, N);
  assertClose(out, wasmView(outPtr, N), `add N=${N}`);

  const nativeMs = bench(() => lib.symbols.add_f32_strided(a, 0n, 1n, b, 0n, 1n, out, 0n, 1n, BigInt(N)), 20, 200);
  const wasmMs = bench(() => w.add_f32_strided(aPtr, 0, 1, bPtr, 0, 1, outPtr, 0, 1, N), 20, 200);
  rows.push({ op: "addInto", size: `N=${N.toLocaleString("en-US")}`, nativeMs, wasmMs, speedup: wasmMs / nativeMs });
}

// ---- gemm --------------------------------------------------------------------

for (const S of [64, 256, 512, 1024]) {
  const a = fill(rng, S * S);
  const b = fill(rng, S * S);
  const out = new Float32Array(S * S);

  const aPtr = wasmAllocF32(S * S);
  const bPtr = wasmAllocF32(S * S);
  const outPtr = wasmAllocF32(S * S);
  wasmView(aPtr, S * S).set(a);
  wasmView(bPtr, S * S).set(b);

  const Sb = BigInt(S);
  lib.symbols.gemm_f32(a, 0n, Sb, 1n, b, 0n, Sb, 1n, out, 0n, Sb, 1n, Sb, Sb, Sb, 1, 0);
  w.gemm_f32(aPtr, 0, S, 1, bPtr, 0, S, 1, outPtr, 0, S, 1, S, S, S, 1, 0);
  assertClose(out, wasmView(outPtr, S * S), `gemm ${S}x${S}`, 1e-5 * Math.sqrt(S));

  const iters = S >= 1024 ? 3 : 10;
  const nativeMs = bench(() => lib.symbols.gemm_f32(a, 0n, Sb, 1n, b, 0n, Sb, 1n, out, 0n, Sb, 1n, Sb, Sb, Sb, 1, 0), iters, 300);
  const wasmMs = bench(() => w.gemm_f32(aPtr, 0, S, 1, bPtr, 0, S, 1, outPtr, 0, S, 1, S, S, S, 1, 0), iters, 300);
  rows.push({ op: "gemm", size: `${S}x${S}`, nativeMs, wasmMs, speedup: wasmMs / nativeMs });
  ws.gemm_f32_simd128(aPtr, 0, S, 1, bPtr, 0, S, 1, outPtr, 0, S, 1, S, S, S, 1, 0);
  assertClose(out, wasmView(outPtr, S * S), `gemm simd128 ${S}x${S}`, 1e-5 * Math.sqrt(S));
  const simdMs = bench(() => ws.gemm_f32_simd128(aPtr, 0, S, 1, bPtr, 0, S, 1, outPtr, 0, S, 1, S, S, S, 1, 0), iters, 300);
  rows.push({ op: "gemm (wasm = simd128)", size: `${S}x${S}`, nativeMs, wasmMs: simdMs, speedup: simdMs / nativeMs });
}

// ---- solve -------------------------------------------------------------------

for (const S of [64, 256]) {
  // Diagonally-dominant -> well-conditioned, no pivoting pathologies.
  const a = fill(rng, S * S);
  for (let i = 0; i < S; i++) a[i * S + i] = 10 + Math.abs(a[i * S + i] as number);
  const b = fill(rng, S);
  const out = new Float32Array(S);

  const aPtr = wasmAllocF32(S * S);
  const bPtr = wasmAllocF32(S);
  const outPtr = wasmAllocF32(S);
  wasmView(aPtr, S * S).set(a);
  wasmView(bPtr, S).set(b);

  const Sb = BigInt(S);
  lib.symbols.solve_f32(a, 0n, Sb, 1n, b, 0n, 1n, out, 0n, 1n, Sb);
  w.solve_f32(aPtr, 0, S, 1, bPtr, 0, 1, outPtr, 0, 1, S);
  assertClose(out, wasmView(outPtr, S), `solve n=${S}`);

  const nativeMs = bench(() => lib.symbols.solve_f32(a, 0n, Sb, 1n, b, 0n, 1n, out, 0n, 1n, Sb), 10, 200);
  const wasmMs = bench(() => w.solve_f32(aPtr, 0, S, 1, bPtr, 0, 1, outPtr, 0, 1, S), 10, 200);
  rows.push({ op: "solve", size: `n=${S}`, nativeMs, wasmMs, speedup: wasmMs / nativeMs });
}

// ---- the copy cost the FFI path structurally avoids --------------------------

{
  const N = 1_000_000;
  const a = fill(rng, N);
  const aPtr = wasmAllocF32(N);
  const copyMs = bench(() => wasmView(aPtr, N).set(a), 50, 200);
  rows.push({ op: "copy-in (wasm residency tax)", size: `N=${N.toLocaleString("en-US")}`, nativeMs: 0, wasmMs: copyMs, speedup: Number.NaN });
}

// ---- report ------------------------------------------------------------------

console.log(`\n${"op".padEnd(30)} ${"size".padEnd(14)} ${"native ms".padStart(11)} ${"wasm ms".padStart(11)} ${"native speedup".padStart(15)}`);
for (const r of rows) {
  console.log(
    `${r.op.padEnd(30)} ${r.size.padEnd(14)} ${r.nativeMs.toFixed(4).padStart(11)} ${r.wasmMs.toFixed(4).padStart(11)} ${Number.isNaN(r.speedup) ? "—".padStart(15) : (r.speedup.toFixed(2) + "x").padStart(15)}`,
  );
}
console.log("\nJSON:", JSON.stringify(rows));
lib.close();
