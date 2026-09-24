/**
 * The WASM-vs-WebGPU GEMM threshold benchmark's shared pieces: the 43
 * measured shapes, their inputs, and the two end-to-end calls being raced.
 * One copy, used by `measure-gemm-threshold.ts` (Dawn in-process, or
 * headless Chrome over CDP) and by `gemm-threshold-page/` (a page for a
 * real, visible browser), so every environment measures the same thing.
 * Browser-safe: no Node imports.
 */
import type { Kernels, WasmTensor as WasmTensorClass } from "@johnhenry/math-plus-tensor-wasm";
import type { WebGpuDevice } from "../src/facade.ts";

export interface Cell {
  group: "square" | "k-sweep" | "linear";
  m: number;
  k: number;
  n: number;
  /** `b` is a Linear weight `[n, k]` (x·Wᵀ) instead of `[k, n]`. */
  transB: boolean;
}

export interface CellLists {
  sizes?: number[];
  sweepMN?: number[];
  sweepK?: number[];
  linearM?: number[];
  groups?: ReadonlySet<string>;
}

/**
 * Square n³ (14), a k sweep at five fixed m = n (24), and Linear rows M
 * against ModernBERT-large's MLP-in weight [3072, 1024] (5): 43 cells.
 */
export function gemmCells(o: CellLists = {}): Cell[] {
  const sizes = o.sizes ?? [8, 16, 32, 48, 64, 96, 128, 160, 192, 256, 384, 512, 1024, 2048];
  const sweepMN = o.sweepMN ?? [32, 64, 96, 128, 192];
  const sweepK = o.sweepK ?? [16, 64, 256, 1024, 4096];
  const linearM = o.linearM ?? [1, 4, 16, 64, 256];
  const groups = o.groups ?? new Set(["square", "k-sweep", "linear"]);
  const cells: Cell[] = [];
  if (groups.has("square")) for (const n of sizes) cells.push({ group: "square", m: n, k: n, n, transB: false });
  if (groups.has("k-sweep")) {
    for (const mn of sweepMN) for (const k of sweepK) if (k !== mn) cells.push({ group: "k-sweep", m: mn, k, n: mn, transB: false });
  }
  if (groups.has("linear")) for (const m of linearM) cells.push({ group: "linear", m, k: 1024, n: 3072, transB: true });
  return cells;
}

export const cellLabel = (c: Cell): string => `${c.m}x${c.k}x${c.n}${c.transB ? "ᵀ" : ""}`;

/** Deterministic inputs in [-1, 1). Source text too, for pages driven over CDP. */
export const LCG_SOURCE = `function lcg(size, seed) {
  let s = seed >>> 0;
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) { s = (s * 1664525 + 1013904223) >>> 0; out[i] = (s / 0xffffffff) * 2 - 1; }
  return out;
}`;
export const lcg = new Function(`${LCG_SOURCE}; return lcg;`)() as (size: number, seed: number) => Float32Array;

/** One cell's inputs. `b` is [K,N], or [N,K] when transB; `bKN` is always [K,N] (what WASM's matmulInto takes). */
export function cellInputs(c: Cell): { a: Float32Array; b: Float32Array; bKN: Float32Array } {
  const a = lcg(c.m * c.k, c.m + 1);
  const b = lcg(c.k * c.n, c.n + 2);
  if (!c.transB) return { a, b, bKN: b };
  const bKN = new Float32Array(c.k * c.n);
  for (let j = 0; j < c.n; j++) for (let p = 0; p < c.k; p++) bKN[p * c.n + j] = b[j * c.k + p] as number;
  return { a, b, bKN };
}

/**
 * The WebGPU side, end to end on host arrays, exactly as a caller of the
 * facade writes it: upload both operands (`gpu.backend.fromHost`), `matmul` for
 * A·B or `linear` for x·Wᵀ, read back (`gpu.toHost`), dispose. Source
 * text, so a CDP-driven page runs the identical code.
 */
export const FACADE_GEMM_SOURCE = `async function facadeGemm(gpu, m, k, n, transB, a, b) {
  const A = await gpu.backend.fromHost({ dtype: "f32", shape: [m, k], data: a });
  const B = await gpu.backend.fromHost({ dtype: "f32", shape: transB ? [n, k] : [k, n], data: b });
  const C = transB ? gpu.backend.linear(A, B) : gpu.backend.matmul(A, B);
  try {
    return (await gpu.toHost(C)).data;
  } finally {
    gpu.dispose(A);
    gpu.dispose(B);
    gpu.dispose(C);
  }
}`;
export const facadeGemm = new Function(`${FACADE_GEMM_SOURCE}; return facadeGemm;`)() as (
  gpu: WebGpuDevice,
  m: number,
  k: number,
  n: number,
  transB: boolean,
  a: Float32Array,
  b: Float32Array,
) => Promise<Float32Array>;

/** The WASM side, end to end: tensor-wasm's `matmulInto` (SIMD128 when available) on [M,K]·[K,N], including the copies in and out. */
export function wasmGemm(kernels: Kernels, WasmTensor: typeof WasmTensorClass, c: Cell, a: Float32Array, bKN: Float32Array): Float32Array {
  const ta = WasmTensor.fromArray(kernels, a, [c.m, c.k]);
  const tb = WasmTensor.fromArray(kernels, bKN, [c.k, c.n]);
  const out = kernels.zeros([c.m, c.n]);
  try {
    kernels.matmulInto(out, ta, tb);
    return out.toFloat32Array();
  } finally {
    ta.free();
    tb.free();
    out.free();
  }
}

/** Which backend pipelines one call dispatches (e.g. `gemm` = tiled, `gemmsg`, `gemmskinny`), recorded by wrapping the runtime's `dispatch`. */
export const KERNEL_PROBE_SOURCE = `async function kernelsOf(gpu, run) {
  const rt = gpu.backend.rt;
  const keys = [];
  const orig = rt.dispatch;
  rt.dispatch = function (k, ...rest) { keys.push(k.key.split(":")[0]); return orig.call(this, k, ...rest); };
  try { await run(); } finally { rt.dispatch = orig; }
  return keys.join("+");
}`;
export const kernelsOf = new Function(`${KERNEL_PROBE_SOURCE}; return kernelsOf;`)() as (gpu: WebGpuDevice, run: () => Promise<unknown>) => Promise<string>;
