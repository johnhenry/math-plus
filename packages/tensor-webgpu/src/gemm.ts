/**
 * WebGPU GEMM entry points: `out[m,n] = sum_k a[m,k] * b[k,n]` (or `b[n,k]`
 * with `transB`), row-major, f32 or f16 storage with **f32 accumulation**.
 *
 * Since issue #146 these run on `@johnhenry/backend-webgpu`'s GEMM (the
 * kernels this package used to carry were ports of it; there is now one
 * copy). `transB` is the backend's `linear` (x·Wᵀ), which picks a kernel
 * per shape: split-K "skinny" for M ≤ 64, subgroup matrices (Apple/Dawn,
 * see gemm-caps.ts) for larger M, else a register-blocked "direct" kernel,
 * with a tiled fallback for unaligned K. Plain A·B is `linear(a, bᵀ)` too
 * when subgroup matrices apply (one strided-copy dispatch for bᵀ buys the
 * faster kernel), else the backend's tiled `matmul`.
 *
 * @deprecated All three entry points are kept through the deprecation
 * window (see the 0.2.0 changelog). New code: `createWebGpuDevice()`, then
 * `gpu.backend.matmul(a, b)` / `gpu.backend.linear(x, w)` on tensors from
 * `await gpu.fromTensor(t)`.
 *
 * Not supported (disclosed per AGENTS.md): batched/broadcast GEMM through
 * these shims (2-D only; the backend's `matmul` batches and broadcasts),
 * mixed dtypes, bias/activation epilogues (the backend's `linear` takes a
 * bias), and f16 on devices without `shader-f16` (throws rather than
 * silently widening).
 */
import type { WebGpuBackend, WebGpuTensor } from "@johnhenry/backend-webgpu";
import { backendFor, uploadSync } from "./bridge.ts";
import { GPUTensor } from "./device.ts";
import type { GemmCapabilities } from "./gemm-caps.ts";

export type GemmKernel = "tiled" | "skinny" | "subgroup-matrix";
export type GemmDType = "f32" | "f16";

export interface GemmOptions {
  /** `b` is `[n, k]` (a Linear weight; computes `a · bᵀ`) instead of `[k, n]`. Default false. */
  transB?: boolean;
  /**
   * Force a kernel family instead of the backend's choice (tests and
   * benchmarks). Throws if its preconditions ({@link gemmKernelApplicable})
   * don't hold. Default `"auto"`.
   *
   * @deprecated Kept for the deprecation window. Forcing maps onto
   * backend-webgpu's per-shape `gemmTuning` table ("skinny", a
   * subgroup-matrix config) or its tiled `matmul` ("tiled").
   */
  kernel?: GemmKernel | "auto";
}

/** backend-webgpu's skinny kernel handles M up to this (its largest `skinny` config). */
const SKINNY_MAX_M = 64;

/**
 * Preconditions of each forcible kernel family (pure; exported for tests).
 * `caps` is what the device's backend can use (`{ f16, subgroupMatrix }`).
 */
export function gemmKernelApplicable(
  kernel: GemmKernel,
  m: number,
  k: number,
  _n: number,
  transB: boolean,
  caps: GemmCapabilities,
): boolean {
  switch (kernel) {
    case "tiled":
      return true;
    case "skinny":
      return transB && k % 4 === 0 && m <= SKINNY_MAX_M;
    case "subgroup-matrix":
      return caps.subgroupMatrix && k % 4 === 0;
  }
}

/** Subgroup-matrix config index in backend-webgpu's `GEMM_DEFAULT.sg` used when that family is forced: the general 32×64-tile entry (no workgroup-count gate). */
const FORCED_SG_INDEX = 1;

/** `x · wᵀ` with a forced tuning choice for exactly this shape, restoring the table afterwards. */
function linearWith(b: WebGpuBackend, x: WebGpuTensor, w: WebGpuTensor, choice: "skinny" | number): WebGpuTensor {
  const [m, k] = x.shape as [number, number];
  const n = w.shape[0] as number;
  // backend-webgpu's documented tuning key: `${storage}:${M}x${N}x${K}`.
  const key = `${b.kind(x.dtype).st}:${m}x${n}x${k}`;
  const table = b.gemmTuning;
  const prev = table.get(key);
  table.set(key, choice);
  try {
    return b.linear(x, w);
  } finally {
    if (prev === undefined) table.delete(key);
    else table.set(key, prev);
  }
}

/**
 * Encode one GEMM on the device's backend. `a` is `[m, k]`; `bt` is `[n, k]`
 * when `transB`, else `[k, n]`. Returns a new backend-owned `[m, n]` tensor
 * (untracked by any scope: the caller disposes it).
 */
export function encodeGemm(b: WebGpuBackend, a: WebGpuTensor, bt: WebGpuTensor, transB: boolean, kernel: GemmKernel | "auto" = "auto"): WebGpuTensor {
  const m = a.shape[0] as number;
  const k = a.shape[1] as number;
  const transposed = (): WebGpuTensor => b.transpose(bt, [1, 0]);
  return b.scope(() => {
    switch (kernel) {
      case "skinny":
        return linearWith(b, a, bt, "skinny");
      case "subgroup-matrix":
        return linearWith(b, a, transB ? bt : transposed(), FORCED_SG_INDEX);
      case "tiled":
        return b.matmul(a, transB ? transposed() : bt);
      case "auto":
        if (transB) return b.linear(a, bt);
        // A·B: the backend's `matmul` always uses its portable tiled kernel;
        // `linear` reaches subgroup matrices. Where they apply (M > 64,
        // K % 4 == 0), paying one transpose of B is the faster route.
        if (b.hasSubgroupMatrix && m > SKINNY_MAX_M && k % 4 === 0) return b.linear(a, transposed());
        return b.matmul(a, bt);
    }
  });
}

function checkF16(device: GPUDevice, dtype: GemmDType): void {
  if (dtype === "f16" && !device.features.has("shader-f16")) {
    throw new TypeError("GEMM: f16 needs a device with the shader-f16 feature (request it, e.g. via detectWebGPU())");
  }
}

function checkKernel(b: WebGpuBackend, kernel: GemmKernel | "auto", m: number, k: number, n: number, transB: boolean): void {
  if (kernel === "auto") return;
  if (!gemmKernelApplicable(kernel, m, k, n, transB, { f16: b.hasF16, subgroupMatrix: b.hasSubgroupMatrix })) {
    throw new RangeError(`GEMM: kernel "${kernel}" is not applicable to m=${m} k=${k} n=${n} transB=${transB} on this device`);
  }
}

/**
 * GPU-resident GEMM: `a` is a 2-D `[m, k]` `GPUTensor`, `b` is `[k, n]`
 * (or `[n, k]` with `transB`), same dtype (f32, or f16 with `shader-f16`).
 * Returns a new `[m, n]` `GPUTensor` of that dtype without any host copy —
 * chain it, read it with `.toTensor()`, and `.free()` it when done (inputs
 * are never freed here).
 *
 * @deprecated See the module doc: use `gpu.backend.matmul`/`linear`.
 */
export async function runGemm(device: GPUDevice, a: GPUTensor, b: GPUTensor, opts: GemmOptions = {}): Promise<GPUTensor> {
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    throw new RangeError(`runGemm: operands must be 2-D, got [${a.shape}] and [${b.shape}]`);
  }
  if (a.dtype !== b.dtype) throw new TypeError(`runGemm: dtype mismatch (${a.dtype} vs ${b.dtype})`);
  const transB = opts.transB ?? false;
  const [m, k] = a.shape as [number, number];
  const [bRows, bCols] = b.shape as [number, number];
  const kB = transB ? bCols : bRows;
  const n = transB ? bRows : bCols;
  if (kB !== k) {
    throw new RangeError(`runGemm: inner dimensions differ: a [${a.shape}] and b [${b.shape}]${transB ? " (transB)" : ""}`);
  }
  checkF16(device, a.dtype);
  const be = backendFor(device);
  const kernel = opts.kernel ?? "auto";
  checkKernel(be, kernel, m, k, n, transB);
  return GPUTensor._fromHandle(device, encodeGemm(be, a._live("runGemm"), b._live("runGemm"), transB, kernel));
}

function checkLengths(fn: string, aLen: number, bLen: number, m: number, k: number, n: number): void {
  if (aLen !== m * k) throw new RangeError(`${fn}: a.length ${aLen} !== m*k ${m * k}`);
  if (bLen !== k * n) throw new RangeError(`${fn}: b.length ${bLen} !== k*n ${k * n}`);
}

async function runGemmHost(
  fn: string,
  device: GPUDevice,
  dtype: GemmDType,
  a: Float32Array | Uint16Array,
  bData: Float32Array | Uint16Array,
  m: number,
  k: number,
  n: number,
  opts: GemmOptions,
): Promise<ArrayBuffer> {
  checkLengths(fn, a.length, bData.length, m, k, n);
  checkF16(device, dtype);
  const transB = opts.transB ?? false;
  const be = backendFor(device);
  const kernel = opts.kernel ?? "auto";
  checkKernel(be, kernel, m, k, n, transB);
  if (m * n === 0) return new ArrayBuffer(0);
  const ta = uploadSync(be, a, [m, k], dtype);
  const tb = uploadSync(be, bData, transB ? [n, k] : [k, n], dtype);
  let out: WebGpuTensor | undefined;
  try {
    out = encodeGemm(be, ta, tb, transB, kernel);
    return await be.rt.readBytes(out.storage.buffer, 0, m * n * (dtype === "f16" ? 2 : 4));
  } finally {
    be.dispose(ta);
    be.dispose(tb);
    if (out) be.dispose(out);
  }
}

/**
 * `a` is `m x k`, `b` is `k x n` (or `n x k` with `transB`), both row-major
 * contiguous `Float32Array`s; returns `m x n` row-major. Upload, dispatch
 * and readback happen per call — {@link runGemm} keeps operands on the GPU.
 *
 * @deprecated See the module doc.
 */
export async function runGemmWGSL(
  device: GPUDevice,
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number,
  opts: GemmOptions = {},
): Promise<Float32Array> {
  return new Float32Array(await runGemmHost("runGemmWGSL", device, "f32", a, b, m, k, n, opts));
}

/**
 * f16 counterpart of {@link runGemmWGSL}: operands and result are raw
 * IEEE-754 binary16 bits (`Uint16Array`, tensor-core's f16 storage).
 * Products accumulate in f32; the result is rounded to f16 once, on store.
 * Needs `shader-f16`.
 *
 * @deprecated See the module doc.
 */
export async function runGemmF16WGSL(
  device: GPUDevice,
  a: Uint16Array,
  b: Uint16Array,
  m: number,
  k: number,
  n: number,
  opts: GemmOptions = {},
): Promise<Uint16Array> {
  return new Uint16Array(await runGemmHost("runGemmF16WGSL", device, "f16", a, b, m, k, n, opts));
}
