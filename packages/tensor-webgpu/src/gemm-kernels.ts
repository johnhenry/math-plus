/**
 * WGSL source generators for the GEMM kernels (pure string functions — no
 * GPU calls, so they unit-test in plain Node). gemm.ts owns kernel
 * selection and dispatch.
 *
 * `C[M,N] = A[M,K] · B`, where B is `[K,N]` row-major, or `[N,K]` row-major
 * when `transB` (a PyTorch-style Linear weight, i.e. `A · Bᵀ`). Storage
 * dtype is f32 or f16 (all three buffers share it); **every kernel converts
 * loads to f32, accumulates in f32, and rounds once on store** — so f16 GEMM
 * is "f16 storage, f32 accumulation", never f16 accumulation.
 *
 * Ported from laya-js's `@johnhenry/backend-webgpu` (packages/backend-webgpu/
 * src/kernels.ts, same author; tile configs tuned there on an Apple M2 via
 * Dawn/Metal and verified against MLX), adapted to this package's runtime:
 * fixed 4-binding layout (A, B, C, uniform dims) with `layout: "auto"`, no
 * batch dimension, no bias/bf16 epilogues, no buffer offsets. Three kernels:
 *
 *  - **tiled**: 64x64x16 workgroup-memory tiles, 4x4 outputs per thread,
 *    vec4 staging along M/N; vec4 global loads when K (for A) / N or K (for
 *    B) is a multiple of 4, scalar loads otherwise. The portable default.
 *  - **skinny**: small-M latency path for `transB` with K % 4 == 0. All M
 *    rows (M <= 64) live in one workgroup so each weight row is read from
 *    memory once; K is split across lanes and reduced in workgroup memory.
 *  - **subgroup-matrix**: Dawn's experimental
 *    `chromium_experimental_subgroup_matrix` (Metal `simdgroup_matrix` on
 *    Apple), f32 8x8x8 fragments, subgroup size 32. Operands are converted
 *    to f32 while being staged through workgroup memory (Dawn only offers
 *    f16 fragments with f16 accumulation, which this package refuses), and
 *    the next K panel is prefetched into registers while the current one is
 *    multiplied. Needs K % 4 == 0 (and N % 4 == 0 when B is `[K,N]`). The
 *    `[K,N]` B layout (`row_major` right fragments) is new here — laya-js
 *    only ships the `transB` form.
 */

export type GemmDType = "f32" | "f16";

/** Workgroup-memory tiled GEMM: BM x BN output tile, BK-deep K panels, TM x TN outputs per thread. */
export interface TiledGemmConfig {
  BM: number;
  BN: number;
  BK: number;
  TM: number;
  TN: number;
}

/** Skinny GEMM (small M, transB): WX column threads x TN columns each, WY row threads, KS-way split K. */
export interface SkinnyGemmConfig {
  WX: number;
  TN: number;
  WY: number;
  KS: number;
}

/** Subgroup-matrix GEMM: BM x BN workgroup tile, BK-deep panels (multiple of 8), WM x WN subgroups of 32 lanes. */
export interface SubgroupMatrixGemmConfig {
  BM: number;
  BN: number;
  BK: number;
  WM: number;
  WN: number;
}

const f4 = (dtype: GemmDType, e: string): string => (dtype === "f32" ? e : `vec4<f32>(${e})`);
const f1 = (dtype: GemmDType, e: string): string => (dtype === "f32" ? e : `f32(${e})`);
const store = (dtype: GemmDType, e: string): string => (dtype === "f32" ? e : `f16(${e})`);

/** Shared header: enables, the uniform `Dims` struct, and the fixed A/B/C/dims binding layout gemm.ts dispatches against. */
function header(dtype: GemmDType, aElem: string, bElem: string, extra: { enables?: string[]; directives?: string[] } = {}): string {
  const lines: string[] = [];
  if (dtype === "f16") lines.push("enable f16;");
  for (const e of extra.enables ?? []) lines.push(`enable ${e};`);
  for (const d of extra.directives ?? []) lines.push(`${d};`);
  lines.push(
    "struct Dims { M: u32, N: u32, K: u32, _pad: u32 };",
    `@group(0) @binding(0) var<storage, read> A: array<${aElem}>;`,
    `@group(0) @binding(1) var<storage, read> B: array<${bElem}>;`,
    `@group(0) @binding(2) var<storage, read_write> C: array<${dtype}>;`,
    "@group(0) @binding(3) var<uniform> P: Dims;",
  );
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// tiled

/** Workgroup size of a tiled config (threads), for validation against device limits. */
export function tiledWorkgroupSize(cfg: TiledGemmConfig): number {
  return (cfg.BN / cfg.TN) * (cfg.BM / cfg.TM);
}

/**
 * `vecA`: A bound as `vec4` (needs K % 4 == 0). `vecB`: B bound as `vec4`
 * (needs K % 4 == 0 when `transB`, N % 4 == 0 otherwise). Dispatch:
 * `[ceil(N / BN), ceil(M / BM)]`.
 */
export function tiledGemmWGSL(dtype: GemmDType, transB: boolean, vecA: boolean, vecB: boolean, cfg: TiledGemmConfig): string {
  const { BM, BN, BK, TM, TN } = cfg;
  if (TN % 4 || TM % 4 || BK % 4 || BM % TM || BN % TN) throw new RangeError("tiledGemmWGSL: TM, TN, BK must be multiples of 4 dividing BM/BN");
  const TX = BN / TN;
  const WG = tiledWorkgroupSize(cfg);
  const BM4 = BM / 4;
  const BN4 = BN / 4;

  // K-contiguous operand (A, or B when transB) -> S[k][row/4] (vec4 over 4
  // rows). Each thread owns whole vec4s in workgroup memory (component
  // writes from different threads into one vec4 would race).
  const loadRowsK = (lim: string, name: string, vec: boolean, S: string, rows: number, rowBase: string): string => {
    const R4 = rows / 4;
    const rowsOf = [0, 1, 2, 3].map((i) => `${rowBase} + m4 * 4u + ${i}u`);
    if (vec) {
      const n = R4 * (BK / 4);
      const loads = rowsOf
        .map(
          (row, i) => `      var q${i} = vec4<f32>(0.0);
      if (${row} < ${lim} && kk < P.K) { q${i} = ${f4(dtype, `${name}[((${row}) * P.K + kk) / 4u]`)}; }`,
        )
        .join("\n");
      return `    for (var t = lid; t < ${n}u; t += ${WG}u) {
      let m4 = t / ${BK / 4}u; let kq = t % ${BK / 4}u;
      let kk = k0 + kq * 4u;
${loads}
      let s = kq * 4u * ${R4}u + m4;
      ${S}[s] = vec4<f32>(q0.x, q1.x, q2.x, q3.x);
      ${S}[s + ${R4}u] = vec4<f32>(q0.y, q1.y, q2.y, q3.y);
      ${S}[s + ${2 * R4}u] = vec4<f32>(q0.z, q1.z, q2.z, q3.z);
      ${S}[s + ${3 * R4}u] = vec4<f32>(q0.w, q1.w, q2.w, q3.w);
    }\n`;
    }
    const n = R4 * BK;
    const loads = rowsOf
      .map(
        (row, i) => `      var v${i} = 0.0;
      if (${row} < ${lim} && k < P.K) { v${i} = ${f1(dtype, `${name}[(${row}) * P.K + k]`)}; }`,
      )
      .join("\n");
    return `    for (var t = lid; t < ${n}u; t += ${WG}u) {
      let m4 = t / ${BK}u; let kk = t % ${BK}u; let k = k0 + kk;
${loads}
      ${S}[kk * ${R4}u + m4] = vec4<f32>(v0, v1, v2, v3);
    }\n`;
  };
  // N-contiguous B ([K, N]) -> Bs[k][n/4].
  const loadBkn = (): string => {
    const n = (BK * BN) / 4;
    if (vecB) {
      return `    for (var t = lid; t < ${n}u; t += ${WG}u) {
      let kk = t / ${BN4}u; let nq = t % ${BN4}u;
      let k = k0 + kk; let col = c0 + nq * 4u;
      var v = vec4<f32>(0.0);
      if (k < P.K && col < P.N) { v = ${f4(dtype, "B[(k * P.N + col) / 4u]")}; }
      Bs[kk * ${BN4}u + nq] = v;
    }\n`;
    }
    const loads = [0, 1, 2, 3]
      .map(
        (q) => `      var v${q} = 0.0;
      if (k < P.K && col + ${q}u < P.N) { v${q} = ${f1(dtype, `B[k * P.N + col + ${q}u]`)}; }`,
      )
      .join("\n");
    return `    for (var t = lid; t < ${n}u; t += ${WG}u) {
      let kk = t / ${BN4}u; let nq = t % ${BN4}u;
      let k = k0 + kk; let col = c0 + nq * 4u;
${loads}
      Bs[kk * ${BN4}u + nq] = vec4<f32>(v0, v1, v2, v3);
    }\n`;
  };

  const MV = TM / 4;
  const NV = TN / 4;
  let decl = "";
  for (let i = 0; i < TM; i++) for (let j = 0; j < NV; j++) decl += `  var c${i}_${j} = vec4<f32>(0.0);\n`;
  let inner = "";
  for (let i = 0; i < MV; i++) inner += `      let a${i} = As[kk * ${BM4}u + ty * ${MV}u + ${i}u];\n`;
  for (let j = 0; j < NV; j++) inner += `      let b${j} = Bs[kk * ${BN4}u + tx * ${NV}u + ${j}u];\n`;
  for (let i = 0; i < TM; i++) for (let j = 0; j < NV; j++) inner += `      c${i}_${j} += a${i >> 2}.${"xyzw"[i & 3]} * b${j};\n`;
  let stores = "";
  for (let i = 0; i < TM; i++) {
    stores += `  { let row = r0 + ty * ${TM}u + ${i}u;\n    if (row < P.M) {\n`;
    for (let j = 0; j < NV; j++) {
      stores += `      { let col = c0 + tx * ${TN}u + ${4 * j}u; let v = c${i}_${j};\n`;
      for (let q = 0; q < 4; q++) {
        stores += `        if (col + ${q}u < P.N) { C[row * P.N + col + ${q}u] = ${store(dtype, `v.${"xyzw"[q]}`)}; }\n`;
      }
      stores += "      }\n";
    }
    stores += "    }\n  }\n";
  }

  const aElem = vecA ? `vec4<${dtype}>` : dtype;
  const bElem = vecB ? `vec4<${dtype}>` : dtype;
  return `${header(dtype, aElem, bElem)}
// tiled GEMM ${BM}x${BN}x${BK}/${TM}x${TN} transB=${transB} vecA=${vecA} vecB=${vecB}
var<workgroup> As: array<vec4<f32>, ${BK * BM4}>;
var<workgroup> Bs: array<vec4<f32>, ${BK * BN4}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let tx = lid % ${TX}u; let ty = lid / ${TX}u;
  let r0 = wid.y * ${BM}u; let c0 = wid.x * ${BN}u;
${decl}
  for (var k0 = 0u; k0 < P.K; k0 += ${BK}u) {
${loadRowsK("P.M", "A", vecA, "As", BM, "r0")}${transB ? loadRowsK("P.N", "B", vecB, "Bs", BN, "c0") : loadBkn()}
    workgroupBarrier();
    for (var kk = 0u; kk < ${BK}u; kk++) {
${inner}    }
    workgroupBarrier();
  }
${stores}}
`;
}

// ---------------------------------------------------------------------------
// skinny

/**
 * Small-M GEMM for `transB` with K % 4 == 0 (the "one short sequence
 * through a Linear layer" latency path). `TM = ceil(M / WY)` rows per
 * thread is baked in, so one shader per distinct TM. Dispatch:
 * `[ceil(N / (WX * TN)), 1]`.
 */
export function skinnyGemmWGSL(dtype: GemmDType, cfg: SkinnyGemmConfig, TM: number): string {
  const { WX, TN, WY, KS } = cfg;
  const WG = WX * WY * KS;
  let s = "";
  for (let j = 0; j < TN; j++) s += `  let br${j} = min(c0 + ${j * WX}u, P.N - 1u) * K4;\n`;
  for (let i = 0; i < TM; i++) s += `  let ar${i} = min(ty * ${TM}u + ${i}u, P.M - 1u) * K4;\n`;
  for (let i = 0; i < TM; i++) for (let j = 0; j < TN; j++) s += `  var c${i}_${j} = 0.0;\n`;
  // A (small) straight from global memory — it stays in cache; no barriers in the K loop.
  s += `  for (var k = ks; k < K4; k += ${KS}u) {\n`;
  for (let j = 0; j < TN; j++) s += `    let b${j} = ${f4(dtype, `B[br${j} + k]`)};\n`;
  for (let i = 0; i < TM; i++) {
    s += `    { let a = ${f4(dtype, `A[ar${i} + k]`)};\n`;
    for (let j = 0; j < TN; j++) s += `      c${i}_${j} += dot(a, b${j});\n`;
    s += "    }\n";
  }
  s += "  }\n";
  if (KS > 1) {
    // Reduce one accumulator at a time through red[WG] (keeps workgroup memory small).
    for (let i = 0; i < TM; i++) {
      for (let j = 0; j < TN; j++) {
        s += `  red[lid] = c${i}_${j};
  workgroupBarrier();
  if (ks == 0u) {
    var acc = 0.0;
    for (var q = 0u; q < ${KS}u; q++) { acc += red[lid + q]; }
    let row = ty * ${TM}u + ${i}u; let col = c0 + ${j * WX}u;
    if (row < P.M && col < P.N) { C[row * P.N + col] = ${store(dtype, "acc")}; }
  }
  workgroupBarrier();\n`;
      }
    }
  } else {
    for (let i = 0; i < TM; i++) {
      s += `  { let row = ty * ${TM}u + ${i}u;\n    if (row < P.M) {\n`;
      for (let j = 0; j < TN; j++) {
        s += `      if (c0 + ${j * WX}u < P.N) { C[row * P.N + c0 + ${j * WX}u] = ${store(dtype, `c${i}_${j}`)}; }\n`;
      }
      s += "    }\n  }\n";
    }
  }
  return `${header(dtype, `vec4<${dtype}>`, `vec4<${dtype}>`)}
// skinny GEMM ${WX}x${WY}x${KS}/${TM}x${TN}
${KS > 1 ? `var<workgroup> red: array<f32, ${WG}>;` : ""}
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let K4 = P.K / 4u;
  let ks = lid % ${KS}u; let tx = (lid / ${KS}u) % ${WX}u; let ty = lid / ${KS * WX}u;
  // columns interleaved across threads: c0 + j*WX
  let c0 = wid.x * ${WX * TN}u + tx;
${s}}
`;
}

// ---------------------------------------------------------------------------
// subgroup matrix

/** Subgroup size the subgroup-matrix kernel is written for (its lane arithmetic assumes it). */
export const SUBGROUP_SIZE = 32;

/**
 * The experimental extension's builtin signatures changed across Dawn
 * versions: current Dawn (the `webgpu@0.6` npm package, recent Chromium)
 * takes the memory layout as a template argument
 * (`subgroupMatrixLoad<T, row_major>(p, offset, stride)`); older builds
 * took a `col_major: bool` argument (`subgroupMatrixLoad<T>(p, offset,
 * false, stride)`). gemm.ts tries `"template"` first and falls back to
 * `"bool"` if the shader fails validation.
 */
export type SubgroupMatrixSyntax = "template" | "bool";

/**
 * Dispatch: `[ceil(N / BN), ceil(M / BM)]`. Requires K % 4 == 0, and
 * N % 4 == 0 when `!transB`.
 */
export function subgroupMatrixGemmWGSL(
  dtype: GemmDType,
  transB: boolean,
  cfg: SubgroupMatrixGemmConfig,
  syntax: SubgroupMatrixSyntax = "template",
): string {
  const { BM, BN, BK, WM, WN } = cfg;
  const SG = SUBGROUP_SIZE;
  const WG = WM * WN * SG;
  const FM = BM / WM / 8; // fragments per subgroup
  const FN = BN / WN / 8;
  if (!Number.isInteger(FM) || !Number.isInteger(FN) || BK % 8) throw new RangeError("subgroupMatrixGemmWGSL: bad tile config");
  const K4 = BK / 4; // vec4s per K panel row
  const BN4 = BN / 4;
  const BKP = BK + 4; // padded row stride (floats) of K-contiguous staged tiles
  const BNP = BN + 4; // padded row stride (floats) of the N-contiguous staged B tile
  const BOFF = BM * BKP; // B tile offset in Sh
  const bTileFloats = transB ? BN * BKP : BK * BNP;
  const SCR = BOFF + bTileFloats; // per-subgroup 8x8 output scratch
  const NA = Math.ceil((BM * K4) / WG);
  const NB = Math.ceil((transB ? BN * K4 : BK * BN4) / WG);
  const aCount = BM * K4;
  const bCount = transB ? BN * K4 : BK * BN4;
  // Guard for the last partial round when the tile's vec4 count isn't a multiple of WG.
  const guard = (count: number, v: number): string => ((v + 1) * WG > count ? `if (lid + ${v * WG}u < ${count}u) ` : "");
  const regs = (p: string, n: number): string => Array.from({ length: n }, (_, v) => `  var ${p}${v} = vec4<f32>(0.0);\n`).join("");

  // Cooperative tile loads, software-pipelined: the next K panel is fetched
  // into registers (a vec4 per thread per round) while the subgroups
  // multiply the current one out of workgroup memory.
  const fetchA = Array.from(
    { length: NA },
    (_, v) => `    ${guard(aCount, v)}{ let t = lid + ${v * WG}u; let gr = r0 + t / ${K4}u; let kq = k0n / 4u + t % ${K4}u;
      pa${v} = vec4<f32>(0.0);
      if (gr < P.M && kq < K4) { pa${v} = ${f4(dtype, "A[gr * K4 + kq]")}; } }\n`,
  ).join("");
  const fetchB = Array.from({ length: NB }, (_, v) =>
    transB
      ? `    ${guard(bCount, v)}{ let t = lid + ${v * WG}u; let gc = c0 + t / ${K4}u; let kq = k0n / 4u + t % ${K4}u;
      pb${v} = vec4<f32>(0.0);
      if (gc < P.N && kq < K4) { pb${v} = ${f4(dtype, "B[gc * K4 + kq]")}; } }\n`
      : `    ${guard(bCount, v)}{ let t = lid + ${v * WG}u; let kr = k0n + t / ${BN4}u; let cq = c0 / 4u + t % ${BN4}u;
      pb${v} = vec4<f32>(0.0);
      if (kr < P.K && cq < N4) { pb${v} = ${f4(dtype, "B[kr * N4 + cq]")}; } }\n`,
  ).join("");
  const stashA = Array.from(
    { length: NA },
    (_, v) => `    ${guard(aCount, v)}{ let t = lid + ${v * WG}u; let o = (t / ${K4}u) * ${BKP}u + (t % ${K4}u) * 4u;
      Sh[o] = pa${v}.x; Sh[o + 1u] = pa${v}.y; Sh[o + 2u] = pa${v}.z; Sh[o + 3u] = pa${v}.w; }\n`,
  ).join("");
  const stashB = Array.from({ length: NB }, (_, v) => {
    const o = transB
      ? `${BOFF}u + (t / ${K4}u) * ${BKP}u + (t % ${K4}u) * 4u`
      : `${BOFF}u + (t / ${BN4}u) * ${BNP}u + (t % ${BN4}u) * 4u`;
    return `    ${guard(bCount, v)}{ let t = lid + ${v * WG}u; let o = ${o};
      Sh[o] = pb${v}.x; Sh[o + 1u] = pb${v}.y; Sh[o + 2u] = pb${v}.z; Sh[o + 3u] = pb${v}.w; }\n`;
  }).join("");

  let decl = "";
  for (let i = 0; i < FM; i++) for (let j = 0; j < FN; j++) decl += `  var c${i}_${j}: subgroup_matrix_result<f32, 8, 8>;\n`;
  const load = (frag: string, colMajor: boolean, offset: string, stride: number): string =>
    syntax === "template"
      ? `subgroupMatrixLoad<${frag}, ${colMajor ? "col_major" : "row_major"}>(&Sh, ${offset}, ${stride}u)`
      : `subgroupMatrixLoad<${frag}>(&Sh, ${offset}, ${colMajor}, ${stride}u)`;
  let mma = "";
  for (let i = 0; i < FM; i++) {
    mma += `      let a${i} = ${load("subgroup_matrix_left<f32, 8, 8>", false, `(sm + ${i * 8}u) * ${BKP}u + kk`, BKP)};\n`;
  }
  for (let j = 0; j < FN; j++) {
    // Right fragment is K x N. transB: staged as [n][k] -> column-major.
    // [K,N] B: staged as [k][n] -> row-major.
    mma += transB
      ? `      let b${j} = ${load("subgroup_matrix_right<f32, 8, 8>", true, `${BOFF}u + (sn + ${j * 8}u) * ${BKP}u + kk`, BKP)};\n`
      : `      let b${j} = ${load("subgroup_matrix_right<f32, 8, 8>", false, `${BOFF}u + kk * ${BNP}u + sn + ${j * 8}u`, BNP)};\n`;
  }
  for (let i = 0; i < FM; i++) for (let j = 0; j < FN; j++) mma += `      c${i}_${j} = subgroupMatrixMultiplyAccumulate(a${i}, b${j}, c${i}_${j});\n`;
  // Epilogue: each subgroup spills one 8x8 fragment at a time into its own
  // 64-float scratch; its 32 lanes then store 2 values each.
  let stores = "";
  for (let i = 0; i < FM; i++) {
    for (let j = 0; j < FN; j++) {
      const st = syntax === "template" ? `subgroupMatrixStore<row_major>(&Sh, scr, c${i}_${j}, 8u)` : `subgroupMatrixStore(&Sh, scr, c${i}_${j}, false, 8u)`;
      stores += `  ${st};
  workgroupBarrier();
  for (var e = lane; e < 64u; e += ${SG}u) {
    let row = r0 + sm + ${i * 8}u + e / 8u; let col = c0 + sn + ${j * 8}u + e % 8u;
    if (row < P.M && col < P.N) { C[row * P.N + col] = ${store(dtype, "Sh[scr + e]")}; }
  }
  workgroupBarrier();\n`;
    }
  }
  return `${header(dtype, `vec4<${dtype}>`, `vec4<${dtype}>`, {
    enables: ["chromium_experimental_subgroup_matrix"],
    // Offsets differ per subgroup (derived from local_invocation_index), which
    // is fine: each subgroup executes the matrix ops in subgroup-uniform control flow.
    directives: ["diagnostic(off, chromium.subgroup_matrix_uniformity)"],
  })}
// subgroup-matrix GEMM ${BM}x${BN}x${BK}/${WM}x${WN} transB=${transB} syntax=${syntax}
// A tile at 0, B tile at ${BOFF}, per-subgroup 8x8 output scratch at ${SCR}.
var<workgroup> Sh: array<f32, ${SCR + (WG / SG) * 64}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let K4 = P.K / 4u;
  let N4 = P.N / 4u;
  let r0 = wid.y * ${BM}u; let c0 = wid.x * ${BN}u;
  let sg = lid / ${SG}u;
  let sm = (sg / ${WN}u) * ${BM / WM}u; let sn = (sg % ${WN}u) * ${BN / WN}u;
  let lane = lid % ${SG}u; let scr = ${SCR}u + sg * 64u;
${decl}
${regs("pa", NA)}${regs("pb", NB)}  {
    let k0n = 0u;
${fetchA}${fetchB}  }
  for (var k0 = 0u; k0 < P.K; k0 += ${BK}u) {
${stashA}${stashB}    workgroupBarrier();
    let k0n = k0 + ${BK}u;
    if (k0n < P.K) {
${fetchA}${fetchB}    }
${Array.from({ length: BK / 8 }, (_, q) => `    {\n      let kk = ${q * 8}u;\n${mma}    }\n`).join("")}
    workgroupBarrier();
  }
${stores}}
`;
}
