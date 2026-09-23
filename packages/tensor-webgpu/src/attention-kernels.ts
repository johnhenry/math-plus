/**
 * WGSL generators for fused (flash-style) scaled-dot-product attention,
 * ported from laya-js's WebGPU backend (issue #126) and narrowed to this
 * package's conventions: f32 only, `(batch, seq, dim)` row-major operands,
 * and an optional f32 mask (nonzero = attend) broadcast to
 * `(batch, seqQ, seqK)` through strides that may be 0.
 *
 * One workgroup handles one block of queries of one batch entry and walks
 * the keys in tiles held in workgroup memory, with an online (running max /
 * running sum) softmax in f32 — the `(seqQ, seqK)` score matrix never
 * exists in global memory.
 *
 * Masked-key-tile skipping: with a mask, each workgroup first scans the
 * mask for the range of keys that ANY of its queries may attend to, and
 * only walks the key tiles overlapping that range. Tiles outside it would
 * contribute exactly nothing (every score masked), so a sliding-window or
 * padding mask stops costing the full `seqQ x seqK` work. Tiles inside the
 * range still apply the mask per element.
 *
 * Fully masked query rows produce 0 (not NaN): softmax over an empty set is
 * undefined, and 0 is the value a caller can keep computing with.
 *
 * Two kernels:
 *  - `fast` (head dim 32 or 64): 32 queries x 16 keys per tile, 128
 *    threads, vec4 loads, each thread a 2x2 score block. Needs ~11 KiB
 *    (D=32) / ~20 KiB (D=64) of workgroup memory — D=64 only fits devices
 *    that raised `maxComputeWorkgroupStorageSize` above the 16 KiB default
 *    (detectWebGPU() requests the adapter's maximum).
 *  - `generic` (any head dim): square BQ = BKV tiles, sized to fit the
 *    device's workgroup-memory limit.
 */

export type AttentionKernel = "fast" | "generic";

export interface GenericAttentionConfig {
  /** Queries per workgroup. */
  BQ: number;
  /** Keys per tile. */
  BKV: number;
  /** Threads per workgroup. */
  WG: number;
}

export interface AttentionVariant {
  /** Head dim (`dim`). */
  D: number;
  /** The kernel reads a mask binding. */
  masked: boolean;
  /** With a mask: skip key tiles no query in the block may attend to. Default true. */
  skipMaskedTiles?: boolean;
}

const GENERIC_CONFIGS: readonly GenericAttentionConfig[] = [
  { BQ: 32, BKV: 32, WG: 128 },
  { BQ: 16, BKV: 16, WG: 128 },
  { BQ: 8, BKV: 8, WG: 64 },
  { BQ: 4, BKV: 4, WG: 64 },
];

/** Workgroup-memory bytes the generic kernel needs for head dim `D` and tile config `c`. */
export function genericAttentionBytes(D: number, c: GenericAttentionConfig, masked: boolean): number {
  return 4 * (c.BQ * D + c.BKV * (D + 1) + c.BKV * D + c.BQ * c.BKV + 3 * c.BQ) + (masked ? 16 : 0);
}

/** Workgroup-memory bytes the fast kernel needs for head dim `D` (32 or 64). */
export function fastAttentionBytes(D: number, masked: boolean): number {
  const D4 = D / 4;
  return 16 * (32 * D4 + 16 * (D4 + 1) + 16 * D4) + 4 * (32 * 17 + 128 + 3 * 32) + (masked ? 16 : 0);
}

/** The largest generic tile config whose workgroup memory fits `limit` bytes, or undefined. */
export function genericAttentionConfig(D: number, limit: number, masked: boolean): GenericAttentionConfig | undefined {
  return GENERIC_CONFIGS.find((c) => genericAttentionBytes(D, c, masked) <= limit);
}

/** Queries per workgroup of the fast kernel. */
export const FAST_BQ = 32;

const ATTN_NEG = "-3.0e38";

/** Parameters struct. `msb`/`msq`/`msk` are the mask's element strides over (batch, query, key); 0 broadcasts. */
function attnParamsStruct(masked: boolean): string {
  return masked
    ? "struct Params { Lq: u32, Lk: u32, msb: u32, msq: u32, msk: u32, scale: f32 };"
    : "struct Params { Lq: u32, Lk: u32, scale: f32 };";
}

function attnHeader(elem: string, masked: boolean): string {
  const lines = [
    attnParamsStruct(masked),
    `@group(0) @binding(0) var<storage, read> Q: array<${elem}>;`,
    `@group(0) @binding(1) var<storage, read> Kt: array<${elem}>;`,
    `@group(0) @binding(2) var<storage, read> V: array<${elem}>;`,
  ];
  if (masked) lines.push("@group(0) @binding(3) var<storage, read> Mk: array<f32>;");
  const o = masked ? 4 : 3;
  lines.push(`@group(0) @binding(${o}) var<storage, read_write> O: array<f32>;`);
  lines.push(`@group(0) @binding(${o + 1}) var<uniform> P: Params;`);
  return lines.join("\n");
}

/**
 * The key-tile loop header. With `skip`, first reduce (over the workgroup)
 * the [lo, hi) range of keys any query in this block may see, then loop
 * over only the tiles covering it. Loop bounds come from
 * `workgroupUniformLoad`, so the barriers inside the loop stay in uniform
 * control flow.
 */
function attnTileLoop(skip: boolean, q0: string, BQ: number, BKV: number, WG: number): string {
  if (!skip) return `  for (var kt = 0u; kt < P.Lk; kt += ${BKV}u) {`;
  return `  if (lid == 0u) { atomicStore(&kRange[0], 0xffffffffu); atomicStore(&kRange[1], 0u); }
  workgroupBarrier();
  {
    var lo = 0xffffffffu; var hi = 0u;
    let nq = min(${BQ}u, P.Lq - ${q0});
    for (var j = lid; j < P.Lk; j += ${WG}u) {
      var vis = false;
      for (var qi = 0u; qi < nq; qi++) {
        vis = vis || Mk[mBase + (${q0} + qi) * P.msq + j * P.msk] != 0.0;
        if (vis || P.msq == 0u) { break; }
      }
      if (vis) { lo = min(lo, j); hi = max(hi, j + 1u); }
    }
    if (hi > 0u) { atomicMin(&kRange[0], lo); atomicMax(&kRange[1], hi); }
  }
  workgroupBarrier();
  if (lid == 0u) { kLo = (atomicLoad(&kRange[0]) / ${BKV}u) * ${BKV}u; kHi = atomicLoad(&kRange[1]); }
  let ktLo = workgroupUniformLoad(&kLo);
  let ktHi = workgroupUniformLoad(&kHi);
  for (var kt = ktLo; kt < ktHi; kt += ${BKV}u) {`;
}

const ATTN_SKIP_DECLS = "var<workgroup> kRange: array<atomic<u32>, 2>;\nvar<workgroup> kLo: u32;\nvar<workgroup> kHi: u32;\n";

/**
 * Generic flash attention (any head dim): one workgroup per (query block,
 * batch entry); dispatch `[ceil(seqQ / BQ), batch]`.
 */
export function genericAttentionWGSL(v: AttentionVariant, c: GenericAttentionConfig): string {
  const { D, masked } = v;
  const skip = masked && (v.skipMaskedTiles ?? true);
  const { BQ, BKV, WG } = c;
  const NS = Math.ceil((BQ * BKV) / WG);
  const NO = Math.ceil((BQ * D) / WG);
  let decl = "";
  for (let i = 0; i < NO; i++) decl += `  var o${i} = 0.0;\n`;
  let scores = "";
  for (let i = 0; i < NS; i++) {
    scores += `    { let e = lid + ${i * WG}u;
      if (e < ${BQ * BKV}u) {
        let qi = e / ${BKV}u; let kj = e % ${BKV}u;
        let qg = q0 + qi; let kg = kt + kj;
        var s = NEG;
        if (qg < P.Lq && kg < P.Lk) {
          var acc = 0.0;
          for (var d = 0u; d < ${D}u; d++) { acc += Qs[qi * ${D}u + d] * Ks[kj * ${D + 1}u + d]; }
          s = acc;
${masked ? "          if (Mk[mBase + qg * P.msq + kg * P.msk] == 0.0) { s = NEG; }\n" : ""}        }
        S[e] = s;
      }
    }\n`;
  }
  let accum = "";
  for (let i = 0; i < NO; i++) {
    accum += `    { let e = lid + ${i * WG}u;
      if (e < ${BQ * D}u) {
        let qi = e / ${D}u; let d = e % ${D}u;
        var acc = o${i} * Al[qi];
        for (var j = 0u; j < ${BKV}u; j++) { acc += S[qi * ${BKV}u + j] * Vs[j * ${D}u + d]; }
        o${i} = acc;
      }
    }\n`;
  }
  let write = "";
  for (let i = 0; i < NO; i++) {
    write += `  { let e = lid + ${i * WG}u;
    if (e < ${BQ * D}u) {
      let qi = e / ${D}u; let d = e % ${D}u;
      if (q0 + qi < P.Lq) { let l = Ls[qi]; O[oBase + (q0 + qi) * ${D}u + d] = select(0.0, o${i} / l, l > 0.0); }
    }
  }\n`;
  }
  return `${attnHeader("f32", masked)}
const NEG = ${ATTN_NEG};
var<workgroup> Qs: array<f32, ${BQ * D}>;
var<workgroup> Ks: array<f32, ${BKV * (D + 1)}>;
var<workgroup> Vs: array<f32, ${BKV * D}>;
var<workgroup> S: array<f32, ${BQ * BKV}>;
var<workgroup> Ms: array<f32, ${BQ}>;
var<workgroup> Ls: array<f32, ${BQ}>;
var<workgroup> Al: array<f32, ${BQ}>;
${skip ? ATTN_SKIP_DECLS : ""}
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let q0 = wid.x * ${BQ}u; let b = wid.y;
  let qBase = b * P.Lq * ${D}u;
  let kBase = b * P.Lk * ${D}u;
  let oBase = qBase;
${masked ? "  let mBase = b * P.msb;\n" : ""}  for (var t = lid; t < ${BQ * D}u; t += ${WG}u) {
    var x = 0.0;
    if (q0 + t / ${D}u < P.Lq) { x = Q[qBase + q0 * ${D}u + t] * P.scale; }
    Qs[t] = x;
  }
  if (lid < ${BQ}u) { Ms[lid] = NEG; Ls[lid] = 0.0; }
${decl}
${attnTileLoop(skip, "q0", BQ, BKV, WG)}
    for (var t = lid; t < ${BKV * D}u; t += ${WG}u) {
      let kj = t / ${D}u;
      var kx = 0.0; var vx = 0.0;
      if (kt + kj < P.Lk) {
        kx = Kt[kBase + kt * ${D}u + t];
        vx = V[kBase + kt * ${D}u + t];
      }
      Ks[kj * ${D + 1}u + t % ${D}u] = kx; Vs[t] = vx;
    }
    workgroupBarrier();
${scores}    workgroupBarrier();
    if (lid < ${BQ}u) {
      let r = lid;
      var mx = NEG;
      for (var j = 0u; j < ${BKV}u; j++) { mx = max(mx, S[r * ${BKV}u + j]); }
      let mprev = Ms[r];
      let mnew = max(mprev, mx);
      var alpha = 1.0;
      var l = Ls[r];
      if (mnew > -1.0e38) {
        alpha = exp(mprev - mnew);
        l = l * alpha;
        for (var j = 0u; j < ${BKV}u; j++) {
          let s = S[r * ${BKV}u + j];
          var p = 0.0;
          if (s > -1.0e38) { p = exp(s - mnew); }
          S[r * ${BKV}u + j] = p;
          l += p;
        }
      } else {
        for (var j = 0u; j < ${BKV}u; j++) { S[r * ${BKV}u + j] = 0.0; }
      }
      Ms[r] = mnew; Ls[r] = l; Al[r] = alpha;
    }
    workgroupBarrier();
${accum}    workgroupBarrier();
  }
${write}}
`;
}

/**
 * Register-blocked flash attention for head dim 32 or 64: 32 queries x 16
 * keys per tile, 128 threads; each thread computes a 2x2 score block from
 * vec4 tiles, the online softmax uses 4 lanes per row, and P·V gives each
 * thread 2 rows x D/32 vec4s. Dispatch `[ceil(seqQ / 32), batch]`.
 */
export function fastAttentionWGSL(v: AttentionVariant): string {
  const { D, masked } = v;
  if (D !== 32 && D !== 64) throw new RangeError(`fastAttentionWGSL: head dim must be 32 or 64, got ${D}`);
  const skip = masked && (v.skipMaskedTiles ?? true);
  const BQ = FAST_BQ, BKV = 16, WG = 128, D4 = D / 4, KP = D4 + 1, TD = D4 / 8;
  let score = "";
  for (let a = 0; a < 2; a++) for (let c = 0; c < 2; c++) score += `    var s${a}${c} = 0.0;\n`;
  score += `    for (var d = 0u; d < ${D4}u; d++) {
      let qa = Qs[sr * ${D4}u + d]; let qb = Qs[(sr + 1u) * ${D4}u + d];
      let ka = Ks[sc * ${KP}u + d]; let kb = Ks[(sc + 1u) * ${KP}u + d];
      s00 += dot(qa, ka); s01 += dot(qa, kb); s10 += dot(qb, ka); s11 += dot(qb, kb);
    }\n`;
  for (let a = 0; a < 2; a++)
    for (let c = 0; c < 2; c++) {
      score += `    { let qg = q0 + sr + ${a}u; let kg = kt + sc + ${c}u; var s = s${a}${c};
      if (qg >= P.Lq || kg >= P.Lk) { s = NEG; }
${masked ? "      else if (Mk[mBase + qg * P.msq + kg * P.msk] == 0.0) { s = NEG; }\n" : ""}      S[(sr + ${a}u) * ${BKV + 1}u + sc + ${c}u] = s; }\n`;
    }
  let decl = "";
  for (let a = 0; a < 2; a++) for (let j = 0; j < TD; j++) decl += `  var o${a}_${j} = vec4<f32>(0.0);\n`;
  let accum = "    let al0 = Al[orow]; let al1 = Al[orow + 1u];\n";
  for (let j = 0; j < TD; j++) accum += `    o0_${j} *= al0; o1_${j} *= al1;\n`;
  accum += `    for (var j = 0u; j < ${BKV}u; j++) {
      let p0 = S[orow * ${BKV + 1}u + j]; let p1 = S[(orow + 1u) * ${BKV + 1}u + j];\n`;
  for (let j = 0; j < TD; j++) accum += `      { let vv = Vs[j * ${D4}u + ocol + ${j * 8}u]; o0_${j} += p0 * vv; o1_${j} += p1 * vv; }\n`;
  accum += "    }\n";
  let write = "";
  for (let a = 0; a < 2; a++) {
    write += `  { let qg = q0 + orow + ${a}u;\n    if (qg < P.Lq) {\n      let l = Ls[orow + ${a}u];\n      let inv = select(0.0, 1.0 / l, l > 0.0);\n`;
    for (let j = 0; j < TD; j++) {
      const idx = `oBase + qg * ${D}u + (ocol + ${j * 8}u) * 4u`;
      write += `      { let o = o${a}_${j} * inv; O[${idx}] = o.x; O[${idx} + 1u] = o.y; O[${idx} + 2u] = o.z; O[${idx} + 3u] = o.w; }\n`;
    }
    write += "    }\n  }\n";
  }
  return `${attnHeader("vec4<f32>", masked)}
const NEG = ${ATTN_NEG};
var<workgroup> Qs: array<vec4<f32>, ${BQ * D4}>;
var<workgroup> Ks: array<vec4<f32>, ${BKV * KP}>;
var<workgroup> Vs: array<vec4<f32>, ${BKV * D4}>;
var<workgroup> S: array<f32, ${BQ * (BKV + 1)}>;
var<workgroup> red: array<f32, ${WG}>;
var<workgroup> Ms: array<f32, ${BQ}>;
var<workgroup> Ls: array<f32, ${BQ}>;
var<workgroup> Al: array<f32, ${BQ}>;
${skip ? ATTN_SKIP_DECLS : ""}
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let q0 = wid.x * ${BQ}u; let b = wid.y;
  let qBase = b * P.Lq * ${D4}u;
  let kBase = b * P.Lk * ${D4}u;
  let oBase = b * P.Lq * ${D}u;
${masked ? "  let mBase = b * P.msb;\n" : ""}  let sr = (lid / 8u) * 2u; let sc = (lid % 8u) * 2u;
  let rr = lid / 4u; let lane = lid % 4u;
  let orow = (lid / 8u) * 2u; let ocol = lid % 8u;
  for (var t = lid; t < ${BQ * D4}u; t += ${WG}u) {
    var x = vec4<f32>(0.0);
    if (q0 + t / ${D4}u < P.Lq) { x = Q[qBase + q0 * ${D4}u + t] * P.scale; }
    Qs[t] = x;
  }
  if (lid < ${BQ}u) { Ms[lid] = NEG; Ls[lid] = 0.0; }
${decl}
${attnTileLoop(skip, "q0", BQ, BKV, WG)}
    for (var t = lid; t < ${BKV * D4}u; t += ${WG}u) {
      let kj = t / ${D4}u; let d = t % ${D4}u;
      var kx = vec4<f32>(0.0); var vx = vec4<f32>(0.0);
      if (kt + kj < P.Lk) {
        kx = Kt[kBase + kt * ${D4}u + t];
        vx = V[kBase + kt * ${D4}u + t];
      }
      Ks[kj * ${KP}u + d] = kx; Vs[t] = vx;
    }
    workgroupBarrier();
${score}    workgroupBarrier();
    // online softmax: 4 lanes per row, ${BKV / 4} keys per lane
    let sb = rr * ${BKV + 1}u + lane * ${BKV / 4}u;
    var lm = NEG;
    for (var j = 0u; j < ${BKV / 4}u; j++) { lm = max(lm, S[sb + j]); }
    red[lid] = lm;
    workgroupBarrier();
    let mprev = Ms[rr];
    let mnew = max(mprev, max(max(red[rr * 4u], red[rr * 4u + 1u]), max(red[rr * 4u + 2u], red[rr * 4u + 3u])));
    let live = mnew > -1.0e38;
    var ls = 0.0;
    for (var j = 0u; j < ${BKV / 4}u; j++) {
      let s = S[sb + j];
      var p = 0.0;
      if (live && s > -1.0e38) { p = exp(s - mnew); }
      S[sb + j] = p; ls += p;
    }
    workgroupBarrier();
    red[lid] = ls;
    workgroupBarrier();
    if (lane == 0u) {
      let alpha = select(1.0, exp(mprev - mnew), live);
      Ls[rr] = Ls[rr] * alpha + red[lid] + red[lid + 1u] + red[lid + 2u] + red[lid + 3u];
      Ms[rr] = mnew; Al[rr] = alpha;
    }
    workgroupBarrier();
${accum}    workgroupBarrier();
  }
${write}}
`;
}
