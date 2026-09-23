/**
 * Transformer building blocks (issue #123), PyTorch-parity by construction:
 * every layer here is differential-tested against PyTorch (forward AND
 * backward, incl. parameter gradients) in test/transformer.test.ts, and the
 * parameterized ones use PyTorch's exact `state_dict()` key names — which is
 * why some fields are snake_case (`in_proj_weight`, `self_attn`, ...): the
 * `Module` reflection walk derives checkpoint keys from field names.
 *
 * Scope boundaries (deliberate, documented rather than silently wrong):
 * - No dropout anywhere (no `dropout_p`, no dropout sub-modules): outputs
 *   equal PyTorch's in `eval()` mode / with `dropout=0`. Compose `nn.Dropout`
 *   yourself if you need it in training.
 * - `MultiheadAttention`: `kdim`/`vdim` must equal `embedDim` (packed
 *   `in_proj_weight` only), no `add_bias_kv`/`add_zero_attn`, and `forward`
 *   returns only the attention output, not the averaged weights.
 * - Fully-masked attention rows produce NaN, exactly as PyTorch's math path
 *   does (softmax over all `-inf`).
 * - `RotaryEmbedding` rotates the full last axis (no partial `rotary_dim`)
 *   and has no RoPE scaling variants; cos/sin are computed in f64, then cast.
 * - Masks are plain (non-differentiable) Tensors.
 */
import { Tensor, type Rng } from "@johnhenry/math-plus-tensor-core";
import { Variable, constant } from "./variable.ts";
import { LayerNorm, Linear, Module, Parameter, asCompute, type ParamDType } from "./nn.ts";

// ---- functional ---------------------------------------------------------------

export interface ScaledDotProductAttentionOptions {
  /**
   * PyTorch `F.scaled_dot_product_attention` semantics: a BOOL mask where
   * `true` means "may attend" (false positions get `-inf` before softmax), or
   * a FLOAT mask added to the scores. Broadcast against `[..., L, S]`.
   */
  attnMask?: Tensor | null;
  /** Causal (lower-triangular) masking. Mutually exclusive with `attnMask`, as in PyTorch. */
  isCausal?: boolean;
  /** Score scale; default `1/sqrt(E)` (E = query's last axis). */
  scale?: number;
}

/**
 * `softmax(q k^T * scale + mask) v` for `q [..., L, E]`, `k [..., S, E]`,
 * `v [..., S, Ev]` → `[..., L, Ev]`. Leading axes broadcast (batched matmul).
 */
export function scaledDotProductAttention(
  q: Variable,
  k: Variable,
  v: Variable,
  options: ScaledDotProductAttentionOptions = {},
): Variable {
  const { attnMask = null, isCausal = false } = options;
  if (attnMask && isCausal) {
    throw new TypeError("scaledDotProductAttention: pass attnMask or isCausal, not both (PyTorch rejects both too)");
  }
  const e = q.shape[q.ndim - 1] as number;
  const scale = options.scale ?? 1 / Math.sqrt(e);
  let scores = q.matmul(k.transpose(-2, -1)).mul(scale);
  if (isCausal) {
    const L = q.shape[q.ndim - 2] as number;
    const S = k.shape[k.ndim - 2] as number;
    scores = scores.maskedFill(upperTriangleMask(L, S), -Infinity);
  } else if (attnMask) {
    scores = applyMask(scores, attnMask, "keep");
  }
  return scores.softmax(-1).matmul(v);
}

/**
 * Split-half ("NeoX"/Hugging Face `rotate_half`) rotary embedding:
 * `x * cos + rotate_half(x) * sin`, with `rotate_half([a, b]) = [-b, a]` over
 * the last axis. `cos`/`sin` broadcast against `x` (typically `[T, dim]`
 * against `[..., T, dim]`). See {@link RotaryEmbedding} for building them.
 */
export function applyRotaryEmbedding(x: Variable, cos: Tensor, sin: Tensor): Variable {
  const d = x.shape[x.ndim - 1] as number;
  if (d % 2 !== 0) throw new RangeError(`applyRotaryEmbedding: last axis must be even, got ${d}`);
  const half = d / 2;
  const x1 = x.narrow(-1, 0, half);
  const x2 = x.narrow(-1, half, half);
  const rotated = Variable.concat([x2.mul(-1), x1], -1);
  return x.mul(constant(cos.cast(x.dtype))).add(rotated.mul(constant(sin.cast(x.dtype))));
}

/**
 * GeGLU gate: split the last axis in half into `[value, gate]` and return
 * `gelu(value) * gate` — the ModernBERT / laya-js order (GELU on the FIRST
 * half). Note diffusers' `GEGLU` applies GELU to the second half instead;
 * swap your projection rows if porting from there. `approximate` defaults to
 * `"none"` (exact erf GELU, PyTorch's default).
 */
export function geglu(x: Variable, options: { approximate?: "none" | "tanh" } = {}): Variable {
  const d = x.shape[x.ndim - 1] as number;
  if (d % 2 !== 0) throw new RangeError(`geglu: last axis must be even, got ${d}`);
  const half = d / 2;
  const value = x.narrow(-1, 0, half);
  const gate = x.narrow(-1, half, half);
  return value.gelu({ approximate: options.approximate ?? "none" }).mul(gate);
}

// ---- modules ------------------------------------------------------------------

/**
 * Rotary position embedding (split-half / NeoX style, Hugging Face
 * `LlamaRotaryEmbedding`/ModernBERT default RoPE): `inv_freq[i] =
 * base^(-2i/dim)`, angle `pos * inv_freq`, duplicated across both halves.
 * No parameters. `forward(x)` rotates `x [..., T, dim]` for positions
 * `offset .. offset+T-1`.
 */
export class RotaryEmbedding extends Module {
  readonly dim: number;
  readonly base: number;

  constructor(dim: number, options: { base?: number } = {}) {
    super();
    if (dim % 2 !== 0) throw new RangeError(`RotaryEmbedding: dim must be even, got ${dim}`);
    this.dim = dim;
    this.base = options.base ?? 10000;
  }

  /** `[cos, sin]`, each `[T, dim]` f64, for positions `offset .. offset+T-1`. */
  cosSin(T: number, offset = 0): [Tensor, Tensor] {
    const half = this.dim / 2;
    const cos = new Float64Array(T * this.dim);
    const sin = new Float64Array(T * this.dim);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < half; i++) {
        const angle = (t + offset) * Math.pow(this.base, (-2 * i) / this.dim);
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        cos[t * this.dim + i] = c;
        cos[t * this.dim + half + i] = c;
        sin[t * this.dim + i] = s;
        sin[t * this.dim + half + i] = s;
      }
    }
    return [
      Tensor.fromTypedArray(cos, [T, this.dim], { dtype: "f64" }),
      Tensor.fromTypedArray(sin, [T, this.dim], { dtype: "f64" }),
    ];
  }

  forward(x: Variable, options: { offset?: number } = {}): Variable {
    if ((x.shape[x.ndim - 1] as number) !== this.dim) {
      throw new RangeError(`RotaryEmbedding: expected last axis ${this.dim}, got ${x.shape[x.ndim - 1]}`);
    }
    const [cos, sin] = this.cosSin(x.shape[x.ndim - 2] as number, options.offset ?? 0);
    return applyRotaryEmbedding(x, cos, sin);
  }
}

export interface MultiheadAttentionForwardOptions {
  /**
   * PyTorch `nn.MultiheadAttention` semantics — NOTE the opposite bool
   * convention from {@link scaledDotProductAttention}: here `true` means
   * "NOT allowed to attend". Float masks are added. Shape `[L, S]` or
   * `[N * numHeads, L, S]`.
   */
  attnMask?: Tensor | null;
  /** `[N, S]` (or `[S]` for unbatched input): bool `true` = padding (ignored), or float (added). */
  keyPaddingMask?: Tensor | null;
}

/**
 * PyTorch `nn.MultiheadAttention` (packed `in_proj_weight [3E, E]`,
 * `in_proj_bias [3E]`, `out_proj` Linear) — same state-dict keys, so a
 * PyTorch checkpoint loads directly. `batchFirst` defaults to `false` like
 * PyTorch (`[L, N, E]`); unbatched `[L, E]` input is accepted too.
 *
 * Extension beyond PyTorch: `rotary` applies a {@link RotaryEmbedding} to
 * the per-head queries and keys (ModernBERT-style attention), after the
 * input projection and before the scores.
 */
export class MultiheadAttention extends Module {
  readonly embedDim: number;
  readonly numHeads: number;
  readonly headDim: number;
  readonly batchFirst: boolean;
  readonly in_proj_weight: Parameter;
  readonly in_proj_bias: Parameter | null;
  readonly out_proj: Linear;
  readonly rotary: RotaryEmbedding | null;

  constructor(
    embedDim: number,
    numHeads: number,
    options: { bias?: boolean; batchFirst?: boolean; dtype?: ParamDType; rng?: Rng; rotary?: RotaryEmbedding } = {},
  ) {
    super();
    if (embedDim % numHeads !== 0) {
      throw new RangeError(`MultiheadAttention: embedDim ${embedDim} must be divisible by numHeads ${numHeads}`);
    }
    this.embedDim = embedDim;
    this.numHeads = numHeads;
    this.headDim = embedDim / numHeads;
    this.batchFirst = options.batchFirst ?? false;
    const bias = options.bias ?? true;
    // Same init DISTRIBUTION family as Linear (uniform, k = 1/sqrt(E));
    // PyTorch uses xavier_uniform for in_proj and zeros for biases. Load a
    // checkpoint for exact values.
    const packed = new Linear(embedDim, 3 * embedDim, { bias, dtype: options.dtype, rng: options.rng });
    this.in_proj_weight = packed.weight;
    this.in_proj_bias = packed.bias;
    this.out_proj = new Linear(embedDim, embedDim, { bias, dtype: options.dtype, rng: options.rng });
    this.rotary = options.rotary ?? null;
    if (this.rotary && this.rotary.dim !== this.headDim) {
      throw new RangeError(`MultiheadAttention: rotary dim ${this.rotary.dim} must equal headDim ${this.headDim}`);
    }
  }

  forward(query: Variable, key: Variable, value: Variable, options: MultiheadAttentionForwardOptions = {}): Variable {
    const unbatched = query.ndim === 2;
    const toBatchFirst = (x: Variable): Variable =>
      unbatched ? x.reshape([1, ...x.shape]) : this.batchFirst ? x : x.transpose(0, 1);
    const q0 = toBatchFirst(query);
    const k0 = toBatchFirst(key);
    const v0 = toBatchFirst(value);
    const [N, L, E] = q0.shape as [number, number, number];
    const S = k0.shape[1] as number;
    const H = this.numHeads;
    const D = this.headDim;

    const dtype = q0.dtype;
    const w = asCompute(this.in_proj_weight, dtype);
    const b = this.in_proj_bias ? asCompute(this.in_proj_bias, dtype) : null;
    const project = (x: Variable, i: number, len: number): Variable => {
      let y = x.matmul(w.narrow(0, i * E, E).transpose());
      if (b) y = y.add(b.narrow(0, i * E, E));
      return y.reshape([N, len, H, D]).transpose(1, 2); // [N, H, len, D]
    };
    let q = project(q0, 0, L);
    let k = project(k0, 1, S);
    const v = project(v0, 2, S);
    if (this.rotary) {
      q = this.rotary.forward(q);
      k = this.rotary.forward(k);
    }

    const mask = this.#combinedMask(options, N, L, S, dtype, unbatched);
    const attn = scaledDotProductAttention(q, k, v, { attnMask: mask }); // [N, H, L, D]
    let out = this.out_proj.forward(attn.transpose(1, 2).reshape([N, L, E]));
    if (unbatched) return out.reshape([L, E]);
    if (!this.batchFirst) out = out.transpose(0, 1);
    return out;
  }

  /** Both PyTorch masks folded into one additive float mask `[N|1, H|1, L, S]` (or null). */
  #combinedMask(
    options: MultiheadAttentionForwardOptions,
    N: number,
    L: number,
    S: number,
    dtype: Tensor["dtype"],
    unbatched: boolean,
  ): Tensor | null {
    let mask: Tensor | null = null;
    const add = (m: Tensor): void => {
      mask = mask ? mask.add(m) : m;
    };
    if (options.attnMask) {
      let m = toAdditive(options.attnMask, dtype, "masked");
      if (m.ndim === 3) m = m.contiguous().reshape([N, this.numHeads, L, S]);
      add(m);
    }
    if (options.keyPaddingMask) {
      const kpm = unbatched ? options.keyPaddingMask.reshape([1, S]) : options.keyPaddingMask;
      add(toAdditive(kpm, dtype, "masked").contiguous().reshape([N, 1, 1, S]));
    }
    return mask;
  }
}

export interface TransformerEncoderLayerOptions {
  /** Feed-forward width; default 2048 (PyTorch's default). */
  dimFeedforward?: number;
  /** `"relu"` (default) or `"gelu"` — exact erf GELU, matching PyTorch's `activation="gelu"`. */
  activation?: "relu" | "gelu";
  layerNormEps?: number;
  batchFirst?: boolean;
  /** Pre-norm (`x + sa(norm1(x))`) when true; post-norm (PyTorch default) when false. */
  normFirst?: boolean;
  /** `false` drops every Linear and LayerNorm bias (PyTorch `bias=False`). */
  bias?: boolean;
  dtype?: ParamDType;
  rng?: Rng;
}

/**
 * PyTorch `nn.TransformerEncoderLayer` (self-attention + ReLU/GELU FFN,
 * post- or pre-norm), with PyTorch's state-dict keys (`self_attn.*`,
 * `linear1.*`, `linear2.*`, `norm1.*`, `norm2.*`). No dropout — equals
 * PyTorch in `eval()`.
 */
export class TransformerEncoderLayer extends Module {
  readonly self_attn: MultiheadAttention;
  readonly linear1: Linear;
  readonly linear2: Linear;
  readonly norm1: LayerNorm;
  readonly norm2: LayerNorm;
  readonly normFirst: boolean;
  readonly activation: "relu" | "gelu";

  constructor(dModel: number, nhead: number, options: TransformerEncoderLayerOptions = {}) {
    super();
    const bias = options.bias ?? true;
    const { dtype, rng } = options;
    const eps = options.layerNormEps ?? 1e-5;
    const ff = options.dimFeedforward ?? 2048;
    this.self_attn = new MultiheadAttention(dModel, nhead, { bias, batchFirst: options.batchFirst ?? false, dtype, rng });
    this.linear1 = new Linear(dModel, ff, { bias, dtype, rng });
    this.linear2 = new Linear(ff, dModel, { bias, dtype, rng });
    this.norm1 = new LayerNorm(dModel, { eps, bias, dtype });
    this.norm2 = new LayerNorm(dModel, { eps, bias, dtype });
    this.normFirst = options.normFirst ?? false;
    this.activation = options.activation ?? "relu";
    if (this.activation !== "relu" && this.activation !== "gelu") {
      throw new RangeError(`TransformerEncoderLayer: activation must be "relu" or "gelu", got ${JSON.stringify(this.activation)}`);
    }
  }

  forward(src: Variable, options: { srcMask?: Tensor | null; srcKeyPaddingMask?: Tensor | null } = {}): Variable {
    const sa = (x: Variable): Variable =>
      this.self_attn.forward(x, x, x, { attnMask: options.srcMask, keyPaddingMask: options.srcKeyPaddingMask });
    const ff = (x: Variable): Variable => {
      const h = this.linear1.forward(x);
      return this.linear2.forward(this.activation === "relu" ? h.relu() : h.gelu({ approximate: "none" }));
    };
    if (this.normFirst) {
      const x = src.add(sa(this.norm1.forward(src)));
      return x.add(ff(this.norm2.forward(x)));
    }
    const x = this.norm1.forward(src.add(sa(src)));
    return this.norm2.forward(x.add(ff(x)));
  }
}

/**
 * Linear projection to `2 * dimOut` followed by {@link geglu}: `proj.weight`
 * is `[2 * dimOut, dimIn]`, rows `[0, dimOut)` feed GELU and rows
 * `[dimOut, 2*dimOut)` the gate (ModernBERT's `Wi` layout).
 */
export class GeGLU extends Module {
  readonly proj: Linear;
  readonly approximate: "none" | "tanh";

  constructor(
    dimIn: number,
    dimOut: number,
    options: { bias?: boolean; approximate?: "none" | "tanh"; dtype?: ParamDType; rng?: Rng } = {},
  ) {
    super();
    this.proj = new Linear(dimIn, 2 * dimOut, { bias: options.bias ?? true, dtype: options.dtype, rng: options.rng });
    this.approximate = options.approximate ?? "none";
  }

  forward(x: Variable): Variable {
    return geglu(this.proj.forward(x), { approximate: this.approximate });
  }
}

// ---- helpers ------------------------------------------------------------------

/** Bool `[L, S]`, true strictly above the diagonal (the positions causal attention hides). */
function upperTriangleMask(L: number, S: number): Tensor {
  const data = new Uint8Array(L * S);
  for (let i = 0; i < L; i++) for (let j = i + 1; j < S; j++) data[i * S + j] = 1;
  return Tensor.fromTypedArray(data, [L, S], { dtype: "bool" });
}

/**
 * A bool or float mask as an additive float mask in `dtype`. `boolMeans`
 * picks the bool convention: `"keep"` (SDPA: true = attend) or `"masked"`
 * (nn.MultiheadAttention: true = hidden).
 */
function toAdditive(mask: Tensor, dtype: Tensor["dtype"], boolMeans: "keep" | "masked"): Tensor {
  if (mask.dtype !== "bool") return mask.dtype === dtype ? mask : mask.cast(dtype);
  const hidden = boolMeans === "keep" ? mask.logicalNot() : mask;
  return Tensor.where(hidden, Tensor.full([], -Infinity, { dtype }), Tensor.zeros([], { dtype }));
}

function applyMask(scores: Variable, mask: Tensor, boolMeans: "keep" | "masked"): Variable {
  if (mask.dtype === "bool") {
    return scores.maskedFill(boolMeans === "keep" ? mask.logicalNot() : mask, -Infinity);
  }
  return scores.add(constant(mask.dtype === scores.dtype ? mask : mask.cast(scores.dtype)));
}
