/**
 * Minimal nn slice (issue #9): Parameter, Module, Linear/Embedding/
 * LayerNorm, mseLoss/crossEntropy. Plain TypeScript class composition, no
 * decorators or metaclass-style registration — `Module.parameters()` finds
 * `Parameter`/`Module` instances via a reflection pass over own properties,
 * per the source design's explicit preference.
 */
import { Tensor, random, allocate, isBigIntDType, isHalfDType, type AnyTypedArray, type DType, type Rng } from "@johnhenry/math-plus-tensor-core";
import { Variable, constant } from "./variable.ts";
import { LEGACY_LINEAR_LAYOUT } from "./io.ts";

/**
 * Parameter storage dtypes (issue #123). `f32` is the default everywhere.
 * `f16`/`bf16` are STORAGE-only: tensor-core cannot compute in half
 * precision, so layers upcast such parameters to the input's dtype on the
 * fly (differentiably, via `Variable.cast`). That makes half parameters fine
 * for inference and checkpoint fidelity, but NOT for training — `optim.*`
 * updates run tensor-core arithmetic on the parameter itself, which throws
 * for half dtypes. Train in f32/f64.
 */
export type ParamDType = "f32" | "f64" | "f16" | "bf16";

/** Uniform(-k, k) init in `dtype` (half dtypes are drawn in f32, then rounded). */
function uniformParam(shape: number[], k: number, dtype: ParamDType, rng: Rng | undefined): Tensor {
  const drawDtype = isHalfDType(dtype) ? "f32" : dtype;
  const t = random.uniform(shape, { min: -k, max: k, dtype: drawDtype, rng });
  return drawDtype === dtype ? t : t.cast(dtype);
}

function filledParam(shape: number[], value: number, dtype: ParamDType): Tensor {
  return Tensor.full(shape, value, { dtype });
}

/**
 * `p` as seen by a computation in `dtype`: unchanged if it already matches
 * or is a full-precision dtype (a real f32-vs-f64 mismatch should still hit
 * tensor-core's loud no-implicit-promotion error), upcast if it is a
 * half-precision storage dtype.
 */
export function asCompute(p: Variable, dtype: DType): Variable {
  return p.dtype !== dtype && isHalfDType(p.dtype) ? p.cast(dtype) : p;
}

export interface LoadStateDictOptions {
  /**
   * Default `true`: every module parameter must be present and every dict key
   * must name a parameter. `false` loads the intersection and ignores the
   * rest (PyTorch's `strict=False`; useful for checkpoints carrying buffers or
   * heads you don't model). Shape mismatches ALWAYS throw.
   */
  strict?: boolean;
  /**
   * Treat `nn.Linear` weights in `dict` as the pre-#123 `[in, out]` layout
   * and transpose them on load. Set automatically for dicts returned by
   * `io.loadCheckpoint` on version-1 files; pass it yourself only for old
   * state dicts obtained some other way.
   */
  legacyLinearLayout?: boolean;
}

/** A leaf Variable that always requires grad and is collected by `Module.parameters()`. */
export class Parameter extends Variable {
  constructor(value: Tensor) {
    super(value, true, null);
  }
}

export abstract class Module {
  // Not `...inputs: Variable[]`: Embedding legitimately takes a raw integer-
  // index Tensor (not a differentiable Variable), so the base signature
  // stays loose and each subclass narrows its own `forward`.
  abstract forward(...inputs: unknown[]): Variable;

  /** Recursively collects every `Parameter` reachable through own-property fields. */
  parameters(): Parameter[] {
    const found: Parameter[] = [];
    for (const key of Object.keys(this)) {
      const v = (this as unknown as Record<string, unknown>)[key];
      if (v instanceof Parameter) found.push(v);
      else if (v instanceof Module) found.push(...v.parameters());
    }
    return found;
  }

  /**
   * Same reflection walk as {@link parameters}, but keyed by dotted path
   * (e.g. `"layer1.weight"` for a `Parameter` nested inside a sub-`Module`
   * field named `layer1`) — the basis for {@link stateDict}/
   * {@link loadStateDict} (issue #42), where a checkpoint needs to know
   * WHICH parameter each saved tensor belongs to, not just a flat list.
   */
  namedParameters(): Record<string, Parameter> {
    const found: Record<string, Parameter> = {};
    for (const key of Object.keys(this)) {
      const v = (this as unknown as Record<string, unknown>)[key];
      if (v instanceof Parameter) {
        found[key] = v;
      } else if (v instanceof Module) {
        for (const [subKey, p] of Object.entries(v.namedParameters())) {
          found[`${key}.${subKey}`] = p;
        }
      }
    }
    return found;
  }

  /** Every named parameter's current (detached) value — a plain, serializable snapshot. See {@link loadStateDict} for the inverse. */
  stateDict(): Record<string, Tensor> {
    const out: Record<string, Tensor> = {};
    for (const [name, p] of Object.entries(this.namedParameters())) out[name] = p.value;
    return out;
  }

  /**
   * Every sub-`Module` reachable through own-property fields, keyed by dotted
   * path, INCLUDING this module itself under `""` (PyTorch's
   * `named_modules()`).
   */
  namedModules(): Record<string, Module> {
    const found: Record<string, Module> = { "": this };
    for (const key of Object.keys(this)) {
      const v = (this as unknown as Record<string, unknown>)[key];
      if (v instanceof Module) {
        for (const [subKey, m] of Object.entries(v.namedModules())) {
          found[subKey === "" ? key : `${key}.${subKey}`] = m;
        }
      }
    }
    return found;
  }

  /**
   * Reassigns each named `Parameter`'s mutable `.value` from `dict` (the
   * SAME "leaf reassignment between steps" mechanism `optim.*` already uses
   * — see `Variable.value`'s own doc comment: a JS object-reference repoint,
   * not an in-place Tensor mutation).
   *
   * PyTorch `load_state_dict` semantics (issue #123):
   * - strict by default — throws naming any parameter missing from `dict`,
   *   or any `dict` key that doesn't match a real parameter (a checkpoint
   *   silently loading onto the wrong architecture should be loud); see
   *   {@link LoadStateDictOptions.strict}.
   * - shapes must match exactly (always checked, throws otherwise).
   * - values are CAST to the parameter's existing dtype (like
   *   `param.copy_(src)`): an f16 safetensors checkpoint loads into f32
   *   parameters, an old f64 checkpoint into today's f32 default. Construct
   *   the module with `{ dtype }` to choose the storage dtype.
   * - pre-#123 `[in, out]` Linear weights are transposed when the dict is
   *   marked legacy — see {@link LoadStateDictOptions.legacyLinearLayout}.
   */
  loadStateDict(dict: Readonly<Record<string, Tensor>>, options: LoadStateDictOptions = {}): void {
    const strict = options.strict ?? true;
    const legacy =
      options.legacyLinearLayout ?? (dict as unknown as Record<symbol, unknown>)[LEGACY_LINEAR_LAYOUT] === true;
    const named = this.namedParameters();
    const moduleKeys = new Set(Object.keys(named));
    const dictKeys = new Set(Object.keys(dict));
    if (strict) {
      for (const key of moduleKeys) {
        if (!dictKeys.has(key)) throw new Error(`loadStateDict: missing parameter "${key}" in the given state dict`);
      }
      for (const key of dictKeys) {
        if (!moduleKeys.has(key)) throw new Error(`loadStateDict: state dict has unexpected parameter "${key}" (not in this module)`);
      }
    }

    const legacyWeights = new Set<string>();
    if (legacy) {
      for (const [path, m] of Object.entries(this.namedModules())) {
        if (m instanceof Linear) legacyWeights.add(path === "" ? "weight" : `${path}.weight`);
      }
    }

    // Validate everything before assigning anything: a failed load must not
    // leave the module half-updated.
    const updates: Array<[Parameter, Tensor]> = [];
    for (const [name, p] of Object.entries(named)) {
      if (!dictKeys.has(name)) continue;
      let t = dict[name] as Tensor;
      if (legacyWeights.has(name) && t.ndim === 2) t = t.transpose().contiguous();
      const want = p.value.shape;
      if (t.ndim !== want.length || t.shape.some((d, i) => d !== want[i])) {
        throw new Error(
          `loadStateDict: shape mismatch for "${name}": checkpoint [${t.shape}] vs parameter [${want}]` +
            (legacy ? "" : " (a pre-#123 [in, out] Linear weight? see the legacyLinearLayout option)"),
        );
      }
      updates.push([p, t.dtype === p.value.dtype ? t : t.cast(p.value.dtype)]);
    }
    for (const [p, t] of updates) p.value = t;
  }

  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad();
  }
}

/**
 * `y = x W^T + b` with PyTorch's layout (issue #123): `weight` is
 * `[outFeatures, inFeatures]`, `bias` is `[outFeatures]` — so PyTorch, Hugging
 * Face, MLX and safetensors checkpoints load with no transposes.
 *
 * BREAKING vs. <= 0.2: weights used to be `[in, out]` and always f64. Old
 * MPCK checkpoints still load (see `io.loadCheckpoint` /
 * `LoadStateDictOptions.legacyLinearLayout`); code that indexed
 * `linear.weight.value` directly must swap its axes.
 *
 * `x` may have any number of leading batch axes (`[..., in] -> [..., out]`),
 * or be 1-D (`[in] -> [out]`). Init matches PyTorch's default distribution
 * (uniform(-k, k), k = 1/sqrt(in)) but not its RNG stream.
 */
export class Linear extends Module {
  readonly weight: Parameter;
  readonly bias: Parameter | null;
  readonly inFeatures: number;
  readonly outFeatures: number;

  constructor(
    inFeatures: number,
    outFeatures: number,
    options: { bias?: boolean; rng?: Rng; dtype?: ParamDType } = {},
  ) {
    super();
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    const useBias = options.bias ?? true;
    const dtype = options.dtype ?? "f32";
    const k = 1 / Math.sqrt(inFeatures);
    this.weight = new Parameter(uniformParam([outFeatures, inFeatures], k, dtype, options.rng));
    this.bias = useBias ? new Parameter(uniformParam([outFeatures], k, dtype, options.rng)) : null;
  }

  forward(x: Variable): Variable {
    if (x.ndim === 1) return this.forward(x.reshape([1, x.shape[0] as number])).reshape([this.outFeatures]);
    const y = x.matmul(asCompute(this.weight, x.dtype).transpose());
    return this.bias ? y.add(asCompute(this.bias, x.dtype)) : y;
  }
}

export class Embedding extends Module {
  readonly weight: Parameter;

  /** `dtype` default `"f32"` (was always f64 before issue #123). Half dtypes gather in half, then upcast the gathered rows to f32. */
  constructor(numEmbeddings: number, embeddingDim: number, options: { rng?: Rng; dtype?: ParamDType } = {}) {
    super();
    const dtype = options.dtype ?? "f32";
    const drawDtype = isHalfDType(dtype) ? "f32" : dtype;
    const w = random.normal([numEmbeddings, embeddingDim], { std: 1, dtype: drawDtype, rng: options.rng });
    this.weight = new Parameter(drawDtype === dtype ? w : w.cast(dtype));
  }

  /**
   * `indices`: integer tensor of row indices to gather. Backward is a
   * scatter-add of the incoming gradient's rows back into the matching
   * weight rows (accumulating duplicates) — tensor-core has no native
   * scatter-add primitive yet, so this loops over plain arrays. Fine for
   * v1/toy-scale embedding tables; a real scatter-add kernel is future work.
   *
   * Accumulation is sparse: contributions land in a `Map<rowIdx, Float64Array>`
   * keyed by the (few) touched rows, not a dense `numEmbeddings x embeddingDim`
   * table walked/filled on every backward call — that dense allocation used
   * to dominate cost (measured ~770ms for a batch-of-3 gradient into a
   * 50,000x256 table) regardless of how few rows the batch actually touched.
   * The only full-table-sized allocation left is the final zero-initialized
   * typed array `Tensor.fromTypedArray` needs (a native allocation, not a
   * JS-level fill loop), scattered into only at the touched rows.
   */
  forward(indices: Tensor): Variable {
    const idxArray = [...(indices.toArray() as (number | bigint)[])].map(Number);
    const storageDtype = this.weight.value.dtype;
    const gatheredRaw = this.weight.value.take(idxArray, { axis: 0 });
    // Half storage: hand back f32 rows; the gradient is built in f32 and
    // rounded back to the storage dtype.
    const dtype: DType = isHalfDType(storageDtype) ? "f32" : storageDtype;
    const gathered = dtype === storageDtype ? gatheredRaw : gatheredRaw.cast(dtype);
    const [numEmbeddings, embeddingDim] = this.weight.value.shape as [number, number];

    return Variable.fromOp(gathered, [this.weight], (g) => {
      const gRows = g.contiguous().toArray() as number[][];

      const acc = new Map<number, Float64Array>();
      idxArray.forEach((rowIdx, i) => {
        let row = acc.get(rowIdx);
        if (!row) {
          row = new Float64Array(embeddingDim);
          acc.set(rowIdx, row);
        }
        const gRow = gRows[i] as number[];
        for (let d = 0; d < embeddingDim; d++) {
          (row as Float64Array)[d] += gRow[d] as number;
        }
      });

      const flat = allocate(dtype, numEmbeddings * embeddingDim);
      if (isBigIntDType(dtype)) {
        const big = flat as unknown as { [i: number]: bigint };
        for (const [rowIdx, row] of acc) {
          const base = rowIdx * embeddingDim;
          for (let d = 0; d < embeddingDim; d++) {
            big[base + d] = BigInt(Math.trunc(row[d] as number));
          }
        }
      } else {
        const numeric = flat as Exclude<AnyTypedArray, BigInt64Array | BigUint64Array>;
        for (const [rowIdx, row] of acc) {
          const base = rowIdx * embeddingDim;
          numeric.set(row, base);
        }
      }

      const grad = Tensor.fromTypedArray(flat, [numEmbeddings, embeddingDim], { dtype });
      return [dtype === storageDtype ? grad : grad.cast(storageDtype)];
    });
  }
}

/**
 * Layer normalization over the LAST axis (biased variance, like PyTorch).
 * `bias: false` drops the additive term entirely (PyTorch >= 2.1's
 * `LayerNorm(bias=False)`, used by ModernBERT and `TransformerEncoderLayer(
 * bias=False)`) — `bias` is then `null` and absent from the state dict.
 * `dtype` default `"f32"` (was always f64 before issue #123).
 */
export class LayerNorm extends Module {
  readonly weight: Parameter;
  readonly bias: Parameter | null;
  readonly eps: number;

  constructor(normalizedShape: number, options: { eps?: number; bias?: boolean; dtype?: ParamDType } = {}) {
    super();
    const dtype = options.dtype ?? "f32";
    this.eps = options.eps ?? 1e-5;
    this.weight = new Parameter(filledParam([normalizedShape], 1, dtype));
    this.bias = (options.bias ?? true) ? new Parameter(filledParam([normalizedShape], 0, dtype)) : null;
  }

  /** Normalizes over the LAST axis of `x`. */
  forward(x: Variable): Variable {
    const axis = x.ndim - 1;
    const mean = x.mean(axis).unsqueeze(axis);
    const centered = x.sub(mean);
    const variance = centered.mul(centered).mean(axis).unsqueeze(axis);
    const std = variance.add(this.eps).sqrt();
    const normalized = centered.div(std);
    // weight/bias shape [normalizedShape] broadcasts against [..., normalizedShape]
    // via ordinary trailing-axis alignment — no unsqueeze needed here.
    const scaled = normalized.mul(asCompute(this.weight, x.dtype));
    return this.bias ? scaled.add(asCompute(this.bias, x.dtype)) : scaled;
  }
}

/**
 * Composes an ordered list of sub-modules, `forward` chaining them
 * (issue #71). Stores each layer as a NUMBERED own-property
 * (`this["0"]`, `this["1"]`, ...) rather than an array field — `Module`'s
 * existing `parameters()`/`namedParameters()` reflection walk only
 * recognizes `Parameter`/`Module` values on own properties (an array field
 * would be invisible to it), so this gets full `parameters()`/
 * `stateDict()`/`loadStateDict()` support with ZERO changes to the base
 * `Module` class. Dotted-path names come out as `"0.weight"`, `"1.weight"`,
 * etc. — the same convention PyTorch's own `nn.Sequential` uses.
 */
export class Sequential extends Module {
  readonly length: number;

  constructor(layers: readonly Module[]) {
    super();
    layers.forEach((layer, i) => {
      (this as unknown as Record<string, Module>)[String(i)] = layer;
    });
    this.length = layers.length;
  }

  forward(x: Variable): Variable {
    let out = x;
    for (let i = 0; i < this.length; i++) {
      out = (this as unknown as Record<string, Module>)[String(i)]!.forward(out) as Variable;
    }
    return out;
  }
}

/**
 * Inverted dropout (issue #71): zeroes each element independently with
 * probability `p`, scaling survivors by `1/(1-p)` so the expected output
 * magnitude is unchanged whether or not dropout is active — the standard
 * "inverted" convention (scale at train time, no-op at eval time, rather
 * than the reverse). `training` is an explicit `forward` parameter, not
 * module-level mode-switching state (`.train()`/`.eval()`) — this repo has
 * no such lifecycle elsewhere, and inventing one for just this module
 * would be scope beyond what's asked.
 */
export class Dropout extends Module {
  readonly p: number;

  constructor(p: number) {
    super();
    if (p < 0 || p >= 1) throw new RangeError(`Dropout: p must be in [0, 1), got ${p}`);
    this.p = p;
  }

  forward(x: Variable, training: boolean, options: { rng?: Rng } = {}): Variable {
    if (!training || this.p === 0) return x;
    // keep[i] = 1 with probability (1-p), else 0 -- P(uniform < p) = p is
    // exactly the drop event, so "keep" is the >= p side.
    const keepMask = random
      .uniform(x.shape, { min: 0, max: 1, dtype: x.dtype, rng: options.rng })
      .gte(this.p)
      .cast(x.dtype);
    const scale = 1 / (1 - this.p);
    return x.mul(constant(keepMask)).mul(scale);
  }
}

/** Mean squared error. */
export function mseLoss(prediction: Variable, target: Variable): Variable {
  const diff = prediction.sub(target);
  return diff.mul(diff).mean();
}

/**
 * Pseudo-Huber (Charbonnier) loss: `delta^2 * (sqrt(1 + ((pred-target)/delta)^2) - 1)`,
 * averaged. The smooth, fully-differentiable variant of Huber loss —
 * behaves like scaled L2 near zero and like scaled L1 far from zero
 * (Huber's whole point: quadratic near the optimum, linear/outlier-robust
 * far from it), but with a smooth transition instead of Huber's classic
 * hard piecewise switch at `delta`. Deliberate, not a shortcut: `Variable`
 * has no conditional/select op yet (see issue #64's own note on this), so
 * the piecewise form isn't buildable from existing ops without one; the
 * pseudo-Huber form needs only `sqrt`/`add`/`mul`/`div`, all already
 * gradient-checked.
 */
export function huberLoss(prediction: Variable, target: Variable, delta = 1): Variable {
  const diff = prediction.sub(target).mul(1 / delta); // Variable.div only accepts another Variable, not a scalar
  const inner = diff.mul(diff).add(1).sqrt().add(-1);
  return inner.mul(delta * delta).mean();
}

/**
 * Binary cross-entropy FROM RAW LOGITS (matches {@link crossEntropy}'s own
 * "from logits" contract — sigmoid applied internally, never fed a
 * pre-squashed probability), computed via the standard numerically-stable
 * "BCEWithLogits" formulation (issue #85):
 *
 * `L(z, y) = relu(z) - z*y + log(1 + exp(-|z|))`
 *
 * A prior version computed `p = sigmoid(z)` first, then `y*log(p) +
 * (1-y)*log(1-p)` — for `|z| >~ 37`, f64 `sigmoid` saturates to exactly
 * `1.0`/`0.0`, so the inactive side's factor becomes `0 * log(0) = 0 *
 * -Inf = NaN` (IEEE 754: `0 * Inf` is NaN regardless of the other
 * factor). This form never evaluates `log` at a saturating probability —
 * `log(1+exp(-|z|))` is rewritten as `-log(sigmoid(|z|))` (no `exp`/`abs`
 * Variable ops exist yet, so `|z|` itself is `relu(z) + relu(-z)`), and
 * `sigmoid(|z|)` is always `>= 0.5` for any finite `|z|`, so its `log`
 * never sees 0. Verified equal to the prior formula to ~1e-15 in the
 * non-saturated regime, and finite (not NaN/Infinity) at `|z|` up to at
 * least 100, before writing this.
 */
export function binaryCrossEntropy(logits: Variable, target: Variable): Variable {
  const absLogits = logits.relu().add(logits.mul(-1).relu());
  const negLogSigmoidAbs = absLogits.sigmoid().log().mul(-1);
  return logits.relu().sub(logits.mul(target)).add(negLogSigmoidAbs).mean();
}

/**
 * Cross-entropy from raw logits (shape `[batch, numClasses]`) and integer
 * class labels (shape `[batch]`). Built on the already gradient-checked
 * `softmax`/`log` ops rather than a hand-rolled log-softmax fusion — softmax
 * is numerically stable (subtracts the row max), and its output is safely
 * > 0 for the logit magnitudes a toy training loop produces.
 */
export function crossEntropy(logits: Variable, labels: Tensor): Variable {
  const [batchSize, numClasses] = logits.shape as [number, number];
  const labelIdx = [...(labels.toArray() as (number | bigint)[])].map(Number);
  const onehotData = new Array(batchSize * numClasses).fill(0);
  labelIdx.forEach((cls, row) => {
    onehotData[row * numClasses + cls] = 1;
  });
  const onehot = constant(
    Tensor.from(onehotData, { dtype: logits.dtype }).reshape([batchSize, numClasses]),
  );

  const logProbs = logits.softmax(1).log();
  const perExampleLoss = logProbs.mul(onehot).sum(1).mul(-1);
  return perExampleLoss.mean();
}
