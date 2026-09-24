/**
 * @johnhenry/math-plus-tensor-autograd — reverse-mode automatic differentiation over
 * @johnhenry/math-plus-tensor-core `Tensor`s (issue #8).
 *
 * Design: `Tensor` stays plain and immutable (tensor-core has no grad
 * bookkeeping fields, by design — it's the pure numeric core). `Variable`
 * wraps a `Tensor` and adds the tape: a define-by-run reverse-mode graph,
 * built as ops execute, walked backward on `.backward()`.
 *
 * Naming note: the source design sketched `Tensor.variable(x)` as the
 * grad-enabled constructor. That would require tensor-core to depend on
 * autograd, which is the wrong direction — the equivalent here is the
 * standalone `variable(x)` export (aliased as `Variable.variable(x)`).
 *
 * Non-differentiable ops (argmax, sort, comparisons, ...) simply aren't
 * `Variable` methods in v1 — call them on `.value` (a plain Tensor) instead,
 * which returns a plain Tensor with no grad tracking. That's an unambiguous
 * way to satisfy "must not silently produce wrong gradients" without a
 * runtime throw for an operation nothing calls.
 *
 * Non-goal 7 (no in-place ops on tracked tensors) is satisfied structurally:
 * `Variable` has no in-place API at all, matching tensor-core's own
 * immutable style — there's nothing to reject at runtime because the
 * mutating method doesn't exist.
 */
import { allocate, checkGeluApproximate, Tensor, type Axis, type DType, type GeluApproximate, type SliceSpec } from "@johnhenry/math-plus-tensor-core";
import { timed } from "@johnhenry/math-plus-telemetry";
import { contiguousOf, sumToShape } from "./shape-utils.ts";

/** Swap the last two axes (a view) — matmul's transpose for batched operands. */
function swapLastTwo(t: Tensor): Tensor {
  const axes = Array.from({ length: t.ndim }, (_, i) => i);
  axes[t.ndim - 1] = t.ndim - 2;
  axes[t.ndim - 2] = t.ndim - 1;
  return t.permute(axes);
}

/**
 * Gradient of `x.slice(...specs)`: zeros shaped like `x`, with `g` written at
 * the sliced positions. Flat target indices come from slicing an `arange`
 * index tensor the same way — tensor-core's public API has no strided
 * scatter, and this keeps the (subtle) slice-resolution rules in exactly one
 * place: `Tensor.slice` itself.
 */
function scatterIntoZeros(g: Tensor, x: Tensor, specs: Array<SliceSpec | null>): Tensor {
  const index = Tensor.arange(0, x.size, 1, { dtype: "f64" })
    .reshape([...x.shape])
    .slice(...specs)
    .contiguous().data as Float64Array;
  const src = g.contiguous().data as Float64Array;
  const out = allocate(g.dtype, x.size) as Float64Array;
  for (let i = 0; i < index.length; i++) out[index[i] as number] = src[i] as number;
  return Tensor.fromTypedArray(out, [...x.shape], { dtype: g.dtype });
}

let gradEnabled = true;

export function isGradEnabled(): boolean {
  return gradEnabled;
}

/** Run `fn` with gradient tracking disabled (inference mode) — PyTorch's `torch.no_grad()`. */
export function noGrad<T>(fn: () => T): T {
  const prev = gradEnabled;
  gradEnabled = false;
  try {
    return fn();
  } finally {
    gradEnabled = prev;
  }
}

/** Run `fn` with gradient tracking forced on, even inside an enclosing `noGrad`. */
export function enableGrad<T>(fn: () => T): T {
  const prev = gradEnabled;
  gradEnabled = true;
  try {
    return fn();
  } finally {
    gradEnabled = prev;
  }
}

interface TapeNode {
  inputs: readonly Variable[];
  backwardFn: (gradOutput: Tensor) => readonly (Tensor | undefined)[];
}

let nextId = 0;

export class Variable {
  /**
   * Intentionally NOT readonly: `optim.*` steps reassign a `Parameter`
   * leaf's value in place between training steps (`param.value =
   * newWeights`) — a JS object-reference repoint, not an in-place Tensor
   * mutation (the underlying `Tensor` objects stay fully immutable; this
   * just swaps which one `.value` points at). Non-goal 7 (no in-place ops
   * on tracked tensors) is about mutating a tensor mid-computation on the
   * tape — nothing here does that; reassignment only happens on leaves,
   * between backward passes, after the previous step's tape is done with.
   */
  value: Tensor;
  readonly requiresGrad: boolean;
  grad: Tensor | null = null;
  readonly id: number;
  readonly node: TapeNode | null;

  /** `protected`, not `private`: allows `nn.Parameter extends Variable` while still blocking `new Variable(...)` from outside the class hierarchy. */
  protected constructor(value: Tensor, requiresGrad: boolean, node: TapeNode | null) {
    this.value = value;
    this.requiresGrad = requiresGrad;
    this.node = node;
    this.id = nextId++;
  }

  /** A leaf that accumulates gradients on `.backward()`. */
  static variable(value: Tensor): Variable {
    return new Variable(value, true, null);
  }

  /** A leaf that never accumulates gradients — wrap a plain Tensor to use it in graph ops. */
  static constant(value: Tensor): Variable {
    return new Variable(value, false, null);
  }

  /** Build a traced non-leaf result. Used internally by every op method below. */
  static fromOp(
    value: Tensor,
    inputs: readonly Variable[],
    backwardFn: (gradOutput: Tensor) => readonly (Tensor | undefined)[],
  ): Variable {
    const tracked = isGradEnabled() && inputs.some((v) => v.requiresGrad || v.node !== null);
    return new Variable(value, false, tracked ? { inputs, backwardFn } : null);
  }

  get shape(): Tensor["shape"] {
    return this.value.shape;
  }
  get dtype(): Tensor["dtype"] {
    return this.value.dtype;
  }
  get ndim(): number {
    return this.value.ndim;
  }

  /** Same value, no gradient history — cuts the tape at this point. */
  detach(): Variable {
    return Variable.constant(this.value);
  }

  zeroGrad(): void {
    this.grad = null;
  }

  /**
   * Walk the graph reachable from `this` in reverse topological order,
   * accumulating into every requires-grad leaf's `.grad`. `gradOutput`
   * defaults to a ones-tensor and requires `this` be scalar (size 1) —
   * matching PyTorch's `.backward()` contract.
   */
  /**
   * `options` is entirely opt-in telemetry (issue #10): when no sink is
   * installed (the default), `timed()` skips even the `performance.now()`
   * call, so this costs nothing beyond the two extra `?.`/`??` reads.
   */
  backward(gradOutput?: Tensor, options: { runId?: string; step?: number } = {}): void {
    const runId = options.runId ?? "default";
    const step = options.step ?? 0;
    timed(runId, step, "backward", "autograd", () => {
      const seed =
        gradOutput ??
        (this.value.size === 1
          ? Tensor.ones(this.value.shape, { dtype: this.value.dtype })
          : (() => {
              throw new RangeError(
                "backward() with no argument requires a scalar (size-1) output; pass an explicit gradOutput for non-scalar tensors",
              );
            })());

      const order: Variable[] = [];
      const visited = new Set<number>();
      const visit = (v: Variable): void => {
        if (visited.has(v.id)) return;
        visited.add(v.id);
        if (v.node) for (const input of v.node.inputs) visit(input);
        order.push(v);
      };
      visit(this);

      const grads = new Map<number, Tensor>();
      grads.set(this.id, seed);

      for (let i = order.length - 1; i >= 0; i--) {
        const v = order[i] as Variable;
        const g = grads.get(v.id);
        if (g === undefined) continue; // not reachable via any actual gradient path
        if (v.requiresGrad) {
          v.grad = v.grad ? v.grad.add(g) : g;
        }
        if (v.node) {
          const inputGrads = v.node.backwardFn(g);
          v.node.inputs.forEach((input, idx) => {
            const contribution = inputGrads[idx];
            if (contribution === undefined) return;
            const existing = grads.get(input.id);
            grads.set(input.id, existing ? existing.add(contribution) : contribution);
          });
        }
      }
    });
  }

  // ---- differentiable ops ----------------------------------------------------

  add(other: Variable | number): Variable {
    if (typeof other === "number") {
      const value = this.value.add(other);
      return Variable.fromOp(value, [this], (g) => [g]);
    }
    const value = this.value.add(other.value);
    return Variable.fromOp(value, [this, other], (g) => [
      sumToShape(g, this.value.shape),
      sumToShape(g, other.value.shape),
    ]);
  }

  sub(other: Variable | number): Variable {
    if (typeof other === "number") {
      const value = this.value.sub(other);
      return Variable.fromOp(value, [this], (g) => [g]);
    }
    const value = this.value.sub(other.value);
    return Variable.fromOp(value, [this, other], (g) => [
      sumToShape(g, this.value.shape),
      sumToShape(g.mul(-1), other.value.shape),
    ]);
  }

  mul(other: Variable | number): Variable {
    if (typeof other === "number") {
      const value = this.value.mul(other);
      return Variable.fromOp(value, [this], (g) => [g.mul(other)]);
    }
    const value = this.value.mul(other.value);
    return Variable.fromOp(value, [this, other], (g) => [
      sumToShape(g.mul(other.value), this.value.shape),
      sumToShape(g.mul(this.value), other.value.shape),
    ]);
  }

  div(other: Variable | number): Variable {
    if (typeof other === "number") {
      const value = this.value.div(other);
      return Variable.fromOp(value, [this], (g) => [g.div(other)]);
    }
    const value = this.value.div(other.value);
    return Variable.fromOp(value, [this, other], (g) => [
      sumToShape(g.div(other.value), this.value.shape),
      // d/db (a/b) = -a/b^2
      sumToShape(g.mul(this.value).div(other.value.mul(other.value)).mul(-1), other.value.shape),
    ]);
  }

  /**
   * NumPy/PyTorch `matmul` for operands with ndim >= 2: the last two axes
   * multiply, leading (batch) axes broadcast — e.g. attention's
   * `[B, H, L, D] @ [B, H, D, S]`, or `[B, T, in] @ [in, out]`. Backward is
   * `g @ b^T` / `a^T @ g` (transposing the last two axes), reduced back over
   * any broadcast batch axes with {@link sumToShape}. 1-D operands are still
   * rejected (unsqueeze first) — their squeeze-back semantics would need
   * their own backward rules for no current caller.
   */
  matmul(other: Variable): Variable {
    if (this.value.ndim < 2 || other.value.ndim < 2) {
      throw new TypeError("Variable.matmul: operands must have ndim >= 2 (unsqueeze a 1-D operand first)");
    }
    const value = this.value.matmul(other.value);
    return Variable.fromOp(value, [this, other], (g) => [
      sumToShape(g.matmul(swapLastTwo(other.value)), this.value.shape),
      sumToShape(swapLastTwo(this.value).matmul(g), other.value.shape),
    ]);
  }

  /**
   * Insert a size-1 axis at `axis` — a view forward, differentiable backward.
   * Needed to broadcast a reduced result back against its pre-reduction
   * shape (e.g. LayerNorm's `mean(axis).unsqueeze(axis)` before subtracting).
   * Backward reduces exactly the inserted axis: squeeze if it stayed size 1,
   * sum if a later op broadcast it wider (mirrors sumToShape's logic, but
   * targeted at one known axis instead of inferred from a shape diff).
   */
  unsqueeze(axis: number): Variable {
    const ax = axis < 0 ? axis + this.value.ndim + 1 : axis;
    const value = this.value.unsqueeze(ax);
    return Variable.fromOp(value, [this], (g) => [
      (g.shape[ax] as number) === 1 ? g.squeeze(ax) : g.sum(ax),
    ]);
  }

  sqrt(): Variable {
    const value = this.value.sqrt();
    // d/dx sqrt(x) = 1/(2*sqrt(x))
    return Variable.fromOp(value, [this], (g) => [g.div(value.mul(2))]);
  }

  log(): Variable {
    const value = this.value.log();
    // d/dx log(x) = 1/x
    return Variable.fromOp(value, [this], (g) => [g.div(this.value)]);
  }

  sum(axis?: Axis): Variable {
    const value = this.value.sum(axis);
    return Variable.fromOp(value, [this], (g) => {
      if (axis === undefined) return [g.broadcastTo(this.value.shape).contiguous()];
      const ax = axis < 0 ? axis + this.value.ndim : axis;
      return [g.unsqueeze(ax).broadcastTo(this.value.shape).contiguous()];
    });
  }

  mean(axis?: Axis): Variable {
    const value = this.value.mean(axis);
    const count =
      axis === undefined
        ? this.value.size
        : (this.value.shape[axis < 0 ? axis + this.value.ndim : axis] as number);
    return Variable.fromOp(value, [this], (g) => {
      const scaled = g.div(count);
      if (axis === undefined) return [scaled.broadcastTo(this.value.shape).contiguous()];
      const ax = axis < 0 ? axis + this.value.ndim : axis;
      return [scaled.unsqueeze(ax).broadcastTo(this.value.shape).contiguous()];
    });
  }

  relu(): Variable {
    const value = this.value.relu();
    return Variable.fromOp(value, [this], (g) => {
      const mask = this.value.gt(0).cast(this.value.dtype);
      return [g.mul(mask)];
    });
  }

  sigmoid(): Variable {
    const value = this.value.sigmoid();
    return Variable.fromOp(value, [this], (g) => {
      const oneMinusValue = value.mul(-1).add(1);
      return [g.mul(value).mul(oneMinusValue)];
    });
  }

  /**
   * GELU with the same `approximate` option and default as `Tensor.gelu()`
   * (`"none"` = exact erf-GELU, the default since #122; `"tanh"` = the tanh
   * approximation). The backward pass differentiates whichever forward was
   * actually computed:
   * - exact: `Φ(x) + x·φ(x)`, with `Φ(x) = 0.5·erfc(-x/√2)` from the
   *   canonical `Tensor.erfc()` (@johnhenry/math-plus-special, re-exported by tensor-core);
   * - tanh: the exact derivative of the tanh approximation, using the
   *   identity tanh(x) = 2*sigmoid(2x)-1.
   */
  gelu(options: { approximate?: GeluApproximate } = {}): Variable {
    const approximate = checkGeluApproximate(options.approximate);
    const value = this.value.gelu({ approximate });
    if (approximate === "none") {
      return Variable.fromOp(value, [this], (g) => {
        const x = this.value;
        const cdf = x.mul(-Math.SQRT1_2).erfc().mul(0.5); // Φ(x) = 0.5·erfc(-x/√2), no cancellation for x << 0
        const pdf = x.mul(x).mul(-0.5).exp().mul(1 / Math.sqrt(2 * Math.PI)); // φ(x)
        return [g.mul(cdf.add(x.mul(pdf)))];
      });
    }
    return Variable.fromOp(value, [this], (g) => {
      const c = Math.sqrt(2 / Math.PI);
      const x = this.value;
      const x2 = x.mul(x);
      const x3 = x2.mul(x);
      const inner = x.add(x3.mul(0.044715)).mul(c);
      const t = inner.mul(2).sigmoid().mul(2).add(-1); // tanh(inner)
      const sech2 = t.mul(t).mul(-1).add(1); // 1 - tanh^2(inner)
      const dInner = x2.mul(3 * 0.044715).add(1).mul(c); // c*(1 + 3*0.044715*x^2)
      const derivative = t.add(1).mul(0.5).add(x.mul(0.5).mul(sech2).mul(dInner));
      return [g.mul(derivative)];
    });
  }

  softmax(axis: Axis = -1): Variable {
    const value = this.value.softmax(axis);
    return Variable.fromOp(value, [this], (g) => {
      const ax = axis < 0 ? axis + this.value.ndim : axis;
      const dot = g.mul(value).sum(ax).unsqueeze(ax).broadcastTo(this.value.shape).contiguous();
      return [value.mul(g.sub(dot))];
    });
  }
  exp(): Variable {
    const value = this.value.exp();
    return Variable.fromOp(value, [this], (g) => [g.mul(value)]);
  }

  tanh(): Variable {
    const value = this.value.tanh();
    // d/dx tanh(x) = 1 - tanh(x)^2
    return Variable.fromOp(value, [this], (g) => [g.mul(value.mul(value).mul(-1).add(1))]);
  }

  // ---- views & structure (issue #123) ----------------------------------------
  //
  // Gradients of view ops are made contiguous before they flow on, so no
  // downstream backward ever has to care whether its incoming gradient is a
  // strided view.

  /**
   * New shape (one -1 allowed), like `Tensor.reshape` — but a
   * non-contiguous input (e.g. after `permute`) is packed first instead of
   * throwing, matching PyTorch's `reshape` (view when possible, else copy).
   */
  reshape(shape: readonly number[]): Variable {
    const src = this.value.isContiguous ? this.value : this.value.contiguous();
    const value = src.reshape(shape as number[]);
    const inShape = [...this.value.shape];
    return Variable.fromOp(value, [this], (g) => [contiguousOf(g).reshape(inShape)]);
  }

  /** Axis permutation (a view). Negative axes allowed. Backward applies the inverse permutation. */
  permute(axes: readonly number[]): Variable {
    const nd = this.value.ndim;
    const norm = axes.map((a) => (a < 0 ? a + nd : a));
    const value = this.value.permute(norm);
    const inverse = new Array<number>(nd);
    norm.forEach((a, i) => {
      inverse[a] = i;
    });
    return Variable.fromOp(value, [this], (g) => [g.permute(inverse).contiguous()]);
  }

  /**
   * PyTorch-style `transpose(dim0, dim1)`: swap two axes (negative allowed).
   * With no arguments, reverses all axes (tensor-core's `Tensor.transpose()`
   * / NumPy `.T`) — for a 2-D Variable both spellings mean the same thing.
   */
  transpose(dim0?: number, dim1?: number): Variable {
    const nd = this.value.ndim;
    if (dim0 === undefined && dim1 === undefined) {
      return this.permute(Array.from({ length: nd }, (_, i) => nd - 1 - i));
    }
    if (dim0 === undefined || dim1 === undefined) {
      throw new TypeError("Variable.transpose: pass both dim0 and dim1, or neither");
    }
    const a = dim0 < 0 ? dim0 + nd : dim0;
    const b = dim1 < 0 ? dim1 + nd : dim1;
    const axes = Array.from({ length: nd }, (_, i) => i);
    axes[a] = b;
    axes[b] = a;
    return this.permute(axes);
  }

  /**
   * Strided slice with tensor-core's `Tensor.slice` semantics (specs align
   * to leading axes; Python `slice.indices()` rules incl. negative
   * start/end/step). Backward scatters the incoming gradient into a zero
   * tensor of the input's shape at exactly the sliced positions.
   */
  slice(...specs: Array<SliceSpec | null>): Variable {
    const value = this.value.slice(...specs);
    return Variable.fromOp(value, [this], (g) => [scatterIntoZeros(g, this.value, specs)]);
  }

  /** PyTorch `narrow(dim, start, length)`: `length` entries of axis `dim` from `start` — a {@link slice} along one axis. */
  narrow(dim: number, start: number, length: number): Variable {
    const nd = this.value.ndim;
    const ax = dim < 0 ? dim + nd : dim;
    const specs: Array<SliceSpec | null> = new Array(ax + 1).fill(null);
    specs[ax] = { start, end: start + length };
    return this.slice(...specs);
  }

  /** Join along an existing axis (like `Tensor.concat`); backward slices the gradient back apart. */
  static concat(vars: readonly Variable[], axis = 0): Variable {
    if (vars.length === 0) throw new RangeError("Variable.concat requires at least one Variable");
    const nd = (vars[0] as Variable).ndim;
    const ax = axis < 0 ? axis + nd : axis;
    const value = Tensor.concat(
      vars.map((v) => v.value),
      { axis: ax },
    );
    return Variable.fromOp(value, vars, (g) => {
      let cursor = 0;
      return vars.map((v) => {
        const size = v.value.shape[ax] as number;
        const specs: Array<SliceSpec | null> = new Array(ax + 1).fill(null);
        specs[ax] = { start: cursor, end: cursor + size };
        cursor += size;
        return g.slice(...specs).contiguous();
      });
    });
  }

  /**
   * `where(mask, value, this)` — PyTorch `masked_fill`. `mask` is a plain
   * bool Tensor (not differentiable), broadcast against `this`. Positions
   * where `mask` is true get `value` (e.g. `-Infinity` for attention) and
   * receive zero gradient.
   */
  maskedFill(mask: Tensor, value: number): Variable {
    if (mask.dtype !== "bool") {
      throw new TypeError(`maskedFill: mask must be a bool tensor, got ${mask.dtype}`);
    }
    const fill = Tensor.full([], value, { dtype: this.value.dtype });
    const out = Tensor.where(mask, fill, this.value);
    return Variable.fromOp(out, [this], (g) => [
      sumToShape(Tensor.where(mask, Tensor.zeros([], { dtype: g.dtype }), g), this.value.shape),
    ]);
  }

  /**
   * Differentiable dtype conversion (`Tensor.cast`); backward casts the
   * gradient back to the source dtype. The mechanism behind half-precision
   * (f16/bf16) parameter STORAGE: layers upcast such parameters to the
   * input's compute dtype on the fly (tensor-core cannot compute in half).
   */
  cast(dtype: DType): Variable {
    if (dtype === this.value.dtype) return this;
    const srcDtype = this.value.dtype;
    const value = this.value.cast(dtype);
    return Variable.fromOp(value, [this], (g) => [g.cast(srcDtype)]);
  }

}

/** `Tensor.variable(x)` from the source design — see the naming note above. */
export function variable(value: Tensor): Variable {
  return Variable.variable(value);
}

export function constant(value: Tensor): Variable {
  return Variable.constant(value);
}
