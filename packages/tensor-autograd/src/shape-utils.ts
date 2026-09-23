import { Tensor } from "@johnhenry/math-plus-tensor-core";

/**
 * Reduce `grad` down to `targetShape` by summing over exactly the axes that
 * broadcasting would have expanded — the standard "undo a forward broadcast"
 * step every elementwise-op gradient needs. Two cases, applied in order:
 * extra LEADING axes (grad has more dims than the target) are summed away
 * entirely; axes where the target is size-1 but grad isn't are summed and
 * the size-1 axis reinserted (keepdims-equivalent).
 */
export function sumToShape(grad: Tensor, targetShape: readonly number[]): Tensor {
  let g = grad;
  while (g.ndim > targetShape.length) {
    g = g.sum(0);
  }
  for (let axis = 0; axis < targetShape.length; axis++) {
    if (targetShape[axis] === 1 && g.shape[axis] !== 1) {
      g = g.sum(axis).unsqueeze(axis);
    }
  }
  // A gradient that reached here unreduced may be a strided view (e.g. from
  // a permute/transpose backward); Tensor.reshape refuses those.
  return contiguousOf(g).reshape(targetShape as number[]);
}

/** `t` itself if already C-contiguous (reshape-able), else a packed copy. */
export function contiguousOf(t: Tensor): Tensor {
  return t.isContiguous ? t : t.contiguous();
}
