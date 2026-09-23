---
"@johnhenry/math-plus-tensor-core": minor
---

Half-precision follow-ups (issue #128):

- New `withCompute("f32" | "f64", inputs, fn)`, an explicit opt-in for arithmetic on f16/bf16. It casts the half inputs to the compute dtype and passes other inputs through unchanged. It runs `fn`, then casts compute-dtype results back to the half dtype, rounding once when the region exits. A single op is bit-identical to NumPy's float16 ufuncs. Mixed f16/bf16 inputs, or no half input, throw. Plain half arithmetic still throws, and its error message now mentions `withCompute`.
- `.npy` bf16 follows the `ml_dtypes` convention. `toNpy()` writes descr `'<V2'`, byte-identical to `np.save` of an `ml_dtypes.bfloat16` array. `Tensor.fromNpy(bytes, { voidAs: "bf16" })` reads such files. Without the option, a 2-byte void descr throws instead of being guessed.
