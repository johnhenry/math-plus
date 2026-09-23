#!/usr/bin/env python3
"""NumPy oracle for @johnhenry/math-plus-tensor-core differential tests.

Reads a JSON job file:
  {
    "op": "add" | "sub" | "mul" | "div" | "sum" | "mean" | "arange" | "permute" | "slice"
        | "matmul" | "dot" | "cast" | "eq" | "ne" | "lt" | "lte" | "gt" | "gte"
        | "min" | "max" | "argmin" | "argmax" | "sqrt" | "variance" | "std"
        | "cumsum" | "cumprod" | "sort" | "argsort" | "topk_values" | "topk_indices"
        | "concat" | "stack" | "where" | "relu" | "sigmoid" | "gelu" | "softmax" | "log"
        | "f16_bits" | "bf16_bits",
    "ddof": 1,                            # optional (variance/std)
    "approximate": "tanh",                # optional (gelu: "none" (default) | "tanh")
    "k": 3, "largest": true,              # optional (topk)
    "condition": "/path/cond.npy",        # optional (where -- npy dtype must be bool)
    "inputs": ["/path/a.npy", ...],       # .npy files written by tensor-core
    "axis": 1,                            # optional (reductions)
    "permutation": [1, 0],                # optional (permute op)
    "scalar": 2.5,                        # optional (binary ops vs scalar)
    "arange": [start, stop, step],        # optional (arange op)
    "dtype": "float32",                   # optional (arange dtype)
    "specs": [[1, 10, 2], null, [null, null, -1]],  # optional (slice op) -- per-axis
                                           # [start, end, step] triples or null for the
                                           # whole axis, matching tensor-core's SliceSpec
    "output": "/path/out.npy"
  }
Computes the equivalent NumPy result and writes it to `output` as .npy.
The exchange format is .npy in both directions, dogfooding tensor-core's I/O.
"""

import json
import sys

import numpy as np

# tensor-core dtype names -> NumPy dtype names. Names already spelled the
# NumPy way (e.g. the "float32" the arange tests pass) fall through unchanged.
DTYPE_MAP = {
    "bool": "bool",
    "u8": "uint8", "i8": "int8",
    "u16": "uint16", "i16": "int16",
    "u32": "uint32", "i32": "int32",
    "u64": "uint64", "i64": "int64",
    "f16": "float16", "f32": "float32", "f64": "float64",
}


def to_numpy_dtype(name: str) -> str:
    return DTYPE_MAP.get(name, name)


def main() -> None:
    with open(sys.argv[1]) as f:
        job = json.load(f)

    op = job["op"]
    inputs = [np.load(path) for path in job.get("inputs", [])]

    if op in ("add", "sub", "mul", "div"):
        a = inputs[0]
        b = job["scalar"] if "scalar" in job else inputs[1]
        result = {
            "add": lambda: a + b,
            "sub": lambda: a - b,
            "mul": lambda: a * b,
            "div": lambda: a / b,
        }[op]()
    elif op == "sum":
        result = inputs[0].sum(axis=job.get("axis"))
    elif op == "mean":
        result = inputs[0].mean(axis=job.get("axis"))
    elif op == "arange":
        start, stop, step = job["arange"]
        result = np.arange(start, stop, step, dtype=job.get("dtype", "float32"))
    elif op == "permute":
        result = np.ascontiguousarray(inputs[0].transpose(job["permutation"]))
    elif op == "slice":
        index = tuple(
            slice(*spec) if spec is not None else slice(None)
            for spec in job["specs"]
        )
        result = np.ascontiguousarray(inputs[0][index])
    elif op == "matmul":
        result = np.ascontiguousarray(np.matmul(inputs[0], inputs[1]))
    elif op == "dot":
        result = np.asarray(np.dot(inputs[0], inputs[1]))
    elif op == "cast":
        result = inputs[0].astype(to_numpy_dtype(job["dtype"]))
    elif op == "bf16_bits":
        # NumPy has no bfloat16; this is the reference float32 -> bfloat16
        # round-to-nearest-even bit trick (PyTorch c10 / TensorFlow /
        # safetensors writers all use it), producing the raw uint16 patterns.
        # Callers only pass NaN-free inputs (the trick is not NaN-safe).
        u = inputs[0].astype(np.float32).view(np.uint32).astype(np.uint64)
        rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
        result = rounded.astype(np.uint16)
    elif op == "f16_bits":
        result = inputs[0].astype(np.float16).view(np.uint16)
    elif op in ("eq", "ne", "lt", "lte", "gt", "gte"):
        a = inputs[0]
        b = job["scalar"] if "scalar" in job else inputs[1]
        result = {
            "eq": lambda: a == b,
            "ne": lambda: a != b,
            "lt": lambda: a < b,
            "lte": lambda: a <= b,
            "gt": lambda: a > b,
            "gte": lambda: a >= b,
        }[op]()
    elif op == "min":
        result = inputs[0].min(axis=job.get("axis"))
    elif op == "max":
        result = inputs[0].max(axis=job.get("axis"))
    elif op == "argmin":
        result = inputs[0].argmin(axis=job.get("axis")).astype("int32")
    elif op == "argmax":
        result = inputs[0].argmax(axis=job.get("axis")).astype("int32")
    elif op == "sqrt":
        result = np.sqrt(inputs[0])
    elif op == "log":
        result = np.log(inputs[0])
    elif op == "unary":
        # Generic dispatch for issue #64's op-table -- one branch instead of
        # 25, keyed by "fn" (a numpy ufunc/np.* name). pow/round/cbrt/sign
        # get their own numpy-name mappings since the JS method name doesn't
        # match numpy's directly.
        fn_name = job["fn"]
        a = inputs[0]
        if fn_name == "pow":
            result = np.power(a, job["scalar"])
        elif fn_name == "neg":
            result = -a
        elif fn_name == "sign":
            result = np.sign(a)
        elif fn_name == "round":
            result = np.round(a)  # round-half-to-even, matching the JS test's away-from-.5 exclusion
        else:
            result = getattr(np, fn_name)(a)
    elif op == "clip":
        result = np.clip(inputs[0], job.get("min"), job.get("max"))
    elif op == "prod":
        result = inputs[0].prod(axis=job.get("axis"))
    elif op == "pad":
        # job["padding"] gives [before, after] pairs for the LAST N axes,
        # matching Tensor.pad's convention -- expand to full-ndim pairs the
        # way np.pad requires.
        a = inputs[0]
        pairs = job["padding"]
        full = [(0, 0)] * (a.ndim - len(pairs)) + [tuple(p) for p in pairs]
        result = np.pad(a, full, mode="constant", constant_values=job.get("value", 0))
    elif op == "split":
        # Multi-output: parts can have DIFFERING sizes (the cut-point form),
        # so np.asarray(result) below can't hold them -- write each part to
        # its own path in job["outputs"] and return early.
        sections = job["sections"]
        axis = job.get("axis", 0)
        parts = np.split(inputs[0], sections, axis=axis)
        for part, path in zip(parts, job["outputs"]):
            np.save(path, np.ascontiguousarray(part))
        return
    elif op == "repeat":
        result = np.repeat(inputs[0], job["counts"], axis=job.get("axis", 0))
    elif op == "flip":
        result = np.ascontiguousarray(np.flip(inputs[0], axis=job.get("axis")))
    elif op == "roll":
        result = np.roll(inputs[0], job["shift"], axis=job.get("axis"))
    elif op == "nonzero":
        # np.argwhere -- matches Tensor.nonzero()'s [count, ndim] shape.
        result = np.ascontiguousarray(np.argwhere(inputs[0]).astype("int64"))
    elif op == "variance":
        result = inputs[0].var(axis=job.get("axis"), ddof=job.get("ddof", 0))
    elif op == "std":
        result = inputs[0].std(axis=job.get("axis"), ddof=job.get("ddof", 0))
    elif op == "cumsum":
        result = inputs[0].cumsum(axis=job.get("axis"))
    elif op == "cumprod":
        result = inputs[0].cumprod(axis=job.get("axis"))
    elif op == "sort":
        result = np.sort(inputs[0], axis=job.get("axis", -1))
    elif op == "argsort":
        result = np.argsort(inputs[0], axis=job.get("axis", -1)).astype("int32")
    elif op in ("topk_values", "topk_indices"):
        axis = job.get("axis", -1)
        k = job["k"]
        largest = job.get("largest", True)
        a = inputs[0]
        order = np.argsort(a, axis=axis)
        if largest:
            order = np.flip(order, axis=axis)
        idx = np.take(order, range(k), axis=axis).astype("int32")
        result = idx if op == "topk_indices" else np.take_along_axis(a, idx, axis=axis)
    elif op == "concat":
        result = np.ascontiguousarray(np.concatenate(inputs, axis=job.get("axis", 0)))
    elif op == "stack":
        result = np.ascontiguousarray(np.stack(inputs, axis=job.get("axis", 0)))
    elif op == "where":
        cond = np.load(job["condition"])
        result = np.where(cond, inputs[0], inputs[1])
    elif op == "relu":
        result = np.maximum(inputs[0], 0)
    elif op == "sigmoid":
        result = 1 / (1 + np.exp(-inputs[0]))
    elif op == "gelu":
        # "approximate" mirrors torch.nn.functional.gelu: "none" (default, exact
        # erf-GELU -- NumPy has no erf, so the C library's math.erf is the
        # reference) or "tanh".
        x = inputs[0]
        if job.get("approximate", "none") == "tanh":
            c = np.sqrt(2 / np.pi)
            result = 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))
        else:
            import math
            erf = np.vectorize(math.erf, otypes=[np.float64])
            result = (0.5 * x.astype(np.float64) * (1 + erf(x.astype(np.float64) / math.sqrt(2)))).astype(x.dtype)
    elif op == "softmax":
        x = inputs[0]
        axis = job.get("axis", -1)
        shifted = x - np.max(x, axis=axis, keepdims=True)
        expd = np.exp(shifted)
        result = expd / np.sum(expd, axis=axis, keepdims=True)
    else:
        raise SystemExit(f"unknown op {op!r}")

    np.save(job["output"], np.asarray(result))


if __name__ == "__main__":
    main()
