#!/usr/bin/env python3
"""Batch NumPy oracle for @johnhenry/math-plus-tensor-mlx differential tests.

One process answers every case (MLX tests build dozens of cases; spawning
Python per op would dominate the run). Reads a JSON list of jobs:

  [{"op": "add", "inputs": ["/tmp/a.npy", "/tmp/b.npy"],
    "args": {"scalar": 2.0, "axis": 1, "keepdims": true, "eps": 1e-5,
             "dtype": "f16", "has_weight": true, "has_bias": true},
    "output": "/tmp/out.npy"}, ...]

Inputs are .npy written by tensor-core; results are written as .npy that
tensor-core reads back. Float results are float32 (the device computes f16
cases on f16-rounded inputs; the harness compares those against this f32
reference with an f16 tolerance). `cast` to f16/bf16 returns the raw bit
pattern as uint16 so the test can compare bits exactly.

Comparison and logical ops return bool, arg-reductions and integer results
int32 (the device's dtypes), so the harness can compare those exactly.

This is separate from tensor-core's scripts/numpy_oracle.py on purpose:
that one is one-op-per-process and has no layer_norm, exact-erf gelu or
half-precision casts, which are this package's core ops.
"""

import json
import math
import sys

import numpy as np

_erf = np.vectorize(math.erf, otypes=[np.float64])


def gelu_erf(x):
    x64 = x.astype(np.float64)
    return 0.5 * x64 * (1.0 + _erf(x64 / math.sqrt(2.0)))


def bf16_bits(x):
    """Round-to-nearest-even float32 -> bfloat16 bits (NaN -> 0x7fc0)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
    nan = np.isnan(x)
    rounded[nan] = 0x7FC0
    return rounded.astype(np.uint16)


BINARY = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide,
    "maximum": np.maximum, "minimum": np.minimum, "pow": np.power,
    "equal": np.equal, "not_equal": np.not_equal, "less": np.less,
    "less_equal": np.less_equal, "greater": np.greater, "greater_equal": np.greater_equal,
}


def run(job):
    op = job["op"]
    a = job.get("args", {})
    xs = [np.load(p) for p in job["inputs"]]
    x = xs[0] if xs else None
    axis = a.get("axis")
    keep = bool(a.get("keepdims", False))

    if op in BINARY:
        y = np.asarray(a["scalar"], dtype=x.dtype) if "scalar" in a else xs[1]
        if op == "pow":
            r = np.power(x.astype(np.float64), y.astype(np.float64))
        else:
            r = BINARY[op](x, y)
    elif op in ("logical_and", "logical_or"):
        r = getattr(np, op)(x, xs[1])
    elif op == "logical_not":
        r = np.logical_not(x)
    elif op == "abs":
        r = np.abs(x)
    elif op in ("sqrt", "tanh"):
        r = getattr(np, op)(x.astype(np.float64))
    elif op == "rsqrt":
        r = 1.0 / np.sqrt(x.astype(np.float64))
    elif op == "sigmoid":
        r = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
    elif op == "erf":
        r = _erf(x.astype(np.float64))
    elif op in ("argmax", "argmin"):
        r = getattr(np, op)(x, axis=axis, keepdims=keep).astype(np.int32)
    elif op == "cumsum":
        r = np.cumsum(x.astype(np.float64) if x.dtype.kind == "f" else x, axis=axis)
        if r.dtype.kind in "iu":
            r = r.astype(np.int32)
    elif op == "neg":
        r = -x
    elif op == "exp":
        r = np.exp(x)
    elif op == "log":
        r = np.log(x)
    elif op == "relu":
        r = np.maximum(x, 0)
    elif op == "gelu":
        r = gelu_erf(x)
    elif op in ("sum", "mean", "max", "min"):
        r = getattr(np, op)(x, axis=axis, keepdims=keep)
    elif op == "softmax":
        ax = a.get("axis", -1)
        x64 = x.astype(np.float64)
        e = np.exp(x64 - x64.max(axis=ax, keepdims=True))
        r = e / e.sum(axis=ax, keepdims=True)
    elif op == "matmul":
        r = np.matmul(x.astype(np.float64), xs[1].astype(np.float64))
    elif op == "layer_norm":
        x64 = x.astype(np.float64)
        mu = x64.mean(axis=-1, keepdims=True)
        var = ((x64 - mu) ** 2).mean(axis=-1, keepdims=True)
        r = (x64 - mu) / np.sqrt(var + a["eps"])
        i = 1
        if a.get("has_weight"):
            r = r * xs[i]
            i += 1
        if a.get("has_bias"):
            r = r + xs[i]
    elif op == "where":
        r = np.where(xs[0].astype(bool), xs[1], xs[2])
    elif op == "transpose":
        r = np.transpose(x, a.get("axes"))
    elif op == "cast":
        dt = a["dtype"]
        if dt == "f16":
            r = x.astype(np.float16).view(np.uint16)
        elif dt == "bf16":
            r = bf16_bits(x.astype(np.float32))
        elif dt == "i32":
            r = x.astype(np.int32)  # C cast: truncation toward zero
        elif dt == "bool":
            r = x.astype(bool)
        else:
            r = x.astype(np.float32)
        np.save(job["output"], np.array(r, order="C"))  # not ascontiguousarray: it promotes 0-d to 1-d
        return
    else:
        raise SystemExit(f"numpy_oracle: unknown op {op!r}")

    r = np.asarray(r)
    if r.dtype.kind == "f":
        r = r.astype(np.float32)
    elif r.dtype.kind in "iu":
        r = r.astype(np.int32)
    np.save(job["output"], np.array(r, order="C"))  # not ascontiguousarray: it promotes 0-d to 1-d


def main():
    with open(sys.argv[1]) as f:
        for job in json.load(f):
            run(job)


if __name__ == "__main__":
    main()
