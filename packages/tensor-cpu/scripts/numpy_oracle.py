#!/usr/bin/env python3
"""Batch NumPy oracle for @johnhenry/math-plus-tensor-cpu differential tests.

One process answers every case. Reads a JSON list of jobs:

  [{"op": "sdpa", "inputs": ["/tmp/q.npy", ...], "args": {"scale": 0.35},
    "output": "/tmp/out.npy"}, ...]

Inputs are .npy written by tensor-core; every float computation here runs
in float64 on the (float32) inputs, and float results are written back as
float32. Integer/bool results keep NumPy's dtype (int32 where the backend
contract says i32).

Separate from tensor-mlx's oracle on purpose: this one covers the fused
transformer ops of the @johnhenry/tensor-backend contract (linear with a
PyTorch [out, in] weight, split-half RoPE with f32 angles, grouped-query
SDPA with bool/additive masks, GEGLU, masked mean pooling, row gathers).
"""

import json
import math
import sys

import numpy as np

_erf = np.vectorize(math.erf, otypes=[np.float64])


def f64(x):
    return x.astype(np.float64)


def gelu(x):
    x = f64(x)
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def softmax(x, axis):
    x = f64(x)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def rope(x, base):
    # Split-half rotation; inv_freq and pos*inv_freq rounded to float32 like MLX / PyTorch.
    B, H, L, D = x.shape
    half = D // 2
    inv = (np.float32(base) ** (-(np.arange(half, dtype=np.float32) / np.float32(half)))).astype(np.float32)
    th = (np.arange(L, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float32).astype(np.float64)
    c, s = np.cos(th), np.sin(th)
    x = f64(x)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def sdpa(q, k, v, mask, scale):
    H, Hk = q.shape[1], k.shape[1]
    k = np.repeat(f64(k), H // Hk, axis=1)
    v = np.repeat(f64(v), H // Hk, axis=1)
    s = np.matmul(f64(q), np.swapaxes(k, -1, -2)) * scale
    if mask is not None:
        s = np.where(mask, s, -np.inf) if mask.dtype == np.bool_ else s + f64(mask)
    return np.matmul(softmax(s, -1), v)


UNARY = {
    "exp": np.exp, "log": np.log, "relu": lambda x: np.maximum(x, 0), "gelu": gelu,
    "sqrt": np.sqrt, "rsqrt": lambda x: 1.0 / np.sqrt(x), "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)), "erf": _erf, "neg": np.negative, "abs": np.abs,
}
BINARY = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide,
    "maximum": np.maximum, "pow": np.power,
    "equal": np.equal, "notEqual": np.not_equal, "less": np.less, "lessEqual": np.less_equal,
    "greater": np.greater, "greaterEqual": np.greater_equal,
    "logicalAnd": np.logical_and, "logicalOr": np.logical_or,
}


def run(job):
    op = job["op"]
    a = job.get("args", {})
    xs = [np.load(p) for p in job["inputs"]]
    x = xs[0] if xs else None

    if op == "unary":
        r = UNARY[a["fn"]](f64(x) if x.dtype.kind == "f" else x)
    elif op == "binary":
        fn = BINARY[a["fn"]]
        l, rr = xs
        r = fn(f64(l), f64(rr)) if a["fn"] in ("div", "pow") or l.dtype.kind == "f" or rr.dtype.kind == "f" else fn(l, rr)
    elif op == "reduce":
        fn = a["fn"]
        ax, keep = a["axis"], a.get("keepDims", False)
        if fn in ("argmax", "argmin"):
            r = getattr(np, fn)(x, axis=ax)
            r = np.expand_dims(r, ax) if keep else r
            r = r.astype(np.int32)
        elif fn == "sum" and x.dtype.kind != "f":
            r = x.astype(np.int64).sum(axis=ax, keepdims=keep).astype(np.int32)
        else:
            r = getattr(np, fn)(f64(x) if fn in ("sum", "mean") else x, axis=ax, keepdims=keep)
    elif op == "cumsum":
        r = np.cumsum(f64(x), axis=a["axis"]) if x.dtype.kind == "f" else np.cumsum(x.astype(np.int64), axis=a["axis"]).astype(np.int32)
    elif op == "softmax":
        r = softmax(x, a["axis"])
    elif op == "sort":
        r = np.sort(x, axis=a["axis"])
    elif op == "linear":
        r = np.matmul(f64(x), f64(xs[1]).T)
        if len(xs) > 2:
            r = r + f64(xs[2])
    elif op == "matmul":
        r = np.matmul(f64(x), f64(xs[1]))
    elif op == "layer_norm":
        x64 = f64(x)
        mu = x64.mean(axis=-1, keepdims=True)
        var = ((x64 - mu) ** 2).mean(axis=-1, keepdims=True)
        r = (x64 - mu) / np.sqrt(var + a["eps"])
        i = 1
        if a.get("has_weight"):
            r = r * f64(xs[i])
            i += 1
        if a.get("has_bias"):
            r = r + f64(xs[i])
    elif op == "rope":
        r = rope(x, a["base"])
    elif op == "sdpa":
        r = sdpa(xs[0], xs[1], xs[2], xs[3] if len(xs) > 3 else None, a["scale"])
    elif op == "geglu":
        d = x.shape[-1] // 2
        r = gelu(x[..., :d]) * f64(x[..., d:])
    elif op == "mean_pool":
        m = xs[1].astype(np.float64)[..., None]
        r = (f64(x) * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1.0)
    elif op == "where":
        r = np.where(xs[0].astype(bool), xs[1], xs[2])
    elif op == "transpose":
        r = np.transpose(x, a["perm"])
    elif op == "slice":
        r = x[tuple(slice(b, e) for b, e in zip(a["begin"], a["end"]))]
    elif op == "concat":
        r = np.concatenate(xs, axis=a["axis"])
    elif op == "embedding":
        r = x[xs[1]]
    elif op == "gather_rows":
        idx = xs[1]
        r = x[np.arange(x.shape[0])[:, None], idx]
    else:
        raise SystemExit(f"numpy_oracle: unknown op {op!r}")

    r = np.asarray(r)
    if r.dtype.kind == "f":
        r = r.astype(np.float32)
    np.save(job["output"], np.array(r, order="C"))  # not ascontiguousarray: it promotes 0-d to 1-d


def main():
    with open(sys.argv[1]) as f:
        for job in json.load(f):
            run(job)


if __name__ == "__main__":
    main()
