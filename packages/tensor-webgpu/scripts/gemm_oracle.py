#!/usr/bin/env python3
"""NumPy oracle for @johnhenry/math-plus-tensor-webgpu's GEMM tests.

Reads JSON on stdin:
  {"cases": [{"a": <base64>, "b": <base64>, "dtype": "f32" | "f16",
              "m": M, "k": K, "n": N, "transB": bool}, ...]}
where a/b are the exact little-endian operand bytes the GPU saw (float32,
or IEEE binary16 bits for f16). a is [M, K]; b is [K, N], or [N, K] when
transB. Writes JSON to stdout:
  {"results": [{"c": <base64 float64 [M, N]>, "absprod": <base64 float64 [M, N]>}, ...]}
c = A @ B computed in float64 from the exact (already-rounded) inputs, and
absprod = |A| @ |B|, the scale the tests' accumulation-error bound is
relative to.
"""

import base64
import json
import sys

import numpy as np


def decode(data: str, dtype: str, shape: tuple) -> np.ndarray:
    np_dtype = {"f32": "<f4", "f16": "<f2"}[dtype]
    return np.frombuffer(base64.b64decode(data), dtype=np_dtype).reshape(shape).astype(np.float64)


def encode(x: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(x, dtype="<f8").tobytes()).decode("ascii")


def main() -> None:
    job = json.load(sys.stdin)
    results = []
    for case in job["cases"]:
        m, k, n = case["m"], case["k"], case["n"]
        a = decode(case["a"], case["dtype"], (m, k))
        b = decode(case["b"], case["dtype"], (n, k) if case["transB"] else (k, n))
        if case["transB"]:
            b = b.T
        results.append({"c": encode(a @ b), "absprod": encode(np.abs(a) @ np.abs(b))})
    json.dump({"results": results}, sys.stdout)


if __name__ == "__main__":
    main()
