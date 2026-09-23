"""Half-precision oracle for tensor-core's test/half-compute.test.ts.

Covers the f16/bf16 behaviour numpy_oracle.py cannot express on its own:
bfloat16 needs the `ml_dtypes` package (NumPy has no bfloat16). f16 jobs need
only NumPy; bf16 jobs also import ml_dtypes, and the test skips them when it
cannot find an interpreter that has it (skip-don't-fail).

Usage: python3 half_oracle.py job.json
    {
      "op": "save_bf16" | "load_bf16" | "compute",
      "inputs": ["/path/a.npy", ...],
      "output": "/path/out.npy",
      # compute only:
      "expr": "add" | "mul" | "chain" | "matmul" | "softmax" | "sum0",
      "half": "f16" | "bf16",         # dtype of the half inputs and of the result
      "compute": "float32" | "float64" | "native",   # "native": no astype, NumPy's own f16 ufunc
    }

- save_bf16: float32 input -> `np.save(output, x.astype(ml_dtypes.bfloat16))`,
  i.e. a genuine NumPy + ml_dtypes bf16 .npy (descr '<V2').
- load_bf16: a bf16 .npy written by JS -> `np.load(...).view(bfloat16)` ->
  float32, saved as '<f4'. Proves Python reads the JS file the documented way.
- compute: half inputs -> astype(compute) -> expr -> astype(half) -> save
  (bf16 results are saved by np.save as '<V2', so the JS side also reads a real
  ml_dtypes file back).
"""
import json
import sys

import numpy as np


def bf16():
    import ml_dtypes  # imported lazily: f16 jobs must not need it

    return ml_dtypes.bfloat16


def load_half(path, half):
    arr = np.load(path)
    if half == "bf16":
        return arr.view(bf16())
    assert arr.dtype == np.float16, arr.dtype
    return arr


def softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


EXPRS = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    # (a*b + a) then relu: a chain, rounded once at the end
    "chain": lambda a, b: np.maximum(a * b + a, np.zeros((), dtype=a.dtype)),
    "matmul": lambda a, b: a @ b,
    "softmax": lambda a: softmax(a),
    "sum0": lambda a: np.sum(a, axis=0),
}


def main() -> None:
    with open(sys.argv[1]) as f:
        job = json.load(f)
    op = job["op"]
    if op == "save_bf16":
        x = np.load(job["inputs"][0])
        assert x.dtype == np.float32, x.dtype
        np.save(job["output"], x.astype(bf16()))
    elif op == "load_bf16":
        x = np.load(job["inputs"][0]).view(bf16())
        np.save(job["output"], x.astype(np.float32))
    elif op == "compute":
        half = job["half"]
        xs = [load_half(p, half) for p in job["inputs"]]
        if job["compute"] != "native":
            xs = [x.astype(job["compute"]) for x in xs]
        out = EXPRS[job["expr"]](*xs)
        target = np.float16 if half == "f16" else bf16()
        np.save(job["output"], np.asarray(out).astype(target))
    else:
        raise SystemExit(f"unknown op {op}")


if __name__ == "__main__":
    main()
