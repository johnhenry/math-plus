#!/usr/bin/env python3
"""SciPy / PyTorch oracle for tensor-core's canonical erf/erfc/GELU (issue #122).

Reads a JSON job file and prints a JSON array of results to stdout:
  {"op": "erf" | "erfc", "xs": [...]}              -> scipy.special.erf / erfc (float64)
  {"op": "gelu_ndtr", "xs": [...]}                 -> x * scipy.special.ndtr(x) (float64): exact
                                                      GELU with full relative accuracy in the left
                                                      tail (ndtr is erfc-based)
  {"op": "torch_gelu", "xs": [...],
   "approximate": "none" | "tanh",
   "dtype": "float64" | "float32"}                 -> torch.nn.functional.gelu(x, approximate=...)

Inputs/outputs must be finite (JSON has no NaN/Infinity); special values are
covered by tensor-core's own unit tests instead.
"""

import json
import sys


def main() -> None:
    with open(sys.argv[1]) as f:
        job = json.load(f)
    op = job["op"]
    xs = job["xs"]
    if op in ("erf", "erfc", "gelu_ndtr"):
        import numpy as np
        from scipy import special

        x = np.asarray(xs, dtype=np.float64)
        if op == "erf":
            out = special.erf(x)
        elif op == "erfc":
            out = special.erfc(x)
        else:
            out = x * special.ndtr(x)
        result = [float(v) for v in out]
    elif op == "torch_gelu":
        import torch

        dtype = torch.float64 if job.get("dtype", "float64") == "float64" else torch.float32
        x = torch.tensor(xs, dtype=dtype)
        out = torch.nn.functional.gelu(x, approximate=job.get("approximate", "none"))
        result = [float(v) for v in out.tolist()]
    else:
        raise SystemExit(f"unknown op {op!r}")
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
