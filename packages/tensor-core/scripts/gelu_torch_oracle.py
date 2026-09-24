#!/usr/bin/env python3
"""PyTorch oracle for tensor-core's Tensor.gelu() (issue #122).

Reads a JSON job file and prints a JSON array of results to stdout:
  {"op": "torch_gelu", "xs": [...],
   "approximate": "none" | "tanh",
   "dtype": "float64" | "float32"}   -> torch.nn.functional.gelu(x, approximate=...)

The scalar erf/erfc/GELU's SciPy oracle lives with them, in
packages/special/scripts/special_oracle.py. Inputs/outputs must be finite.
"""

import json
import sys


def main() -> None:
    with open(sys.argv[1]) as f:
        job = json.load(f)
    if job["op"] != "torch_gelu":
        raise SystemExit(f"unknown op {job['op']!r}")
    import torch

    dtype = torch.float64 if job.get("dtype", "float64") == "float64" else torch.float32
    x = torch.tensor(job["xs"], dtype=dtype)
    out = torch.nn.functional.gelu(x, approximate=job.get("approximate", "none"))
    json.dump([float(v) for v in out.tolist()], sys.stdout)


if __name__ == "__main__":
    main()
