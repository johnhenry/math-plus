#!/usr/bin/env python3
"""SciPy oracle for @johnhenry/math-plus-special's canonical erf/erfc/GELU (issue #122).

Reads a JSON job file and prints a JSON array of results to stdout:
  {"op": "erf" | "erfc", "xs": [...]}   -> scipy.special.erf / erfc (float64)
  {"op": "gelu_ndtr", "xs": [...]}      -> x * scipy.special.ndtr(x) (float64): exact GELU with
                                           full relative accuracy in the left tail (ndtr is
                                           erfc-based)

Inputs/outputs must be finite (JSON has no NaN/Infinity); special values are
covered by the package's own unit tests instead.
"""

import json
import sys

import numpy as np
from scipy import special


def main() -> None:
    with open(sys.argv[1]) as f:
        job = json.load(f)
    op = job["op"]
    x = np.asarray(job["xs"], dtype=np.float64)
    if op == "erf":
        out = special.erf(x)
    elif op == "erfc":
        out = special.erfc(x)
    elif op == "gelu_ndtr":
        out = x * special.ndtr(x)
    else:
        raise SystemExit(f"unknown op {op!r}")
    json.dump([float(v) for v in out], sys.stdout)


if __name__ == "__main__":
    main()
