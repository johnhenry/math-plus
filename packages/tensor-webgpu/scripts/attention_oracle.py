#!/usr/bin/env python3
"""NumPy oracle for @johnhenry/math-plus-tensor-webgpu's fused attention tests.

Reads JSON on stdin:
  {"cases": [{"q": <b64 f32>, "k": <b64 f32>, "v": <b64 f32>,
              "batch": B, "seqQ": LQ, "seqK": LK, "dim": D, "scale": s,
              "mask": <b64 f32> | null, "maskShape": [..] | null}, ...]}
q is [B, LQ, D], k and v are [B, LK, D]; the mask (nonzero = attend) is
broadcast against [B, LQ, LK] with NumPy's rules. Writes JSON to stdout:
  {"results": [{"out": <b64 float64 [B, LQ, D]>}, ...]}
out = softmax(scale * q @ k^T, masked keys excluded) @ v in float64 from the
exact f32 inputs. Query rows with no visible key are 0 (the kernel's
documented convention). Masked keys contribute exactly nothing, even when
their V rows hold non-finite values.
"""

import base64
import json
import sys

import numpy as np


def decode(data: str, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype="<f4").reshape(shape).astype(np.float64)


def encode(x: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(x, dtype="<f8").tobytes()).decode("ascii")


def attention(case) -> np.ndarray:
    b, lq, lk, d = case["batch"], case["seqQ"], case["seqK"], case["dim"]
    q = decode(case["q"], (b, lq, d))
    k = decode(case["k"], (b, lk, d))
    v = decode(case["v"], (b, lk, d))
    s = case["scale"] * np.einsum("bqd,bkd->bqk", q, k)
    if case.get("mask") is not None:
        visible = np.broadcast_to(decode(case["mask"], tuple(case["maskShape"])) != 0, (b, lq, lk))
    else:
        visible = np.ones((b, lq, lk), dtype=bool)
    s = np.where(visible, s, -np.inf)
    m = s.max(axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    p = np.where(visible, np.exp(s - m), 0.0)
    denom = p.sum(axis=-1, keepdims=True)
    w = np.divide(p, denom, out=np.zeros_like(p), where=denom > 0)
    # Masked (w == 0) keys must not propagate non-finite V: 0 * nan = nan.
    out = np.zeros((b, lq, d))
    for i in range(b):
        vis_rows = visible[i].any(axis=0)
        out[i] = w[i][:, vis_rows] @ v[i][vis_rows]
    return out


def main() -> None:
    job = json.load(sys.stdin)
    json.dump({"results": [{"out": encode(attention(c))} for c in job["cases"]]}, sys.stdout)


if __name__ == "__main__":
    main()
