#!/usr/bin/env python3
"""PyTorch oracle for @johnhenry/math-plus-tensor-autograd (issue #123).

Usage:
  torch_oracle.py probe
      {"torch": version}
  torch_oracle.py run            (JSON request on stdin, JSON response on stdout)
      Request:  {"cases": [CASE, ...]}
      CASE:     {"id", "kind", "dtype": "f32"|"f64", "config": {...},
                 "inputs": {name: T}, "masks": {name: T}, "state": {name: T},
                 "gradOut": T,
                 "stateFile": path   (optional: load state from a safetensors
                                      file instead of "state"),
                 "exportState": path (optional: keep torch's init, rounded
                                      to f16, and save it there as f16
                                      safetensors before running)}
      T:        {"shape": [...], "data": [flat row-major values]}
      Response: {"results": {id: {"out": T, "grads": {name: T}} | {"error": str}}}

For every case the oracle builds the op/layer in PyTorch, loads `state`
with load_state_dict(strict=True) -- so a JS parameter name that doesn't
match PyTorch's state_dict key fails loudly -- runs forward, then backward
of sum(out * gradOut), and returns the output plus the gradient of every
float input and every named parameter. Layers run in eval() with dropout 0
and inputs that require grad, which keeps PyTorch off its fused inference
fast paths (the JS side implements the reference math path).
"""

import json
import math
import sys

import torch
import torch.nn.functional as F

DTYPES = {"f32": torch.float32, "f64": torch.float64}


def to_t(spec, dtype):
    return torch.tensor(spec["data"], dtype=dtype).reshape(spec["shape"])


def to_bool(spec):
    return torch.tensor([bool(v) for v in spec["data"]], dtype=torch.bool).reshape(spec["shape"])


def to_spec(t):
    t = t.detach().to(torch.float64).contiguous()
    return {"shape": list(t.shape), "data": t.reshape(-1).tolist()}


def mask_of(masks, name, dtype):
    if name not in masks:
        return None
    spec = masks[name]
    return to_bool(spec) if spec.get("bool") else to_t(spec, dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_cos_sin(T, dim, base, offset, dtype):
    # Hugging Face default RoPE, computed in f64 then cast (matches the JS side).
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    pos = torch.arange(offset, offset + T, dtype=torch.float64)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, base, offset):
    cos, sin = rope_cos_sin(x.shape[-2], x.shape[-1], base, offset, x.dtype)
    return x * cos + rotate_half(x) * sin


class GeGLU(torch.nn.Module):
    """Reference: Linear to 2*dim_out, GELU on the FIRST half (ModernBERT order)."""

    def __init__(self, dim_in, dim_out, bias, approximate, dtype):
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, 2 * dim_out, bias=bias, dtype=dtype)
        self.approximate = approximate

    def forward(self, x):
        value, gate = self.proj(x).chunk(2, dim=-1)
        return F.gelu(value, approximate=self.approximate) * gate


class RopeMHA(torch.nn.Module):
    """nn.MultiheadAttention's parameters + RoPE on per-head q/k (the JS `rotary` option)."""

    def __init__(self, E, H, bias, batch_first, base, dtype):
        super().__init__()
        self.E, self.H, self.batch_first, self.base = E, H, batch_first, base
        self.in_proj_weight = torch.nn.Parameter(torch.empty(3 * E, E, dtype=dtype))
        self.in_proj_bias = torch.nn.Parameter(torch.empty(3 * E, dtype=dtype)) if bias else None
        self.out_proj = torch.nn.Linear(E, E, bias=bias, dtype=dtype)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        if not self.batch_first:
            q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        N, L, E = q.shape
        S = k.shape[1]
        H, D = self.H, E // self.H
        wq, wk, wv = self.in_proj_weight.chunk(3)
        bq, bk, bv = self.in_proj_bias.chunk(3) if self.in_proj_bias is not None else (None, None, None)
        qh = F.linear(q, wq, bq).reshape(N, L, H, D).transpose(1, 2)
        kh = F.linear(k, wk, bk).reshape(N, S, H, D).transpose(1, 2)
        vh = F.linear(v, wv, bv).reshape(N, S, H, D).transpose(1, 2)
        qh, kh = apply_rope(qh, self.base, 0), apply_rope(kh, self.base, 0)
        out = F.scaled_dot_product_attention(qh, kh, vh)
        out = self.out_proj(out.transpose(1, 2).reshape(N, L, E))
        return out if self.batch_first else out.transpose(0, 1)


def build(case):
    """Returns (module_or_None, fn(inputs: dict) -> Tensor)."""
    kind = case["kind"]
    cfg = case.get("config", {})
    dtype = DTYPES[case["dtype"]]
    masks = case.get("masks", {})

    if kind == "reshape":
        return None, lambda i: i["x"].reshape(cfg["shape"])
    if kind == "permuteReshape":
        return None, lambda i: i["x"].permute(cfg["axes"]).reshape(cfg["shape"])
    if kind == "permute":
        return None, lambda i: i["x"].permute(cfg["axes"])
    if kind == "transpose":
        return None, lambda i: i["x"].transpose(cfg["dim0"], cfg["dim1"])
    if kind == "slice":
        def f(i):
            x = i["x"]
            for axis, spec in enumerate(cfg["specs"]):
                if spec is None:
                    continue
                idx = list(range(*slice(spec.get("start"), spec.get("end"), spec.get("step")).indices(x.shape[axis])))
                x = x.index_select(axis, torch.tensor(idx, dtype=torch.long))
            return x
        return None, f
    if kind == "narrow":
        return None, lambda i: i["x"].narrow(cfg["dim"], cfg["start"], cfg["length"])
    if kind == "concat":
        return None, lambda i: torch.cat([i[n] for n in cfg["order"]], dim=cfg["axis"])
    if kind == "exp":
        return None, lambda i: i["x"].exp()
    if kind == "tanh":
        return None, lambda i: i["x"].tanh()
    if kind == "gelu":
        return None, lambda i: F.gelu(i["x"], approximate=cfg["approximate"])
    if kind == "maskedFill":
        return None, lambda i: i["x"].masked_fill(to_bool(masks["mask"]), cfg["value"])
    if kind == "matmul":
        return None, lambda i: i["a"] @ i["b"]
    if kind == "divScalar":
        return None, lambda i: i["x"] / cfg["value"]
    if kind == "cast":
        return None, lambda i: i["x"].to(DTYPES[cfg["to"]])
    if kind == "sdpa":
        def f(i):
            return F.scaled_dot_product_attention(
                i["q"], i["k"], i["v"], attn_mask=mask_of(masks, "attnMask", dtype),
                is_causal=cfg.get("isCausal", False), scale=cfg.get("scale"))
        return None, f
    if kind == "rope":
        return None, lambda i: apply_rope(i["x"], cfg["base"], cfg.get("offset", 0))
    if kind == "geglu":
        return None, lambda i: (lambda v, g: F.gelu(v, approximate=cfg["approximate"]) * g)(*i["x"].chunk(2, dim=-1))
    if kind == "linear":
        m = torch.nn.Linear(cfg["in"], cfg["out"], bias=cfg["bias"], dtype=dtype)
        return m, lambda i: m(i["x"])
    if kind == "layerNorm":
        m = torch.nn.LayerNorm(cfg["dim"], eps=cfg["eps"], bias=cfg["bias"], dtype=dtype)
        return m, lambda i: m(i["x"])
    if kind == "mha":
        m = torch.nn.MultiheadAttention(cfg["embedDim"], cfg["numHeads"], bias=cfg["bias"],
                                        batch_first=cfg["batchFirst"], dropout=0.0, dtype=dtype)
        def f(i):
            # Self-attention cases send only "q" (passed as query, key AND value).
            out, _ = m(i["q"], i.get("k", i["q"]), i.get("v", i["q"]), attn_mask=mask_of(masks, "attnMask", dtype),
                       key_padding_mask=mask_of(masks, "keyPaddingMask", dtype), need_weights=False)
            return out
        return m, f
    if kind == "mhaRope":
        m = RopeMHA(cfg["embedDim"], cfg["numHeads"], cfg["bias"], cfg["batchFirst"], cfg["base"], dtype)
        return m, lambda i: m(i["q"], i.get("k", i["q"]), i.get("v", i["q"]))
    if kind == "encoderLayer":
        m = torch.nn.TransformerEncoderLayer(
            cfg["dModel"], cfg["nhead"], dim_feedforward=cfg["dimFeedforward"], dropout=0.0,
            activation=cfg["activation"], layer_norm_eps=cfg["eps"], batch_first=cfg["batchFirst"],
            norm_first=cfg["normFirst"], bias=cfg["bias"], dtype=dtype)
        def f(i):
            return m(i["x"], src_mask=mask_of(masks, "srcMask", dtype),
                     src_key_padding_mask=mask_of(masks, "srcKeyPaddingMask", dtype))
        return m, f
    if kind == "geGLU":
        m = GeGLU(cfg["dimIn"], cfg["dimOut"], cfg["bias"], cfg["approximate"], dtype)
        return m, lambda i: m(i["x"])
    raise ValueError(f"unknown kind {kind!r}")


def run_case(case):
    dtype = DTYPES[case["dtype"]]
    module, fn = build(case)
    if module is not None:
        if "stateFile" in case:
            from safetensors.torch import load_file
            state = {k: v.to(dtype) for k, v in load_file(case["stateFile"]).items()}
            module.load_state_dict(state, strict=True)
        elif "state" in case:
            state = {k: to_t(v, dtype) for k, v in case["state"].items()}
            module.load_state_dict(state, strict=True)
        if "exportState" in case:
            # Keep torch's own init, rounded to f16 so both sides see identical
            # weights, and save it as an f16 safetensors file for JS to load.
            from safetensors.torch import save_file
            with torch.no_grad():
                for prm in module.parameters():
                    prm.copy_(prm.to(torch.float16).to(prm.dtype))
            half = {k: v.detach().to(torch.float16).contiguous() for k, v in module.state_dict().items()}
            save_file(half, case["exportState"], metadata={"format": "pt"})
        module.eval()
    inputs = {}
    for name, spec in case["inputs"].items():
        in_dtype = DTYPES[spec.get("dtype", case["dtype"])]
        inputs[name] = to_t(spec, in_dtype).requires_grad_(True)
    out = fn(inputs)
    grad_out = to_t(case["gradOut"], out.dtype)
    (out * grad_out).sum().backward()
    grads = {f"input:{n}": to_spec(t.grad) for n, t in inputs.items() if t.grad is not None}
    if module is not None:
        for n, p in module.named_parameters():
            if p.grad is not None:
                grads[n] = to_spec(p.grad)
    return {"out": to_spec(out), "grads": grads}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "probe":
        info = {"torch": torch.__version__}
        try:
            import safetensors
            info["safetensors"] = safetensors.__version__
        except ImportError:
            info["safetensors"] = None
        print(json.dumps(info))
        return
    torch.manual_seed(0)
    req = json.load(sys.stdin)
    results = {}
    for case in req["cases"]:
        try:
            results[case["id"]] = run_case(case)
        except Exception as exc:  # reported per case, so one bad case can't hide the rest
            results[case["id"]] = {"error": f"{type(exc).__name__}: {exc}"}
    for r in results.values():
        for t in ([r["out"]] + list(r["grads"].values())) if "out" in r else []:
            if any(not math.isfinite(v) for v in t["data"]):
                raise SystemExit("oracle produced a non-finite value; keep test cases finite (JSON can't carry them)")
    json.dump({"results": results}, sys.stdout)


if __name__ == "__main__":
    main()
