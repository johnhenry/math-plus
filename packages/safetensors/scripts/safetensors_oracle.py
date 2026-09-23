#!/usr/bin/env python3
"""Python `safetensors` oracle for @johnhenry/math-plus-safetensors tests.

Subcommands (all output JSON on stdout; raw bytes are base64 so NaN/Inf and
exact bit patterns survive the trip):

  probe
      {"numpy": ver, "safetensors": ver, "bf16": "torch" | "ml_dtypes" | null}
  make OUTDIR
      Writes fixtures with the reference writers and prints a manifest:
        mixed.safetensors       safetensors.numpy.save_file, every numpy dtype
                                (F16 incl. subnormal/inf/nan/-0, F32, F64, I64
                                up to 2^52, BOOL, empty [0] and [2,0], scalar [])
        bf16.safetensors        safetensors.torch.save_file (BF16 + F32)
        misaligned.safetensors  hand-built: unpadded header so 8+N % 8 != 0,
                                as MLX writes it; verified readable by
                                safetensors.safe_open before being listed
  dump FILE
      {"metadata": {...} | null, "tensors": {name: {"dtype", "shape",
       "bytes_b64", "f32_b64"}}} -- read by the reference reader
      (safetensors.deserialize); f32_b64 is the reference float32 conversion
      (numpy astype(float32); torch .float() for BF16).
  serialize SPEC OUT
      SPEC = {"metadata": {...} | null, "tensors": {name: {"dtype": "F16",
      "shape": [...], "bytes_b64": "..."}}}; writes the reference
      serialization (safetensors.serialize) of exactly those tensors to OUT.
"""

import base64
import json
import struct
import sys

import numpy as np
import safetensors
from safetensors import deserialize, safe_open, serialize
from safetensors.numpy import save_file as np_save_file

SAFE_TO_NUMPY = {
    "BOOL": "bool", "U8": "uint8", "I8": "int8", "U16": "uint16", "I16": "int16",
    "F16": "float16", "U32": "uint32", "I32": "int32", "F32": "float32",
    "U64": "uint64", "I64": "int64", "F64": "float64",
}
SAFE_TO_SERIALIZE = {**SAFE_TO_NUMPY, "BF16": "bfloat16"}


def bf16_backend():
    try:
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        pass
    try:
        import ml_dtypes  # noqa: F401

        return "ml_dtypes"
    except ImportError:
        return None


def b64(data: bytes) -> str:
    return base64.b64encode(bytes(data)).decode("ascii")


def to_f32_bytes(dtype: str, shape, data: bytes) -> str:
    if dtype == "BF16":
        backend = bf16_backend()
        if backend == "torch":
            import torch

            t = torch.frombuffer(bytearray(data), dtype=torch.bfloat16) if data else torch.empty(0, dtype=torch.bfloat16)
            return b64(t.to(torch.float32).numpy().tobytes())
        if backend == "ml_dtypes":
            import ml_dtypes

            return b64(np.frombuffer(data, dtype=ml_dtypes.bfloat16).astype(np.float32).tobytes())
        return ""
    arr = np.frombuffer(data, dtype=SAFE_TO_NUMPY[dtype])
    with np.errstate(over="ignore"):  # F64 beyond float32 range -> +-inf, as intended
        return b64(arr.astype(np.float32).tobytes())


def dump(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + n])
    tensors = {}
    for name, info in deserialize(raw):
        data = bytes(info["data"])
        tensors[name] = {
            "dtype": header[name]["dtype"],
            "shape": list(info["shape"]),
            "bytes_b64": b64(data),
            "f32_b64": to_f32_bytes(header[name]["dtype"], info["shape"], data),
        }
    return {"metadata": header.get("__metadata__"), "tensors": tensors}


def make(outdir: str) -> dict:
    rng = np.random.default_rng(1234)
    mixed = {
        "f32": rng.standard_normal((3, 4)).astype(np.float32),
        "f16": np.array([1.5, -0.1, 65504, 6e-8, np.inf, -np.inf, np.nan, -0.0, 2**-14], dtype=np.float16),
        "f64": np.array([[np.pi, -1e300], [1e-310, 0.1]], dtype=np.float64),
        "i64": np.array([-(2**40), 2**52, -1, 7], dtype=np.int64),
        "u64": np.array([0, 2**52, 3], dtype=np.uint64),
        "i32": np.array([-(2**31), 2**31 - 1, 5], dtype=np.int32),
        "u32": np.array([0, 2**32 - 1], dtype=np.uint32),
        "i16": np.array([-32768, 32767], dtype=np.int16),
        "u16": np.array([65535, 1], dtype=np.uint16),
        "i8": np.array([-128, 127, 0], dtype=np.int8),
        "u8": np.array([255, 0, 9], dtype=np.uint8),
        "bool": np.array([[True, False, True], [False, False, True]]),
        "empty": np.zeros((0,), dtype=np.float32),
        "empty2d": np.zeros((2, 0), dtype=np.float16),
        "scalar": np.array(3.25, dtype=np.float32),
    }
    files = {}
    np_save_file(mixed, f"{outdir}/mixed.safetensors", metadata={"format": "np", "note": "fixture"})
    files["mixed"] = f"{outdir}/mixed.safetensors"

    backend = bf16_backend()
    if backend == "torch":
        import torch
        from safetensors.torch import save_file as pt_save_file

        g = torch.Generator().manual_seed(7)
        bf = torch.cat([
            torch.tensor([1.0, -3.0, 0.1, 3.0e38, 1e-40, float("inf"), float("nan")]),
            torch.randn(20, generator=g) * 100,
        ]).to(torch.bfloat16)
        pt_save_file({"bf16": bf, "f32": torch.arange(6, dtype=torch.float32).reshape(2, 3)}, f"{outdir}/bf16.safetensors", metadata={"format": "pt"})
        files["bf16"] = f"{outdir}/bf16.safetensors"
    elif backend == "ml_dtypes":
        import ml_dtypes

        bf = np.array([1.0, -3.0, 0.1, 3.0e38, 1e-40, np.inf, np.nan], dtype=np.float32).astype(ml_dtypes.bfloat16)
        blob = serialize({"bf16": {"dtype": "bfloat16", "shape": list(bf.shape), "data": bf.tobytes()}}, {"format": "np"})
        with open(f"{outdir}/bf16.safetensors", "wb") as f:
            f.write(bytes(blob))
        files["bf16"] = f"{outdir}/bf16.safetensors"

    # Misaligned: compact JSON, no padding (what MLX writes). Pick a header
    # whose length makes 8+N odd so every F16/F32 view is misaligned.
    a = np.array([1.25, -2.5, 3.75], dtype=np.float32)
    h = np.array([0.5, -0.25], dtype=np.float16)
    for extra in range(0, 8):
        header = {
            "__metadata__": {"x": "y" * extra},
            "a": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]},
            "h": {"dtype": "F16", "shape": [2], "data_offsets": [12, 16]},
        }
        text = json.dumps(header, separators=(",", ":")).encode()
        if (8 + len(text)) % 2 == 1:
            break
    with open(f"{outdir}/misaligned.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + a.tobytes() + h.tobytes())
    with safe_open(f"{outdir}/misaligned.safetensors", framework="np") as f:
        assert np.array_equal(f.get_tensor("a"), a) and np.array_equal(f.get_tensor("h"), h)
    files["misaligned"] = f"{outdir}/misaligned.safetensors"
    return {"files": files, "bf16": backend}


def serialize_spec(spec_path: str, out: str) -> None:
    with open(spec_path) as f:
        spec = json.load(f)
    tensors = {
        name: {
            "dtype": SAFE_TO_SERIALIZE[t["dtype"]],
            "shape": t["shape"],
            "data": base64.b64decode(t["bytes_b64"]),
        }
        for name, t in spec["tensors"].items()
    }
    blob = serialize(tensors, spec.get("metadata"))
    with open(out, "wb") as f:
        f.write(bytes(blob))


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "probe":
        result = {"numpy": np.__version__, "safetensors": safetensors.__version__, "bf16": bf16_backend()}
    elif cmd == "make":
        result = make(sys.argv[2])
    elif cmd == "dump":
        result = dump(sys.argv[2])
    elif cmd == "serialize":
        serialize_spec(sys.argv[2], sys.argv[3])
        result = {"ok": True}
    else:
        raise SystemExit(f"unknown command {cmd!r}")
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
