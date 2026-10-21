#!/usr/bin/env python3
"""scipy.signal oracle for @johnhenry/math-plus-signal's differential tests (issue #44).

Same job-file convention as tensor-core's numpy_oracle.py: a single JSON
argument path describing the operation, writes a JSON result to stdout
(chosen over .npy here since results are small and sometimes structured --
e.g. SOS arrays -- not bulk tensor data).
"""
import json
import sys
import warnings

import numpy as np
from scipy import signal

# SciPy warns when nperseg > len(x) (it then clamps); that is expected here.
warnings.simplefilter("ignore")


def _window(w):
    """JSON window -> scipy window: a list is an ndarray, {"tuple": [...]} a tuple (null -> None), a string or number as is."""
    if isinstance(w, list):
        return np.array(w, dtype=float)
    if isinstance(w, dict):
        return tuple(np.array(a, dtype=float) if isinstance(a, list) else a for a in w["tuple"])
    return w


# Detrend callables for the differential tests (JSON cannot carry a function). Each acts along the LAST axis, which is
# the segment axis whenever scipy calls it (welch always; stft with its default axis=-1).
def _first3(d):
    return d - d[..., :3].mean(axis=-1, keepdims=True)


def _ends(d):
    n = d.shape[-1]
    t = np.arange(n) / (n - 1)
    return d - (d[..., :1] + (d[..., -1:] - d[..., :1]) * t)


DETREND_FNS = {"first3": _first3, "ends": _ends}


def _kwargs(kw: dict) -> dict:
    """JSON kwargs -> scipy kwargs: `window` per _window, a {"fn": name} `detrend` becomes a callable."""
    kw = dict(kw)
    if "window" in kw:
        kw["window"] = _window(kw["window"])
    if isinstance(kw.get("detrend"), dict):
        kw["detrend"] = DETREND_FNS[kw["detrend"]["fn"]]
    return kw


def _signal(job: dict, key: str = "x") -> np.ndarray:
    """A (possibly N-D, possibly complex) array from `job[key]` (+ `job[key + '_imag']`)."""
    # An empty array of a given shape (JSON cannot carry (3, 0) or (0, 5) data): `<key>_shape`.
    x = np.zeros(job[key + "_shape"]) if key + "_shape" in job else np.array(job[key], dtype=float)
    if key + "_imag" in job:
        x = x + 1j * np.array(job[key + "_imag"], dtype=float)
    return x


def _complex_parts(z: np.ndarray) -> dict:
    return {"shape": list(z.shape), "real": np.real(z).tolist(), "imag": np.imag(z).tolist()}


def run(job: dict):
    op = job["op"]

    if op == "butter_sos":
        sos = signal.butter(job["order"], job["wn"], btype=job["btype"], output="sos")
        result = {"sos": sos.tolist()}

    elif op == "sosfilt":
        sos = np.array(job["sos"], dtype=float)
        x = np.array(job["x"], dtype=float)
        y = signal.sosfilt(sos, x)
        result = {"y": y.tolist()}

    elif op == "find_peaks":
        x = np.array(job["x"], dtype=float)
        kwargs = {}
        if "height" in job:
            kwargs["height"] = job["height"]
        if "distance" in job:
            kwargs["distance"] = job["distance"]
        if "prominence" in job:
            kwargs["prominence"] = job["prominence"]
        indices, props = signal.find_peaks(x, **kwargs)
        result = {"indices": indices.tolist()}
        if "peak_heights" in props:
            result["heights"] = props["peak_heights"].tolist()
        if "prominences" in props:
            result["prominences"] = props["prominences"].tolist()

    elif op == "peak_prominences":
        x = np.array(job["x"], dtype=float)
        indices = np.array(job["indices"], dtype=int)
        prominences, left_bases, right_bases = signal.peak_prominences(x, indices)
        result = {"prominences": prominences.tolist()}

    elif op == "convolve":
        a = np.array(job["a"], dtype=float)
        b = np.array(job["b"], dtype=float)
        mode = job.get("mode", "full")
        y = np.convolve(a, b, mode=mode)
        result = {"y": y.tolist()}

    elif op == "correlate":
        a = np.array(job["a"], dtype=float)
        b = np.array(job["b"], dtype=float)
        mode = job.get("mode", "full")
        y = signal.correlate(a, b, mode=mode)
        result = {"y": y.tolist()}

    elif op == "correlate2d":
        a = np.array(job["a"], dtype=float)
        b = np.array(job["b"], dtype=float)
        y = signal.correlate2d(a, b, mode="full")
        result = {"y": y.tolist()}

    elif op == "sosfreqz":
        # scipy.signal.sosfreqz's own default grid (whole=False, fs=2*pi):
        # w[i] = i*pi/worN -- matches freqz.ts's frequency grid exactly
        # (verified numerically before writing the TS side).
        sos = np.array(job["sos"], dtype=float)
        w, h = signal.sosfreqz(sos, worN=job.get("worN", 512))
        result = {"frequencies": w.tolist(), "real": h.real.tolist(), "imag": h.imag.tolist()}

    elif op == "welch":
        # return_onesided=False matches welch.ts's documented v1 scope
        # (two-sided PSD, not scipy's default one-sided-with-doubling).
        x = np.array(job["x"], dtype=float)
        window = np.array(job["window"], dtype=float)
        f, pxx = signal.welch(
            x,
            fs=1.0,
            window=window,
            nperseg=job["nperseg"],
            noverlap=job["noverlap"],
            return_onesided=False,
            scaling="density",
            detrend=False,
        )
        result = {"freqs": f.tolist(), "Pxx": pxx.tolist()}

    elif op == "resample_poly":
        # Identity-path oracle only (issue #113): resample_poly(x, n, n) must
        # return x's values unchanged. The non-identity paths use a Hamming
        # FIR rather than scipy's Kaiser default (resample.ts), so they are
        # checked by reconstruction quality, not against this op.
        x = np.array(job["x"], dtype=np.dtype(job.get("dtype", "float64")))
        y = signal.resample_poly(x, job["up"], job["down"])
        result = {"y": y.astype(np.float64).tolist(), "dtype": str(y.dtype)}

    elif op == "welch_kw":
        # scipy.signal.welch called with ONLY the keyword arguments the job
        # lists, so omitted options take SciPy's own defaults. `x` may be N-D,
        # and complex when the job carries `x_imag`; `window` may be a name,
        # {"tuple": [...]} or an explicit array; `detrend` may be {"fn": name}.
        x = _signal(job)
        f, pxx = signal.welch(x, **_kwargs(job.get("kwargs", {})))
        result = {"freqs": f.tolist(), "freqs_shape": list(f.shape), "shape": list(pxx.shape), "Pxx": pxx.tolist()}

    elif op == "stft_kw":
        # scipy.signal.stft with only the listed kwargs. Zxx is (..., freq, time) with the frequency axis where `axis` was.
        x = _signal(job)
        f, t, z = signal.stft(x, **_kwargs(job.get("kwargs", {})))
        result = {"freqs": f.tolist(), "times": t.tolist(), **_complex_parts(z)}

    elif op == "istft_kw":
        # scipy.signal.istft of a spectrogram given as real/imag (any dimensionality).
        z = _signal(job, "z") if ("z" in job or "z_shape" in job) else np.array(job["real"], dtype=float) + 1j * np.array(job["imag"], dtype=float)
        t, x = signal.istft(z, **_kwargs(job.get("kwargs", {})))
        result = {"times": t.tolist(), "complex": bool(np.iscomplexobj(x)), **_complex_parts(x)}

    elif op == "get_window":
        w = signal.get_window(_window(job["window"]), job["n"], fftbins=job.get("fftbins", True))
        result = {"w": w.tolist()}

    elif op == "check_nola":
        result = {"ok": bool(signal.check_NOLA(_window(job["window"]), job["nperseg"], job["noverlap"], **({"tol": job["tol"]} if "tol" in job else {})))}

    elif op == "check_cola":
        result = {"ok": bool(signal.check_COLA(_window(job["window"]), job["nperseg"], job["noverlap"], **({"tol": job["tol"]} if "tol" in job else {})))}

    elif op == "window_table":
        # SciPy's own table of window names (the source of truth for the aliases getWindow accepts).
        from scipy.signal.windows import _windows
        result = {"table": [[list(names), bool(needs)] for names, (_f, needs) in _windows._win_equiv_raw.items()]}

    elif op == "periodogram_kw":
        # scipy.signal.periodogram with only the listed kwargs; `x` may be empty (`x_shape`), N-D or complex.
        x = _signal(job)
        f, pxx = signal.periodogram(x, **_kwargs(job.get("kwargs", {})))
        result = {"freqs": np.asarray(f).tolist(), "freqs_shape": list(np.shape(f)), "shape": list(pxx.shape), "Pxx": pxx.tolist()}

    elif op == "csd_kw":
        # scipy.signal.csd(x, y, ...) -- `y` from `y`/`y_imag`/`y_shape`, or the very same object as x when `same` is set.
        x = _signal(job)
        y = x if job.get("same") else _signal(job, "y")
        f, pxy = signal.csd(x, y, **_kwargs(job.get("kwargs", {})))
        result = {"freqs": np.asarray(f).tolist(), "freqs_shape": list(np.shape(f)), "complex": bool(np.iscomplexobj(pxy)), **_complex_parts(pxy)}

    elif op == "coherence_kw":
        x = _signal(job)
        y = x if job.get("same") else _signal(job, "y")
        f, cxy = signal.coherence(x, y, **_kwargs(job.get("kwargs", {})))
        result = {"freqs": np.asarray(f).tolist(), "freqs_shape": list(np.shape(f)), "shape": list(cxy.shape), "Cxy": cxy.tolist()}

    elif op == "spectrogram_kw":
        x = _signal(job)
        f, t, sxx = signal.spectrogram(x, **_kwargs(job.get("kwargs", {})))
        result = {"freqs": np.asarray(f).tolist(), "times": np.asarray(t).tolist(), "complex": bool(np.iscomplexobj(sxx)), **_complex_parts(sxx)}

    elif op == "windowed_frame_fft":
        # A primitive, version-stable oracle for our own stft's exact
        # algorithm (windowed-frame -> full FFT, no extra normalization) --
        # deliberately NOT scipy.signal.stft, whose own internal scaling
        # convention varies across scipy versions and would test "did we
        # replicate scipy's scaling" rather than "is the FFT of a windowed
        # frame correct," which is the property that actually matters here.
        x = np.array(job["x"], dtype=float)
        window = np.array(job["window"], dtype=float)
        frame = x * window
        spectrum = np.fft.fft(frame)
        result = {"real": spectrum.real.tolist(), "imag": spectrum.imag.tolist()}

    else:
        raise ValueError(f"scipy_oracle: unknown op {op!r}")

    return result


def run_recorded(job: dict):
    """run(), plus the text of every warning SciPy raised, as `warnings` (a job that SciPy rejects reports `error`)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = run(job)
        except Exception as e:  # noqa: BLE001 -- reported to the test, which decides whether that was expected
            if not job.get("expect_error"):
                raise
            result = {"error": f"{type(e).__name__}: {e}", "error_type": type(e).__name__}
    # SciPy's own `warnings.warn` is a UserWarning; NumPy's RuntimeWarnings (divide by zero ...) are kept apart.
    result["warnings"] = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    result["runtime_warnings"] = [str(w.message) for w in caught if not issubclass(w.category, UserWarning)]
    return result


def main() -> None:
    job_path = sys.argv[1]
    with open(job_path) as f:
        job = json.load(f)
    # {"op": "batch", "jobs": [...]} runs many jobs in one interpreter start
    # (python + scipy import dominates a single job's cost).
    if job["op"] == "batch":
        print(json.dumps([run_recorded(j) for j in job["jobs"]]))
    else:
        print(json.dumps(run_recorded(job)))


if __name__ == "__main__":
    main()
