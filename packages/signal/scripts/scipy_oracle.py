#!/usr/bin/env python3
"""scipy.signal oracle for @johnhenry/math-plus-signal's differential tests (issue #44).

Same job-file convention as tensor-core's numpy_oracle.py: a single JSON
argument path describing the operation, writes a JSON result to stdout
(chosen over .npy here since results are small and sometimes structured --
e.g. SOS arrays -- not bulk tensor data).
"""
import json
import sys

import numpy as np
from scipy import signal


def main() -> None:
    job_path = sys.argv[1]
    with open(job_path) as f:
        job = json.load(f)

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
        result = {"frequencies": f.tolist(), "psd": pxx.tolist()}

    elif op == "resample_poly":
        # Identity-path oracle only (issue #113): resample_poly(x, n, n) must
        # return x's values unchanged. The non-identity paths use a Hamming
        # FIR rather than scipy's Kaiser default (resample.ts), so they are
        # checked by reconstruction quality, not against this op.
        x = np.array(job["x"], dtype=np.dtype(job.get("dtype", "float64")))
        y = signal.resample_poly(x, job["up"], job["down"])
        result = {"y": y.astype(np.float64).tolist(), "dtype": str(y.dtype)}

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

    print(json.dumps(result))


if __name__ == "__main__":
    main()
