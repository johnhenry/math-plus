/** Hann window (issue #44's stft/istft dependency). `periodic=true` (the default) matches `scipy.signal.get_window`'s "periodic" convention (denominator `n`, not `n-1`) -- what `scipy.signal.stft` uses internally by default for its own Hann window, needed to differential-test against it exactly. */
export function hannWindow(n: number, periodic = true): Float64Array {
  const w = new Float64Array(n);
  const denom = periodic ? n : n - 1;
  if (denom === 0) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / denom);
  return w;
}

/** Hamming window (symmetric, `sym=True` convention -- denominator `n-1`), `resamplePoly`'s anti-aliasing FIR filter dependency. */
export function hammingWindow(n: number): Float64Array {
  const w = new Float64Array(n);
  if (n === 1) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/** The windows `welch`/`stft` accept by name; `"rectangular"` is an alias of `"boxcar"`. */
export type WindowName = "hann" | "hamming" | "blackman" | "boxcar" | "rectangular";
export type WindowSpec = WindowName | Float64Array;

/**
 * `scipy.signal.get_window(name, n)` with its default `fftbins=True`: the PERIODIC form (denominator `n`), the kind spectral analysis uses.
 * (`hammingWindow` above is the symmetric one `resamplePoly` needs.) A `Float64Array` is returned as given, after checking its length is `n`.
 */
export function getWindow(spec: WindowSpec, n: number): Float64Array {
  if (!Number.isInteger(n) || n < 1) throw new RangeError(`getWindow: length must be a positive integer, got ${n}`);
  if (spec instanceof Float64Array) {
    if (spec.length !== n) throw new RangeError(`getWindow: window length (${spec.length}) must equal ${n}`);
    return spec;
  }
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = (2 * Math.PI * i) / n;
    switch (spec) {
      case "hann":
        w[i] = 0.5 - 0.5 * Math.cos(x);
        break;
      case "hamming":
        w[i] = 0.54 - 0.46 * Math.cos(x);
        break;
      case "blackman":
        w[i] = 0.42 - 0.5 * Math.cos(x) + 0.08 * Math.cos(2 * x);
        break;
      case "boxcar":
      case "rectangular":
        w[i] = 1;
        break;
      default:
        throw new RangeError(`getWindow: unknown window ${String(spec)} (hann, hamming, blackman, boxcar, rectangular, or a Float64Array)`);
    }
  }
  return w;
}
