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

/** The windows accepted by name (`scipy.signal.get_window`'s primary names; `"rectangular"` is an alias of `"boxcar"`). Those taking parameters have defaults here only for `"tukey"`, `"exponential"` and `"taylor"`. */
export type WindowName =
  | "hann"
  | "hamming"
  | "blackman"
  | "boxcar"
  | "rectangular"
  | "bartlett"
  | "flattop"
  | "parzen"
  | "bohman"
  | "nuttall"
  | "blackmanharris"
  | "barthann"
  | "cosine"
  | "triang"
  | "tukey"
  | "exponential"
  | "taylor";

/**
 * `(name, param, ...)` window tuples, `scipy.signal.get_window`'s tuple form (parameters are positional, as in SciPy):
 * `["kaiser", beta]`, `["gaussian", std]`, `["general_gaussian", p, sig]`, `["general_hamming", alpha]`,
 * `["general_cosine", coefficients]`, `["tukey", alpha = 0.5]`, `["exponential", center = null, tau = 1]`
 * (`center` is the FIRST parameter, as in SciPy: `["exponential", null, tau]` for a centred window; any explicit
 * `center` needs `fftbins: true`) and `["taylor", nbar = 4, sll = 30, norm = true]`. A parameter-free name may also be
 * wrapped in a 1-tuple.
 */
export type WindowTuple =
  | readonly ["kaiser", beta: number]
  | readonly ["gaussian", std: number]
  | readonly ["general_gaussian", p: number, sig: number]
  | readonly ["general_hamming", alpha: number]
  | readonly ["general_cosine", coefficients: readonly number[]]
  | readonly ["tukey", alpha?: number]
  | readonly ["exponential", center?: number | null, tau?: number]
  | readonly ["taylor", nbar?: number, sll?: number, norm?: boolean]
  | readonly [WindowName];

/** What `welch`/`stft`/`istft`/`checkNOLA`/`checkCOLA` take as `window`: a name, a `(name, param)` tuple, or the window itself (its length then sets `nperseg`). */
export type WindowSpec = WindowName | WindowTuple | Float64Array;

type Params = ReadonlyArray<unknown>;

const isNum = (v: unknown): v is number => typeof v === "number" && !Number.isNaN(v);

/** Modified Bessel function of the first kind, order 0: the power series `sum (x²/4)^k / (k!)²` (all terms positive, so no cancellation). */
function besselI0(x: number): number {
  const q = (x * x) / 4;
  let term = 1;
  let sum = 1;
  for (let k = 1; k < 500; k++) {
    term *= q / (k * k);
    sum += term;
    if (term < sum * 1e-18) break;
  }
  return sum;
}

/** `np.linspace(-pi, pi, m)` */
function linspacePi(m: number): Float64Array {
  const out = new Float64Array(m);
  const step = (2 * Math.PI) / (m - 1);
  for (let i = 0; i < m; i++) out[i] = -Math.PI + i * step;
  out[m - 1] = Math.PI;
  return out;
}

/** `scipy.signal.windows.general_cosine`'s symmetric form of length `m` (`m >= 2`). */
function generalCosine(m: number, a: readonly number[]): Float64Array {
  const fac = linspacePi(m);
  const w = new Float64Array(m);
  for (let k = 0; k < a.length; k++) {
    for (let i = 0; i < m; i++) w[i] = (w[i] as number) + (a[k] as number) * Math.cos(k * (fac[i] as number));
  }
  return w;
}

interface Spec {
  /** Minimum and maximum number of parameters. */
  readonly min: number;
  readonly max: number;
  /** Symmetric window of length `m >= 2` (SciPy has already extended it by one for a periodic window). */
  readonly make: (m: number, p: Params, sym: boolean) => Float64Array;
}

const cosineSum = (a: readonly number[]): Spec => ({ min: 0, max: 0, make: (m) => generalCosine(m, a) });

function tukey(m: number, alpha: number): Float64Array {
  const w = new Float64Array(m);
  if (alpha <= 0) return w.fill(1);
  if (alpha >= 1) return generalCosine(m, [0.5, 0.5]);
  const width = Math.floor((alpha * (m - 1)) / 2);
  for (let n = 0; n < m; n++) {
    if (n <= width) w[n] = 0.5 * (1 + Math.cos(Math.PI * (-1 + (2 * n) / alpha / (m - 1))));
    else if (n < m - width - 1) w[n] = 1;
    else w[n] = 0.5 * (1 + Math.cos(Math.PI * (-2 / alpha + 1 + (2 * n) / alpha / (m - 1))));
  }
  return w;
}

function taylor(m: number, nbar: number, sll: number, norm: boolean): Float64Array {
  const B = 10 ** (sll / 20);
  const A = Math.acosh(B) / Math.PI;
  const s2 = nbar ** 2 / (A ** 2 + (nbar - 0.5) ** 2);
  const count = nbar - 1;
  const Fm = new Float64Array(count);
  for (let mi = 0; mi < count; mi++) {
    const mm = (mi + 1) ** 2;
    let numer = mi % 2 === 0 ? 1 : -1;
    for (let j = 0; j < count; j++) numer *= 1 - mm / s2 / (A ** 2 + (j + 1 - 0.5) ** 2);
    let denom = 2;
    for (let j = 0; j < mi; j++) denom *= 1 - mm / (j + 1) ** 2;
    for (let j = mi + 1; j < count; j++) denom *= 1 - mm / (j + 1) ** 2;
    Fm[mi] = numer / denom;
  }
  const W = (n: number): number => {
    let s = 0;
    for (let j = 0; j < count; j++) s += (Fm[j] as number) * Math.cos((2 * Math.PI * (j + 1) * (n - m / 2 + 0.5)) / m);
    return 1 + 2 * s;
  };
  const w = new Float64Array(m);
  for (let n = 0; n < m; n++) w[n] = W(n);
  if (norm) {
    const scale = 1 / W((m - 1) / 2);
    for (let n = 0; n < m; n++) w[n] = (w[n] as number) * scale;
  }
  return w;
}

const SPECS: Record<string, Spec> = {
  boxcar: { min: 0, max: 0, make: (m) => new Float64Array(m).fill(1) },
  rectangular: { min: 0, max: 0, make: (m) => new Float64Array(m).fill(1) },
  hann: cosineSum([0.5, 0.5]),
  hamming: cosineSum([0.54, 1 - 0.54]),
  blackman: cosineSum([0.42, 0.5, 0.08]),
  nuttall: cosineSum([0.3635819, 0.4891775, 0.1365995, 0.0106411]),
  blackmanharris: cosineSum([0.35875, 0.48829, 0.14128, 0.01168]),
  flattop: cosineSum([0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]),
  general_hamming: { min: 1, max: 1, make: (m, p) => generalCosine(m, [p[0] as number, 1 - (p[0] as number)]) },
  general_cosine: { min: 1, max: 1, make: (m, p) => generalCosine(m, p[0] as number[]) },
  triang: {
    min: 0,
    max: 0,
    make(m) {
      const w = new Float64Array(m);
      const half = Math.floor((m + 1) / 2);
      for (let k = 1; k <= half; k++) {
        const v = m % 2 === 0 ? (2 * k - 1) / m : (2 * k) / (m + 1);
        w[k - 1] = v;
        w[m - k] = v;
      }
      return w;
    },
  },
  bartlett: {
    min: 0,
    max: 0,
    make(m) {
      const w = new Float64Array(m);
      for (let n = 0; n < m; n++) w[n] = n <= (m - 1) / 2 ? (2 * n) / (m - 1) : 2 - (2 * n) / (m - 1);
      return w;
    },
  },
  parzen: {
    min: 0,
    max: 0,
    make(m) {
      const w = new Float64Array(m);
      for (let i = 0; i < m; i++) {
        const a = Math.abs(i - (m - 1) / 2) / (m / 2);
        w[i] = Math.abs(i - (m - 1) / 2) <= (m - 1) / 4 ? 1 - 6 * a ** 2 + 6 * a ** 3 : 2 * (1 - a) ** 3;
      }
      return w;
    },
  },
  bohman: {
    min: 0,
    max: 0,
    make(m) {
      const w = new Float64Array(m); // the end points stay 0
      const step = 2 / (m - 1);
      for (let i = 1; i < m - 1; i++) {
        const fac = Math.abs(-1 + i * step);
        w[i] = (1 - fac) * Math.cos(Math.PI * fac) + (1 / Math.PI) * Math.sin(Math.PI * fac);
      }
      return w;
    },
  },
  barthann: {
    min: 0,
    max: 0,
    make(m) {
      const w = new Float64Array(m);
      for (let n = 0; n < m; n++) {
        const fac = Math.abs(n / (m - 1) - 0.5);
        w[n] = 0.62 - 0.48 * fac + 0.38 * Math.cos(2 * Math.PI * fac);
      }
      return w;
    },
  },
  cosine: {
    min: 0,
    max: 0,
    make(m) {
      const w = new Float64Array(m);
      for (let n = 0; n < m; n++) w[n] = Math.sin((Math.PI / m) * (n + 0.5));
      return w;
    },
  },
  kaiser: {
    min: 1,
    max: 1,
    make(m, p) {
      const beta = p[0] as number;
      const alpha = (m - 1) / 2;
      const w = new Float64Array(m);
      const denom = besselI0(beta);
      for (let n = 0; n < m; n++) w[n] = besselI0(beta * Math.sqrt(Math.max(0, 1 - ((n - alpha) / alpha) ** 2))) / denom;
      return w;
    },
  },
  gaussian: {
    min: 1,
    max: 1,
    make(m, p) {
      const std = p[0] as number;
      const w = new Float64Array(m);
      for (let n = 0; n < m; n++) w[n] = Math.exp(-((n - (m - 1) / 2) ** 2) / (2 * std * std));
      return w;
    },
  },
  general_gaussian: {
    min: 2,
    max: 2,
    make(m, p) {
      const [pw, sig] = p as [number, number];
      const w = new Float64Array(m);
      for (let n = 0; n < m; n++) w[n] = Math.exp(-0.5 * Math.abs((n - (m - 1) / 2) / sig) ** (2 * pw));
      return w;
    },
  },
  tukey: { min: 0, max: 1, make: (m, p) => tukey(m, (p[0] as number | undefined) ?? 0.5) },
  exponential: {
    min: 0,
    max: 2,
    make(m, p, sym) {
      const given = p[0] as number | null | undefined;
      if (sym && given != null) throw new RangeError("getWindow: for a symmetric window (fftbins: false) the exponential window's center must be omitted");
      const center = given ?? (m - 1) / 2;
      const tau = (p[1] as number | undefined) ?? 1;
      const w = new Float64Array(m);
      for (let n = 0; n < m; n++) w[n] = Math.exp(-Math.abs(n - center) / tau);
      return w;
    },
  },
  taylor: {
    min: 0,
    max: 3,
    make(m, p) {
      const nbar = (p[0] as number | undefined) ?? 4;
      if (!Number.isInteger(nbar) || nbar < 1) throw new RangeError(`getWindow: taylor's nbar must be a positive integer, got ${nbar}`);
      return taylor(m, nbar, (p[1] as number | undefined) ?? 30, (p[2] as boolean | undefined) ?? true);
    },
  },
};

/** Parameter checks that SciPy gets for free from Python's argument binding. */
function checkParams(name: string, spec: Spec, params: Params): void {
  if (params.length < spec.min || params.length > spec.max) {
    const want = spec.min === spec.max ? `${spec.min}` : `${spec.min} to ${spec.max}`;
    throw new RangeError(`getWindow: the "${name}" window takes ${want} parameter(s), got ${params.length}`);
  }
  params.forEach((v, i) => {
    const ok =
      name === "general_cosine"
        ? Array.isArray(v) && v.length > 0 && v.every(isNum)
        : name === "exponential" && i === 0
          ? v == null || isNum(v)
          : name === "taylor" && i === 2
            ? typeof v === "boolean" || v === undefined
            : name === "tukey" || name === "exponential" || name === "taylor"
              ? v === undefined || isNum(v)
              : isNum(v);
    if (!ok) throw new RangeError(`getWindow: invalid parameter ${i + 1} (${String(v)}) for the "${name}" window`);
  });
}

/**
 * `scipy.signal.get_window(window, n, fftbins)`. With the default `fftbins: true` the window is PERIODIC (the
 * symmetric one of length `n + 1`, last sample dropped), the kind spectral analysis uses; `fftbins: false` gives the
 * symmetric window (`hammingWindow` is the symmetric Hamming `resamplePoly` needs). `window` is a name
 * ({@link WindowName}), a `(name, param, ...)` tuple ({@link WindowTuple}), a number (a Kaiser `beta`, as in SciPy;
 * only here, not in `welch` and friends) or an array, returned as given after checking its length is `n`. A name that needs parameters (`"kaiser"`,
 * `"gaussian"`, `"general_hamming"`, ...) but is given bare throws a RangeError (SciPy raises a ValueError). Length 1
 * gives `[1]`. Not supported: SciPy's short aliases (`"hanning"`, `"blk"`, ...), `chebwin`, `dpss`, `lanczos`,
 * `kaiser_bessel_derived`.
 */
export function getWindow(spec: WindowSpec | number, n: number, fftbins = true): Float64Array {
  if (!Number.isInteger(n) || n < 1) throw new RangeError(`getWindow: length must be a positive integer, got ${n}`);
  if (spec instanceof Float64Array) {
    if (spec.length !== n) throw new RangeError(`getWindow: window length (${spec.length}) must equal ${n}`);
    return spec;
  }
  let name: string;
  let params: Params;
  if (typeof spec === "number") {
    name = "kaiser";
    params = [spec];
  } else if (typeof spec === "string") {
    name = spec;
    params = [];
  } else if (Array.isArray(spec) && typeof (spec as readonly unknown[])[0] === "string") {
    const tuple = spec as readonly unknown[];
    name = tuple[0] as string;
    params = tuple.slice(1);
  } else {
    throw new RangeError(`getWindow: unsupported window ${String(spec)} (a name, a (name, param) tuple, a number or a Float64Array)`);
  }
  const def = Object.prototype.hasOwnProperty.call(SPECS, name) ? (SPECS[name] as Spec) : undefined;
  if (def === undefined) {
    throw new RangeError(
      `getWindow: unknown window ${JSON.stringify(name)} (hann, hamming, blackman, blackmanharris, nuttall, flattop, boxcar, rectangular, bartlett, triang, parzen, bohman, barthann, cosine, tukey, taylor, exponential, kaiser, gaussian, general_gaussian, general_hamming, general_cosine)`,
    );
  }
  if (params.length === 0 && def.min > 0) {
    throw new RangeError(`getWindow: the "${name}" window needs one or more parameters -- pass a tuple`);
  }
  checkParams(name, def, params);
  if (n === 1) return Float64Array.of(1);
  const sym = !fftbins;
  const w = def.make(sym ? n : n + 1, params, sym);
  return sym ? w : w.slice(0, n);
}
