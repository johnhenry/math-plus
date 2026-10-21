import { planDft } from "./dft.ts";
import { emitWarning, type WarningOptions } from "./warn.ts";

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

// ---- names ------------------------------------------------------------------------------------------------------

/**
 * `scipy.signal.windows._win_equiv_raw`, SciPy 1.16's own table: `[names, needsParameters]`. The first name is the
 * canonical one; every name in a row is accepted by `get_window` (and so by `getWindow` and every `window` option).
 * `needsParameters` rows throw when given bare, as a string; they need a `(name, ...)` tuple.
 */
export const WINDOW_NAMES: ReadonlyArray<readonly [names: readonly string[], needsParameters: boolean]> = [
  [["barthann", "brthan", "bth"], false],
  [["bartlett", "bart", "brt"], false],
  [["blackman", "black", "blk"], false],
  [["blackmanharris", "blackharr", "bkh"], false],
  [["bohman", "bman", "bmn"], false],
  [["boxcar", "box", "ones", "rect", "rectangular"], false],
  [["chebwin", "cheb"], true],
  [["cosine", "halfcosine"], false],
  [["dpss"], true],
  [["exponential", "poisson"], false],
  [["flattop", "flat", "flt"], false],
  [["gaussian", "gauss", "gss"], true],
  [["general cosine", "general_cosine"], true],
  [["general gaussian", "general_gaussian", "general gauss", "general_gauss", "ggs"], true],
  [["general hamming", "general_hamming"], true],
  [["hamming", "hamm", "ham"], false],
  [["hann", "han"], false],
  [["kaiser", "ksr"], true],
  [["kaiser bessel derived", "kbd"], true],
  [["lanczos", "sinc"], false],
  [["nuttall", "nutl", "nut"], false],
  [["parzen", "parz", "par"], false],
  [["taylor", "taylorwin"], false],
  [["triangle", "triang", "tri"], false],
  [["tukey", "tuk"], false],
];

/** Window names that need no parameters, so may be given bare (`"hann"`) or in a 1-tuple. */
export type WindowName =
  | "barthann" | "brthan" | "bth"
  | "bartlett" | "bart" | "brt"
  | "blackman" | "black" | "blk"
  | "blackmanharris" | "blackharr" | "bkh"
  | "bohman" | "bman" | "bmn"
  | "boxcar" | "box" | "ones" | "rect" | "rectangular"
  | "cosine" | "halfcosine"
  | "exponential" | "poisson"
  | "flattop" | "flat" | "flt"
  | "hamming" | "hamm" | "ham"
  | "hann" | "han"
  | "lanczos" | "sinc"
  | "nuttall" | "nutl" | "nut"
  | "parzen" | "parz" | "par"
  | "taylor" | "taylorwin"
  | "triangle" | "triang" | "tri"
  | "tukey" | "tuk";

/**
 * `(name, param, ...)` window tuples, `scipy.signal.get_window`'s tuple form (parameters are positional, as in SciPy):
 * `["kaiser", beta]`, `["gaussian", std]`, `["general_gaussian", p, sig]`, `["general_hamming", alpha]`,
 * `["general_cosine", coefficients]`, `["chebwin", attenuation]`, `["dpss", NW]`, `["kaiser bessel derived", beta]`
 * (symmetric only), `["tukey", alpha = 0.5]`, `["exponential", center = null, tau = 1]` (`center` is the FIRST
 * parameter, as in SciPy: `["exponential", null, tau]` for a centred window; any explicit `center` needs
 * `fftbins: true`) and `["taylor", nbar = 4, sll = 30, norm = true]`. Every alias in {@link WINDOW_NAMES} works in
 * its place. A parameter-free name may also be wrapped in a 1-tuple.
 */
export type WindowTuple =
  | readonly ["kaiser" | "ksr", beta: number]
  | readonly ["kaiser bessel derived" | "kbd", beta: number]
  | readonly ["gaussian" | "gauss" | "gss", std: number]
  | readonly ["general_gaussian" | "general gaussian" | "general_gauss" | "general gauss" | "ggs", p: number, sig: number]
  | readonly ["general_hamming" | "general hamming", alpha: number]
  | readonly ["general_cosine" | "general cosine", coefficients: readonly number[]]
  | readonly ["chebwin" | "cheb", attenuation: number]
  | readonly ["dpss", NW: number]
  | readonly ["tukey" | "tuk", alpha?: number]
  | readonly ["exponential" | "poisson", center?: number | null, tau?: number]
  | readonly ["taylor" | "taylorwin", nbar?: number, sll?: number, norm?: boolean]
  | readonly [WindowName];

/** What `welch`/`stft`/`istft`/`checkNOLA`/`checkCOLA` and friends take as `window`: a name, a `(name, param)` tuple, or the window itself (its length then sets `nperseg`). */
export type WindowSpec = WindowName | WindowTuple | Float64Array;

type Params = ReadonlyArray<unknown>;

const isNum = (v: unknown): v is number => typeof v === "number" && !Number.isNaN(v);

/** Modified Bessel function of the first kind, order 0: the power series `sum (x²/4)^k / (k!)²` (all terms positive, so no cancellation). */
function besselI0(x: number): number {
  const q = (x * x) / 4;
  let term = 1;
  let sum = 1;
  for (let k = 1; k < 1000; k++) {
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

/** `np.sinc` */
function sinc(x: number): number {
  const y = Math.PI * (x === 0 ? 1e-20 : x);
  return Math.sin(y) / y;
}

/** What each positional parameter must be (SciPy gets these from Python's argument binding and arithmetic). */
type Kind = "number" | "optNumber" | "optCenter" | "optBool" | "numbers";

interface Spec {
  /** The positional parameters after the length: `required` of them are mandatory. */
  readonly params: readonly Kind[];
  readonly required: number;
  /** Runs first, before SciPy's own length guard (a warning, or a check SciPy makes before it). `n` is the requested length. */
  readonly pre?: (n: number, p: Params, sym: boolean, options: WarningOptions | undefined) => void;
  /** Symmetric window of length `m >= 2` (SciPy has already extended it by one for a periodic window). */
  readonly make: (m: number, p: Params, sym: boolean) => Float64Array;
  /** Replaces the whole length handling (no extension, no length guard): `kaiser_bessel_derived`. */
  readonly direct?: (n: number, p: Params, sym: boolean) => Float64Array;
}

const NONE: readonly Kind[] = [];
const cosineSum = (a: readonly number[]): Spec => ({ params: NONE, required: 0, make: (m) => generalCosine(m, a) });

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

function kaiser(m: number, beta: number): Float64Array {
  if (m === 1) return Float64Array.of(1);
  const alpha = (m - 1) / 2;
  const w = new Float64Array(m);
  const denom = besselI0(beta);
  for (let n = 0; n < m; n++) w[n] = besselI0(beta * Math.sqrt(Math.max(0, 1 - ((n - alpha) / alpha) ** 2))) / denom;
  return w;
}

/** `scipy.signal.windows.chebwin`'s window of length `m >= 2` (Dolph-Chebyshev, by the inverse DFT of the Chebyshev polynomial samples). */
function chebwin(m: number, at: number): Float64Array {
  const order = m - 1;
  const beta = Math.cosh((1 / order) * Math.acosh(10 ** (Math.abs(at) / 20)));
  const pRe = new Float64Array(m);
  const pIm = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const x = beta * Math.cos((Math.PI * k) / m);
    let v: number;
    if (x > 1) v = Math.cosh(order * Math.acosh(x));
    else if (x < -1) v = (2 * (m % 2) - 1) * Math.cosh(order * Math.acosh(-x));
    else v = Math.cos(order * Math.acos(x));
    if (m % 2) {
      pRe[k] = v;
    } else {
      const ang = (Math.PI / m) * k; // p * exp(1j * pi / M * k)
      pRe[k] = v * Math.cos(ang);
      pIm[k] = v * Math.sin(ang);
    }
  }
  const spectrum = planDft(m).transform(pRe, pIm, false).re;
  const w = new Float64Array(m);
  if (m % 2) {
    const n = (m + 1) / 2;
    for (let i = 0; i < n; i++) {
      w[n - 1 + i] = spectrum[i] as number; // w[:n]
      w[n - 1 - i] = spectrum[i] as number; // flip(w[1:n]) in front
    }
  } else {
    const n = m / 2 + 1;
    for (let i = 1; i < n; i++) {
      w[n - 1 - i] = spectrum[i] as number; // flip(w[1:n])
      w[n - 2 + i] = spectrum[i] as number; // w[1:n]
    }
  }
  let max = -Infinity;
  for (const v of w) if (v > max) max = v;
  for (let i = 0; i < m; i++) w[i] = (w[i] as number) / max;
  return w;
}

/**
 * The eigenvector of the largest eigenvalue of the symmetric tridiagonal matrix with diagonal `d` and off-diagonal `e`
 * (what `scipy.linalg.eigh_tridiagonal(..., select="i", select_range=(M - 1, M - 1))` finds, LAPACK's `stebz` + `stein`
 * the same way): the eigenvalue by Sturm-sequence bisection, the vector by inverse iteration with a pivoting
 * tridiagonal solve. Unit 2-norm, sign arbitrary.
 */
function topEigenvector(d: Float64Array, e: Float64Array): Float64Array {
  const n = d.length;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const r = Math.abs(i > 0 ? (e[i - 1] as number) : 0) + Math.abs(i < n - 1 ? (e[i] as number) : 0);
    lo = Math.min(lo, (d[i] as number) - r);
    hi = Math.max(hi, (d[i] as number) + r);
  }
  const scale = Math.max(Math.abs(lo), Math.abs(hi), Number.MIN_VALUE);
  const pivmin = Number.EPSILON * Number.EPSILON * scale;
  const negatives = (x: number): number => {
    let count = 0;
    let q = 1;
    for (let i = 0; i < n; i++) {
      const sq = i > 0 ? (e[i - 1] as number) ** 2 : 0;
      q = (d[i] as number) - x - (i > 0 ? sq / q : 0);
      if (Math.abs(q) < pivmin) q = -pivmin;
      if (q < 0) count++;
    }
    return count;
  };
  for (let it = 0; it < 2000 && hi - lo > 2 * Number.EPSILON * Math.max(Math.abs(lo), Math.abs(hi)); it++) {
    const mid = lo + (hi - lo) / 2;
    if (mid <= lo || mid >= hi) break;
    if (negatives(mid) >= n) hi = mid;
    else lo = mid;
  }
  const mu = (lo + hi) / 2;

  // Inverse iteration: solve (T - mu I) y = b by Gaussian elimination with partial pivoting (LAPACK dgtsv).
  const tiny = Number.EPSILON * scale;
  let b = new Float64Array(n);
  let seed = 12345;
  for (let i = 0; i < n; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    b[i] = seed / 4294967296 + 0.5;
  }
  for (let iter = 0; iter < 6; iter++) {
    const dd = new Float64Array(n);
    const du = new Float64Array(n);
    const dl = new Float64Array(n); // sub-diagonal, then the second super-diagonal once pivoted
    for (let i = 0; i < n; i++) dd[i] = (d[i] as number) - mu;
    for (let i = 0; i < n - 1; i++) {
      du[i] = e[i] as number;
      dl[i] = e[i] as number;
    }
    const y = Float64Array.from(b);
    for (let i = 0; i < n - 1; i++) {
      if (Math.abs(dd[i] as number) >= Math.abs(dl[i] as number)) {
        if (dd[i] === 0) dd[i] = tiny;
        const fact = (dl[i] as number) / (dd[i] as number);
        dd[i + 1] = (dd[i + 1] as number) - fact * (du[i] as number);
        y[i + 1] = (y[i + 1] as number) - fact * (y[i] as number);
        dl[i] = 0;
      } else {
        const fact = (dd[i] as number) / (dl[i] as number);
        dd[i] = dl[i] as number;
        const temp = dd[i + 1] as number;
        dd[i + 1] = (du[i] as number) - fact * temp;
        if (i < n - 2) {
          dl[i] = du[i + 1] as number;
          du[i + 1] = -fact * (dl[i] as number);
        } else {
          dl[i] = 0;
        }
        du[i] = temp;
        const tb = y[i] as number;
        y[i] = y[i + 1] as number;
        y[i + 1] = tb - fact * (y[i + 1] as number);
      }
    }
    if (dd[n - 1] === 0) dd[n - 1] = tiny;
    y[n - 1] = (y[n - 1] as number) / (dd[n - 1] as number);
    if (n > 1) {
      if (dd[n - 2] === 0) dd[n - 2] = tiny;
      y[n - 2] = ((y[n - 2] as number) - (du[n - 2] as number) * (y[n - 1] as number)) / (dd[n - 2] as number);
    }
    for (let i = n - 3; i >= 0; i--) {
      if (dd[i] === 0) dd[i] = tiny;
      y[i] = ((y[i] as number) - (du[i] as number) * (y[i + 1] as number) - (dl[i] as number) * (y[i + 2] as number)) / (dd[i] as number);
    }
    let norm = 0;
    for (const v of y) norm = Math.hypot(norm, v);
    for (let i = 0; i < n; i++) y[i] = (y[i] as number) / norm;
    b = y;
  }
  return b;
}

/** `scipy.signal.windows.dpss(M, NW)` (one window, `Kmax=None`, `norm="approximate"`) for the extended length `m >= 2`; `sym` says whether `m` is the requested length. */
function dpss(m: number, NW: number, sym: boolean): Float64Array {
  const requested = sym ? m : m - 1;
  if (NW >= requested / 2) throw new RangeError("getWindow: dpss: NW must be less than M/2.");
  if (NW <= 0) throw new RangeError("getWindow: dpss: NW must be positive");
  const W = NW / m;
  const c = Math.cos(2 * Math.PI * W);
  const d = new Float64Array(m);
  for (let t = 0; t < m; t++) d[t] = ((m - 1 - 2 * t) / 2) ** 2 * c;
  const e = new Float64Array(m - 1);
  for (let t = 1; t < m; t++) e[t - 1] = (t * (m - t)) / 2;
  const w = topEigenvector(d, e);
  let sum = 0;
  for (const v of w) sum += v;
  if (sum < 0) for (let i = 0; i < m; i++) w[i] = -(w[i] as number); // symmetric tapers have a positive average
  let max = -Infinity;
  for (const v of w) if (v > max) max = v;
  const correction = m % 2 === 0 ? m ** 2 / (m ** 2 + NW) : 1;
  for (let i = 0; i < m; i++) w[i] = ((w[i] as number) / max) * correction;
  return w;
}

const CHEBWIN_WARNING =
  "This window is not suitable for spectral analysis for attenuation values lower than about 45dB because the equivalent noise bandwidth of a Chebyshev window does not grow monotonically with increasing sidelobe attenuation when the attenuation is smaller than about 45 dB.";

const SPECS: Record<string, Spec> = {
  boxcar: { params: NONE, required: 0, make: (m) => new Float64Array(m).fill(1) },
  hann: cosineSum([0.5, 0.5]),
  hamming: cosineSum([0.54, 1 - 0.54]),
  blackman: cosineSum([0.42, 0.5, 0.08]),
  nuttall: cosineSum([0.3635819, 0.4891775, 0.1365995, 0.0106411]),
  blackmanharris: cosineSum([0.35875, 0.48829, 0.14128, 0.01168]),
  flattop: cosineSum([0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]),
  general_hamming: { params: ["number"], required: 1, make: (m, p) => generalCosine(m, [p[0] as number, 1 - (p[0] as number)]) },
  general_cosine: { params: ["numbers"], required: 1, make: (m, p) => generalCosine(m, p[0] as number[]) },
  triang: {
    params: NONE,
    required: 0,
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
    params: NONE,
    required: 0,
    make(m) {
      const w = new Float64Array(m);
      for (let n = 0; n < m; n++) w[n] = n <= (m - 1) / 2 ? (2 * n) / (m - 1) : 2 - (2 * n) / (m - 1);
      return w;
    },
  },
  parzen: {
    params: NONE,
    required: 0,
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
    params: NONE,
    required: 0,
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
    params: NONE,
    required: 0,
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
    params: NONE,
    required: 0,
    make(m) {
      const w = new Float64Array(m);
      for (let n = 0; n < m; n++) w[n] = Math.sin((Math.PI / m) * (n + 0.5));
      return w;
    },
  },
  lanczos: {
    params: NONE,
    required: 0,
    make(m) {
      // the right half from np.sinc(2 k / (m - 1) - 1), mirrored, so the window is exactly symmetric
      const w = new Float64Array(m);
      const start = m % 2 === 0 ? m / 2 : (m + 1) / 2;
      for (let k = start; k < m; k++) {
        const v = sinc((2 * k) / (m - 1) - 1);
        w[k] = v;
        w[m - 1 - k] = v;
      }
      if (m % 2) w[(m - 1) / 2] = 1;
      return w;
    },
  },
  kaiser: { params: ["number"], required: 1, make: (m, p) => kaiser(m, p[0] as number) },
  kaiser_bessel_derived: {
    params: ["number"],
    required: 1,
    make: () => {
      throw new Error("unreachable");
    },
    direct(n, p, sym) {
      if (!sym) throw new RangeError("getWindow: Kaiser-Bessel Derived windows are only defined for symmetric shapes");
      if (n < 1) return new Float64Array(0);
      if (n % 2) throw new RangeError("getWindow: Kaiser-Bessel Derived windows are only defined for even number of points");
      const k = kaiser(n / 2 + 1, p[0] as number);
      const half = new Float64Array(n / 2);
      let total = 0;
      for (const v of k) total += v; // csum[-1], accumulated left to right like np.cumulative_sum
      let run = 0;
      for (let i = 0; i < n / 2; i++) {
        run += k[i] as number;
        half[i] = Math.sqrt(run / total);
      }
      const w = new Float64Array(n);
      for (let i = 0; i < n / 2; i++) {
        w[i] = half[i] as number;
        w[n - 1 - i] = half[i] as number;
      }
      return w;
    },
  },
  gaussian: {
    params: ["number"],
    required: 1,
    make(m, p) {
      const std = p[0] as number;
      const w = new Float64Array(m);
      for (let n = 0; n < m; n++) w[n] = Math.exp(-((n - (m - 1) / 2) ** 2) / (2 * std * std));
      return w;
    },
  },
  general_gaussian: {
    params: ["number", "number"],
    required: 2,
    make(m, p) {
      const [pw, sig] = p as [number, number];
      const w = new Float64Array(m);
      for (let n = 0; n < m; n++) w[n] = Math.exp(-0.5 * Math.abs((n - (m - 1) / 2) / sig) ** (2 * pw));
      return w;
    },
  },
  chebwin: {
    params: ["number"],
    required: 1,
    pre(_n, p, _sym, options) {
      if (Math.abs(p[0] as number) < 45) emitWarning(options, CHEBWIN_WARNING);
    },
    make: (m, p) => chebwin(m, p[0] as number),
  },
  dpss: { params: ["number"], required: 1, make: (m, p, sym) => dpss(m, p[0] as number, sym) },
  tukey: { params: ["optNumber"], required: 0, make: (m, p) => tukey(m, (p[0] as number | undefined) ?? 0.5) },
  exponential: {
    params: ["optCenter", "optNumber"],
    required: 0,
    pre(_n, p, sym) {
      if (sym && p[0] != null) throw new RangeError("getWindow: If sym==True, center must be None.");
    },
    make(m, p) {
      const given = p[0] as number | null | undefined;
      const center = given ?? (m - 1) / 2;
      const tau = (p[1] as number | undefined) ?? 1;
      const w = new Float64Array(m);
      for (let n = 0; n < m; n++) w[n] = Math.exp(-Math.abs(n - center) / tau);
      return w;
    },
  },
  taylor: {
    params: ["optNumber", "optNumber", "optBool"],
    required: 0,
    make(m, p) {
      const nbar = (p[0] as number | undefined) ?? 4;
      if (!Number.isInteger(nbar)) throw new TypeError(`getWindow: taylor's nbar must be an integer, got ${nbar}`);
      if (nbar < 1) throw new RangeError(`getWindow: taylor's nbar must be a positive integer, got ${nbar}`);
      return taylor(m, nbar, (p[1] as number | undefined) ?? 30, (p[2] as boolean | undefined) ?? true);
    },
  },
};

SPECS.triangle = SPECS.triang as Spec; // the first name of SciPy's row is "triangle"

/** Every accepted spelling -> `[the SPECS key, needs parameters]`. */
const BY_NAME = new Map<string, readonly [string, boolean]>();
for (const [names, needs] of WINDOW_NAMES) {
  const key = names[0] === "kaiser bessel derived" ? "kaiser_bessel_derived" : (names[0] as string).replace(" ", "_");
  for (const name of names) BY_NAME.set(name, [key, needs]);
}

/** Argument binding as Python does it: the wrong number of parameters is a TypeError, as is a parameter of the wrong type. */
function bindParams(name: string, spec: Spec, params: Params): void {
  if (params.length < spec.required) throw new TypeError(`getWindow: the "${name}" window is missing ${spec.required - params.length} required parameter(s)`);
  if (params.length > spec.params.length) {
    throw new TypeError(`getWindow: the "${name}" window takes ${spec.params.length === spec.required ? spec.required : `${spec.required} to ${spec.params.length}`} parameter(s), got ${params.length}`);
  }
  params.forEach((v, i) => {
    const kind = spec.params[i] as Kind;
    const ok =
      kind === "numbers"
        ? Array.isArray(v) && v.length > 0 && v.every(isNum)
        : kind === "optCenter"
          ? v == null || isNum(v)
          : kind === "optBool"
            ? v === undefined || typeof v === "boolean"
            : kind === "optNumber"
              ? v === undefined || isNum(v)
              : isNum(v);
    if (!ok) throw new TypeError(`getWindow: invalid parameter ${i + 1} (${String(v)}) for the "${name}" window`);
  });
}

const NUMERIC_STRING = /^\s*[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?|inf(?:inity)?|nan)\s*$/i;

/**
 * `scipy.signal.get_window(window, n, fftbins)`. With the default `fftbins: true` the window is PERIODIC (the
 * symmetric one of length `n + 1`, last sample dropped), the kind spectral analysis uses; `fftbins: false` gives the
 * symmetric window (`hammingWindow` is the symmetric Hamming `resamplePoly` needs). `window` is a name
 * ({@link WindowName}, with all of SciPy's aliases), a `(name, param, ...)` tuple ({@link WindowTuple}), a number (a
 * Kaiser `beta`, as in SciPy; only here, not in `welch` and friends), or an array, returned as given after checking
 * its length is `n` (an extension: `get_window` itself rejects arrays; every spectral function takes one).
 *
 * Errors follow SciPy's classes: an unknown name, a bare name that needs parameters, a bad length and a bad
 * parameter value throw a `RangeError` (a `ValueError` there); the wrong number or type of parameters throws a
 * `TypeError`. Length 0 gives an empty window and length 1 gives `[1]`. `chebwin` with an attenuation under 45 dB
 * warns (through `options.onWarning`, else `console.warn`), as SciPy does. `["dpss", NW]` is one window (`Kmax=None`):
 * `get_window` has no way to ask for more (it raises a `TypeError` for a `Kmax`, as this does for a second
 * parameter). `kaiser bessel derived` (`"kbd"`) is symmetric only (`fftbins: false`) and needs an even length.
 */
export function getWindow(spec: WindowSpec | number, n: number, fftbins = true, options?: WarningOptions): Float64Array {
  if (typeof n !== "number" || !Number.isInteger(n) || n < 0) throw new RangeError("getWindow: Window length M must be a non-negative integer");
  if (spec instanceof Float64Array) {
    if (spec.length !== n) throw new RangeError(`getWindow: window length (${spec.length}) must equal ${n}`);
    return spec;
  }
  let winstr: string;
  let params: Params;
  if (typeof spec === "number") {
    winstr = "kaiser";
    params = [spec];
  } else if (typeof spec === "string" && NUMERIC_STRING.test(spec)) {
    winstr = "kaiser"; // Python's float("4.0")
    params = [Number(spec.trim().replace(/^([+-]?)inf(inity)?$/i, "$1Infinity"))];
  } else if (typeof spec === "string") {
    const bare = BY_NAME.get(spec);
    if (bare?.[1]) throw new RangeError(`getWindow: The '${spec}' window needs one or more parameters -- pass a tuple.`);
    winstr = spec;
    params = [];
  } else if (Array.isArray(spec) && (spec as readonly unknown[]).length > 0) {
    const tuple = spec as readonly unknown[];
    if (typeof tuple[0] !== "string") throw new RangeError(`getWindow: Unknown window type.`);
    winstr = tuple[0];
    params = tuple.slice(1);
  } else {
    throw new RangeError(`getWindow: ${String(spec)} as window type is not supported.`);
  }
  const found = BY_NAME.get(winstr);
  if (found === undefined) throw new RangeError("getWindow: Unknown window type.");
  const def = SPECS[found[0]] as Spec;
  bindParams(winstr, def, params);
  const sym = !fftbins;
  def.pre?.(n, params, sym, options);
  if (def.direct) return def.direct(n, params, sym);
  if (n <= 1) return new Float64Array(n).fill(1);
  const w = def.make(sym ? n : n + 1, params, sym);
  return sym ? w : w.slice(0, n);
}
