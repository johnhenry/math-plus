/**
 * scipy.signal oracle helper (issue #44). Same skip-don't-fail convention
 * as tensor-core's numpy_oracle.py: if no Python with scipy is found, tests
 * using this SKIP rather than fail. On trycooy:
 * `nix-shell -p "python3.withPackages(ps: [ps.scipy ps.numpy])"` provides
 * one (pip wheels don't work on NixOS) -- see docs/TESTING.md.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ComplexTensor } from "@johnhenry/math-plus-fft";
import { Tensor } from "@johnhenry/math-plus-tensor-core";

const ORACLE_SCRIPT = new URL("../scripts/scipy_oracle.py", import.meta.url).pathname;

function findOraclePython(): string | undefined {
  const candidates = [process.env.MATH_PLUS_SCIPY_ORACLE_PYTHON, process.env.MATH_PLUS_ORACLE_PYTHON, "python3"].filter(
    (c): c is string => Boolean(c),
  );
  for (const candidate of candidates) {
    try {
      execFileSync(candidate, ["-c", "import scipy.signal, numpy"], { stdio: "ignore" });
      return candidate;
    } catch {
      // try next
    }
  }
  return undefined;
}

export const SCIPY_PYTHON = findOraclePython();
export const SCIPY_SKIP_REASON = SCIPY_PYTHON ? undefined : "no python with scipy found (set MATH_PLUS_SCIPY_ORACLE_PYTHON)";

function scipyVersion(): [number, number] | undefined {
  if (!SCIPY_PYTHON) return undefined;
  try {
    const out = execFileSync(SCIPY_PYTHON, ["-c", "import scipy; print(scipy.__version__)"], { stdio: ["ignore", "pipe", "ignore"] }).toString("utf8").trim();
    const [major, minor] = out.split(".").map(Number);
    return [major as number, minor as number];
  } catch {
    return undefined;
  }
}
const version = scipyVersion();
/**
 * SciPy 1.17 changed `get_window` (and so every spectral function that builds a window from a name): it refuses a
 * length of 0 and numeric strings, also accepts "kaiser_bessel_derived" and "<name>_periodic" / "<name>_symmetric",
 * and raises ValueError for every bad parameter list (1.16: TypeError for a wrong count). This package follows
 * SciPy 1.16 (see the README), so the few expectations that depend on those changes are only compared with an
 * oracle older than 1.17; with a newer one they are skipped or their error class is relaxed. Everything else is
 * compared with whichever SciPy is installed.
 */
export const SCIPY_17_PLUS = version !== undefined && (version[0] > 1 || version[1] >= 17);

export function runScipyOracle<T>(job: Record<string, unknown>): T {
  if (!SCIPY_PYTHON) throw new Error("runScipyOracle called without a resolved python (check SCIPY_SKIP_REASON first)");
  const dir = mkdtempSync(join(tmpdir(), "math-plus-signal-oracle-"));
  try {
    const jobPath = join(dir, "job.json");
    writeFileSync(jobPath, JSON.stringify(job));
    const stdout = execFileSync(SCIPY_PYTHON, [ORACLE_SCRIPT, jobPath], { stdio: ["ignore", "pipe", "pipe"], maxBuffer: 256 * 1024 * 1024 });
    return JSON.parse(stdout.toString("utf8")) as T;
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// ---- shared by the spectral differential suites ---------------------------

/** Default relative tolerance for comparisons against SciPy (relative to the reference's peak). */
export const REL = 1e-9;

/** A reproducible test signal: a trend and an offset (so detrend choices differ), two tones and noise. */
export function signalOf(n: number, seed: number): number[] {
  let s = seed >>> 0;
  const rng = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
  return Array.from({ length: n }, (_, i) => 3 + 0.01 * i + Math.sin(i / 7) + 0.5 * Math.sin(i / 2.3) + rng());
}
export const tensorOf = (x: number[]) => Tensor.from(x, { dtype: "f64" });
export const flat = (t: Tensor) => Array.from(t.contiguous().data as Float64Array);
export const zParts = (z: ComplexTensor) => ({ re: flat(z.real), im: flat(z.imag) });
export const maxAbs = (a: ArrayLike<number>) => Array.from(a as ArrayLike<number> as number[]).reduce((m, v) => Math.max(m, Math.abs(v)), 0);

/** Max |a-b| relative to `scale` (the reference's own peak, floored). */
export function closeScaled(a: ArrayLike<number>, b: ArrayLike<number>, scale: number, what: string, tol = REL): void {
  assert.equal(a.length, b.length, `${what}: lengths ${a.length} vs ${b.length}`);
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs((a[i] as number) - (b[i] as number)));
  assert.ok(worst <= tol * Math.max(scale, 1e-300), `${what}: differs by ${worst} (scale ${scale})`);
}

export type Kw = Record<string, unknown>;

/** JS option names -> the scipy keyword arguments (omitted JS options stay omitted, so SciPy's own defaults apply). */
export function pyKw(o: Kw, spectrogram: boolean): Kw {
  const rename: Record<string, string> = { onesided: "return_onesided", inputOnesided: "input_onesided", freqAxis: "freq_axis", timeAxis: "time_axis" };
  const out: Kw = {};
  for (const [k, v] of Object.entries(o)) {
    if (v === undefined || k === "onWarning") continue;
    let val: unknown = v;
    if (k === "window") val = pyWindow(v);
    if (k === "scaling" && spectrogram && v === "density") val = "psd";
    out[rename[k] ?? k] = val;
  }
  return out;
}

/** A JS window spec -> the oracle's JSON form (array, {tuple}, string or number). */
export function pyWindow(v: unknown): unknown {
  if (v instanceof Float64Array) return Array.from(v);
  if (Array.isArray(v)) return { tuple: v };
  return v === "rectangular" ? "boxcar" : v;
}

/** Nested arrays -> a flat list. */
export const flatten = (a: unknown): number[] => (a as number[]).flat(Infinity) as number[];

/** What the oracle returns for a welch job. */
export interface WelchWant {
  freqs: number[];
  shape: number[];
  Pxx: unknown;
  warnings: string[];
}
/** What the oracle returns for an stft job, and the `real` / `imag` of an istft job (nested lists of `shape`). */
export interface StftWant {
  freqs: number[];
  times: number[];
  shape: number[];
  real: unknown;
  imag: unknown;
  warnings: string[];
}
export interface IstftWant {
  times: number[];
  complex: boolean;
  shape: number[];
  real: unknown;
  imag: unknown;
  warnings: string[];
}

export function checkWelch(got: { freqs: number[]; Pxx: Tensor }, want: WelchWant, what: string): void {
  closeScaled(got.freqs, want.freqs, maxAbs(want.freqs), `freqs ${what}`);
  assert.deepEqual(got.Pxx.shape, want.shape, `Pxx shape ${what}`);
  const w = flatten(want.Pxx);
  closeScaled(flat(got.Pxx), w, maxAbs(w), `Pxx ${what}`);
}

export function checkStft(got: { freqs: number[]; times: number[]; Zxx: ComplexTensor }, want: StftWant, what: string): void {
  assert.deepEqual(got.Zxx.shape, want.shape, `Zxx shape ${what}`);
  closeScaled(got.freqs, want.freqs, maxAbs(want.freqs), `freqs ${what}`);
  closeScaled(got.times, want.times, maxAbs(want.times), `times ${what}`);
  const { re, im } = zParts(got.Zxx);
  const wr = flatten(want.real);
  const wi = flatten(want.imag);
  const scale = Math.max(maxAbs(wr), maxAbs(wi));
  closeScaled(re, wr, scale, `real ${what}`);
  closeScaled(im, wi, scale, `imag ${what}`);
}
