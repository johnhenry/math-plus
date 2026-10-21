import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { getWindow, WINDOW_NAMES } from "../src/index.ts";
import { closeScaled, maxAbs, runScipyOracle, SCIPY_17_PLUS, SCIPY_SKIP_REASON } from "./helpers.ts";

// Every window and alias `scipy.signal.get_window` accepts, against SciPy 1.16 itself at 1e-12: the name table
// (read from SciPy's own `_win_equiv_raw`, not typed from memory), every alias with every parameter set, periodic and
// symmetric, lengths 0 to 255, plus SciPy's errors (and their classes) and warnings.

const TOL = 1e-12;

interface Case {
  readonly window: unknown;
  readonly n: number;
  readonly fftbins: boolean;
}
interface Want {
  w?: number[];
  warnings: string[];
  error?: string;
  error_type?: string;
}

const pyWin = (w: unknown): unknown => (Array.isArray(w) ? { tuple: w } : w);

function runWindows(cases: readonly Case[]): Want[] {
  return runScipyOracle<Want[]>({
    op: "batch",
    jobs: cases.map((c) => ({ op: "get_window", window: pyWin(c.window), n: c.n, fftbins: c.fftbins, expect_error: true })),
  });
}

/** The spellings SciPy 1.17+ accepts that 1.16 (which this package follows) does not. */
const NEWER_ONLY = /^(kaiser_bessel_derived|.*_(periodic|symmetric))$/;
const nameOf = (w: unknown): unknown => (Array.isArray(w) ? w[0] : w);

/** Our `getWindow` must agree with SciPy on a case: the same values at 1e-12, or an error of the same class, and the same warnings. */
function agree(c: Case, want: Want, tol = TOL): void {
  const what = `${JSON.stringify(c.window)} n=${c.n} fftbins=${c.fftbins}`;
  if (SCIPY_17_PLUS) {
    // SciPy >= 1.17 refuses n = 0 and numeric strings and accepts newer spellings; this package is 1.16's (see helpers.ts)
    if (c.n === 0 || (typeof c.window === "string" && /^[\s+\-.\d]/.test(c.window)) || NEWER_ONLY.test(String(nameOf(c.window)))) return;
  }
  const warned: string[] = [];
  let got: Float64Array | undefined;
  let err: unknown;
  try {
    got = getWindow(c.window as never, c.n, c.fftbins, { onWarning: (m) => warned.push(m) });
  } catch (e) {
    err = e;
  }
  if (want.error !== undefined) {
    assert.ok(err !== undefined, `${what}: SciPy raised ${want.error} but getWindow returned`);
    const typeError = /TypeError$/.test(want.error_type ?? "");
    // (SciPy >= 1.17 raises ValueError where 1.16 raises TypeError for a wrong parameter count)
    const ok = SCIPY_17_PLUS ? err instanceof TypeError || err instanceof RangeError : typeError ? err instanceof TypeError : err instanceof RangeError;
    assert.ok(ok, `${what}: SciPy raised ${want.error}; got ${String(err)}`);
  } else {
    assert.ok(err === undefined, `${what}: SciPy returned a window; getWindow threw ${String(err)}`);
    const w = want.w as number[];
    assert.equal((got as Float64Array).length, w.length, `${what}: length`);
    closeScaled(got as Float64Array, w, Math.max(maxAbs(w), 1), what, tol);
  }
  assert.deepEqual(warned, want.warnings, `${what}: warnings`);
}

function checkAll(cases: readonly Case[], tol = TOL): void {
  const wants = runWindows(cases);
  cases.forEach((c, i) => agree(c, wants[i] as Want, tol));
}

const LENGTHS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 32, 33, 64, 65, 100, 127, 128, 255];
const both = (window: unknown, lengths: readonly number[]): Case[] =>
  lengths.flatMap((n) => [true, false].map((fftbins) => ({ window, n, fftbins })));

test("window names: the table of accepted names is SciPy's own _win_equiv_raw", { skip: SCIPY_SKIP_REASON }, () => {
  const { table: raw } = runScipyOracle<{ table: [string[], boolean][] }>({ op: "window_table" });
  // (SciPy >= 1.17 adds the spelling "kaiser_bessel_derived", which 1.16 does not accept)
  const table: [string[], boolean][] = raw.map(([names, needs]) => [names.filter((n) => !NEWER_ONLY.test(n)), needs]);
  assert.deepEqual(
    WINDOW_NAMES.map(([names, needs]) => [names, needs]),
    table,
  );
  const all = table.flatMap(([names]) => names);
  assert.ok(all.length >= 60 && new Set(all).size === all.length);
});

test("every parameter-free name and alias matches get_window, periodic and symmetric, lengths 0 to 255", { skip: SCIPY_SKIP_REASON }, () => {
  const names = WINDOW_NAMES.filter(([, needs]) => !needs).flatMap(([n]) => n);
  assert.ok(names.includes("hann") && names.includes("lanczos") && names.includes("sinc") && names.includes("tri"));
  checkAll(names.flatMap((name) => both(name, LENGTHS)));
  // a 1-tuple is the same as the bare name
  checkAll(names.flatMap((name) => both([name], [1, 2, 16, 33])));
});

test("every name that needs parameters throws when given bare (RangeError, as SciPy's ValueError)", { skip: SCIPY_SKIP_REASON }, () => {
  const names = WINDOW_NAMES.filter(([, needs]) => needs).flatMap(([n]) => n);
  assert.ok(names.includes("dpss") && names.includes("cheb") && names.includes("kbd") && names.includes("general cosine"));
  checkAll(names.flatMap((name) => [0, 1, 8, 9].map((n) => ({ window: name, n, fftbins: true }))));
});

/** Parameter sets for every window that takes them, by canonical name (used with each of its aliases). */
const PARAM_SETS: Record<string, unknown[][]> = {
  kaiser: [[0], [2.5], [8.6], [14]],
  gaussian: [[1.5], [7], [20]],
  "general cosine": [[[0.5, 0.5]], [[0.35875, 0.48829, 0.14128, 0.01168]], [[1]]],
  "general gaussian": [[1.5, 7], [0.5, 20], [3, 10]],
  "general hamming": [[0.54], [0.7], [0.5]],
  chebwin: [[30], [45], [60], [80], [100], [-70]],
  dpss: [[0.5], [1], [2.5], [4], [8]],
  "kaiser bessel derived": [[2], [4], [8.5]],
  tukey: [[], [0], [0.25], [0.5], [1]],
  exponential: [[], [null], [null, 3], [null, 1], [3, 2], [0, 5]],
  taylor: [[], [4, 30, true], [6, 40, false], [2], [8, 20]],
};

test("every alias of every parametrised window matches get_window (periodic, symmetric, lengths 0 to 255)", { skip: SCIPY_SKIP_REASON }, () => {
  const cases: Case[] = [];
  for (const [names] of WINDOW_NAMES) {
    const sets = PARAM_SETS[names[0] as string];
    if (sets === undefined) continue;
    for (const name of names) for (const params of sets) cases.push(...both([name, ...params], LENGTHS));
  }
  assert.ok(cases.length > 3000);
  checkAll(cases);
});

test("chebwin, dpss, lanczos and kaiser bessel derived in detail (the windows that were missing)", { skip: SCIPY_SKIP_REASON }, () => {
  const cases: Case[] = [];
  for (const at of [0, 1, 20, 44.9, 45, 55.5, 70, 90, 120, -50]) cases.push(...both(["chebwin", at], [0, 1, 2, 3, 4, 5, 10, 11, 24, 25, 100, 101, 255]));
  for (const NW of [0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 10, 20, 50]) cases.push(...both(["dpss", NW], [0, 1, 2, 3, 4, 5, 8, 9, 10, 21, 32, 50, 64, 65, 100, 101, 200, 255]));
  cases.push(...both("lanczos", [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 100, 101, 255]));
  for (const beta of [0, 0.5, 4, 8.5, 20]) cases.push(...both(["kbd", beta], [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 100, 101, 254, 255]));
  checkAll(cases);
  // Long windows (the spectral functions use nperseg in the thousands). The Slepian eigenproblem is ill conditioned
  // there, so dpss is held to 1e-10 (its error against SciPy's LAPACK result is ~1e-12 at 1000 samples).
  const long: Case[] = [];
  for (const at of [60, 100]) long.push(...both(["chebwin", at], [512, 1000, 1024, 2001]));
  long.push(...both("lanczos", [512, 1001]), ...both(["kbd", 6], [512, 1000]));
  checkAll(long);
  const slepian: Case[] = [];
  for (const NW of [2.5, 4, 8, 25]) slepian.push(...both(["dpss", NW], [512, 1000, 1024, 2000]));
  checkAll(slepian, 1e-10);
});

test("dpss is the leading Slepian taper: approximate norm, peak near 1, symmetric, positive; Kmax is not available (as in get_window)", () => {
  const w = getWindow(["dpss", 3], 64, false);
  assert.equal(w.length, 64);
  for (let i = 0; i < 32; i++) assert.ok(Math.abs((w[i] as number) - (w[63 - i] as number)) < 1e-12);
  assert.ok((w[0] as number) > 0 && Math.max(...w) <= 1.0000001 && Math.max(...w) > 0.99);
  assert.throws(() => getWindow(["dpss", 3, 2] as never, 64), TypeError);
});

test("SciPy's errors, by class: unknown names, wrong parameter counts and types, bad lengths, symmetric-only and even-only windows", { skip: SCIPY_SKIP_REASON }, () => {
  const bad: [unknown, number, boolean][] = [
    ["hanning", 8, true], // not in SciPy 1.16's table, so an error here too
    ["Hann", 8, true],
    ["foo", 8, true],
    ["", 8, true],
    [["kaiser_bessel_derived", 4], 8, false], // SciPy's spelling has spaces: "kaiser bessel derived"
    [["kbd", 4], 8, true], // periodic: only symmetric windows are defined
    [["kbd", 4], 7, false], // odd length
    [["kbd", 4], 1, false],
    [["foo", 1], 8, true],
    [["hann", 1], 8, true],
    [["lanczos", 1], 8, true],
    [["boxcar", 1, 2], 8, true],
    [["kaiser"], 8, true],
    [["kaiser", "a"], 8, true],
    [["kaiser", 1, 2], 8, true],
    [["gaussian"], 8, true],
    [["general_gaussian", 1], 8, true],
    [["chebwin"], 8, true],
    [["dpss"], 8, true],
    [["dpss", 3, 2], 20, true], // Kmax: get_window cannot pass it
    [["dpss", 5], 10, true], // NW must be < M/2
    [["dpss", 0], 10, true], // NW must be positive
    [["dpss", -1], 10, true],
    [["tukey", 0.5, 3], 8, true],
    [["exponential", 3, 2], 8, false], // a centre for a symmetric window
    [["taylor", 0], 8, true],
    [["taylor", 1.5], 8, true],
    ["hann", -1, true],
    ["hann", 1.5, true],
    [[], 8, true],
    [[3], 8, true],
    [null, 8, true],
  ];
  checkAll(bad.map(([window, n, fftbins]) => ({ window, n, fftbins })));
});

test("a number (or a numeric string) is a Kaiser beta, as in get_window", { skip: SCIPY_SKIP_REASON }, () => {
  checkAll([0, 1, 2, 4.5, 8.6, "4.0", " 3 ", "1e1", "0.5"].flatMap((window) => both(window, [0, 1, 2, 9, 64, 65])));
});

test("chebwin warns below 45 dB attenuation (through onWarning, SciPy's text), even for lengths 0 and 1; never at 45 and above", { skip: SCIPY_SKIP_REASON }, () => {
  checkAll([...both(["cheb", 10], [0, 1, 2, 9]), ...both(["chebwin", -44.99], [1, 9]), ...both(["chebwin", 45], [1, 9]), ...both(["chebwin", 100], [0, 9])]);
  const logged: string[] = [];
  const original = console.warn;
  console.warn = (m: unknown) => void logged.push(String(m));
  try {
    getWindow(["chebwin", 30], 8);
  } finally {
    console.warn = original;
  }
  assert.equal(logged.length, 1, "with no onWarning the message goes to console.warn");
  assert.match(logged[0] as string, /not suitable for spectral analysis/);
});

test("windows are exactly symmetric where SciPy's are (lanczos, chebwin, dpss, kbd) and kaiser bessel derived satisfies Princen-Bradley", () => {
  for (const spec of ["lanczos", ["chebwin", 60], ["dpss", 2.5]] as const) {
    for (const n of [7, 8, 31, 32]) {
      const w = getWindow(spec as never, n, false);
      for (let i = 0; i < n; i++) assert.ok(Math.abs((w[i] as number) - (w[n - 1 - i] as number)) < 1e-12, `${JSON.stringify(spec)} n=${n} i=${i}`);
    }
  }
  const k = getWindow(["kbd", 5], 64, false);
  for (let i = 0; i < 32; i++) assert.ok(Math.abs((k[i] as number) ** 2 + (k[i + 32] as number) ** 2 - 1) < 1e-12);
});

test("1.16 corner cases that SciPy 1.17+ changed, checked by hand: empty windows, numeric strings, the underscored kaiser_bessel_derived", () => {
  for (const w of ["hann", "boxcar", "lanczos", ["kaiser", 5], ["chebwin", 80], ["dpss", 1], ["kbd", 4], ["tukey", 0.2]] as const) {
    assert.equal(getWindow(w as never, 0, w[0] === "kbd" ? false : true).length, 0, `${JSON.stringify(w)}`);
  }
  assert.deepEqual(Array.from(getWindow("4.0" as never, 9)), Array.from(getWindow(4.0, 9)));
  assert.deepEqual(Array.from(getWindow(" 1e1 " as never, 9)), Array.from(getWindow(10, 9)));
  assert.throws(() => getWindow(["kaiser_bessel_derived", 4] as never, 8, false), RangeError);
  assert.throws(() => getWindow(["hann_periodic"] as never, 8), RangeError);
  assert.throws(() => getWindow(["kbd", 4], 8), RangeError, "periodic kbd");
});
