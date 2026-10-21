/**
 * `check_NOLA` / `check_COLA` (`scipy.signal`): whether a window and overlap can be inverted by overlap-add.
 * NOLA (nonzero overlap-add) is what `istft` needs; COLA (constant overlap-add) is the stronger condition under
 * which the plain, un-normalised overlap-add reconstructs the signal. Both follow SciPy's code line for line,
 * including its tolerance handling.
 */
import { getWindow, type WindowSpec } from "./window.ts";

function resolve(who: string, window: WindowSpec, nperseg: number, noverlap: number): { win: Float64Array; step: number } {
  if (!Number.isInteger(nperseg) || nperseg < 1) throw new RangeError(`${who}: nperseg must be a positive integer`);
  if (noverlap >= nperseg) throw new RangeError(`${who}: noverlap must be less than nperseg`);
  if (!Number.isInteger(noverlap) || noverlap < 0) throw new RangeError(`${who}: noverlap must be a nonnegative integer`);
  const win = getWindow(window, nperseg);
  return { win, step: nperseg - noverlap };
}

/** SciPy's `binsums`: the window (or its square) folded onto one hop. */
function binSums(win: Float64Array, step: number, square: boolean): Float64Array {
  const nperseg = win.length;
  const p = (v: number) => (square ? v * v : v);
  const sums = new Float64Array(step);
  for (let ii = 0; ii < Math.floor(nperseg / step); ii++) {
    for (let j = 0; j < step; j++) sums[j] = (sums[j] as number) + p(win[ii * step + j] as number);
  }
  const rest = nperseg % step;
  if (rest !== 0) {
    for (let j = 0; j < rest; j++) sums[j] = (sums[j] as number) + p(win[nperseg - rest + j] as number);
  }
  return sums;
}

/**
 * `scipy.signal.check_NOLA(window, nperseg, noverlap, tol = 1e-10)`: `true` when the squared window, overlap-added at
 * hop `nperseg - noverlap`, stays above `tol` everywhere, i.e. when `istft` can invert an `stft` made with these
 * parameters. `window` is anything `getWindow` takes.
 */
export function checkNOLA(window: WindowSpec, nperseg: number, noverlap: number, tol = 1e-10): boolean {
  const { win, step } = resolve("checkNOLA", window, nperseg, noverlap);
  const sums = binSums(win, step, true);
  let min = Infinity;
  for (const v of sums) min = Math.min(min, v);
  return min > tol;
}

/**
 * `scipy.signal.check_COLA(window, nperseg, noverlap, tol = 1e-10)`: `true` when the window overlap-added at hop
 * `nperseg - noverlap` is constant (every bin within `tol` of the median).
 */
export function checkCOLA(window: WindowSpec, nperseg: number, noverlap: number, tol = 1e-10): boolean {
  const { win, step } = resolve("checkCOLA", window, nperseg, noverlap);
  const sums = binSums(win, step, false);
  const sorted = Float64Array.from(sums).sort();
  const mid = sorted.length >> 1;
  const median = sorted.length % 2 ? (sorted[mid] as number) : ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2;
  let worst = 0;
  for (const v of sums) worst = Math.max(worst, Math.abs(v - median));
  return worst < tol;
}
