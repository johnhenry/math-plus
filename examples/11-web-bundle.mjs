// @johnhenry/math-plus-web: fft, signal and unit from ONE import, the same file a browser loads.
// `@johnhenry/math-plus-web/bundle` is a single self-contained ESM file (no import specifiers), so this runs
// unchanged in a <script type="module"> or a Worker; see packages/web/examples/index.html for the browser page.
//
// Run: npm install && npm run build && node examples/11-web-bundle.mjs
import assert from "node:assert/strict";
import { fft, signal, tensor, unit } from "@johnhenry/math-plus-web/bundle";

// A 5 Hz tone on a DC offset, sampled at 64 Hz for 1 s.
const fs = 64;
const x = tensor.Tensor.from(Array.from({ length: fs }, (_, i) => 2 + Math.sin((2 * Math.PI * 5 * i) / fs)), { dtype: "f64" });

const spectrum = fft.rfft(x); // fft
assert.equal(spectrum.size, fs);

const { freqs, Pxx } = signal.welch(x, { fs, nperseg: 64 }); // signal
const power = Pxx.toArray();
const peak = freqs[power.indexOf(Math.max(...power.slice(1)))];
console.log("dominant frequency (Hz):", peak); // 5

const length = unit.Unit.of(5, "m").add(unit.Unit.of(20, "cm")); // unit
console.log(length.to("m").value); // 5.2
assert.ok(Math.abs(length.to("m").value - 5.2) < 1e-12);
