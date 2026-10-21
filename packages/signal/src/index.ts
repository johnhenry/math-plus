/**
 * @johnhenry/math-plus-signal — the signal-processing slice of the v2/v3 SciPy-
 * equivalent bundle (issue #44): `convolve`, `stft`/`istft`, `findPeaks`,
 * `sosFilter`, `butter`, `resamplePoly`. See each module's own doc comment
 * for algorithm sources and v1 scope decisions.
 */
export { applyTimeDomainOp, convolve, convolve1D, type ConvolveMode, type ConvolveOptions } from "./convolve.ts";
export { correlate, correlate1D } from "./correlate.ts";
export { correlate2D } from "./correlate2d.ts";
export { butter, type FilterType } from "./filter-design.ts";
export { findPeaks, type FindPeaksOptions, type FindPeaksResult } from "./find-peaks.ts";
export { freqz, type FreqzResult } from "./freqz.ts";
export { resamplePoly } from "./resample.ts";
export { sosFilter, type Sos, type SosSection } from "./sos-filter.ts";
export { sosfiltfilt, type SosfiltfiltOptions } from "./sosfiltfilt.ts";
export { istft, stft, type IstftOptions, type StftOptions, type StftResult, type StftScaling } from "./stft.ts";
export type { Boundary, Detrend, SegmentOptions } from "./spectral.ts";
export { welch, type WelchOptions } from "./welch.ts";
export { getWindow, hammingWindow, hannWindow, type WindowName, type WindowSpec } from "./window.ts";
