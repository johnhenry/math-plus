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
export type { SignalInput } from "./nd.ts";
export { checkCOLA, checkNOLA } from "./overlap.ts";
export { istft, stft, type IstftOptions, type IstftResult, type StftOptions, type StftResult, type StftScaling } from "./stft.ts";
export type { Boundary, Detrend, DetrendFunction, SegmentOptions, WarningOptions } from "./spectral.ts";
export { coherence, type CoherenceOptions, type CoherenceResult } from "./coherence.ts";
export { csd, type CsdOptions, type CsdResult } from "./csd.ts";
export { periodogram, type PeriodogramOptions, type PeriodogramResult } from "./periodogram.ts";
export { spectrogram, type SpectrogramMode, type SpectrogramOptions, type SpectrogramResult } from "./spectrogram.ts";
export { welch, type WelchOptions, type WelchResult } from "./welch.ts";
export { getWindow, hammingWindow, hannWindow, WINDOW_NAMES, type WindowName, type WindowSpec, type WindowTuple } from "./window.ts";
