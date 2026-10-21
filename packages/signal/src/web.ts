/**
 * Browser / Web Worker entry (math-plus#173): `import { rfft, welch, stft,
 * butter, sosfiltfilt, findPeaks, resamplePoly } from
 * "@johnhenry/math-plus-signal/web"`.
 *
 * One import that pulls in exactly the spectral + filtering surface a
 * browser or Worker needs, with `Tensor`/`ComplexTensor` to build inputs.
 * Everything reachable from here is plain ESM over typed arrays: no Node
 * built-ins, no `process`/`Buffer`, no top-level side effects
 * (`test/web-entry.test.ts` enforces this by walking the built import graph
 * and by loading the entry with those globals removed).
 */
export { ComplexTensor, rfft } from "@johnhenry/math-plus-fft";
export { Tensor } from "@johnhenry/math-plus-tensor-core";
export { butter, type FilterType } from "./filter-design.ts";
export { findPeaks, type FindPeaksOptions, type FindPeaksResult } from "./find-peaks.ts";
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
export { getWindow, WINDOW_NAMES, type WindowName, type WindowSpec, type WindowTuple } from "./window.ts";
