/**
 * @johnhenry/math-plus-web: one import for the family in a browser or Worker (math-plus#173, #176).
 *
 *   import { fft, signal, unit } from "@johnhenry/math-plus-web";
 *
 * Each member is a namespace (the adapter's own `fft` function would otherwise collide with the fft package).
 * All of them share one copy of `Tensor`, so what `signal` returns, `tensor-autograd` accepts.
 * Scope: this package adds no maths of its own; it re-exports the packages as they are published.
 * `data` is the async Dataset pipeline (shuffle, epochs, prefetch, mapConcurrent, fold) of @johnhenry/math-plus-data.
 * Not included: tensor-wasm, tensor-webgpu, tensor-mlx (backends, picked by the app), frame-parquet (needs node:fs),
 * safetensors, mcp, adapter-onnx (pulls in onnxruntime-web).
 */
export * as fft from "./fft.ts";
export * as signal from "./signal.ts";
export * as unit from "./unit.ts";
export * as image from "./image.ts";
export * as tensor from "./tensor.ts";
export * as autograd from "./tensor-autograd.ts";
export * as adapterMath from "./adapter-math.ts";
export * as frameArrow from "./frame-arrow.ts";
export * as data from "./data.ts";
