# @johnhenry/math-plus-web

## 0.2.0

### Minor Changes

- d40c1ef: Add the `data` namespace: `Dataset` and `fromAsync` from `@johnhenry/math-plus-data` (async pipelines with shuffle, epochs, prefetch, mapConcurrent and fold), as `data` in the one-import entry, as `./data` and as a self-contained `./bundle/data` file. Browser and Worker panes can now stream a dataset without Node.

## 0.1.0

### Minor Changes

- 64c970d: New package: one browser/Worker entry for fft, signal, unit, image, tensor-core, tensor-autograd, adapter-math and frame-arrow, plus self-contained single-file ESM bundles (`@johnhenry/math-plus-web/bundle/*`, no import specifiers, run offline from a Blob URL) and a per-package size table (`dist/bundle/sizes.json`, README).
