---
"@johnhenry/math-plus-scalar-types": patch
"@johnhenry/math-plus-tensor-core": patch
"@johnhenry/math-plus-telemetry": patch
"@johnhenry/math-plus-tensor-wasm": patch
"@johnhenry/math-plus-tensor-autograd": patch
"@johnhenry/math-plus-tensor-compile": patch
"@johnhenry/math-plus-tensor-webgpu": patch
"@johnhenry/math-plus-frame-arrow": patch
"@johnhenry/math-plus-frame-parquet": patch
"@johnhenry/math-plus-adapter-math": patch
"@johnhenry/math-plus-unit": patch
"@johnhenry/math-plus-adapter-onnx": patch
"@johnhenry/math-plus-fft": patch
"@johnhenry/math-plus-image": patch
"@johnhenry/math-plus-signal": patch
"@johnhenry/math-plus-mcp": patch
"@johnhenry/math-plus-data": patch
---

Lower `engines.node` from `>=26.0.0` to `>=24.0.0`. Nothing in these packages needs Node 26: the full test suite passes on Node 24.9, and CI now tests Node 24. Every suite also runs under Bun 1.2.17 (`npm run test:bun`).

`@johnhenry/math-plus-frame-parquet`: `scanParquet`/`scanParquetLazy` now accept an absolute file path without wildcards under Bun. Bun 1.2's `fs.promises.glob` returns no matches for such a path, so a pattern with no glob metacharacters is now resolved with `stat` on every runtime.
