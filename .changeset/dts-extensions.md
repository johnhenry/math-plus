---
"@johnhenry/math-plus-data": patch
"@johnhenry/math-plus-fft": patch
"@johnhenry/math-plus-frame-arrow": patch
"@johnhenry/math-plus-frame-parquet": patch
"@johnhenry/math-plus-image": patch
"@johnhenry/math-plus-mcp": patch
"@johnhenry/math-plus-safetensors": patch
"@johnhenry/math-plus-scalar-types": patch
"@johnhenry/math-plus-signal": patch
"@johnhenry/math-plus-special": patch
"@johnhenry/math-plus-telemetry": patch
"@johnhenry/math-plus-tensor-autograd": patch
"@johnhenry/math-plus-tensor-compile": patch
"@johnhenry/math-plus-tensor-core": patch
"@johnhenry/math-plus-tensor-cpu": patch
"@johnhenry/math-plus-tensor-mlx": patch
"@johnhenry/math-plus-tensor-wasm": patch
"@johnhenry/math-plus-tensor-webgpu": patch
"@johnhenry/math-plus-adapter-math": patch
"@johnhenry/math-plus-adapter-onnx": patch
"@johnhenry/math-plus-unit": patch
---

Published declarations no longer import `./x.ts` (closes #157). tsc's `rewriteRelativeImportExtensions` rewrites `.ts` specifiers to `.js` in emitted JS but not in emitted `.d.ts`, so `dist/*.d.ts` referenced files that aren't in the package, and Deno's type check failed on them. Every package's build now runs `scripts/rewrite-dts-extensions.mjs` after `tsc`:

- relative `.ts` / `.mts` / `.cts` specifiers in `dist/**/*.d.ts` become `.js` / `.mjs` / `.cjs` (as in laya-js),
- each `dist/*.js` with a declaration file starts with `// @ts-self-types="./x.d.ts"` (after the `#!` line of a bin), so Deno finds the types when it loads `dist/` as plain files or from a URL instead of through `npm:`. Source maps are shifted by the inserted line.

No runtime change. The manifest drift test checks every built `dist/` for both.
