# @johnhenry/math-plus-web

One import for the math-plus family in a browser or a Web Worker, and self-contained single-file ESM bundles that run offline
([#173](https://github.com/johnhenry/math-plus/issues/173), [#176](https://github.com/johnhenry/math-plus/issues/176)).

```js
import { fft, signal, unit } from "@johnhenry/math-plus-web";        // for a bundler (Vite, esbuild, ...)
import { fft, signal, unit } from "@johnhenry/math-plus-web/bundle"; // the ready-built bundle: one file, no imports
```

## What is in it

| Namespace (`index`, `bundle`) | Subpath | Package |
| --- | --- | --- |
| `fft` | `/fft` | `@johnhenry/math-plus-fft` (+ `Tensor`) |
| `signal` | `/signal` | `@johnhenry/math-plus-signal/web` (rfft, welch, stft, butter, sosfiltfilt, findPeaks, resamplePoly, ...) |
| `unit` | `/unit` | `@johnhenry/math-plus-unit` |
| `image` | `/image` | `@johnhenry/math-plus-image` (+ `Tensor`) |
| `tensor` | `/tensor` | `@johnhenry/math-plus-tensor-core` |
| `autograd` | `/tensor-autograd` | `@johnhenry/math-plus-tensor-autograd` (+ `Tensor`, `random`) |
| `adapterMath` | `/adapter-math` | `@johnhenry/math-plus-adapter-math` (+ `Tensor`) |
| `frameArrow` | `/frame-arrow` | `@johnhenry/math-plus-frame-arrow` |

`@johnhenry/math-plus-web/<name>` is the module entry (bare imports left for your bundler); `@johnhenry/math-plus-web/bundle/<name>`
is the built file for that one package (its exports are the package's own, not namespaced); `@johnhenry/math-plus-web/bundle` is the
all-in-one file, whose members are the namespaces above.

## The bundles

Every file in `dist/bundle/` is minified ES2023 ESM with no `import` or `export ... from` specifier, no Node built-in and no Node
global, so its text can be given to a Worker, an iframe or a sandbox as a Blob URL and run with no network. `test/bundle.test.ts`
parses each one for imports and loads each from its text with `process`, `Buffer` and `require` removed.

**One `Tensor`.** A standalone bundle carries its own copy of tensor-core, so two standalone bundles loaded together hold two
`Tensor` classes, and a tensor made by one is not an instance of the other's. Use `bundle/all.js` (one shared copy) when values
cross packages, or build every input with the `Tensor` that the bundle you are calling exports (every bundle that takes a tensor
exports one).

## Sizes

Minified, as built by `npm run build -w @johnhenry/math-plus-web`; `dist/bundle/sizes.json` has the same numbers in bytes.
`all.js` keeps whole namespaces, so it cannot drop unused exports. The table is rewritten by `npm run sizes -w @johnhenry/math-plus-web`.

<!-- sizes:start -->
| Bundle | Carries | Minified | gzip | brotli |
| --- | --- | ---: | ---: | ---: |
| `fft.js` | @johnhenry/math-plus-fft | 50.8 kB | 16.2 kB | 14.4 kB |
| `signal.js` | @johnhenry/math-plus-signal/web (includes fft) | 88.9 kB | 29.6 kB | 26.0 kB |
| `unit.js` | @johnhenry/math-plus-unit | 6.4 kB | 2.2 kB | 1.9 kB |
| `image.js` | @johnhenry/math-plus-image | 40.4 kB | 13.4 kB | 11.8 kB |
| `tensor.js` | @johnhenry/math-plus-tensor-core | 41.4 kB | 13.8 kB | 12.2 kB |
| `tensor-autograd.js` | @johnhenry/math-plus-tensor-autograd (includes tensor-core) | 62.7 kB | 20.3 kB | 18.1 kB |
| `adapter-math.js` | @johnhenry/math-plus-adapter-math (includes @johnhenry/math, tensor-compile, frame-arrow) | 160.7 kB | 50.9 kB | 44.0 kB |
| `frame-arrow.js` | @johnhenry/math-plus-frame-arrow (includes apache-arrow) | 289.1 kB | 73.4 kB | 62.4 kB |
| `all.js` | all of the above, one shared copy of each | 478.5 kB | 134.6 kB | 113.5 kB |
<!-- sizes:end -->

## Licences

The family is MIT. Minifying strips every comment, so `dist/bundle/LICENSES.txt` carries the licence texts, and each entry in
`sizes.json` lists the third-party packages inside its bundle (`thirdParty`): `@johnhenry/math` (MIT, in fft, signal, adapter-math and
all, for `ComplexNumber`), and `apache-arrow`, `flatbuffers` (Apache-2.0) and `tslib` (0BSD) in frame-arrow, adapter-math and all.
Ship `LICENSES.txt` with any bundle you redistribute.

## Scope

- No maths of its own: it re-exports the packages as published.
- Not included: the compute backends (`tensor-wasm`, `tensor-webgpu`, `tensor-mlx`: the app picks one and registers it), `frame-parquet`
  (needs `node:fs`), `safetensors`, `mcp`, and `adapter-onnx` (it pulls in `onnxruntime-web`).
- `tensor-autograd`'s `./safetensors` subpath is not in the bundle.
- The `adapter-math` and `frame-arrow` bundles carry `@johnhenry/math` and `apache-arrow`; that is most of their size.
- The bundles are built from the compiled `dist/` of each package, so a stale package build gives a stale bundle (`turbo` orders them).
