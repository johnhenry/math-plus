# Examples

Runnable walkthroughs, at least one per cluster of the family. They import
the packages by **bare specifier** through the workspace symlinks, so they
exercise the real `exports` maps. Build first:

```bash
npm install
npm run build       # includes the WASM kernels (needs Rust + lld; see below)
npm run examples    # 01-09, the env-independent set (CI runs this)
npm run example:04  # or one at a time
```

| Example | Cluster | Shows |
|---|---|---|
| `01-tensor-basics.mjs` | tensor | dtypes, views vs copies, broadcasting, matmul, `.npy`, seeded RNG — pure-JS path |
| `02-autograd-training.mjs` | tensor | tape + `backward()`, `grad.valueAndGrad`, training `nn.Linear` with SGD, opt-in telemetry |
| `03-fused-expressions.mjs` | tensor | `compile()` trace-once/execute-fused, broadcasting, `asVariableOp()` on the tape |
| `04-fft-and-signal.mjs` | signal | fft/ifft/rfft conventions, convolution modes, `butter`+`sosFilter`, `findPeaks` |
| `05-image-ops.mjs` | signal | channel-last `resize` (nearest / bilinear half-pixel centers), per-channel `normalize` |
| `06-dataframes.mjs` | data | lazy Frame plans, expressions, `groupBy`/aggregate, bigint/JSON and null traps |
| `07-parquet-roundtrip.mjs` | data | `writeParquet`/`readParquet` with real projection + predicate pushdown |
| `08-data-pipelines.mjs` | data | `Dataset` chains, per-epoch reshuffle, `collate.xy({dtype:"f64"})` into `trainer.fit` |
| `09-mcp-tools.mjs` | interop | the MCP server over an in-memory transport: symbolic + guarded numeric tools |
| `10-wasm-kernels.mjs` | tensor | **env-dependent** — zero-alloc `...Into` ops, SIMD detection, stride-aware matmul |

## Environment-dependent examples

`10-wasm-kernels.mjs` is excluded from the `npm run examples` loop (and from
the CI smoke step): it needs the built `.wasm` artifact, which is gitignored
— `rustup target add wasm32-unknown-unknown`, install `lld`, then
`npm run build:wasm`. Run everything including it with `npm run examples:all`.

Not represented as runnable scripts, on purpose:

- **tensor-webgpu** needs a Chromium-family browser (no plain-Node WebGPU in
  v1) — see `packages/tensor-webgpu/README.md` for the honest state of the
  GEMM threshold before you reach for it.
- **interop-python** needs a Python ≥3.11 with pyarrow/pandas/numpy — see
  `packages/interop-python/README.md` for the conformance-fixture setup.
