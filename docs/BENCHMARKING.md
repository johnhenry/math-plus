# Benchmarking

How to measure performance in this repo so the numbers describe the code and not the laptop's
temperature. It applies to every performance spike (`docs/spikes/*.md`) and every "N× faster" claim
in a README or PR. Like the spikes, benchmarks are run by hand and recorded. CI never gates on GPU
performance, because a standard GitHub Actions runner has no real GPU (see `docs/TESTING.md`).

## Why: machines throttle

A fanless laptop does not keep its cold speed under sustained load. Measured in
[laya-js](https://github.com/johnhenry/laya-js) on a MacBook Air M2 (issue #127):

- After about **10 s** of sustained GPU load, throughput falls to about **35 %** of the cold figure.
- About **5 s** of idle restores it.

A benchmark that loops over a large grid for a minute is measuring the throttled machine for
most of that minute. Two effects follow:

1. **Absolute numbers drift.** Later cells look slower than earlier ones, and so do larger sizes,
   because they come later in the run.
2. **Comparisons are biased.** If backend A runs its whole grid before backend B, B runs on a hotter
   machine and looks worse than it is. Two backends in two processes run minutes apart have the
   same problem.

Fanned desktops and CI runners throttle less, but they are not immune: sustained all-core load, a
shared host, or power limits cause the same drift. Follow the method everywhere.

## The method

1. **Cool down before each cell.** Idle **5 s** before every (cell, backend) measurement, so each
   one starts from the same cold state.
2. **Keep each timing window at 1 s or less.** Run one untimed warmup call (for JIT, pipeline
   compilation and buffer pools), then time repeated calls for at most **1 s**, with a minimum of 3
   and a maximum of 30 samples. Report the **median**, and keep min, max and `n` next to it.
3. **Alternate backends inside one process.** For each cell, measure every backend back to back:
   `A, B` on one cell, `B, A` on the next. Do not run one backend's grid to completion first, and do
   not compare numbers from separate processes or separate sittings.
4. **Record the machine and its thermal state** with every table: model, CPU, core count, memory,
   OS, runtime and version, battery or AC power, load average, the platform's thermal report, date,
   and the cooldown and window settings you used. `machineInfo()` (below) collects all of these.
5. **Say what the timing includes.** For example, `measure-gemm-threshold.ts` times allocate →
   upload → compute → read back → free for every call. A number that leaves out transfers answers a
   different question. State which one you measured.

A benchmark that breaks one of these rules for a reason (for example, a deliberate sustained-load
test to measure throttling itself) says so in its write-up.

## The helper: `scripts/bench/thermal.ts`

The helper implements steps 1–4 with no dependencies, and runs under Node (type stripping) and Bun:

```ts
import { formatGrid, machineInfo, runGrid } from "../../../scripts/bench/thermal.ts";

const rows = await runGrid({
  cells: [64, 256, 1024],
  label: (n) => `n=${n}`,
  backends: [
    { name: "wasm", run: (n) => wasmMatmul(n) },
    { name: "webgpu", run: (n) => gpuMatmul(n) },   // may return a Promise; it is awaited
  ],
});
console.log(formatGrid(rows));                        // Markdown median table for the spike doc
console.log(JSON.stringify({ machine: machineInfo(), rows }));   // raw data, including run order
```

- `timeCell(fn, opts)` times one cell: cooldown, then warmup, then a window of at most `windowMs`.
  Its options are `cooldownMs`, `windowMs`, `minRuns`, `maxRuns` and `warmup`.
- `runGrid(spec)` runs every cell on every backend. It alternates backends within each cell,
  reverses their order on every other cell, and prints one line per measurement to stderr.
- Each row records `order`, its position in the run, so you can check afterwards whether
  results drift with time.
- `machineInfo()` / `thermalSnapshot()` read the thermal state with `pmset -g therm` on macOS and
  `/sys/class/thermal` on Linux. They return `null` where the platform offers nothing without root.
- To override the defaults without editing code, set `BENCH_COOL_S` (default 5) and
  `BENCH_WINDOW_MS` (default 1000). `BENCH_COOL_S=0` is only for checking that a script works.
  Never publish numbers from a run without cooldown.

The helper's own tests (`test/bench-thermal.test.ts`, part of `npm run test:manifest`) check the
method itself: a cooldown before every measurement, the window bound, and the alternation order.
They use a fake clock.

## Existing numbers that predate this method

- **`docs/spikes/webgpu-baseline.md`** was measured on an **Intel iGPU (ADL-N) through ANGLE under
  Xvfb**. It used a fixed 5-iteration loop with no cooldown and ran the backends one after the other.
  Treat it as a lower bound for that machine only. Re-run it with this method on real GPUs (Apple
  M-series, a discrete NVIDIA/AMD card) before using it to set `tensor-webgpu` thresholds.
  `packages/tensor-webgpu/scripts/measure-gemm-threshold.ts` now uses `runGrid`; its 2026-09-24
  re-measurements (the latest in Dawn, headless Chrome and a visible browser) are in
  `docs/spikes/webgpu-tiled-gemm.md`.
- `docs/spikes/wasm-baseline.md` and `docs/spikes/wasm-simd.md` are CPU-bound, short, and
  single-backend-per-run. Throttling affects them less, but a re-measurement should still use the
  helper.

## Running GPU benchmarks without a browser

`@johnhenry/math-plus-tensor-webgpu` reaches WebGPU from Node through Dawn (backend-webgpu's
`getGpu`, the `webgpu` npm package), so a GPU benchmark can call `runGrid` in-process with no browser round trip
inside the timed call — `measure-gemm-threshold.ts` does this by default. When a benchmark must run
in a browser page over CDP, have each call time itself in the page and return `{ selfTimedMs }`
(see `timeCell`): that value becomes the sample, so the DevTools round trip stays out of it, while
the window bound still uses the caller's clock. For a real, visible browser, run the whole grid in
the page instead (`packages/tensor-webgpu/scripts/gemm-threshold-page/`: `runGrid` bundled for
the browser, served cross-origin isolated so `performance.now()` has 5 µs rather than 100 µs
resolution), and read the rows back from the page.
