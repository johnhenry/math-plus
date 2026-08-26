# @johnhenry/math-plus-telemetry

Shared event schema + sink registry for the math-plus family (issue #10): a
stable stream any UI can consume later, with a zero-cost no-op default. Zero
runtime dependencies.

The differentiated panel this eventually enables isn't loss curves
(TensorBoard already does those well) — it's JS↔WASM memory residency and
copy accounting: which tensors live in linear memory and how many bytes cross
the boundary per step. `docs/spikes/wasm-baseline.md` measured this exact
cost causing a 4x swing that was invisible until benchmarked; Python tooling
has no equivalent because NumPy never crosses that boundary.

## Install

```bash
npm install @johnhenry/math-plus-telemetry
```

## Quick start

```ts
import { setSink, hasSink, metric, type TrainingEvent } from "@johnhenry/math-plus-telemetry";

const events: TrainingEvent[] = [];
setSink((e) => events.push(e)); // install a collector
hasSink(); // true

metric("r1", 3, "loss", 0.42); // -> { type: "metric", runId: "r1", step: 3, name: "loss", value: 0.42, time: ... }

setSink(null); // restore the no-op default
```

Consuming what siblings emit — installing a sink is all it takes:

```ts
import { setSink } from "@johnhenry/math-plus-telemetry";

setSink((e) => console.log(e.type, e));

// tensor-autograd: backward() emits a "trace" span (name "backward",
// category "autograd"); every optimizer's step() emits "optim/gradNorm".
// tensor-wasm: the arena allocator emits "wasm/alloc.bytes" and
// "wasm/alloc.calls" per allocation.
```

## Event schema

`TrainingEvent` is a discriminated union on `type`:

| `type` | Fields |
|---|---|
| `run.start` | `runId`, `time`, `config`, `environment?` |
| `metric` | `runId`, `step`, `time`, `name`, `value` |
| `tensor.summary` | `runId`, `step`, `name`, `tensor: TensorSummary` |
| `artifact` | `runId`, `step?`, `name`, `kind` (`checkpoint`/`image`/`audio`/`table`/`json`), `ref` |
| `trace` | `runId`, `step`, `spans: [{ name, start, duration, category }]` |

`TensorSummary` carries `shape`/`dtype`/`device` plus stats (`min`/`max`/
`mean`/`std`/`finite` fraction) and an optional histogram. It structurally
**cannot** carry raw values — emit summaries, never tensor dumps, by default.

## API

- `setSink(sink | null)` — install a sink; `null` restores the no-op.
- `hasSink()` — whether a real sink is installed.
- `emit(event)` — raw emission.
- Convenience emitters: `startRun`, `metric`, `tensorSummary`, `artifact`, `trace`.
- `timed(runId, step, spanName, category, fn)` — run `fn`, emitting a trace
  span; when no sink is installed it skips even the `performance.now()` calls.

## How "zero-cost" actually works

The default sink is a real empty function (null-object pattern), so `emit()`
is an unconditional call with no branch. The *real* savings come from
producers guarding payload construction with `hasSink()` — e.g. the
optimizers only compute the global gradient L2 norm when a sink exists, and
tensor-wasm's per-alloc metrics stay exactly zero-cost on the zero-allocation
`...Into` path (issue #3) when unused.

## Things to know before relying on it

- **Single global slot, not a list.** `setSink` *replaces* the previous sink
  silently; two consumers cannot coexist, and there is no unsubscribe token.
  In tests, always `setSink(null)` in a `finally`/`afterEach`.
- **Installing a sink switches on real work globally.** `hasSink()` is a
  global check: a sink installed for loss curves also enables optim's
  grad-norm computation and the WASM allocator's per-alloc metric pair.
- **Two time bases.** `startRun`/`metric` stamp `Date.now()`; `timed()` spans
  use `performance.now()`. `trace.spans[].start` is not comparable to
  `metric.time`.
- tensor-wasm hardcodes `runId: "wasm"` with the alloc counter as `step` —
  its metrics won't correlate with an autograd run's `runId`/`step`.

## Provenance

Part of the [math-plus](https://github.com/johnhenry/math-plus) monorepo —
the engineering side of the `@johnhenry/math` family. Docs for the whole
family: <https://opensource.johnhenry.me/math/>.
