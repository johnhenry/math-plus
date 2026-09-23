/**
 * PyTorch differential oracle (issue #123) — forward AND backward parity for
 * Variable ops and transformer layers against `scripts/torch_oracle.py`.
 *
 * Skip-don't-fail (docs/TESTING.md): the interpreter is
 * `$MATH_PLUS_TORCH_ORACLE_PYTHON`, else `$MATH_PLUS_ORACLE_PYTHON`, else
 * `python3` on PATH; if none can `import torch`, every oracle test skips.
 * A real verification run must show 0 skipped.
 *
 * Protocol: each test file declares its cases, runs them in JS first
 * (forward, then backward of `sum(out * gradOut)` for a seeded random
 * `gradOut`), then sends ONE batched request to Python (torch import is the
 * slow part) and compares outputs plus every input/parameter gradient.
 * Layers ship their JS `stateDict()`, which the oracle loads with
 * `load_state_dict(strict=True)` — so parameter NAMES are checked against
 * PyTorch's too, not just values.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { random, Tensor } from "@johnhenry/math-plus-tensor-core";
import { variable, type Variable, type nn } from "../src/index.ts";

export const ORACLE_SCRIPT = new URL("../scripts/torch_oracle.py", import.meta.url).pathname;

function findTorchPython(requireSafetensors: boolean): string | undefined {
  const candidates = [process.env.MATH_PLUS_TORCH_ORACLE_PYTHON, process.env.MATH_PLUS_ORACLE_PYTHON, "python3"].filter(
    (c): c is string => Boolean(c),
  );
  for (const candidate of candidates) {
    try {
      execFileSync(candidate, ["-c", requireSafetensors ? "import torch, safetensors" : "import torch"], { stdio: "ignore" });
      return candidate;
    } catch {
      // try the next candidate
    }
  }
  return undefined;
}

export const TORCH_PYTHON = findTorchPython(false);
export const torchSkip: string | false = TORCH_PYTHON
  ? false
  : "no python with torch found (set MATH_PLUS_TORCH_ORACLE_PYTHON or MATH_PLUS_ORACLE_PYTHON)";
export const TORCH_ST_PYTHON = findTorchPython(true);
export const torchSafetensorsSkip: string | false = TORCH_ST_PYTHON
  ? false
  : "no python with torch + safetensors found (set MATH_PLUS_TORCH_ORACLE_PYTHON or MATH_PLUS_ORACLE_PYTHON)";

export interface OracleTensor {
  shape: number[];
  data: number[];
  bool?: boolean;
  dtype?: string;
}

export interface OracleResult {
  out?: OracleTensor;
  grads?: Record<string, OracleTensor>;
  error?: string;
}

function flatten(value: unknown): number[] {
  if (Array.isArray(value)) return value.flatMap(flatten);
  return [Number(value)];
}

export function toOracle(t: Tensor, extra: Partial<OracleTensor> = {}): OracleTensor {
  const readable = t.dtype === "f16" || t.dtype === "bf16" ? t.cast("f32") : t;
  const out: OracleTensor = { shape: [...t.shape], data: flatten(readable.contiguous().toArray()), ...extra };
  if (t.dtype === "bool") out.bool = true;
  return out;
}

export interface TolSpec {
  rtol: number;
  atol: number;
}

export const TOL: Record<"f32" | "f64", TolSpec> = {
  f64: { rtol: 1e-9, atol: 1e-11 },
  f32: { rtol: 2e-4, atol: 2e-5 },
};

/** Elementwise `|a - e| <= atol + rtol * |e|`, reporting the worst offender. */
export function assertClose(actual: Tensor, expected: OracleTensor, tol: TolSpec, label: string): void {
  assert.deepEqual([...actual.shape], expected.shape, `${label}: shape`);
  const a = flatten((actual.dtype === "f16" || actual.dtype === "bf16" ? actual.cast("f32") : actual).contiguous().toArray());
  let worst = -1;
  let worstExcess = 0;
  for (let i = 0; i < a.length; i++) {
    const e = expected.data[i] as number;
    const excess = Math.abs((a[i] as number) - e) - (tol.atol + tol.rtol * Math.abs(e));
    if (!(excess <= worstExcess) || Number.isNaN(a[i])) {
      worst = i;
      worstExcess = Number.isNaN(a[i]) ? Infinity : excess;
    }
  }
  assert.ok(
    worst < 0,
    `${label}: element ${worst} is ${a[worst]}, PyTorch says ${expected.data[worst]} (tol atol ${tol.atol} + rtol ${tol.rtol})`,
  );
}

export function runTorchOracle(cases: unknown[], python = TORCH_PYTHON): Record<string, OracleResult> {
  const out = execFileSync(python as string, [ORACLE_SCRIPT, "run"], {
    input: JSON.stringify({ cases }),
    encoding: "utf8",
    maxBuffer: 256 * 1024 * 1024,
  });
  return (JSON.parse(out) as { results: Record<string, OracleResult> }).results;
}

// ---- declarative case runner ---------------------------------------------------

export interface Case {
  id: string;
  kind: string;
  dtype: "f32" | "f64";
  config?: Record<string, unknown>;
  inputs: Record<string, Tensor>;
  masks?: Record<string, Tensor>;
  module?: nn.Module;
  forward: (inputs: Record<string, Variable>) => Variable;
  tol?: TolSpec;
}

interface Prepared {
  c: Case;
  out: Tensor;
  grads: Record<string, Tensor | null>;
  request: Record<string, unknown>;
}

let gradSeed = 1000;

/**
 * Runs every case in JS, builds the batched oracle request, and returns a
 * `check(id)` that compares one case against PyTorch (call inside a test).
 */
export function prepareCases(cases: Case[]): {
  prepared: Map<string, Prepared>;
  requests: unknown[];
} {
  const prepared = new Map<string, Prepared>();
  const requests: unknown[] = [];
  for (const c of cases) {
    const vars: Record<string, Variable> = {};
    for (const [name, t] of Object.entries(c.inputs)) vars[name] = variable(t);
    c.module?.zeroGrad();
    const outVar = c.forward(vars);
    const gradOut = random.normal([...outVar.shape], { std: 1, dtype: outVar.dtype, rng: random.seed(gradSeed++) });
    outVar.backward(gradOut);
    const grads: Record<string, Tensor | null> = {};
    for (const [name, v] of Object.entries(vars)) grads[`input:${name}`] = v.grad;
    if (c.module) for (const [name, p] of Object.entries(c.module.namedParameters())) grads[name] = p.grad;

    const request: Record<string, unknown> = {
      id: c.id,
      kind: c.kind,
      dtype: c.dtype,
      config: c.config ?? {},
      inputs: Object.fromEntries(
        Object.entries(c.inputs).map(([n, t]) => [n, toOracle(t, t.dtype !== c.dtype ? { dtype: t.dtype } : {})]),
      ),
      masks: Object.fromEntries(Object.entries(c.masks ?? {}).map(([n, t]) => [n, toOracle(t)])),
      gradOut: toOracle(gradOut),
    };
    if (c.module) {
      request.state = Object.fromEntries(Object.entries(c.module.stateDict()).map(([n, t]) => [n, toOracle(t)]));
    }
    prepared.set(c.id, { c, out: outVar.value, grads, request });
    requests.push(request);
  }
  return { prepared, requests };
}

/** Compare one prepared case against its oracle result: output, and the SAME set of gradients with matching values. */
export function checkCase(p: Prepared, r: OracleResult | undefined): void {
  assert.ok(r, `${p.c.id}: no oracle result`);
  assert.equal(r.error, undefined, `${p.c.id}: oracle error: ${r.error}`);
  const tol = p.c.tol ?? TOL[p.c.dtype];
  assertClose(p.out, r.out as OracleTensor, tol, `${p.c.id} forward`);
  const expected = r.grads ?? {};
  const jsKeys = Object.keys(p.grads).filter((k) => p.grads[k] !== null);
  assert.deepEqual(new Set(jsKeys), new Set(Object.keys(expected)), `${p.c.id}: gradient key sets differ`);
  for (const [name, spec] of Object.entries(expected)) {
    assertClose(p.grads[name] as Tensor, spec, tol, `${p.c.id} grad ${name}`);
  }
}

/** Seeded uniform tensor. */
export function rand(shape: number[], dtype: "f32" | "f64", seed: number, scale = 1): Tensor {
  return random.uniform(shape, { min: -scale, max: scale, dtype, rng: random.seed(seed) });
}

/** Bool tensor from 0/1 values. */
export function boolMask(values: number[], shape: number[]): Tensor {
  return Tensor.fromTypedArray(Uint8Array.from(values), shape, { dtype: "bool" });
}

/** Replace every parameter with seeded random values (LayerNorm's ones/zeros init would hide gradient bugs). */
export function randomizeParams(module: nn.Module, seed: number, scale = 0.5): void {
  let s = seed;
  for (const p of module.parameters()) {
    p.value = random.uniform([...p.shape], { min: -scale, max: scale, dtype: p.dtype, rng: random.seed(s++) });
  }
}
