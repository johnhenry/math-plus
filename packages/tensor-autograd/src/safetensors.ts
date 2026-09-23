/**
 * `@johnhenry/math-plus-tensor-autograd/safetensors` (issue #123):
 * `Module` state dicts to/from safetensors — the checkpoint format PyTorch,
 * Hugging Face and MLX use — via `@johnhenry/math-plus-safetensors`.
 *
 * A separate subpath so the main entry keeps zero dependency on the
 * safetensors package: it is an OPTIONAL peer dependency, needed only if you
 * import this subpath.
 *
 * Because `nn.Linear` stores PyTorch's `[out, in]` layout and every
 * transformer layer uses PyTorch's state-dict key names, a PyTorch
 * `model.state_dict()` saved with `safetensors.torch.save_file` loads with
 * no renaming or transposing (differential-tested against PyTorch in
 * test/safetensors.test.ts). dtypes are preserved on read (F16 -> "f16",
 * BF16 -> "bf16", ...); `Module.loadStateDict` then casts each tensor to the
 * parameter's own dtype.
 */
import type { Tensor } from "@johnhenry/math-plus-tensor-core";
import {
  openSafetensors,
  readSafetensors,
  writeSafetensors,
  type OpenOptions,
  type SafetensorsSource,
} from "@johnhenry/math-plus-safetensors";
import { fromTensor, readTensor, toTensor } from "@johnhenry/math-plus-safetensors/tensor";
import type { LoadStateDictOptions, Module } from "./nn.ts";

export interface StateDictReadOptions {
  /**
   * Keep only keys starting with `prefix`, and strip it (e.g. `"encoder."`
   * to load one sub-model out of a larger checkpoint).
   */
  prefix?: string;
}

function stripPrefix(names: readonly string[], prefix: string | undefined): Array<[string, string]> {
  if (!prefix) return names.map((n) => [n, n]);
  return names.filter((n) => n.startsWith(prefix)).map((n) => [n, n.slice(prefix.length)]);
}

/**
 * An in-memory safetensors file as a state dict. Zero-copy: each Tensor
 * aliases `bytes` when aligned (tensors are immutable, so this is safe as
 * long as you don't mutate `bytes` yourself).
 */
export function stateDictFromSafetensors(
  bytes: ArrayBuffer | Uint8Array,
  options: StateDictReadOptions = {},
): Record<string, Tensor> {
  const file = readSafetensors(bytes);
  const out: Record<string, Tensor> = {};
  for (const [name, key] of stripPrefix(file.names(), options.prefix)) out[key] = toTensor(file, name);
  return out;
}

/**
 * Any safetensors source `openSafetensors` accepts — a path (Node/Bun/Deno),
 * `Blob`/`File`, `http(s):` URL (Range requests), bytes, a `FileHandle`, a
 * custom `ByteSource` — read lazily as a state dict (one read per kept
 * tensor; with `prefix`, only matching tensors are fetched at all).
 */
export async function loadSafetensors(
  source: SafetensorsSource,
  options: StateDictReadOptions & OpenOptions = {},
): Promise<Record<string, Tensor>> {
  const file = await openSafetensors(source, options);
  try {
    const out: Record<string, Tensor> = {};
    for (const [name, key] of stripPrefix(file.names(), options.prefix)) out[key] = await readTensor(file, name);
    return out;
  } finally {
    await file.close();
  }
}

/** `loadSafetensors` + `module.loadStateDict` in one call (strict by default; see {@link LoadStateDictOptions}). */
export async function loadSafetensorsInto(
  module: Module,
  source: SafetensorsSource,
  options: StateDictReadOptions & OpenOptions & LoadStateDictOptions = {},
): Promise<void> {
  module.loadStateDict(await loadSafetensors(source, options), options);
}

/**
 * A module (its `stateDict()`) or a plain state dict as safetensors bytes,
 * byte-identical to Python's `safetensors.serialize` for the same tensors.
 * `metadata` defaults to `{ format: "pt" }` so `safetensors.torch` /
 * `transformers` accept the file without complaint.
 */
export function saveSafetensors(
  moduleOrStateDict: Module | Readonly<Record<string, Tensor>>,
  metadata: Readonly<Record<string, string>> = { format: "pt" },
): Uint8Array {
  const dict = isModule(moduleOrStateDict) ? moduleOrStateDict.stateDict() : moduleOrStateDict;
  const inputs: Record<string, ReturnType<typeof fromTensor>> = {};
  for (const [name, t] of Object.entries(dict)) inputs[name] = fromTensor(t);
  return writeSafetensors(inputs, metadata);
}

function isModule(v: unknown): v is Module {
  return typeof (v as Module).stateDict === "function" && typeof (v as Module).namedParameters === "function";
}
