/**
 * Shared test helpers: hand-built files for header-validation cases, and
 * the Python `safetensors` oracle (skip-don't-fail, same resolution as the
 * rest of Math Plus: $MATH_PLUS_ORACLE_PYTHON, else `python3` on PATH).
 */
import { execFileSync } from "node:child_process";

/** A raw file: u64 LE header length + header text (+ optional space padding) + data. */
export function craft(header: string | object, data: Uint8Array = new Uint8Array(0), pad = false): Uint8Array {
  let text = typeof header === "string" ? header : JSON.stringify(header);
  if (pad) text += " ".repeat((8 - (new TextEncoder().encode(text).byteLength % 8)) % 8);
  const json = new TextEncoder().encode(text);
  const out = new Uint8Array(8 + json.byteLength + data.byteLength);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.byteLength), true);
  out.set(json, 8);
  out.set(data, 8 + json.byteLength);
  return out;
}

export const ORACLE_SCRIPT = new URL("../scripts/safetensors_oracle.py", import.meta.url).pathname;

function findOraclePython(): string | undefined {
  const candidates = [process.env.MATH_PLUS_ORACLE_PYTHON, "python3"].filter((c): c is string => Boolean(c));
  for (const candidate of candidates) {
    try {
      execFileSync(candidate, ["-c", "import numpy, safetensors"], { stdio: "ignore" });
      return candidate;
    } catch {
      // try next
    }
  }
  return undefined;
}

export const PYTHON = findOraclePython();
export const oracleSkip: string | false = PYTHON ? false : "no python with numpy + safetensors found (set MATH_PLUS_ORACLE_PYTHON)";

export function runOracle(...args: string[]): unknown {
  const out = execFileSync(PYTHON as string, [ORACLE_SCRIPT, ...args], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  return JSON.parse(out);
}

export function fromB64(s: string): Uint8Array {
  return new Uint8Array(Buffer.from(s, "base64"));
}

export function toB64(b: Uint8Array): string {
  return Buffer.from(b.buffer, b.byteOffset, b.byteLength).toString("base64");
}

/** Bitwise equality of two float32 arrays (NaN payloads compared as "both NaN"). */
export function sameF32(actual: Float32Array, expected: Float32Array): string | undefined {
  if (actual.length !== expected.length) return `length ${actual.length} vs ${expected.length}`;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i] as number;
    const e = expected[i] as number;
    if (!(Object.is(a, e) || (Number.isNaN(a) && Number.isNaN(e)))) return `element ${i}: ${a} vs ${e}`;
  }
  return undefined;
}
