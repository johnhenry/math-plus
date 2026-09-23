/**
 * Seeded PRNG for @johnhenry/math-plus-tensor-core (issue #5).
 *
 * PCG32 (XSH-RR variant) — see https://www.pcg-random.org/. Chosen over
 * Math.random() specifically because it's a documented, seedable algorithm:
 * the same seed always produces the same sequence, which is the entire
 * point (reproducible experiments, deterministic tests). No shared global
 * generator state — every entry point takes an explicit `{ rng }`, so two
 * tests running in any order never interfere with each other. `random.seed(n)`
 * returns a fresh `Rng`; omit `{ rng }` and you get a non-deterministic one.
 */
import { allocate, encodeHalf, isBigIntDType, isHalfDType, type AnyTypedArray, type DType } from "./dtype.ts";

const MULT = 6364136223846793005n;
const MASK64 = (1n << 64n) - 1n;

export class Rng {
  #state = 0n;
  #inc: bigint;

  constructor(seedValue: number | bigint = 0n, sequence: number | bigint = 1n) {
    this.#inc = ((BigInt(sequence) << 1n) | 1n) & MASK64;
    this.#step();
    this.#state = (this.#state + (BigInt(seedValue) & MASK64)) & MASK64;
    this.#step();
  }

  #step(): bigint {
    const oldState = this.#state;
    this.#state = (oldState * MULT + this.#inc) & MASK64;
    return oldState;
  }

  /** Next value in [0, 2^32). */
  nextUint32(): number {
    const oldState = this.#step();
    const xorshifted = BigInt.asUintN(32, ((oldState >> 18n) ^ oldState) >> 27n);
    const rot = Number(oldState >> 59n); // top 5 bits of a 64-bit state: 0..31
    const rotated =
      rot === 0
        ? xorshifted
        : BigInt.asUintN(
            32,
            (xorshifted >> BigInt(rot)) | (xorshifted << BigInt(32 - rot)),
          );
    return Number(rotated);
  }

  /** Uniform float in [0, 1). */
  nextFloat(): number {
    return this.nextUint32() / 4294967296; // 2^32
  }

  /** Uniform integer in [0, bound) without modulo bias (rejection sampling). */
  nextBelow(bound: number): number {
    if (!Number.isInteger(bound) || bound <= 0) {
      throw new RangeError(`nextBelow: bound must be a positive integer, got ${bound}`);
    }
    const limit = Math.floor(4294967296 / bound) * bound;
    let x: number;
    do {
      x = this.nextUint32();
    } while (x >= limit);
    return x % bound;
  }
}

/** A fresh, seeded, reproducible generator — pass the result as `{ rng }`. */
export function seed(value: number | bigint): Rng {
  return new Rng(value);
}

/** Non-deterministic default: used only when a caller omits `{ rng }`. */
function defaultRng(): Rng {
  return new Rng(Date.now() ^ Math.floor(Math.random() * 0xffffffff));
}

export interface RandomOptions {
  dtype?: DType;
  rng?: Rng;
}

function totalSize(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Fill a freshly allocated array of `dtype` from `sample(rng)`, called once per element. */
export function fillFrom(
  shape: readonly number[],
  dtype: DType,
  rng: Rng | undefined,
  sample: (rng: Rng) => number,
): AnyTypedArray {
  const active = rng ?? defaultRng();
  const data = allocate(dtype, totalSize(shape));
  const big = isBigIntDType(dtype);
  const half = isHalfDType(dtype) ? dtype : undefined;
  for (let i = 0; i < data.length; i++) {
    const v = sample(active);
    data[i] = (big ? BigInt(Math.trunc(v)) : half ? encodeHalf(half, v) : v) as never;
  }
  return data;
}

export function uniformSample(rng: Rng, min: number, max: number): number {
  return min + rng.nextFloat() * (max - min);
}

export function normalSample(rng: Rng, mean: number, std: number): number {
  // Box-Muller. Draws 2 uniforms per call (only the cosine branch is used) --
  // simpler and less error-prone than caching a spare value across calls,
  // which would need to live on Rng and complicate its contract.
  let u1 = rng.nextFloat();
  if (u1 === 0) u1 = Number.MIN_VALUE; // guard log(0)
  const u2 = rng.nextFloat();
  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z0 * std;
}

export function randintSample(rng: Rng, low: number, high: number): number {
  if (!Number.isInteger(low) || !Number.isInteger(high) || high <= low) {
    throw new RangeError(`randint: expects integers with high > low, got [${low}, ${high})`);
  }
  return low + rng.nextBelow(high - low);
}
