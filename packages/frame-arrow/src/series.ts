/**
 * Series<T> — a single named, typed column. Immutable: every method returns
 * a new Series. Always fully materialized (there is no lazy Series plan;
 * laziness lives at the Frame level — a Series is what you get out of an
 * already-`.collect()`ed Frame).
 *
 * Note: this module and frame.ts import each other (Series.valueCounts()
 * returns a Frame; Frame.getSeries() returns a Series). That's a genuine
 * ESM circular import, safe here because each reference to the other
 * module's export happens inside a method body, not at module-top-level
 * evaluation — by the time either is actually called, both modules have
 * finished initializing.
 */
import { Bool, Field, Table, type Vector } from "apache-arrow";
import { cellAt, columnToArray, timestampDateAt } from "./access.ts";
import {
  arrowTypeFor,
  describeField,
  isNumericDType,
  isTimestampDType,
  type DType,
  type FieldDescriptor,
} from "./dtype.ts";
import { Frame } from "./frame.ts";
import { buildVector } from "./vector-build.ts";

function groupKey(v: unknown): string {
  if (v === null || v === undefined) return " null";
  if (typeof v === "bigint") return `b:${v}`;
  if (typeof v === "object") return `o:${JSON.stringify(v)}`;
  return `${typeof v}:${String(v)}`;
}

function convertScalar(value: unknown, from: DType, to: DType): unknown {
  if (value === null || value === undefined) return null;
  if (to === "int64") {
    return typeof value === "bigint" ? value : BigInt(Math.trunc(value as number));
  }
  if (from === "int64" && isNumericDType(to)) {
    const big = value as bigint;
    if (big > BigInt(Number.MAX_SAFE_INTEGER) || big < BigInt(Number.MIN_SAFE_INTEGER)) {
      throw new Error(
        `Series.cast: int64 value ${big}n is outside Number.MIN_SAFE_INTEGER..Number.MAX_SAFE_INTEGER ` +
          `and cannot be cast to "${to}" without silent precision loss`,
      );
    }
    return Number(big);
  }
  if (isNumericDType(to)) return Number(value);
  if (to === "utf8" || to === "dictionary") return String(value);
  if (to === "bool") return Boolean(value);
  throw new Error(`cannot cast a value of dtype "${from}" to "${to}"`);
}

// cast() is only supported WITHIN a group — numeric<->numeric (bool included, since
// bool<->int is a common, unambiguous cast) or utf8<->dictionary — never across
// groups (e.g. utf8 -> float64 is rejected, not silently coerced via Number("abc") -> NaN).
const NUMERIC_LIKE: ReadonlySet<DType> = new Set([
  "bool",
  "int8",
  "int16",
  "int32",
  "uint8",
  "uint16",
  "uint32",
  "int64",
  "float32",
  "float64",
]);
const STRING_LIKE: ReadonlySet<DType> = new Set(["utf8", "dictionary"]);

export class Series<T = unknown> {
  readonly name: string;
  readonly descriptor: FieldDescriptor;
  private readonly vector: Vector;

  constructor(name: string, vector: Vector, nullable = true) {
    this.name = name;
    this.vector = vector;
    this.descriptor = describeField(new Field(name, vector.type, nullable));
  }

  get dtype(): DType {
    return this.descriptor.dtype;
  }

  get length(): number {
    return this.vector.length;
  }

  /** Escape hatch to the underlying apache-arrow Vector. */
  toVector(): Vector {
    return this.vector;
  }

  get(index: number): T | null {
    return cellAt(this.vector, index, this.dtype) as T | null;
  }

  toArray(): (T | null)[] {
    return columnToArray(this.vector, this.dtype) as (T | null)[];
  }

  /** ms-truncated Date view of a timestamp column — documented lossy convenience;
   * see access.ts's timestampDateAt / docs/spikes/arrow-parity.md sharp edge #3. */
  toDates(): (Date | null)[] {
    if (!isTimestampDType(this.dtype)) {
      throw new Error(`Series.toDates(): "${this.name}" is dtype "${this.dtype}", not a timestamp column`);
    }
    const out: (Date | null)[] = new Array(this.length);
    for (let i = 0; i < this.length; i++) out[i] = timestampDateAt(this.vector, i);
    return out;
  }

  isNull(): Series<boolean> {
    const out: boolean[] = new Array(this.length);
    for (let i = 0; i < this.length; i++) out[i] = !this.vector.isValid(i);
    return new Series<boolean>(this.name, buildVector(out, new Bool()), false);
  }

  fillNull(value: T): Series<T> {
    const values = this.toArray().map((v) => (v === null ? value : v));
    return new Series<T>(this.name, buildVector(values, this.vector.type), this.descriptor.nullable);
  }

  unique(): Series<T> {
    const values = this.toArray();
    const seen = new Set<string>();
    const out: (T | null)[] = [];
    for (const v of values) {
      const key = groupKey(v);
      if (!seen.has(key)) {
        seen.add(key);
        out.push(v);
      }
    }
    return new Series<T>(this.name, buildVector(out, this.vector.type), this.descriptor.nullable);
  }

  /** Returns a two-column Frame: `value` (this Series' dtype) and `count` (int64), most frequent first. */
  valueCounts(): Frame {
    const values = this.toArray();
    const counts = new Map<string, { value: unknown; count: number }>();
    for (const v of values) {
      const key = groupKey(v);
      const entry = counts.get(key);
      if (entry) entry.count++;
      else counts.set(key, { value: v, count: 1 });
    }
    const sorted = [...counts.values()].sort((a, b) => b.count - a.count);
    const valueVec = buildVector(
      sorted.map((e) => e.value),
      this.vector.type,
    );
    const countVec = buildVector(
      sorted.map((e) => BigInt(e.count)),
      arrowTypeFor("int64"),
    );
    return Frame.fromArrow(new Table({ value: valueVec, count: countVec }));
  }

  cast(target: DType): Series {
    if (target === this.dtype) return this as Series;
    const sameGroup =
      (NUMERIC_LIKE.has(this.dtype) && NUMERIC_LIKE.has(target)) ||
      (STRING_LIKE.has(this.dtype) && STRING_LIKE.has(target));
    if (!sameGroup) {
      throw new Error(
        `Series.cast: unsupported cast from "${this.dtype}" to "${target}" in v1 ` +
          `(only numeric<->numeric and utf8<->dictionary casts are supported — timestamp/list/struct ` +
          `casts, and casts across the numeric/string boundary, are not implemented)`,
      );
    }
    const values = this.toArray().map((v) => convertScalar(v, this.dtype, target));
    return new Series(this.name, buildVector(values, arrowTypeFor(target)), this.descriptor.nullable);
  }

  /**
   * Lazy dynamic import of @johnhenry/math-plus-tensor-core — see package.json:
   * @johnhenry/math-plus-tensor-core is an OPTIONAL peerDependency, never a regular
   * dependency, so this package's static import graph has zero edge to the
   * tensor track. Throws for dtypes Tensor can't represent (utf8,
   * dictionary, timestamp, list, struct) — no silent misconversion.
   */
  async toTensor(): Promise<unknown> {
    const { seriesToTensor } = await import("./tensor.ts");
    return seriesToTensor(this);
  }
}
