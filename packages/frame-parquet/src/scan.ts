/**
 * `scanParquet("/events/*.parquet")` — glob/partitioned Parquet scans.
 *
 * DESIGN DECISION (issue #20 leaves this open, "your call, document which
 * you chose and why"): this is an EAGER v1, not a lazy one. `scanParquet`
 * resolves the glob, calls `readParquet` on every match (each pushing its
 * own `columns`/`filter` down to hyparquet, so per-file I/O reduction is
 * still real), and concatenates the results via `Frame.concat` immediately
 * — nothing is deferred to `.collect()`.
 *
 * True laziness — registering multiple files as plan-level sources and only
 * reading the row data a later `.collect()`-equivalent actually needs — DID
 * require a new plan-level extension point in @johnhenry/math-plus-frame-arrow itself
 * (`"lazySource"`, `Frame.fromLazySource()`/`collectAsync()`), tracked as a
 * follow-up and now shipped (issue #32). See `scan-lazy.ts`'s
 * `scanParquetLazy` for that genuinely-lazy counterpart to this function.
 * This eager `scanParquet` stays as-is — a well-scoped, honest v1 and still
 * the simpler choice for a caller who's going to read every column anyway.
 *
 * Uses Node's built-in `fs.promises.glob` (stable-enough since Node 22.0,
 * comfortably covered by this repo's `engines.node: ">=22.12.0"` floor;
 * verified working, no experimental-warning noise, on Node v26.5.0 in this
 * package's own dev environment) rather than adding a new glob-matching
 * dependency, per issue #20's own preference.
 *
 * A real bug found while building this (not hypothetical): apache-arrow
 * 21.2.0's `Table.concat()` produces a Table whose `getChild()` throws
 * ("Vector constructor expects an Array of Data instances") once ANY of the
 * concatenated tables has zero rows — reproducible directly against
 * apache-arrow with no frame-arrow/frame-parquet code involved at all. A
 * `filter` that matches rows in some partition files but not others (a
 * completely ordinary `scanParquet` use case — that's the whole point of
 * partitioned data) routinely produces exactly that mix once each file is
 * read+filtered individually. Filed against frame-arrow as issue #31 and
 * fixed there (`Frame.concat`'s own "concat" execute case now filters
 * zero-row inputs itself), so `scanParquet` just calls `Frame.concat`
 * directly below — no local workaround needed anymore.
 */
import { glob, stat } from "node:fs/promises";
import { Frame } from "@johnhenry/math-plus-frame-arrow";
import { readParquet, type ReadParquetOptions } from "./read.ts";

export type ScanParquetOptions = ReadParquetOptions;

/** Shared by `scanParquet` and `scan-lazy.ts`'s `scanParquetLazy` — resolves
 * a glob pattern to a deterministically-ordered list of matching paths
 * (filesystem readdir order isn't guaranteed, so both scanners need the same
 * explicit sort for reproducible results/tests).
 *
 * A pattern with no glob metacharacters is resolved with `stat` instead of
 * `glob`: Bun 1.2's `fs.promises.glob` returns NOTHING for an absolute
 * literal path (`glob("/abs/part-0.parquet")` -> [], while
 * `glob("/abs/part-*.parquet")` and relative literals work) — verified on
 * Bun 1.2.17, caught by this package's own "single file" tests under
 * `bun test`. Node's glob returns the path itself, which is what this
 * reproduces on both runtimes. */
export async function globSortedParquetPaths(pattern: string): Promise<string[]> {
  // Metacharacters: `*?[]{}` plus extglob openers like `@(`/`!(`/`+(` (a
  // bare `@` is literal — e.g. an npm-scope directory in the path).
  if (!/[*?[\]{}]|[!+@]\(/.test(pattern)) {
    try {
      await stat(pattern);
      return [pattern];
    } catch {
      return [];
    }
  }
  const paths: string[] = [];
  for await (const p of glob(pattern)) paths.push(p);
  paths.sort();
  return paths;
}

export async function scanParquet(pattern: string, options: ScanParquetOptions = {}): Promise<Frame> {
  const paths = await globSortedParquetPaths(pattern);
  if (paths.length === 0) {
    throw new Error(`scanParquet: no files matched pattern "${pattern}"`);
  }
  const frames = await Promise.all(paths.map((p) => readParquet(p, options)));
  return frames.length > 1 ? Frame.concat(frames) : (frames[0] as Frame);
}
