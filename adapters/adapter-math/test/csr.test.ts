import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Graph } from "@johnhenry/math";
import { toCSR, toDense } from "../src/index.ts";

function rowSlice(csr: ReturnType<typeof toCSR>, row: number): Array<[number, number]> {
  const start = csr.rowPointers[row] as number;
  const end = csr.rowPointers[row + 1] as number;
  const out: Array<[number, number]> = [];
  for (let k = start; k < end; k++) out.push([csr.columnIndices[k] as number, csr.values[k] as number]);
  return out;
}

test("toCSR: directed graph — each edge appears once, in the FROM vertex's row only", () => {
  const g = new Graph<string>(true);
  g.addEdge("a", "b", 1).addEdge("a", "c", 2).addEdge("b", "c", 3);
  const csr = toCSR(g);
  assert.deepEqual(csr.order, ["a", "b", "c"]);
  assert.deepEqual(rowSlice(csr, 0), [
    [1, 1],
    [2, 2],
  ]); // a -> b(1), a -> c(2)
  assert.deepEqual(rowSlice(csr, 1), [[2, 3]]); // b -> c(3)
  assert.deepEqual(rowSlice(csr, 2), []); // c has no outgoing edges
  assert.equal(csr.rowPointers[csr.rowPointers.length - 1], 3); // 3 total stored edges
});

test("toCSR: undirected graph — each edge appears in BOTH endpoints' rows (mirrors addEdge's own symmetric storage)", () => {
  const g = new Graph<string>(false);
  g.addEdge("a", "b", 5);
  const csr = toCSR(g);
  assert.deepEqual(csr.order, ["a", "b"]);
  assert.deepEqual(rowSlice(csr, 0), [[1, 5]]); // a -> b(5)
  assert.deepEqual(rowSlice(csr, 1), [[0, 5]]); // b -> a(5)
  assert.equal(csr.rowPointers[csr.rowPointers.length - 1], 2); // stored once per direction, not deduped away
});

test("toCSR: an undirected self-loop is stored ONCE, not doubled", () => {
  const g = new Graph<string>(false);
  g.addEdge("a", "a", 7);
  const csr = toCSR(g);
  assert.deepEqual(rowSlice(csr, 0), [[0, 7]]);
});

test("toCSR: an explicit zero-weight edge is stored (not conflated with 'absent')", () => {
  const g = new Graph<string>(true);
  g.addEdge("a", "b", 0);
  const csr = toCSR(g);
  assert.equal(csr.rowPointers[csr.rowPointers.length - 1], 1); // one REAL edge, weight 0
  assert.deepEqual(rowSlice(csr, 0), [[1, 0]]);
});

test("toCSR: an isolated vertex (added but no edges) gets an empty row, not a missing one", () => {
  const g = new Graph<string>(true);
  g.addVertex("isolated");
  g.addEdge("a", "b", 1);
  const csr = toCSR(g);
  assert.deepEqual(csr.order, ["isolated", "a", "b"]);
  assert.deepEqual(rowSlice(csr, 0), []);
});

test("toCSR: a graph with no vertices produces a valid empty CSR", () => {
  const g = new Graph<string>();
  const csr = toCSR(g);
  assert.deepEqual(csr.order, []);
  assert.deepEqual([...csr.rowPointers], [0]);
  assert.equal(csr.columnIndices.length, 0);
  assert.equal(csr.values.length, 0);
});

test("toDense({missing: 'zero'}): standard sparse convention — absent pairs are 0, no diagonal special-case", () => {
  const g = new Graph<string>(true);
  g.addEdge("a", "b", 1).addEdge("b", "a", 2);
  const csr = toCSR(g);
  const dense = toDense(csr, { missing: "zero" });
  assert.deepEqual(dense, [
    [0, 1],
    [2, 0],
  ]);
});

test("toDense({missing: 'infinity'}) matches Graph.toAdjacencyMatrix()'s own convention exactly (0 diagonal, Infinity elsewhere)", () => {
  const g = new Graph<string>(false);
  g.addEdge("a", "b", 4).addEdge("b", "c", 6);
  g.addVertex("d"); // isolated

  const { matrix: expected, order: adjOrder } = g.toAdjacencyMatrix();
  const csr = toCSR(g);
  assert.deepEqual(csr.order, adjOrder); // both derive from vertices(), same order

  const dense = toDense(csr, { missing: "infinity" });
  assert.deepEqual(dense, expected);
});

test("toDense: a real self-loop edge overrides the default 0 diagonal under missing:'infinity'", () => {
  const g = new Graph<string>(false);
  g.addEdge("a", "a", 9);
  const csr = toCSR(g);
  const dense = toDense(csr, { missing: "infinity" });
  assert.deepEqual(dense, [[9]]);
});

test("toCSR then toDense round-trips a weighted directed graph exactly", () => {
  const g = new Graph<string>(true);
  g.addEdge("x", "y", 1.5).addEdge("y", "z", 2.5).addEdge("z", "x", 3.5).addEdge("x", "z", 0);
  const csr = toCSR(g);
  const dense = toDense(csr, { missing: "zero" });
  assert.deepEqual(dense, [
    [0, 1.5, 0], // x->y=1.5, x->z=0 (explicit zero-weight edge, distinct from "no edge")
    [0, 0, 2.5], // y->z=2.5
    [3.5, 0, 0], // z->x=3.5
  ]);
});
