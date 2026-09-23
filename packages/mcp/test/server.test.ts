/**
 * Exercises the real MCP wire path (issue #45): a genuine SDK Client talks
 * to the server over InMemoryTransport's linked pair -- schemas validate,
 * results serialize, and errors surface as isError tool results (never
 * protocol-level exceptions), exactly as an agent host would see them.
 */
import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { buildServer } from "../src/server.ts";

interface TextResult {
  content: Array<{ type: string; text: string }>;
  isError?: boolean;
}

async function connectedClient(): Promise<Client> {
  const server = buildServer();
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  const client = new Client({ name: "test-client", version: "0.0.0" });
  await client.connect(clientTransport);
  return client;
}

async function call(client: Client, name: string, args: Record<string, unknown>): Promise<{ raw: TextResult; json: any }> {
  const raw = (await client.callTool({ name, arguments: args })) as TextResult;
  const text = raw.content[0]?.text ?? "";
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    json = text; // error results are plain "Error: ..." strings
  }
  return { raw, json };
}

test("lists the full v1 tool set", async () => {
  const client = await connectedClient();
  const { tools } = await client.listTools();
  const names = tools.map((t) => t.name).sort();
  assert.deepEqual(names, [
    "linalg_solve",
    "stats_summary",
    "symbolic_differentiate",
    "symbolic_evaluate",
    "symbolic_integrate",
    "symbolic_parse",
    "symbolic_simplify",
    "symbolic_solve",
    "tensor_pipeline",
  ]);
  await client.close();
});

test("symbolic_parse returns normalized text, latex, and free variables", async () => {
  const client = await connectedClient();
  const { raw, json } = await call(client, "symbolic_parse", { expression: "x^2 + sin(y)*3" });
  assert.ok(!raw.isError);
  assert.equal(typeof json.text, "string");
  assert.equal(typeof json.latex, "string");
  assert.deepEqual([...json.freeVariables].sort(), ["x", "y"]);
  await client.close();
});

test("symbolic_differentiate: d/dx x^3 = 3x^2 (verified by evaluating both at x=2)", async () => {
  const client = await connectedClient();
  const { json } = await call(client, "symbolic_differentiate", { expression: "x^3", variable: "x" });
  const { json: value } = await call(client, "symbolic_evaluate", { expression: json.text, variables: { x: 2 } });
  assert.equal(value.value, 12);
  await client.close();
});

test("symbolic_simplify: (x^2 - 1)/(x - 1) style algebra returns a valid expression", async () => {
  const client = await connectedClient();
  const { raw, json } = await call(client, "symbolic_simplify", { expression: "x + x + 2*x" });
  assert.ok(!raw.isError);
  const { json: value } = await call(client, "symbolic_evaluate", { expression: json.text, variables: { x: 5 } });
  assert.equal(value.value, 20);
  await client.close();
});

test("symbolic_integrate: indefinite integral of 2x evaluates like x^2; definite integral returns the number", async () => {
  const client = await connectedClient();
  const { json: indef } = await call(client, "symbolic_integrate", { expression: "2*x" });
  const { json: at3 } = await call(client, "symbolic_evaluate", { expression: indef.text, variables: { x: 3 } });
  assert.equal(at3.value, 9);
  const { json: def } = await call(client, "symbolic_integrate", { expression: "2*x", lower: 0, upper: 3 });
  assert.ok(Math.abs(def.value - 9) < 1e-9);
  await client.close();
});

test("symbolic_solve accepts 'lhs = rhs' equations and the zero-equals form identically", async () => {
  const client = await connectedClient();
  const { json: eq } = await call(client, "symbolic_solve", { expression: "x^2 = 4" });
  const { json: zero } = await call(client, "symbolic_solve", { expression: "x^2 - 4" });
  const values = (s: any) => s.solutions.map((x: { text: string }) => Number(x.text)).sort((a: number, b: number) => a - b);
  assert.deepEqual(values(eq), [-2, 2]);
  assert.deepEqual(values(zero), [-2, 2]);
  await client.close();
});

test("linalg_solve solves a hand-checked 2x2 system", async () => {
  const client = await connectedClient();
  // 2x + y = 5, 4x + 3y = 9 -> x = 3, y = -1
  const { json } = await call(client, "linalg_solve", { a: [[2, 1], [4, 3]], b: [5, 9] });
  assert.ok(Math.abs(json.x[0] - 3) < 1e-12 && Math.abs(json.x[1] - -1) < 1e-12);
  await client.close();
});

test("tensor_pipeline runs a closed-op chain and reports shape", async () => {
  const client = await connectedClient();
  const { json } = await call(client, "tensor_pipeline", {
    data: [
      [1, 2],
      [3, 4],
    ],
    ops: [{ op: "transpose" }, { op: "mulScalar", scalar: 10 }, { op: "sum", axis: 0 }],
  });
  // transpose -> [[1,3],[2,4]]; *10 -> [[10,30],[20,40]]; sum axis 0 -> [30, 70]
  assert.deepEqual(json.result, [30, 70]);
  await client.close();
});

test("tensor_pipeline rejects unknown ops with the closed-set message, as an isError result (not a protocol error)", async () => {
  const client = await connectedClient();
  const { raw } = await call(client, "tensor_pipeline", { data: [[1]], ops: [{ op: "evalJs" }] });
  assert.equal(raw.isError, true);
  assert.match(raw.content[0]?.text ?? "", /unknown op "evalJs"/);
  assert.match(raw.content[0]?.text ?? "", /closed op set/);
  await client.close();
});

test("oversized inputs are rejected fast by the element cap", async () => {
  const client = await connectedClient();
  const { raw } = await call(client, "tensor_pipeline", {
    data: [[2]],
    ops: [{ op: "reshape", shape: [1, 1] }],
  });
  assert.ok(!raw.isError); // small is fine
  const { raw: big } = await call(client, "linalg_solve", {
    a: [Array.from({ length: 1_000_001 }, () => 1)],
    b: [1],
  });
  assert.equal(big.isError, true);
  assert.match(big.content[0]?.text ?? "", /element cap/);
  await client.close();
});

test("linalg_solve's b vector is rejected fast by the element cap too (not just a)", async () => {
  const client = await connectedClient();
  const { raw } = await call(client, "linalg_solve", {
    a: [[1]],
    b: Array.from({ length: 1_000_001 }, () => 1),
  });
  assert.equal(raw.isError, true);
  assert.match(raw.content[0]?.text ?? "", /element cap/);
  await client.close();
});

test("malformed expressions surface as isError results with the parser's message", async () => {
  const client = await connectedClient();
  const { raw } = await call(client, "symbolic_simplify", { expression: "x +* 2)(" });
  assert.equal(raw.isError, true);
  assert.match(raw.content[0]?.text ?? "", /^Error: /);
  await client.close();
});

test("stats_summary computes hand-checked descriptive statistics", async () => {
  const client = await connectedClient();
  const { json } = await call(client, "stats_summary", { values: [2, 4, 4, 4, 5, 5, 7, 9] });
  assert.equal(json.count, 8);
  assert.equal(json.mean, 5);
  assert.equal(json.populationStd, 2); // the classic textbook example
  assert.equal(json.min, 2);
  assert.equal(json.max, 9);
  assert.equal(json.median, 4.5);
  await client.close();
});
