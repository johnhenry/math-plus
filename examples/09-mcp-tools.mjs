// mcp: the math-plus MCP server, driven programmatically over an in-memory
// transport — the same wire path an agent host uses (`npx math-plus-mcp`
// for the stdio binary). Symbolic CAS tools + guarded numeric tools; no
// arbitrary code execution, every limit a hard constant.
//
// Run: npm install && npm run build && node examples/09-mcp-tools.mjs
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { buildServer } from "@johnhenry/math-plus-mcp";

const server = buildServer();
const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
await server.connect(serverTransport);
const client = new Client({ name: "example-client", version: "0.0.0" });
await client.connect(clientTransport);

const call = async (name, args) => {
  const raw = await client.callTool({ name, arguments: args });
  const text = raw.content[0]?.text ?? "";
  try { return JSON.parse(text); } catch { return text; } // errors are plain strings
};

console.log((await client.listTools()).tools.map((t) => t.name).sort());
// linalg_solve, stats_summary, symbolic_* (6 of them), tensor_pipeline

// Symbolic CAS
console.log(await call("symbolic_differentiate", { expression: "x^3" })); // { text: '3*x^2', ... }
console.log(await call("symbolic_solve", { expression: "x^2 - 5*x + 6" }));
console.log(await call("symbolic_integrate", { expression: "x^2", lower: 0, upper: 2 })); // { value: 2.666... }

// Guarded numerics: closed op table, 1e6-element cap, max 16 pipeline steps
console.log(await call("linalg_solve", { a: [[2, 1], [4, 3]], b: [5, 9] })); // { x: [3, -1] }
console.log(await call("tensor_pipeline", {
  data: [[1, 2], [3, 4]],
  ops: [{ op: "transpose" }, { op: "mulScalar", scalar: 10 }, { op: "sum", axis: 0 }],
})); // { result: [30, 70], ... }

// Errors surface as isError tool results, never protocol exceptions
console.log(await call("tensor_pipeline", { data: [[1]], ops: [{ op: "nope" }] }));

await client.close();
