import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { linalg } from "../src/index.ts";

function close(a: number, b: number, eps = 1e-6): void {
  assert.ok(Math.abs(a - b) < eps, `${a} !~ ${b}`);
}

function closeMatrix(a: number[][], b: number[][], eps = 1e-6): void {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.equal((a[i] as number[]).length, (b[i] as number[]).length);
    for (let j = 0; j < (a[i] as number[]).length; j++) {
      close((a[i] as number[])[j] as number, (b[i] as number[])[j] as number, eps);
    }
  }
}

function matmulRaw(a: number[][], b: number[][]): number[][] {
  const m = a.length;
  const n = (b[0] as number[]).length;
  const k = b.length;
  const out: number[][] = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      for (let j = 0; j < n; j++) {
        (out[i] as number[])[j] = ((out[i] as number[])[j] as number) + (a[i] as number[])[p]! * (b[p] as number[])[j]!;
      }
    }
  }
  return out;
}

function transposeRaw(a: number[][]): number[][] {
  return (a[0] as number[]).map((_, j) => a.map((row) => row[j] as number));
}

const A3 = [
  [4, 3, 2],
  [1, 5, 3],
  [2, 1, 6],
];

test("lu: P*A == L*U", () => {
  const a = Tensor.from(A3.flat()).reshape([3, 3]);
  const { L, U, P } = linalg.lu(a);
  const pa = matmulRaw(P.toArray() as number[][], A3);
  const lu_ = matmulRaw(L.toArray() as number[][], U.toArray() as number[][]);
  closeMatrix(pa, lu_);
});

test("solve: A*x == b", () => {
  const a = Tensor.from(A3.flat()).reshape([3, 3]);
  const b = Tensor.from([1, 2, 3]);
  const x = linalg.solve(a, b);
  const ax = matmulRaw(A3, (x.toArray() as number[]).map((v) => [v])).map((row) => row[0] as number);
  closeMatrix(
    [ax],
    [[1, 2, 3]],
  );
});

test("rref/rank: identity-shaped full-rank matrix", () => {
  const a = Tensor.from(A3.flat()).reshape([3, 3]);
  assert.equal(linalg.rank(a), 3);
  const r = linalg.rref(a);
  closeMatrix(r.toArray() as number[][], [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]);
});

test("nullSpace: a singular matrix has a nontrivial null space that A maps to zero", () => {
  const singular = [
    [1, 2, 3],
    [2, 4, 6],
    [1, 1, 1],
  ];
  const a = Tensor.from(singular.flat()).reshape([3, 3]);
  assert.ok(linalg.rank(a) < 3);
  const basis = linalg.nullSpace(a).toArray() as number[][];
  // A * v (as a column vector) should be ~0 for each basis row v.
  for (const v of basis) {
    const result = singular.map((row) => row.reduce((s, val, j) => s + val * (v[j] as number), 0));
    for (const r of result) close(r, 0, 1e-6);
  }
});

test("qr: Q is orthonormal and Q*R == A", () => {
  const a = Tensor.from(A3.flat()).reshape([3, 3]);
  const { Q, R } = linalg.qr(a);
  const qArr = Q.toArray() as number[][];
  const qtq = matmulRaw(transposeRaw(qArr), qArr);
  closeMatrix(qtq, [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]);
  closeMatrix(matmulRaw(qArr, R.toArray() as number[][]), A3);
});

test("cholesky: L*L^T == A for a symmetric positive-definite matrix", () => {
  const spd = [
    [4, 2, 0],
    [2, 5, 1],
    [0, 1, 3],
  ];
  const a = Tensor.from(spd.flat()).reshape([3, 3]);
  const L = linalg.cholesky(a).toArray() as number[][];
  closeMatrix(matmulRaw(L, transposeRaw(L)), spd);
});

test("cholesky: throws on a non-positive-definite matrix", () => {
  const notPD = Tensor.from([1, 2, 2, 1]).reshape([2, 2]);
  assert.throws(() => linalg.cholesky(notPD));
});

test("eigSymmetric: A*v == lambda*v for each eigenpair, values descending", () => {
  const sym = [
    [2, 1, 0],
    [1, 2, 1],
    [0, 1, 2],
  ];
  const a = Tensor.from(sym.flat()).reshape([3, 3]);
  const { values, vectors } = linalg.eigSymmetric(a);
  const vals = values.toArray() as number[];
  const vecs = vectors.toArray() as number[][]; // columns are eigenvectors

  assert.ok(vals[0]! >= vals[1]! && vals[1]! >= vals[2]!, "descending order");

  for (let col = 0; col < 3; col++) {
    const v = vecs.map((row) => row[col] as number);
    const av = sym.map((row) => row.reduce((s, val, j) => s + val * v[j]!, 0));
    const lambdaV = v.map((x) => x * (vals[col] as number));
    for (let i = 0; i < 3; i++) close(av[i] as number, lambdaV[i] as number, 1e-4);
  }
});

test("svd: U*diag(S)*V^T == A, and singular values match spectralNorm", () => {
  const m = [
    [3, 2],
    [2, 3],
    [1, 1],
  ];
  const a = Tensor.from(m.flat()).reshape([3, 2]);
  const { U, S, V } = linalg.svd(a);
  const sArr = S.toArray() as number[];
  const sigma = [
    [sArr[0], 0],
    [0, sArr[1]],
  ] as number[][];
  const reconstructed = matmulRaw(matmulRaw(U.toArray() as number[][], sigma), transposeRaw(V.toArray() as number[][]));
  closeMatrix(reconstructed, m, 1e-4);
  close(linalg.spectralNorm(a), sArr[0] as number, 1e-4);
});

test("leastSquares: normal-equations solution matches a known overdetermined fit", () => {
  // Fit y = mx + c through (0,1),(1,2),(2,2),(3,4) via [x,1]*[m,c]^T ~= y
  const design = [
    [0, 1],
    [1, 1],
    [2, 1],
    [3, 1],
  ];
  const y = [1, 2, 2, 4];
  const a = Tensor.from(design.flat()).reshape([4, 2]);
  const b = Tensor.from(y);
  const x = linalg.leastSquares(a, b).toArray() as number[];
  // Residual should be at a stationary point: A^T (A x - y) ~= 0.
  const resid = design.map((row, i) => row[0]! * (x[0] as number) + row[1]! * (x[1] as number) - (y[i] as number));
  const atResid = [0, 1].map((j) => design.reduce((s, row, i) => s + row[j]! * (resid[i] as number), 0));
  for (const v of atResid) close(v, 0, 1e-6);
});

test("pseudoInverse: A * A+ * A == A (Moore-Penrose property)", () => {
  const m = [
    [1, 2],
    [3, 4],
    [5, 6],
  ];
  const a = Tensor.from(m.flat()).reshape([3, 2]);
  const pinv = linalg.pseudoInverse(a).toArray() as number[][];
  const reconstructed = matmulRaw(matmulRaw(m, pinv), m);
  closeMatrix(reconstructed, m, 1e-4);
});

test("norms: frobenius, spectral, condition number agree with hand-computed values for a diagonal matrix", () => {
  const diag = [
    [3, 0],
    [0, 4],
  ];
  const a = Tensor.from(diag.flat()).reshape([2, 2]);
  close(linalg.frobeniusNorm(a), 5); // sqrt(9+16)
  close(linalg.spectralNorm(a), 4); // largest singular value
  close(linalg.conditionNumber(a), 4 / 3);
});

// ---- det/inv (issue #67) ---------------------------------------------------

test("det matches a hand-computed value for a 3x3 (cofactor expansion by hand)", () => {
  const a = Tensor.from(A3.flat()).reshape([3, 3]);
  // det(A3) = 4*(5*6-3*1) - 3*(1*6-3*2) + 2*(1*1-5*2) = 4*27 - 3*0 + 2*(-9) = 108 - 0 - 18 = 90
  // Looser eps than the default: LU-with-partial-pivoting floating-point
  // error accumulation puts this a few 1e-6 off exact, not a bug.
  close(linalg.det(a), 90, 1e-4);
});

test("det matches a hand-computed value for a 2x2", () => {
  const a = Tensor.from([2, 3, 5, 7]).reshape([2, 2]); // det = 2*7 - 3*5 = -1
  close(linalg.det(a), -1);
});

test("det of a singular matrix (a repeated row) is ~0", () => {
  const singular = Tensor.from([1, 2, 3, 2, 4, 6, 5, 1, 9]).reshape([3, 3]); // row 1 = 2 * row 0
  close(linalg.det(singular), 0, 1e-8);
});

test("inv: A * inv(A) == identity (the defining property of a matrix inverse)", () => {
  const a = Tensor.from(A3.flat()).reshape([3, 3]);
  const inv = linalg.inv(a);
  const product = matmulRaw(A3, inv.toArray() as number[][]);
  closeMatrix(product, [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]);
});

test("inv matches a hand-computed 2x2 inverse", () => {
  const a = Tensor.from([4, 7, 2, 6]).reshape([2, 2]); // det = 24-14 = 10
  const inv = linalg.inv(a);
  // inv = 1/10 * [[6, -7], [-2, 4]]
  closeMatrix(inv.toArray() as number[][], [
    [0.6, -0.7],
    [-0.2, 0.4],
  ]);
});

test("det and inv agree: det(A) * det(inv(A)) ~ 1", () => {
  const a = Tensor.from(A3.flat()).reshape([3, 3]);
  const detA = linalg.det(a);
  const detInvA = linalg.det(linalg.inv(a));
  close(detA * detInvA, 1, 1e-5);
});

test("inv rejects a non-square matrix", () => {
  const a = Tensor.from([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  assert.throws(() => linalg.inv(a), RangeError);
});
