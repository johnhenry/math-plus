import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import { Distributions, SpecialFunctions, Statistics, Vector } from "@johnhenry/math";
import {
  correlation,
  Distributions as BridgedDistributions,
  linearRegression,
  mean,
  median,
  percentile,
  SpecialFunctions as BridgedSpecialFunctions,
  standardDeviation,
  variance,
} from "../src/index.ts";

test("SpecialFunctions is re-exported verbatim (same class, not a copy)", () => {
  assert.equal(BridgedSpecialFunctions, SpecialFunctions);
});

test("Distributions is re-exported verbatim (same class, not a copy)", () => {
  assert.equal(BridgedDistributions, Distributions);
});

test("gamma: known reference values", () => {
  assert.ok(Math.abs(SpecialFunctions.gamma(0.5) - Math.sqrt(Math.PI)) < 1e-9);
  assert.ok(Math.abs(SpecialFunctions.gamma(5) - 24) < 1e-6); // 4! = 24
  assert.ok(Math.abs(SpecialFunctions.gamma(1) - 1) < 1e-9);
});

test("beta(a, b) = gamma(a)*gamma(b)/gamma(a+b), self-consistency check", () => {
  const a = 2.5;
  const b = 3.5;
  const expected = (SpecialFunctions.gamma(a) * SpecialFunctions.gamma(b)) / SpecialFunctions.gamma(a + b);
  assert.ok(Math.abs(SpecialFunctions.beta(a, b) - expected) < 1e-6);
});

test("mean/variance/standardDeviation accept a plain Float64Array (bridged from @johnhenry/math's Vector-only Statistics.ts)", () => {
  const data = new Float64Array([2, 4, 4, 4, 5, 5, 7, 9]);
  assert.ok(Math.abs(mean(data) - 5) < 1e-9);
  // Population variance for this classic example is 4, sample variance is 4 * 8/7.
  assert.ok(Math.abs(variance(data) - (4 * 8) / 7) < 1e-9);
  assert.ok(Math.abs(standardDeviation(data) - Math.sqrt((4 * 8) / 7)) < 1e-9);
});

test("mean/variance also accept a plain number[]", () => {
  assert.equal(mean([1, 2, 3, 4, 5]), 3);
});

test("mean correctly handles a single-element input (the Vector single-number constructor gotcha this bridge avoids)", () => {
  assert.equal(mean([42]), 42);
  assert.equal(mean(new Float64Array([42])), 42);
});

test("median/percentile match @johnhenry/math's own values directly (percentile is a 0-1 fraction, matching its convention exactly)", () => {
  const data = [1, 2, 3, 4, 5];
  const reference = Vector.fromArray(data);
  assert.equal(median(data), Statistics.median(reference));
  assert.equal(percentile(data, 0.5), Statistics.percentile(reference, 0.5));
  assert.equal(percentile(data, 0.25), Statistics.percentile(reference, 0.25));
  assert.equal(percentile(data, 0.75), Statistics.percentile(reference, 0.75));
});

test("correlation: perfectly correlated data gives 1", () => {
  const x = [1, 2, 3, 4, 5];
  const y = [2, 4, 6, 8, 10];
  assert.ok(Math.abs(correlation(x, y) - 1) < 1e-9);
});

test("linearRegression: recovers a known y = 2x + 1 line exactly", () => {
  const x = [0, 1, 2, 3, 4];
  const y = x.map((v) => 2 * v + 1);
  const fit = linearRegression(x, y);
  assert.ok(Math.abs(fit.slope - 2) < 1e-9);
  assert.ok(Math.abs(fit.intercept - 1) < 1e-9);
});

test("Distributions.normal: pdf/cdf/mean/variance match known values for the standard normal", () => {
  const std = Distributions.normal(0, 1);
  assert.ok(Math.abs(std.mean() - 0) < 1e-9);
  assert.ok(Math.abs(std.variance() - 1) < 1e-9);
  assert.ok(Math.abs(std.cdf(0) - 0.5) < 1e-9);
  // pdf(0) for the standard normal is 1/sqrt(2*pi).
  assert.ok(Math.abs(std.pdf(0) - 1 / Math.sqrt(2 * Math.PI)) < 1e-9);
});

test("Distributions.binomial: mean/variance match n*p / n*p*(1-p)", () => {
  const n = 20;
  const p = 0.3;
  const dist = Distributions.binomial(n, p);
  assert.ok(Math.abs(dist.mean() - n * p) < 1e-9);
  assert.ok(Math.abs(dist.variance() - n * p * (1 - p)) < 1e-9);
});

test("Distributions.poisson: mean equals variance equals lambda", () => {
  const lambda = 4.2;
  const dist = Distributions.poisson(lambda);
  assert.ok(Math.abs(dist.mean() - lambda) < 1e-9);
  assert.ok(Math.abs(dist.variance() - lambda) < 1e-9);
});

test("Distributions.normal: sample() is deterministic with a seeded rng", () => {
  let calls = 0;
  const seeded = () => {
    calls++;
    return 0.5; // fixed draw
  };
  const dist = Distributions.normal(0, 1, seeded);
  const a = dist.sample();
  const b = dist.sample();
  assert.equal(a, b, "same seeded rng sequence should give the same sample");
  assert.ok(calls > 0);
});
