import assert from "node:assert/strict";
import { makeTest } from "../../../test/harness.ts";
// @ts-ignore -- bun types are not installed; only evaluated under Bun (see test/harness.ts)
const { test } = makeTest((globalThis as { Bun?: unknown }).Bun ? await import("bun:test") : null);
import {
  DimensionMismatchError,
  Unit,
  UnitParseError,
  UnknownUnitError,
} from "../src/index.ts";

function close(a: number, b: number, eps = 1e-9): void {
  assert.ok(Math.abs(a - b) < eps, `${a} !~ ${b}`);
}

test("Unit.of(...).to(...): the issue's own example", () => {
  const m = Unit.of(55, "cm").to("m");
  close(m.value, 0.55);
  assert.equal(m.symbol, "m");
});

test("length round-trips: in/ft/yd/mi against m", () => {
  close(Unit.of(1, "in").to("m").value, 0.0254);
  close(Unit.of(1, "ft").to("in").value, 12);
  close(Unit.of(1, "yd").to("ft").value, 3);
  close(Unit.of(1, "mi").to("ft").value, 5280, 1e-6);
});

test("mass round-trips: lb/oz/g/kg", () => {
  close(Unit.of(1, "kg").to("g").value, 1000);
  close(Unit.of(16, "oz").to("lb").value, 1, 1e-6);
  close(Unit.of(1, "lb").to("g").value, 453.59237);
});

test("time round-trips: min/h/day against s", () => {
  close(Unit.of(1, "h").to("min").value, 60);
  close(Unit.of(1, "day").to("h").value, 24);
  close(Unit.of(90, "min").to("h").value, 1.5);
});

test("SI prefixes compose with prefixable base units", () => {
  close(Unit.of(1, "km").to("m").value, 1000);
  close(Unit.of(1500, "mg").to("g").value, 1.5);
  close(Unit.of(1, "kHz").to("Hz").value, 1000);
  close(Unit.of(1, "kPa").to("Pa").value, 1000);
  close(Unit.of(1000, "mL").to("L").value, 1);
  close(Unit.of(1, "Mg").to("kg").value, 1000); // megagram == tonne
});

test("kg is not double-prefixable (direct table entry wins, no 'kkg' nonsense)", () => {
  // "kg" resolves directly; prefixing it again isn't a supported symbol.
  assert.throws(() => Unit.of(1, "kkg"), UnknownUnitError);
});

test("temperature: affine conversions at known fixed points", () => {
  close(Unit.of(0, "degC").to("K").value, 273.15);
  close(Unit.of(100, "degC").to("degF").value, 212);
  close(Unit.of(-40, "degC").to("degF").value, -40);
  close(Unit.of(32, "degF").to("degC").value, 0);
  close(Unit.of(273.15, "K").to("degC").value, 0);
});

test("composite unit parsing: N == kg*m/s^2 dimensionally and numerically", () => {
  const n = Unit.of(1, "N");
  const composite = n.to("kg*m/s^2");
  close(composite.value, 1);
  assert.deepEqual(n.dimension, Unit.of(1, "kg*m/s^2").dimension);
});

test("composite unit parsing: m/s^2 and Pa", () => {
  const accel = Unit.of(9.8, "m/s^2");
  assert.deepEqual(accel.dimension, Unit.of(1, "m").div(Unit.of(1, "s").pow(2)).dimension);

  const pa = Unit.of(1, "Pa").to("N/m^2");
  close(pa.value, 1);
});

test("dimension mismatch throws on .to()", () => {
  assert.throws(() => Unit.of(1, "m").to("kg"), DimensionMismatchError);
});

test("dimension mismatch throws on .add()/.sub()", () => {
  assert.throws(() => Unit.of(1, "m").add(Unit.of(1, "kg")), DimensionMismatchError);
  assert.throws(() => Unit.of(1, "m").sub(Unit.of(1, "s")), DimensionMismatchError);
});

test("unknown unit throws UnknownUnitError", () => {
  assert.throws(() => Unit.of(1, "banana"), UnknownUnitError);
});

test("malformed expressions throw UnitParseError", () => {
  assert.throws(() => Unit.of(1, ""), UnitParseError);
  assert.throws(() => Unit.of(1, "m^abc"), UnitParseError);
  assert.throws(() => Unit.of(1, "degC/s"), UnitParseError); // affine unit inside a composite expression
});

test("add/sub across compatible units, expressed in the left operand's unit", () => {
  const total = Unit.of(1, "m").add(Unit.of(50, "cm"));
  close(total.value, 1.5);
  assert.equal(total.symbol, "m");

  const diff = Unit.of(1, "h").sub(Unit.of(15, "min"));
  close(diff.value, 0.75);
  assert.equal(diff.symbol, "h");
});

test("mul/div combine dimensions and produce a correctly-scaled result", () => {
  const distance = Unit.of(5, "m").mul(Unit.of(2, "s")); // contrived, but exercises the math
  assert.deepEqual(distance.dimension, Unit.of(1, "m*s").dimension);
  close(distance.to("m*s").value, 10);

  // physically meaningful: speed = distance / time
  const speed = Unit.of(100, "m").div(Unit.of(10, "s"));
  close(speed.to("m/s").value, 10);

  // scale correctness across mixed units: 5 cm * 2 s -> 0.1 m*s
  const mixed = Unit.of(5, "cm").mul(Unit.of(2, "s"));
  close(mixed.to("m*s").value, 0.1);
});

test("mul/div by a plain number scales the magnitude, unit unchanged", () => {
  const doubled = Unit.of(3, "m").mul(2);
  close(doubled.value, 6);
  assert.equal(doubled.symbol, "m");

  const halved = Unit.of(10, "kg").div(2);
  close(halved.value, 5);
  assert.equal(halved.symbol, "kg");
});

test("mul with an affine unit throws", () => {
  assert.throws(() => Unit.of(1, "degC").mul(Unit.of(2, "s")));
});

test("pow: simple units compute correctly", () => {
  const area = Unit.of(5, "m").pow(2);
  close(area.value, 25);
  assert.equal(area.symbol, "m^2");
  assert.deepEqual(area.dimension, Unit.of(1, "m*m").dimension);
});

test("pow: throws on a composite symbol (documented v1 limitation)", () => {
  const speed = Unit.of(10, "m").div(Unit.of(2, "s"));
  assert.throws(() => speed.pow(2), UnitParseError);
});

test("dimensionless arithmetic works without a unit symbol", () => {
  const a = Unit.dimensionless(3);
  const b = Unit.dimensionless(4);
  close(a.add(b).value, 7);
  close(a.mul(b).value, 12);
  assert.equal(a.add(b).symbol, "");
});

test("toString formats value and symbol", () => {
  assert.equal(Unit.of(55, "cm").toString(), "55 cm");
  assert.equal(Unit.dimensionless(3).toString(), "3");
  const cm = Unit.of(1, "m").to("cm"); // 100 cm
  assert.equal(cm.toString(2), `${(100).toPrecision(2)} cm`);
});

test("dimensional analysis: derived units check out against their definitions", () => {
  close(Unit.of(1, "J").to("N*m").value, 1);
  close(Unit.of(1, "W").to("J/s").value, 1);
  assert.deepEqual(Unit.of(1, "W").dimension, Unit.of(1, "J/s").dimension);
});
