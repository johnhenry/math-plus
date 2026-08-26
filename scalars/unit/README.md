# @johnhenry/math-plus-unit

A unit/dimension scalar type: a magnitude plus dimension metadata, with
parsing (`"kg*m/s^2"`), SI-prefix handling, formatting, and arithmetic that
is dimensional-analysis-checked — adding metres to kilograms throws instead
of silently producing a number. Pure TypeScript, zero dependencies.

## Install

```bash
npm install @johnhenry/math-plus-unit
```

## Quick start

```js
import { Unit, DimensionMismatchError } from "@johnhenry/math-plus-unit";

Unit.of(55, "cm").to("m").value;        // 0.55
Unit.of(0, "degC").to("degF").value;    // 32
Unit.of(1, "kHz").to("Hz").value;       // 1000 — SI prefixes compose

// Arithmetic converts through the left operand's unit
Unit.of(1, "m").add(Unit.of(50, "cm")).toString(); // "1.5 m"

// mul/div combine dimensions and rescale correctly
const speed = Unit.of(100, "m").div(Unit.of(10, "s"));
speed.to("m/s").value;                  // 10
Unit.of(5, "cm").mul(Unit.of(2, "s")).to("m*s").value; // 0.1

// Composite symbols parse, and derived units check out
Unit.of(1, "N").to("kg*m/s^2").value;   // 1
Unit.of(1, "W").to("J/s").value;        // 1

// The point of the package: wrong physics is an error, not a number
Unit.of(1, "m").to("kg");               // throws DimensionMismatchError
Unit.of(1, "m").add(Unit.of(1, "kg"));  // throws DimensionMismatchError
```

## API surface

- **`Unit`:** `Unit.of(value, symbol)`, `Unit.dimensionless(value)`,
  `.to(symbol)`, `.add`/`.sub` (dimension-checked, result in the left
  operand's unit), `.mul`/`.div` (by another `Unit` — dimensions combine —
  or by a plain number — magnitude scales), `.pow(n)`,
  `.toString(digits?)`, plus `.value`, `.symbol`, `.dimension`.
- **Units table:** 25 unit symbols (`BASE_UNITS`) across length, mass,
  time, temperature, frequency, force, pressure, energy, power, volume, and
  more, with US-customary entries (`in`/`ft`/`yd`/`mi`, `lb`/`oz`) defined
  against SI. `PREFIXES` covers the SI prefixes for prefixable units;
  `lookupUnitSymbol` resolves a symbol the way the parser does.
- **Dimensions:** `dim`, `DIMENSIONLESS`, `multiplyDimensions` /
  `divideDimensions` / `powDimension`, `dimensionsEqual`,
  `isDimensionless`, `dimensionToString` — usable standalone for your own
  dimensional bookkeeping.
- **Errors:** `DimensionMismatchError`, `UnknownUnitError`
  (`Unit.of(1, "banana")`), `UnitParseError` (malformed expressions).

## Traps

- **Affine units don't compose.** `degC`/`degF` conversions are affine
  (offset, not just scale), so they convert correctly point-to-point —
  `-40 degC` is `-40 degF` — but cannot appear inside a composite
  expression (`Unit.of(1, "degC/s")` throws `UnitParseError`) or in
  `mul`/`div`. Use `K` for rate-of-change math.
- **`pow` on composite symbols is a documented v1 limitation.**
  `Unit.of(5, "m").pow(2)` works (`"m^2"`); powing a value whose symbol is
  already composite (say, the result of `m/s`) throws `UnitParseError`.
- **`kg` resolves as a table entry, not prefix + `g`** — so there is no
  double prefixing (`"kkg"` throws `UnknownUnitError`; a megagram is
  `"Mg"`).
- **Units never enter tensor storage.** This is a scalar type for API
  edges and metadata; bulk numeric work stays plain-`number` in
  tensor-core. That boundary is deliberate (see the family's acceleration
  policy in the repo docs).

Part of the [math-plus](https://github.com/johnhenry/math-plus) family —
docs at [opensource.johnhenry.me/math/](https://opensource.johnhenry.me/math/).
