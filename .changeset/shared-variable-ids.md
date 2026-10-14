---
"@johnhenry/math-plus-tensor-autograd": patch
---

Variable ids now come from a process-wide counter (a `Symbol.for` slot on `globalThis`), so a second bundled copy of the module (e.g. from esm.sh) no longer produces colliding ids that break `backward()` (#170).
