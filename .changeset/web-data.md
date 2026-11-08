---
"@johnhenry/math-plus-web": minor
---

Add the `data` namespace: `Dataset` and `fromAsync` from `@johnhenry/math-plus-data` (async pipelines with shuffle, epochs, prefetch, mapConcurrent and fold), as `data` in the one-import entry, as `./data` and as a self-contained `./bundle/data` file. Browser and Worker panes can now stream a dataset without Node.
