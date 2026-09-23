---
"@johnhenry/math-plus-tensor-autograd": patch
---

`io.loadCheckpoint` now reads bf16 entries. `writeCheckpoint` stores them as `.npy` with the `ml_dtypes` `'<V2'` descr, now that tensor-core's `toNpy()` supports bf16. Without this change a bf16 checkpoint could be saved but not loaded.
