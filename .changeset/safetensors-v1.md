---
"@johnhenry/math-plus-safetensors": minor
---

New package `@johnhenry/math-plus-safetensors`: safetensors reader/writer for any JS runtime. Reference-equivalent header validation (offset tiling, end-of-file coverage, duplicate names, typed `SafetensorsError` codes), typed views (F16 as `Float16Array`, BF16 bits), `toFloat32`/`toF32` for every dtype, a writer byte-identical to Python's `safetensors.serialize`, lazy `openSafetensors()` over Blob/File, HTTP Range requests (full-download fallback), Node/Bun file paths and FileHandles, and `@johnhenry/math-plus-safetensors/tensor` interop with tensor-core (optional peer).
