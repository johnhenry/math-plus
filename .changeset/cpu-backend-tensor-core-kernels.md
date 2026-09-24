---
"@johnhenry/math-plus-tensor-core": minor
---

New `@johnhenry/math-plus-tensor-core/kernels` subpath export (#144): the flat typed-array kernels under tensor-core's fast paths (`gemmNT`, `packPanel`, `softmaxAxis`, `sumAxis`, `extremumAxis`, …) plus new strided/broadcast kernels (`alignedStrides`, `rowOffsets`, `stridedCopy`, `binaryStrided`, `compareStrided`, `whereStrided`, `argExtremumAxis`, `cumsumAxis`, `sortAxis`) and fused NN kernels (`linearNT`, `layerNormRows`, `ropeHalf`, `attention`, `gegluRows`, `maskedMeanPool`, `takeRows`, `unaryFlat`), for device packages such as the new CPU `Backend`. The package entry is unchanged. New bit-identical fast paths: `contiguous()` of a strided view copies row by row instead of walking element offsets, and `argmin`/`argmax` along an axis and `cumsum` use flat kernels for contiguous non-BigInt inputs.
