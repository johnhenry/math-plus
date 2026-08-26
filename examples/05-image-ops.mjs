// image: resize + normalize tensor ops for ML/media pipelines.
// Channel-LAST layout ([H, W, C] / [N, H, W, C]), float dtypes only.
//
// Run: npm install && npm run build && node examples/05-image-ops.mjs
import { normalize, resize } from "@johnhenry/math-plus-image";
import { Tensor } from "@johnhenry/math-plus-tensor-core";

// 2x2 single-channel image: [[1,2],[3,4]]
const img = Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2, 1]);

// Nearest-neighbor upscale (floor-based index mapping)
const up = resize(img, { height: 4, width: 4 }, { method: "nearest" });
console.log(up.shape); // [4, 4, 1]
console.log(up.toArray()[0].map((px) => px[0])); // [1, 1, 2, 2]

// Bilinear (the default): half-pixel centers with edge clamping —
// i.e. TF/PyTorch align_corners=false. A horizontal gradient to 1x4:
const grade = Tensor.from([0, 10, 0, 10], { dtype: "f64" }).reshape([2, 2, 1]);
const bil = resize(grade, { height: 1, width: 4 });
console.log(bil.toArray()[0].map((px) => px[0])); // [0, 2.5, 7.5, 10]

// Per-channel standardization: (x - mean[c]) / std[c].
// mean/std lengths must equal the channel count; std of 0 throws.
const px = Tensor.from([10, 20], { dtype: "f64" }).reshape([1, 1, 2]);
console.log(normalize(px, { mean: [5, 10], std: [2, 5] }).toArray()); // [[[2.5, 2]]]
