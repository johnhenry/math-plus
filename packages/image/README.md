# @johnhenry/math-plus-image

Resize and normalize tensor ops for practical ML/media pipelines (issue #41).
Scoped tightly: this is the image *slice* of the math-plus family, not a
general image-processing library. Pure JS; depends only on
`@johnhenry/math-plus-tensor-core`.

## Install

```bash
npm install @johnhenry/math-plus-image
```

## Quick start

```js
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { resize, normalize } from "@johnhenry/math-plus-image";

// Channel-LAST layout: [H, W, C] or [N, H, W, C]
const img = Tensor.from([1, 2, 3, 4], { dtype: "f64" }).reshape([2, 2, 1]);

const up = resize(img, { height: 4, width: 4 }, { method: "nearest" });
// rows: [1,1,2,2] / [1,1,2,2] / [3,3,4,4] / [3,3,4,4]

const bil = resize(img, { height: 1, width: 4 }); // default "bilinear"

// Per-channel (x - mean[c]) / std[c]
const px = Tensor.from([10, 20], { dtype: "f64" }).reshape([1, 1, 2]);
normalize(px, { mean: [5, 10], std: [2, 5] }); // -> [2.5, 2]
```

## API surface

| Export | What it does |
|---|---|
| `resize(input, { height, width }, { method? })` | `"nearest"` or `"bilinear"` (default). Half-pixel-center mapping with edge clamping — i.e. TF/PyTorch `align_corners=false`. |
| `normalize(input, { mean, std })` | Per-channel standardization; `mean`/`std` lengths must equal the channel count; `std` of 0 throws. |

## Traps

- **Channel-last only.** `[H, W, C]` / `[N, H, W, C]`. There is no layout
  option — an NCHW tensor of the right rank is silently misinterpreted.
  `height` comes first in the size object (`{ height, width }`, not a tuple).
- **Float dtypes only.** `f32`/`f64`; anything else throws `TypeError` with
  "cast to a float dtype first". uint8 raw pixel data is explicitly out of
  scope in v1 (rounding/clamping semantics not designed yet).
- **`Tensor.from` defaults to `f32`** — output dtype mirrors input dtype, and
  arithmetic runs internally in f64 either way.
- **No alpha, color-space, or channel-order awareness.** Channels are opaque
  independent scalars; an RGBA alpha channel is interpolated like any other
  channel (no premultiplication).
- **No antialiasing on downscale**, and no `bicubic`/`lanczos`/`area`
  methods. `"nearest"` uses a floor (downscale 4→2 picks indices 0 and 2).

## Tests

`npm test` — pure hand-computed assertions (no Python oracle needed), always
run.

## Provenance

Built for issue #41, the v2 "practical ML/media compute" bundle's image
slice. Part of the [math-plus](https://github.com/johnhenry/math-plus)
monorepo; family docs at <https://opensource.johnhenry.me/math/>.
