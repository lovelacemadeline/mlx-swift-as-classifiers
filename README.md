---
type: guide
status: active
title: mlx-swift-as-classifiers — README
owner: maddie
updated: 2026-10-17
---

# mlx-swift-as-classifiers

`MLXClassifiers` is a Swift port of the golden Python
[mlx-lm-as-classifiers](https://github.com/lovelacemadeline/mlx-lm-as-classifiers):
"System One" decisions (typed, calibrated, one forward pass, no decoding) over a
declared option set, on MLX Swift, for iPhone and Mac. Its definition of done is
reproducing the golden fixtures, which are vendored under `fixtures/`.

Status: scaffold. The wire schema types and the fixture loader are real and
tested; the runner, readout and model are not ported yet (see
[`docs/epics/index.md`](docs/epics/index.md)).

## Build and test

Requires Xcode 26 (Swift 6.2 or newer) on Apple silicon.

```bash
swift build
swift test
```

The first build compiles MLX from source and takes a few minutes.

## Layout

- `Sources/MLXClassifiers/`: `Schema.swift` (the `/v1/systemone` types),
  `Fixtures.swift` (typed golden fixtures), `JSONValue.swift` and
  `JSONCoding.swift` (order-preserving JSON; option order is semantic).
- `schema/`, `fixtures/`: vendored from the golden repo, pinned in each
  `VENDORED.txt`. Do not edit; re-vendor.
- `docs/`: PRD, ADRs, epics, stories, recon. Docs follow
  [`docs/doc-conventions.md`](docs/doc-conventions.md) and are checked by
  `docloom check`. Start at [`docs/product/prd-swift-port.md`](docs/product/prd-swift-port.md).

Licence: MIT.
