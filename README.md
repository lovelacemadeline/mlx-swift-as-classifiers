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

Status: a wire request goes in and a full response comes out (`decide`):
tokenizer and chat template, prompt compilation, label binding, the vendored
Gemma 4 text model on the split-prefill `MLXRunner`, first-token and sequence
readout, permutations and temperature calibration. All eight text fixtures
reproduce within 1e-3 end to end from their request (stories 2.2 and 2.3). The
delta prefix cache, vision and the INT8 kernels are not ported yet (see
[`docs/epics/index.md`](docs/epics/index.md)).

## Build and test

Requires Xcode 26 (Swift 6.2 or newer) with the Metal toolchain, on Apple silicon.

**`swift build` does not compile MLX's Metal shaders**, so the GPU tests only run
under `xcodebuild`:

```bash
# everything, GPU fixtures included (needs the E2B snapshot in the HF cache)
xcodebuild test -scheme MLXClassifiers-Package -destination 'platform=macOS' \
  -derivedDataPath .build/xcode -skipMacroValidation -skipPackagePluginValidation

# no GPU: the model tests skip themselves (or set MLXCLASSIFIERS_SKIP_GPU=1)
swift build && swift test
```

The first build compiles MLX from source and takes a few minutes. Details,
env vars and failure modes: [`docs/runbooks/build-and-test.md`](docs/runbooks/build-and-test.md).

## Layout

- `Sources/MLXClassifiers/Models/`: Gemma 4 text model vendored from
  mlx-swift-lm 3.32.3 (ADR-0004, provenance in each file header).
- `Sources/MLXClassifiers/`: `Runner.swift` (the Runner seam), `MLXRunner.swift`
  (split prefill), `Readout.swift` (label rules, masked softmax), `Schema.swift` (the `/v1/systemone` types),
  `Fixtures.swift` (typed golden fixtures), `JSONValue.swift` and
  `JSONCoding.swift` (order-preserving JSON; option order is semantic).
- `schema/`, `fixtures/`: vendored from the golden repo, pinned in each
  `VENDORED.txt`. Do not edit; re-vendor.
- `docs/`: PRD, ADRs, epics, stories, recon. Docs follow
  [`docs/doc-conventions.md`](docs/doc-conventions.md) and are checked by
  `docloom check`. Start at [`docs/product/prd-swift-port.md`](docs/product/prd-swift-port.md).

Licence: MIT.
