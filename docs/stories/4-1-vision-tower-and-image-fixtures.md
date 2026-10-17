---
type: story
status: in-progress
title: Vendored vision tower, PIL-exact preprocessing and the image fixtures
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 4.1 — Vendored vision tower, PIL-exact preprocessing and the image fixtures

Vendor the Gemma 4 vision tower, pooler, `embed_vision` and the masked-scatter
merge from mlx-swift-lm 3.32.3 (ADR-0004), compare against golden
`src/mlx_classifiers/vision/` (vendored from mlx-vlm 0.7.8), and write the
preprocessing to match PIL `Image.BICUBIC` on uint8 (golden `vision/processor.py`,
`images.resized_size`), not CoreImage.

Swift anchors: `Sources/MLXClassifiers/Vision/ImagePreprocess.swift::resizedUInt8`,
`Sources/MLXClassifiers/Vision/ImagePreprocess.swift::pilResizeBicubic`,
`Sources/MLXClassifiers/Models/Gemma4Vision.swift::Gemma4VisionModel`,
`Sources/MLXClassifiers/Vision/VisionEncoder.swift::VisionEncoder`,
`Sources/MLXClassifiers/MLXRunner.swift::MLXRunner.rows`,
`Tests/MLXClassifiersTests/ImageFixtureTests.swift::testImageFixtures`,
`Tests/MLXClassifiersTests/ImageFixtureTests.swift::testResizedUInt8MatchesPIL`,
`Tests/MLXClassifiersTests/ImageFixtureTests.swift::testSoftTokensMatchGolden`.

**Status: in-progress.** Preprocessing is byte-identical to PIL and two of the
five image fixtures pass. The three gridworld fixtures miss the 1e-3 gate
(below). The tolerance is unchanged; the three misses are strict expected
failures in `testImageFixtures`, so that test fails as soon as one of them
passes.

## As built

- **Preprocessing** (`Vision/ImagePreprocess.swift`).
  - A PNG decoder for 8-bit, non-interlaced images (grey, RGB, palette,
    grey+alpha, RGBA; alpha dropped, as PIL's `convert("RGB")` does). It uses
    the Compression framework's raw DEFLATE after the zlib header, plus the five
    PNG filters.
  - A port of Pillow 12.3.0's `ImagingResample`: horizontal pass first, uint8
    between passes, the bicubic kernel (a = -0.5, support 2 times the downscale
    factor), per-pixel weights normalised in double, then 22-bit fixed point
    with a half-unit offset and `clip8`.
  - Then float32 `* float32(1/255)`, channel-first. Neither CoreImage nor
    CoreGraphics is involved.
- **Vision tower** (`Models/Gemma4Vision.swift`).
  - Vendored from mlx-swift-lm 3.32.3 `MLXVLM/Models/Gemma4.swift`, with the
    provenance header and `LICENSE-mlx-vlm` beside the existing
    `LICENSE-mlx-swift-lm`.
  - Local changes, both to golden's arithmetic: position embeddings summed as
    `hidden + (x + y)`; the float32 q/k/v RMSNorm squares with `pow(x, 2)`
    (golden `x_float**2`) rather than `square()`.
  - `Gemma4Model` decodes `vision_config`, holds `vision_tower` and
    `embed_vision`, and its `sanitize` keeps their weights. `rotary_emb` keys are
    dropped, and the clip bounds are kept only with `use_clipped_linears`
    (true on E2B). MLXLMCommon's loader quantizes `embed_vision`'s projection,
    the only module with `.scales`.
- **`VisionEncoder`**: golden `VisionTower.encode_png`. It memoises
  `[N, 1536]` soft tokens per PNG sha256 and budget (16 entries, LRU) and
  records each encode's milliseconds.
- **Runner seam.**
  - `Runner` gains `cachePrefix(_:images:)` and `prefill(...images:)`. Their
    default implementations refuse images ("no vision path").
  - `PrefixCache.images`, `checkImages`, and the image rule in
    `checkPrefillArgs` follow golden `runner.py`.
  - `ImageSlot` carries the PNG bytes, which are not part of equality.
- **`MLXRunner`.**
  - `imageTokens` comes from the snapshot's JSON; there is no vision path
    without a tower.
  - `rows` is golden `_rows` and `vision/merge.py`. Text rows are
    `embedTokens(ids) * embedScale`, image rows the soft tokens, and the
    per-layer ids are the ids with image rows set to 0. They go through the
    backbone's `inputEmbeddings` entry, under the same cache-only / readout
    split.
  - The delta store keys on `keyedTokens` (golden `image_key_id`: the image's
    sha256, and its budget, in a negative stand-in id). A hit inside an image
    run is re-looked-up capped at the run's start.
  - A checkpoint with bidirectional image attention (26B-A4B) is refused at
    load; it is not ported.
- **Soft-token budgets.** `options.vision_soft_tokens` 280 / 140 / 70 compiles
  to the budget's run length (story 1.2) and resizes to it. The slot's
  `maxSoftTokens` picks the encode budget and the store key.
- **Golden reference.** `scripts/golden_vision_reference.py` (golden venv)
  writes `Tests/MLXClassifiersTests/GoldenVision/vision-reference.json`. For
  each fixture PNG at each budget it records the sha256 of the PIL-resized
  uint8 image, of the tower output and of the soft tokens. With a directory
  argument it also dumps the full arrays as `.npy`.
- **Diagnostics** (environment-gated, skipped by default).
  `VisionDumpTests` writes the tower's intermediates as safetensors
  (`MLXCLASSIFIERS_VISION_DUMP`). `GoldenSoftTokenTests` runs the image
  fixtures on injected soft tokens (`MLXCLASSIFIERS_GOLDEN_SOFT`).

## Results

Measured 2026-10-17 on an M5 Mac with low-power mode on (`pmset -g`:
`lowpowermode 1`), Debug build, `xcodebuild test`: 61 tests, 59 pass, 2
skipped (the diagnostics), 0 failures. The 3 gridworld misses are recorded as
expected failures.

| fixture | image run | max \|Δp\| | gate 1e-3 | raw-logit deltas |
|---|---|---|---|---|
| `choice-first-token-image-colour` | 36+256 | 1.99e-4 | pass | +0.375, +0.25, +0.25, +0.25 |
| `noul-sequence-image` | 36+256 | 0 | pass | +4.0e-4 on both |
| `choice-first-token-image-gridworld` (280) | 128+270 | 7.77e-3 | **miss** | −0.125, −0.125, −0.25, −0.125, −0.25, −0.125 |
| `choice-first-token-image-gridworld-vt140` | 128+120 | 3.48e-2 | **miss** | −0.25, 0, −0.375, −0.125, −0.125, −0.3125 |
| `choice-first-token-image-gridworld-vt70` | 128+63 | 3.90e-2 | **miss** | +0.25, +0.5, +0.5, +0.125, +0.5, +0.25 |

- In all five, ids, image runs (`start`, `length`, sha256,
  `max_soft_tokens`) and the answer match exactly.
- The image-keyed store works: the same prompt again hits 338 of 338 rows. A
  different PNG of the same size (same token ids) hits only up to its image
  run (36). The same PNG at budget 140 hits 36.

**Where the divergence enters**, measured by diffing the Swift and golden
intermediates on all four fixture image/budget dumps:

1. **Preprocess: none.** The resized uint8 image is byte-identical to PIL's
   for both PNGs at 280, 140 and 70 (`testResizedUInt8MatchesPIL`). The
   float32 pixels and the patch embedding are identical too.
2. **Tower: from layer 0.** With identical inputs, the float32 q/k/v
   RMSNorm differs by 1 ulp on about 1e-5 of the elements. The bf16 RMSNorm
   before it also differs on the gridworld frame (one element, 1.2e-4).
   - No reformulation reproduces golden. Inside golden's MLX, `power`,
     `square`, `x*x`, `sum/n` and `sum*(1/n)` give the same counts, CPU
     included.
   - So the difference is the MLX core kernels: golden runs mlx 0.32.0,
     while mlx-swift 0.32.3 bundles core 0.32.2.
   - Sixteen bf16 layers amplify it. By layer 15, 85% of the elements
     differ; the soft tokens differ by up to 0.04–0.08 on 75–85% of their
     elements. The tower and soft tokens are not bit-identical at any budget
     (`testSoftTokensMatchGolden`, reported).
   - SDPA is not the cause: fed Swift's inputs, golden's SDPA reproduces
     Swift's output exactly. The all-zero additive mask golden's reference
     path passes makes no difference either.
3. **Merge and text backbone: also enough on their own.** With golden's
   exact soft tokens injected (`GoldenSoftTokenTests`), the gridworld fixtures
   still miss: 4.1e-2 at 280, 3.5e-2 at 140, 2.6e-2 at 70. Colour then
   measures 7.5e-4 and noul-sequence-image 4.8e-3. The text backbone already
   differs from golden by a bf16 step on the text fixtures (0.0625–0.125 at
   logits 16–32, story 2.1).
4. **Why only the gridworld misses.** Its six action logits are near-tied
   (21.375, 20.875, 20.25, 20.0 …). One bf16 step (0.125) on such a logit is
   about 0.03 in probability, so the 1e-3 gate on these fixtures needs
   bit-identical logits. The text fixtures and the colour case have one
   dominant option, which absorbs the same steps. Golden reproduces all five
   fixtures exactly in its own venv, on both its fast and vendored paths.
5. **What would close it.** The same MLX core as golden (0.32.0 in Swift, or
   golden on 0.32.2 with the fixtures regenerated); that is a pin decision for
   ADR-0004 or golden ADR-0002, not taken here. Alternatively, fixtures whose
   answers are not near-tied. The tolerance stays 1e-3.
6. **Cross-check (2026-10-18, low-power mode on).** The golden checker was run
   in a scratch venv of the golden package with `mlx==0.32.2` (same core as
   mlx-swift 0.32.3) via `mlx-classifiers fixtures check fixtures --runner mlx`:
   11/13 pass; `image-gridworld` fails at max|dp| 0.0757 (max|dlogit| 0.375)
   and `image-gridworld-vt140` at 0.021 (0.25); `vt70` happens to pass at 0.
   Golden Python on the Swift side's MLX core reproduces the Swift misses, so
   the divergence is the MLX core version, not the port.

**Encode time per image** (`testEncodeTiming`: memo cleared, median of 3 after
a warm-up, low-power mode on, Debug build; preprocessing is the Swift-side
decode and resize, unoptimised Debug code):

| image | budget | soft tokens | encode ms | of which preprocess ms |
|---|---|---|---|---|
| blue square 96x96 | 280 | 256 | 774.3 | 462.2 |
| blue square 96x96 | 140 | 121 | 396.7 | 233.5 |
| blue square 96x96 | 70 | 64 | 272.5 | 132.8 |
| gridworld 160x192 | 280 | 270 | 993.7 | 541.8 |
| gridworld 160x192 | 140 | 120 | 439.5 | 270.7 |
| gridworld 160x192 | 70 | 63 | 309.7 | 161.4 |

## Acceptance Criteria
- **Given** the five image fixtures **when** run **then** the image token runs match exactly and `raw_probabilities` are within 1e-3. Partly met: runs, ids and answers are exact in all five; probabilities pass for colour (1.99e-4) and noul-sequence-image (0), and miss for the three gridworld cases (7.8e-3, 3.5e-2, 3.9e-2; above).
- **Given** `vision_soft_tokens` 70 and 140 **when** run **then** `max_soft_tokens` and the run lengths match the -vt70 and -vt140 fixtures. Met: 63 and 120 tokens, `max_soft_tokens` 70 and 140.
- **Given** a fixture PNG at 280, 140 or 70 **when** preprocessed **then** the uint8 image equals PIL's byte for byte. Met: 6 of 6.
