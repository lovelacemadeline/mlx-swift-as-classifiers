---
type: story
status: pending
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

Target Swift anchors: `Sources/MLXClassifiers/Vision/ImagePreprocess.swift::resizedUInt8`,
`Tests/MLXClassifiersTests/ImageFixtureTests.swift::testImageFixtures`.

## Acceptance Criteria
- **Given** the five image fixtures **when** run **then** the image token runs match exactly and `raw_probabilities` are within 1e-3.
- **Given** `vision_soft_tokens` 70 and 140 **when** run **then** `max_soft_tokens` and the run lengths match the -vt70 and -vt140 fixtures.
