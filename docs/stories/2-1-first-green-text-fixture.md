---
type: story
status: pending
title: Vendored Gemma 4 text backbone and the first green text fixture
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 2.1 — Vendored Gemma 4 text backbone and the first green text fixture

Vendor the Gemma 4 text model from mlx-swift-lm 3.32.3 (ADR-0004) with a
provenance header, add the embeddings-in / hidden-out entry points, load the
E2B qat-4bit weights (skipping the audio tower), and implement the golden split
prefill and first-token readout. First green fixture: `noul-first-token`.

Target Swift anchors: `Sources/MLXClassifiers/Runner.swift::MLXRunner`,
`Tests/MLXClassifiersTests/TextFixtureTests.swift::testNoulFirstToken`.

## Acceptance Criteria
- **Given** `noul-first-token` **when** run on the split path **then** `raw_probabilities` are within 1e-3 of the fixture and `answer` matches.
- **Given** the same case run twice **when** compared **then** the results are bit-identical.
