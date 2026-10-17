---
type: story
status: in-progress
title: Sequence readout, noul and calibration
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 2.3 — Sequence readout, noul and calibration

Port golden `sequence.py` (each option's text teacher-forced after the prompt,
scored by its length-normalised loglikelihood from one cached prefix of the
prompt minus its last token), wire it into `decide` and the fixture runner, and
port `calibrate.py` (the masked softmax with temperature, NLL, top-label ECE and
the golden-section `TemperatureScaler`) behind `options.calibrated`. The noul
question kind runs in both readouts. Done when every text fixture (all but the
five image fixtures) passes end to end from `request`, and the image fixtures
skip saying the vision path is epic 4.

Target Swift anchors: `Sources/MLXClassifiers/Sequence.swift::scoreSequences`,
`Sources/MLXClassifiers/Calibrate.swift::TemperatureScaler`,
`Tests/MLXClassifiersTests/EndToEndFixtureTests.swift::testAllTextFixtures`.

## Acceptance Criteria
- **Given** every text fixture (first-token and sequence, noul, choice, score) **when** run from its `request` through `runCase` **then** ids match exactly, `raw_probabilities` within 1e-3, and the answer (or error type and message) equal.
- **Given** an image fixture **when** the runner reaches it **then** it skips with "vision path is Epic 4", never fails.
- **Given** labelled logit rows **when** `TemperatureScaler.fit` runs **then** the temperature, NLL and ECE equal golden `calibrate.py`'s.
- **Given** a sequence-mode wire request **when** decided **then** the response matches golden `decide` (replayed logits exactly; GPU within 1e-3 for single orderings).
