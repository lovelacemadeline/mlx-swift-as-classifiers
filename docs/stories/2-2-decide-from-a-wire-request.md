---
type: story
status: in-progress
title: End-to-end decide from a wire request
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 2.2 — End-to-end decide from a wire request

Replace the `MLXRunner.tokenize` stub with the story 1.2 tokenizer and chat
template, and port golden `decide.py` (with `answers.py`) so a raw wire
`SystemOneRequest` (a fixture's `request` field) goes in and a full
`SystemOneResponse` comes out: prompt compiled, options bound, scored, answers
shaped (probabilities, calibrated and raw, confidence, `expected_score`,
`legend`, the debug block), usage and timing. First-token path only, with
permutations and the calibration temperature as golden applies them there;
sequence readout is story 2.3. Golden `fixtures.run_case` is ported as the
fixture runner and every first-token text fixture runs end to end from
`request`.

Target Swift anchors: `Sources/MLXClassifiers/Decide.swift::decide`,
`Sources/MLXClassifiers/FixtureRunner.swift::runCase`,
`Tests/MLXClassifiersTests/EndToEndFixtureTests.swift::testFirstTokenTextFixtures`.

## Acceptance Criteria
- **Given** every first-token text fixture **when** run from its `request` through `runCase` **then** `tokens`, `prefix_len`, `readout_position`, `options`, `labels` and `option_token_ids` match exactly, `raw_probabilities` within 1e-3, `answer` equal, and the error fixture reports the same error type and message.
- **Given** a wire request **when** passed to `decide` with the MLX runner **then** the response carries the golden answer shape, usage and debug block, matching golden `decide` within 1e-3 on probabilities.
- **Given** `options.permutations` or a fitted temperature **when** decided **then** orderings, averaging and calibration follow golden `decide`.
