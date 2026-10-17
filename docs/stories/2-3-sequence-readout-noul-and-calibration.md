---
type: story
status: completed
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

Swift anchors: `Sources/MLXClassifiers/Sequence.swift::scoreSequences`,
`Sources/MLXClassifiers/Calibrate.swift::TemperatureScaler`,
`Tests/MLXClassifiersTests/EndToEndFixtureTests.swift::testAllTextFixtures`,
`Tests/MLXClassifiersTests/EndToEndFixtureTests.swift::testImageFixtures`.

## As built

- `Sequence.swift::scoreSequences` is golden `score_sequences`: each option's
  continuation ids in context (story 1.2's `sequenceOptionTokenIds`, identical
  texts an error with golden's message), `cachePrefix` over the prompt minus its
  last token, then per option one `prefill` of `tokens + ids` reading the rows
  at `n - 1 ..< n - 1 + len(ids)`; the loglikelihood is the compensated sum of
  `tokenLogprob` (row logit minus the row's full-vocab logsumexp, which
  `MLXRunner` computes in float32 on the GPU as golden does), the score that
  over the token count. `decide` (each prompt caches itself, no shared system
  turn) and `runCase` use it in sequence mode.
- `Calibrate.swift`: golden `calibrate.py` whole: `gather`, `softmax` with
  temperature and `argmax` (moved unchanged out of `Readout.swift`), plus
  `tokenLogprob`, `nll`, `expectedCalibrationError`, `CalibrationReport` and the
  golden-section `TemperatureScaler` (`fit`, `apply`). `options.calibrated`
  applies the caller's fitted temperature in `decide` (story 2.2); serving a
  fitted temperature per model is outside this package.
- noul runs in both readouts: Yes/No labels, never permuted, the answer reports
  P(yes) calibrated and raw.
- The golden decide reference gains five sequence requests (the three sequence
  fixtures, permutations 3 at T 1.5, T 0.7) and records each row's logsumexp, so
  `ReplayRunner` covers sequence mode.
- Tests: `EndToEndFixtureTests.testAllTextFixtures` (the eight text fixtures
  from `request`), `testImageFixtures` (asserts the five image fixtures fail to
  compile with `PromptError` on a runner without a vision path, then skips:
  "vision path is Epic 4: ..."), `DecideLogicTests.testTemperatureScalerMatchesGolden`
  (temperature, NLL and ECE before and after, to 1e-12 against golden on a fixed
  synthetic set), `testTokenLogprob`.

## Results

Measured 2026-10-17, M5 Mac, low-power mode on (`pmset -g`: `lowpowermode 1`),
Debug build, `xcodebuild test`: 40 tests, 39 pass, 1 skipped (the image fixtures).

| fixture (from `request`) | max abs Δp | answer | raw-logit deltas |
|---|---|---|---|
| `choice-first-token-cached-list-state` | 1.55e-4 | technical ✓ | +0.125, 0, 0 |
| `choice-first-token-multi-token-label-error` | n/a | `LabelTokenizationError`, message identical ✓ | n/a |
| `choice-first-token-object-state` | 8.9e-7 | billing ✓ | 0, 0, −0.0625 |
| `choice-sequence-multi-token-options` | 4.8e-6 | refund ✓ | −9.9e-5, −2.5e-4, +0.0624 |
| `noul-first-token` | 4.77e-4 | yes ✓ | 0, +0.125 |
| `noul-sequence` | 3.72e-4 | no ✓ | −0.1247, +3.4e-4 |
| `score-first-token-no-prefix` | 2.49e-4 | 3 ✓ | +0.125, 0, +0.125, 0 |
| `score-sequence-object-state` | 0 | 1 ✓ | −1.9e-4 on all four |

All eight text fixtures pass; ids exact; max |Δp| 4.77e-4. Sequence-mode
deltas are in mean log-probabilities: a common shift (score-sequence: the
row's logsumexp differs by 1.9e-4, so every option moves alike and Δp is 0) or
one bf16 logit step on one option's token. Against golden `decide`, the replay
is bit-identical on all fifteen cases; on the GPU the single-ordering sequence
cases are within 3.72e-4 and usage is exact (noul-sequence 2 computed / 108
cached: two one-token passes over a 54-token cache). The permuted sequence case
(permutations 3, T 1.5) measures 1.27e-3, reported and not asserted, like
story 2.2's permuted cases. A sequence fixture takes 165 to 310 ms (two to
four option passes after the prefix).

## Acceptance Criteria
- **Given** every text fixture (first-token and sequence, noul, choice, score) **when** run from its `request` through `runCase` **then** ids match exactly, `raw_probabilities` within 1e-3, and the answer (or error type and message) equal.
- **Given** an image fixture **when** the runner reaches it **then** it skips with "vision path is Epic 4", never fails.
- **Given** labelled logit rows **when** `TemperatureScaler.fit` runs **then** the temperature, NLL and ECE equal golden `calibrate.py`'s.
- **Given** a sequence-mode wire request **when** decided **then** the response matches golden `decide` (replayed logits exactly; GPU within 1e-3 for single orderings).
