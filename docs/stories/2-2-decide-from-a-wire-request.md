---
type: story
status: completed
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

Swift anchors: `Sources/MLXClassifiers/Decide.swift::decide`,
`Sources/MLXClassifiers/FixtureRunner.swift::runCase`,
`Tests/MLXClassifiersTests/EndToEndFixtureTests.swift::testFirstTokenTextFixtures`,
`Tests/MLXClassifiersTests/EndToEndFixtureTests.swift::DecideTests`.

## As built

- `Runner.swift`: the seam now carries `chatTemplate` (golden
  `runner.chat_template`), `imageTokens` (default nil: golden's optional
  `image_tokens`) and `modelId` (default "default"); `tokenize` is non-throwing.
- `MLXRunner.swift`: `load` is async and loads story 1.2's `ModelTokenizer`
  beside the weights (optionally checking the `tokenizer.json` sha256); the
  `notImplemented` stub is gone. `imageTokens` is nil until epic 4, so a request
  with image blocks fails to compile with golden's "no vision path" `PromptError`.
- `Decide.swift`: golden `decide.py` and `answers.shape_answer`, first-token
  path: compile per question (all orderings first), the system turn cached once
  and shared, bind in context, prefill, gather, raw and calibrated softmax
  averaged over orderings, the debug block (`option_token_ids` one id per
  ordering, mean `raw_logits`, applied `temperature`, `permutation_spread`), usage
  sums and wall-clock timing. `permutationOrders` is golden's (rotations, then
  shuffles); `PythonRandom.swift` reproduces CPython's `random.Random(str)`
  (SHA-512 seeding, MT19937, `_randbelow_with_getrandbits`) so the shuffles are
  the same. `options.delta_cache` changes nothing yet (no store; epic 3), as
  golden's `without_delta` on a store-less runner. Sequence readout throws
  `notImplemented` (story 2.3).
- `FixtureRunner.swift`: golden `run_case` (with `_with_label_style`; a binding
  error is a result carrying the per-label ids) and `compare` (`CaseReport`).
- `scripts/golden_decide_reference.py` runs golden `decide` in the golden venv
  (golden 9b64a14, real model, `delta_cache: false`, `debug: true`) over ten
  requests built from the fixtures (the five first-token fixtures, permutations
  3 and 5 with T 1.7 and 0.8, uncalibrated with T 2, calibrated score with
  T 1.3, a three-question request with permutations 2) and writes
  `Tests/MLXClassifiersTests/GoldenDecide/decide-responses.json`: each wire
  response (timing dropped) or error, plus every prefill's gathered logits.
- Tests: `EndToEndFixtureTests.testFirstTokenTextFixtures` (fixtures from
  `request` through `runCase` and `compareCase`, error message compared too);
  `DecideTests.testDecideReplaysGoldenLogitsExactly` (a `ReplayRunner` serves
  golden's logits with the real tokenizer: decide's arithmetic, ids, answers and
  usage checked to 1e-12; tokenizer only, runs under `swift test`);
  `DecideTests.testDecideMatchesGoldenDecide` (GPU); `DecideLogicTests`
  (permutation orders against values from CPython 3.12, `shapeAnswer`).

## Results

Measured 2026-10-17, M5 Mac, low-power mode on (`pmset -g`: `lowpowermode 1`),
Debug build, `xcodebuild test`; 36 tests pass.

| fixture (from `request`) | max abs Δp | answer | raw-logit deltas |
|---|---|---|---|
| `choice-first-token-cached-list-state` | 1.55e-4 | technical ✓ | +0.125, 0, 0 |
| `choice-first-token-multi-token-label-error` | n/a | `LabelTokenizationError`, message identical ✓ | n/a |
| `choice-first-token-object-state` | 8.9e-7 | billing ✓ | 0, 0, −0.0625 |
| `noul-first-token` | 4.77e-4 | yes ✓ | 0, +0.125 |
| `score-first-token-no-prefix` | 2.49e-4 | 3 ✓ | +0.125, 0, +0.125, 0 |

End to end from the wire request reproduces story 2.1's id-fed numbers exactly.
Against golden `decide`: the replay of golden's logits is bit-identical on all
ten cases (Δ = 0). On the GPU the single-ordering cases match within 1e-3 (max
Δp 9.07e-4 on the T 1.3 score case, 0 on noul and object-state, where caching
the prefix removes 2.1's one-step deltas); ids, answers, temperature and usage
are exact on all cases. The permuted cases measure max Δp 3.28e-4
(permutations 3), 3.59e-2 (permutations 5, T 0.8) and 7.39e-2 (three questions,
permutations 2), reported and not asserted: their rotated prompts are new
token sequences (identical on both sides) whose ordering `(1, 2, 0)` is a
near-tie, billing 26.625 vs technical 26.5 in golden, where Swift reads 26.75
and 26.0 (cached path). Golden itself moves by up to 0.375 between its own
cached and uncached paths on these prompts (`(2, 0, 1)`, sales 18.75 vs
18.375), and Swift's uncached path agrees with golden's uncached path within
one bf16 step; Release builds give bit-identical logits to Debug. So this is
bf16 path noise amplified at a near-tie, not a decide difference. Load 5.2 s
including the tokenizer; a fixture 95 to 180 ms, a five-ordering decide 670 ms.

## Acceptance Criteria
- **Given** every first-token text fixture **when** run from its `request` through `runCase` **then** `tokens`, `prefix_len`, `readout_position`, `options`, `labels` and `option_token_ids` match exactly, `raw_probabilities` within 1e-3, `answer` equal, and the error fixture reports the same error type and message.
- **Given** a wire request **when** passed to `decide` **then** the response carries the golden answer shape, usage and debug block: bit-identical to golden `decide` given golden's logits, and within 1e-3 on probabilities on the MLX runner for single-ordering requests (permuted requests measured above).
- **Given** `options.permutations` or a fitted temperature **when** decided **then** orderings, averaging and calibration follow golden `decide`.
