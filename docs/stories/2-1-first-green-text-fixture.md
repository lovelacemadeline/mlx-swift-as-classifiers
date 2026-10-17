---
type: story
status: completed
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

Swift anchors: `Sources/MLXClassifiers/MLXRunner.swift::MLXRunner`,
`Sources/MLXClassifiers/Runner.swift::Runner`,
`Sources/MLXClassifiers/Readout.swift::scoreFirstToken`,
`Tests/MLXClassifiersTests/TextFixtureTests.swift::testNoulFirstToken`.

## As built

- `Sources/MLXClassifiers/Models/Gemma4Text.swift`, `Gemma4.swift`: vendored from
  mlx-swift-lm 3.32.3 (`3b339ad`, `Libraries/MLXLLM/Models/`), MIT, upstream
  licence beside them; each header records the upstream sha256 and the local
  changes: no `@_spi(GemmaEncoder)`, `BaseLanguageModel` instead of MLXLLM's
  `LLMModel`, a public backbone call returning final-norm hidden states with an
  optional `inputEmbeddings` override (PLE still from the ids), `head(_:)` (tied
  vocab projection + soft-cap) for sliced rows, `filterLMHeadWeights` inlined.
  Text only: audio and vision tensors are dropped by the upstream `sanitize`.
  The directory is `Models/` (ADR-0004 updated to match).
- `Runner.swift`: the `Runner` protocol, `PrefixCache`, `PrefillReadout` (golden
  `runner.Readout`; `Readout` is already the wire readout mode), `LogitRow`,
  `checkPrefillArgs`, as golden `runner.py`.
- `MLXRunner.swift`: loads a snapshot from the local HF cache (never downloads)
  with the checkpoint's per-layer quantisation (4-bit, MLPs 8-bit) through
  MLXLMCommon `loadWeights`; golden's split prefill (cache-only chunks of
  `prefillStep`, remainder first, evaluating only the KV state; then one pass
  over `[p0, p1]` with the vocab projection on the readout rows only); a prefix
  cache is copied before extending. `tokenize` throws `notImplemented` (story 1.2).
- `Readout.swift`: golden `bind_labels`' tokenizer-free half (duplicate, shared
  first token via `PrefixTree`, multi-token; `LabelTokenizationError` with
  golden's messages and hints), `gather`, `softmax`, `argmax`, and
  `scoreFirstToken` (golden `decide._score` + raw softmax, in `run_case` order:
  bind, optional `cachePrefix`, prefill, gather).
- `TextFixtureTests`: model loaded once per run; feeds `expected.tokens` and
  `expected.option_token_ids` (the in-context label ids) directly, since prompt
  compilation is story 1.2. GPU tests skip with `MLXCLASSIFIERS_SKIP_GPU=1`,
  without the MLX metallib (plain `swift test`), or without the snapshot.
- Build: `swift build` compiles no Metal shaders; the GPU tests run under
  `xcodebuild test` ([runbook](../runbooks/build-and-test.md)).

## Results

Measured 2026-10-17, M5 Mac, low-power mode on (`pmset lowpowermode 1`), Debug
build, `xcodebuild test`. Tolerance 1e-3 on raw probabilities plus answer match
(golden `fixtures.py`).

| fixture | max abs Δp | answer | raw-logit deltas |
|---|---|---|---|
| `noul-first-token` | 4.77e-4 | yes ✓ | 0, +0.125 |
| `choice-first-token-object-state` | 8.9e-7 | billing ✓ | 0, 0, −0.0625 |
| `choice-first-token-cached-list-state` (prefix cached) | 1.55e-4 | technical ✓ | +0.125, 0, 0 |
| `score-first-token-no-prefix` | 2.49e-4 | 3 ✓ | +0.125, 0, +0.125, 0 |
| `choice-first-token-multi-token-label-error` | n/a | `LabelTokenizationError`, message identical ✓ | n/a |

Logit deltas are single bf16 steps (0.0625 / 0.125 at logits of 16 to 32): the
Swift and Python graphs agree up to bf16 rounding. The same case run twice is
bit-identical. Load 0.7 s warm (3.0 s first); each fixture 90 to 180 ms after
the first, which pays about 9 s of one-off kernel compilation.

## Acceptance Criteria
- **Given** `noul-first-token` **when** run on the split path **then** `raw_probabilities` are within 1e-3 of the fixture and `answer` matches.
- **Given** the same case run twice **when** compared **then** the results are bit-identical.
