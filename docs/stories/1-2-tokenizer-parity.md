---
type: story
status: pending
title: Tokenizer and chat-template parity with the fixture token ids
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 1.2 — Tokenizer and chat-template parity with the fixture token ids

Tokenize the compiled prompt exactly as golden does: render (system = `prefix`,
user = rendered state + question, assistant opened, thinking off) with Gemma 4's
chat template, encode with `add_special_tokens = false`, and bind each label in
context (tokenize `prompt + label`, take the ids after the prompt's own). Use
swift-transformers `Tokenizers` (`GemmaTokenizer` maps to its BPE tokenizer) and
check its `tokenizer.json` sha256 against `model.tokenizer_sha256`. Golden
reference: `prompt.py`, `template.py`, `readout.py` (`bind_options`,
`label_token_ids`). Needs the tokenizer files only, no weights.

Target Swift anchors: `Sources/MLXClassifiers/Prompt.swift::compileQuestion`,
`Tests/MLXClassifiersTests/TokenParityTests.swift::testFixtureTokensMatch`.

## Acceptance Criteria
- **Given** each fixture request **when** compiled **then** `tokens`, `prefix_len` and `readout_position` equal the fixture (first mismatch named as the golden checker does).
- **Given** each fixture **when** labels are bound **then** `options`, `labels` and `option_token_ids` equal the fixture.
- **Given** the option-text case with a two-token label **when** compiled first-token **then** it fails with `LabelTokenizationError`.
