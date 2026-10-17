---
type: story
status: completed
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

## As built

- `Sources/MLXClassifiers/ChatTemplate.swift::JinjaChatTemplate` is golden
  `HFChatTemplate`: it renders the snapshot's `chat_template.jinja` with
  swift-jinja 2.5.1 (the engine swift-transformers' `applyChatTemplate` uses,
  now a direct exact pin) with `trimBlocks`/`lstripBlocks` on, `messages`,
  `add_generation_prompt`, `enable_thinking = false` and the special-token
  strings in the context. It is driven directly because `applyChatTemplate`
  returns token ids only, and compilation needs the text (the prefix split and
  the image markers).
- **No Jinja divergence.** swift-jinja's rendering of Gemma 4's template equals
  HF `apply_chat_template` on every fixture (all 13 prompts and prefixes
  tokenize to `expected.tokens` and `prefix_len`) and on the bare markup
  (`testChatTemplateRendersLikeHF`, strings captured from the golden venv:
  `<bos><|turn>system\n…<turn|>\n<|turn>user\n…<turn|>\n<|turn>model\n`,
  generation prompt `<|turn>model\n`). No hand-written template was needed.
- **No BPE divergence.** swift-transformers 1.3.5 `GemmaTokenizer` (BPE, byte
  fallback, added tokens split out of the text) with `addSpecialTokens: false`
  reproduces golden `tokenizer.encode(text, add_special_tokens=False)` on every
  prompt, prefix, label continuation and image placeholder run.
- `ModelTokenizer.swift::ModelTokenizer`: tokenizer, chat template and
  `ImageTokens` from a local HF snapshot (no weights, no network), with the
  `tokenizer.json` sha256 checked against `model.tokenizer_sha256`.
- `Prompt.swift::compileQuestion` ports golden `compile_question` (with
  `renderState`, `renderUserTurn`, `optionLabels`, `questionOptions`, the image
  sentinels and `ImageSlot`s). `LabelBinding.swift` ports `emitted_label`,
  `label_token_ids`, `PrefixTree`, `bind_labels`, `bind_options` and the id half
  of `score_sequences`; `tokenResult` is the id half of `run_case`.
  `ImageTokens.swift` ports the placeholder-count arithmetic (`png_size`,
  `resized_size`, `soft_token_count`, `find_image_runs`); pixels stay Epic 4.
- `PythonText.swift`: the Python semantics the prompt text depends on:
  `json.dumps(indent=2, sort_keys=True, ensure_ascii=False)` with code-point key
  order and `float.__repr__`, code-point `startswith`/`count`/`replace`, and
  `str.isspace` for the label's leading space.
- Tests (`Tests/MLXClassifiersTests/TokenParityTests.swift`):
  `testFixtureTokensMatch` (all 13: `tokens`, `prefix_len`, `readout_position`,
  `images`, `options`, `labels`, `option_token_ids`, `error.type`, first
  mismatch named as golden `_compare_ids` does), `testImageRunStructure` (the 5
  image fixtures: runs bracketed by begin/end-of-image at `expected.images`),
  `testMultiTokenOptionTextLabelFails`, `testChatTemplateRendersLikeHF`, and
  `PythonTextTests`. A deliberate mutation (always prefixing the label with a
  space) fails `testFixtureTokensMatch`, so the comparison is live. The tests
  need the local snapshot `mlx-community/gemma-4-E2B-it-qat-4bit@42f6273`
  (tokenizer files only) and skip, saying so, without it.

## Acceptance Criteria
- **Given** each fixture request **when** compiled **then** `tokens`, `prefix_len` and `readout_position` equal the fixture (first mismatch named as the golden checker does).
- **Given** each fixture **when** labels are bound **then** `options`, `labels` and `option_token_ids` equal the fixture.
- **Given** the option-text case with a two-token label **when** compiled first-token **then** it fails with `LabelTokenizationError`.
