---
type: epic
status: in-progress
title: Schema, prompt compilation and label binding
epic: 1
owner: maddie
updated: 2026-10-17
---

# Epic 1 — Schema, prompt compilation and label binding

The wire types, the golden fixtures loaded as typed cases, and prompt compilation: Gemma 4 chat template, state rendering (string, sorted-key JSON object, `[i]` blocks, image placeholders), tokenizer, and label binding in context. Done when every fixture's `tokens`, `prefix_len`, `readout_position`, `options`, `labels`, `option_token_ids` and the label error reproduce exactly, with no model loaded.

## Stories
| id | title | status |
|---|---|---|
| 1.1 | Scaffold — package, docloom, vendored schema and fixtures | completed |
| 1.2 | Tokenizer and chat-template parity with the fixture token ids | pending |

## Related
- [PRD](../product/prd-swift-port.md), [recon](../research/swift-port-recon-2026-10.md), ADR-0001, ADR-0002
