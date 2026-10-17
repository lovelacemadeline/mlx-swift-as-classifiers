---
type: story
status: pending
title: Exact KV rewind and the longest-prefix store
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 3.1 — Exact KV rewind and the longest-prefix store

Port golden `prompt_cache.py` (`PromptCacheStore`, `common_prefix_len`) and
`mlx_runner.rewind_cache`: a `KVCacheSimple` trims to any length; a
`RotatingKVCache` rewinds exactly only under the golden W-1 rule, otherwise the
store tries the next entry. `RotatingKVCache.trim` is not exact past the wrap
(see the recon), so it is not used.

Target Swift anchors: `Sources/MLXClassifiers/PromptCache.swift::PromptCacheStore`,
`Tests/MLXClassifiersTests/PromptCacheTests.swift::testDeltaHitMatchesColdRun`.

## Acceptance Criteria
- **Given** two prompts sharing a prefix longer than the sliding window **when** the second hits the store **then** its raw probabilities equal a cold run.
- **Given** a rewind the window cannot represent **when** looked up **then** the store reports no hit from that entry.
