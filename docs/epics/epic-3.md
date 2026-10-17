---
type: epic
status: completed
title: Delta prefix cache
epic: 3
owner: maddie
updated: 2026-10-17
---

# Epic 3 — Delta prefix cache

The golden `PromptCacheStore`: longest-common-prefix lookup over stored KV states, an exact rewind of `KVCacheSimple` and of `RotatingKVCache` (the W-1-rows rule of golden `mlx_runner.py`), LRU by entries and bytes, `cached_tokens` in usage. Done when a delta hit gives the same raw probabilities as a cold run.

## Stories
| id | title | status |
|---|---|---|
| 3.1 | Exact KV rewind and the longest-prefix store | completed |

## Related
- [PRD](../product/prd-swift-port.md), [recon](../research/swift-port-recon-2026-10.md), ADR-0002
