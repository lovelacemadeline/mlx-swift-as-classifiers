---
type: epic
status: in-progress
title: 26B on Swift
epic: 7
owner: maddie
updated: 2026-10-17
---

# Epic 7 — 26B on Swift

Prove the port on the model that matters: Gemma 4 26B-A4B (MoE, bidirectional
image blocks) through `MLXRunner` against golden's `fixtures/26b/`, then serve
it over golden's `POST /v1/systemone` wire route and run the gridworld through
it beside the Python server.

## Stories
| id | title | status |
|---|---|---|
| 7.1 | 26B through MLXRunner — bidirectional image blocks, the five 26B fixtures pass | completed |

## Related
- [Epic 4 — Vision path](epic-4.md)
- Golden story 4.4 (26B fixture subset), golden story 6.4 (the 26B image-block mask)
