---
type: epic
status: in-progress
title: Runner — split prefill, readout and calibration
epic: 2
owner: maddie
updated: 2026-10-17
---

# Epic 2 — Runner — split prefill, readout and calibration

The Runner seam (golden ADR-0002 §4) over the vendored Gemma 4 text model (ADR-0004): cache-only prefill of the prefix and body rows, an L=1 readout pass sliced before `lm_head`, first-token masked softmax and sequence loglikelihood, permutations and temperature calibration ported unchanged. Done when the text fixtures reproduce within 1e-3.

## Stories
| id | title | status |
|---|---|---|
| 2.1 | Vendored Gemma 4 text backbone and the first green text fixture | completed |
| 2.2 | [End-to-end decide from a wire request](../stories/2-2-decide-from-a-wire-request.md) | in-progress |

## Related
- [PRD](../product/prd-swift-port.md), [recon](../research/swift-port-recon-2026-10.md), ADR-0002, ADR-0003, ADR-0004
