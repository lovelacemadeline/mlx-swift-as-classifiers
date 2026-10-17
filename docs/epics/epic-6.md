---
type: epic
status: in-progress
title: EV3 demo app
epic: 6
owner: maddie
updated: 2026-10-17
---

# Epic 6 — EV3 demo app

The first consumer: an iPhone 17 app that links `MLXClassifiers`, loads Gemma 4 E2B, turns a camera frame plus a status line into one decision over the robot's actions, and drives an EV3 servo. Measures memory with `os_proc_available_memory` under the increased-memory-limit entitlement.

## Stories
| id | title | status |
|---|---|---|
| 6.1 | On-device load, memory budget and one decision from a camera frame | in-progress |

## Related
- [iPhone deploy runbook](../runbooks/iphone-deploy.md)
- [PRD](../product/prd-swift-port.md), [recon](../research/swift-port-recon-2026-10.md), ADR-0002
