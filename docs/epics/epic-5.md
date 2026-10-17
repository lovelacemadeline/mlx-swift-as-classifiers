---
type: epic
status: pending
title: INT8 kernels and MoE on A19
epic: 5
owner: maddie
updated: 2026-10-17
---

# Epic 5 — INT8 kernels and MoE on A19

Feasibility first: do the golden W8A8 and W4A8 kernels (MPP `matmul2d` cooperative tensors) compile and match on an A19 GPU through `MLXFast.metalKernel`, and is the INT8 patch worth its memory on an 8 GB phone ([story 5.1](../stories/5-1-int8-kernels-on-a19.md)). The 26B MoE is out of scope on iPhone (it does not fit); the segmented NAX kernel is Mac-only if ported at all.

## Stories
| id | title | status |
|---|---|---|
| 5.1 | Compile the golden INT8 kernels on A19 and measure | pending |

## Related
- [PRD](../product/prd-swift-port.md), [recon](../research/swift-port-recon-2026-10.md), ADR-0004
