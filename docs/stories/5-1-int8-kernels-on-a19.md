---
type: story
status: pending
title: Compile the golden INT8 kernels on A19 and measure
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 5.1 — Compile the golden INT8 kernels on A19 and measure

Feasibility: build each golden kernel (`int8/wiring.py`, `int8/experts.py`)
through `MLXFast.metalKernel` on an A19 device, check the outputs against the
stock path, and measure prefill speed and memory against the 8 GB budget. The
`moe.py` segmented NAX kernel is pinned to MLX 0.32 headers; mlx-swift 0.32.3
bundles MLX 0.32.2, so check the header before any port.

Target Swift anchors: `Tests/MLXClassifiersTests/Int8KernelTests.swift::testInt8LinearMatchesStock`.

## Acceptance Criteria
- **Given** an iPhone 17 **when** the INT8 GEMM kernel is compiled **then** it either builds and matches the stock linear within bf16 noise, or the failure is recorded with the Metal error.
- **Given** E2B with and without the INT8 patch **when** measured on device **then** decisions/s and `os_proc_available_memory` headroom are recorded.
