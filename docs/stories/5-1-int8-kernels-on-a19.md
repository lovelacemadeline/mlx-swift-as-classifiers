---
type: story
status: in-progress
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

## Progress (2026-10-17) — all six golden kernels compile, run and match M5 bit for bit on A19

`scripts/capture_int8_kernels.py` (golden venv, M5, mlx 0.32.0) patches `mx.fast.metal_kernel`
before the golden modules build their kernels, so the MSL source and header strings are the
golden ones exactly. It then calls each kernel once through its golden wrapper on a small
problem and records the launch geometry, template, input bytes and M5 output bytes in
`App/ClassifierProbe/ClassifierProbe/int8-kernels.json`. On the phone,
`KernelProbe.swift::KernelProbe.run` (`scripts/deploy.sh --run --kernels`) rebuilds each kernel
with `MLXFast.metalKernel` (mlx-swift 0.32.3), replays the call and compares the outputs.

iPhone 17 (A19, iOS 26.6.1):

| kernel | golden source | shape run | result | first call (incl. JIT) |
|---|---|---|---|---|
| `int8_linear_epilogue_w8a8opt_nk` (MPP `matmul2d`, int8) | `int8/wiring.py` | M64 N128 K256 | bit-identical | 366 ms |
| `fused_rowquant_int8_w8a8opt` | `int8/wiring.py` | 4×256 | bit-identical | 33 ms |
| `mlx_classifiers_int8_gemv_nk` | `int8/wiring.py` | 2×512 → 16 | bit-identical | 36 ms |
| `mlx_classifiers_w4a8_rowquant_rowsum` | `int8/experts.py` | 16×128 | bit-identical (3 outputs) | 33 ms |
| `mlx_classifiers_w4a8_affine_gather_sm16` (MPP `matmul2d`, `int4b`) | `int8/experts.py` | 4 experts, 16 rows, N64 K128 | bit-identical | 138 ms |
| `mlx_classifiers_segmented_gather_qmm` (inlined NAX headers) | `moe.py` | 8 experts, 48 rows, N128 K256 | bit-identical | 179 ms |

So MetalPerformancePrimitives `matmul2d` cooperative tensors (int8 and `int4b_format`) and the
NAX tiles exist on the A19 and give the M5's bytes. The segmented kernel's header was inlined
from the golden venv's MLX 0.32.0 headers and compiled under mlx-swift's MLX 0.32.2 runtime.
No speed or model-level work yet: the second criterion (E2B with and without the INT8 patch,
decisions/s and headroom) is open. The 6.1 measurements leave ~2.1 GiB of headroom after
text+vision, against the INT8 patch's +1.84 GB on M5.
