---
type: decision
status: active
title: Vendor the Gemma 4 text and vision model code from mlx-swift-lm; depend on mlx-swift, MLXLMCommon and swift-transformers
owner: maddie
updated: 2026-10-17
---

# ADR-0004 — Vendored Gemma 4 model code vs the mlx-swift-lm dependency

## Status
Active.

## Context
The golden runner (ADR-0002) needs four things from the model: run the backbone
from input embeddings with per-layer inputs (image soft tokens in place of the
placeholders, PLE pad id 0 there); stop before `lm_head` and project only the
readout rows; split prefill (cache-only rows, then the readout pass) over a
`[KVCache]`; and an exact rewind of sliding-window caches for the delta cache.
The [recon](../research/swift-port-recon-2026-10.md) found that mlx-swift-lm
3.32.3 has Gemma 4 text (PLE, KV sharing, MoE) and Gemma 4 vision, but the
backbone forward is token-ids-only and internal (`@_spi(GemmaEncoder)` at
most), every vision class is `private`, the VLM `prepare` runs one monolithic
pass, and its image resize is CoreImage bicubic where golden is PIL bicubic on
uint8. mlx-swift-examples no longer ships these libraries at all (2.29.1).

## Decision
1. **Vendor** `Libraries/MLXLLM/Models/Gemma4Text.swift` and the Gemma 4 parts
   of `Libraries/MLXVLM/Models/Gemma4.swift` from mlx-swift-lm **3.32.3**
   (`3b339ad`, MIT, © 2024 ml-explore) into `Sources/MLXClassifiers/Models/`,
   each file with a provenance header (upstream path, tag, commit, licence, list
   of local changes), and the upstream licence beside them. Local changes are
   the minimum the runner needs: public embeddings-in / hidden-out entry
   points and the per-layer-input override.
2. **Depend** (exact pins) on mlx-swift 0.32.3 (`MLX`, `MLXNN`, `MLXFast`
   kernels), mlx-swift-lm 3.32.3 for `MLXLMCommon` only (`KVCache`,
   `KVCacheSimple`, `RotatingKVCache`, weight loading and quantisation helpers),
   and swift-transformers 1.3.5 `Tokenizers`. Not `MLXLLM` or `MLXVLM`.
3. The image preprocessing (decode, aspect-preserving resize, normalise) is
   written here to match PIL `Image.BICUBIC` on uint8, not CoreImage, because the
   image fixtures pin raw probabilities to 1e-3.
4. The exact rotating-cache rewind is ported from golden `mlx_runner.py`
   (`_rotating_rewind`), not taken from `RotatingKVCache.trim`.
5. Re-vendoring is deliberate: bump the provenance headers, rerun the fixtures.

## Consequences
- We own roughly 2-3k lines of model code (Gemma4Text.swift is 978 lines; the vision and multimodal parts of the 3382-line VLM file are most of the rest) and its upstream drift; fixture parity
  is the guard. The golden repo did the same for mlx-vlm's vision path.
- mlx-swift-lm's other models, generation loop and Foundation Models adapter are
  not compiled into our target; the package still resolves them (swift-syntax
  prebuilt) because SwiftPM resolves whole packages.
- If upstream later exposes embeddings-in and hidden-out publicly, ADR-0004 can
  be superseded and the vendored copy dropped.

## Alternatives rejected
- **Depend on MLXLLM/MLXVLM as is**: the entry points we need are internal or
  private; `@_spi` is not a stable contract. Rejected.
- **Fork mlx-swift-lm**: a whole fork to change two files, and a fork to rebase
  on every tag. Rejected.
- **Upstream PR first**: worth sending later, but it blocks E2 on review. Not now.
- **Port from Python `mlx_lm/models/gemma4_text.py` by hand**: the Swift port
  already exists and is tested upstream; vendoring it is less code to get wrong.
