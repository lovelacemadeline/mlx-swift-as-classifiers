---
type: research
status: active
title: Recon — what MLX Swift already has for a Gemma 4 System One port, and what we must vendor
owner: maddie
updated: 2026-10-17
---

# Recon — MLX Swift for the Gemma 4 port (2026-10-17)

Time-boxed, no GPU, no model loaded. Sources: shallow clones of
[ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) at tag `0.32.3`
(`1960120`, 2026-09-30), [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)
at tag `3.32.3` (`3b339ad`, 2026-09-30), [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers)
`1.3.5` (`4847310`), the golden repo at `9b64a14`, and the local HF cache. Paths
below are relative to those checkouts. This is an input to [ADR-0004](../adr/0004-vendor-gemma4-model-code.md).

**Packaging fact first.** The LLM/VLM libraries left mlx-swift-examples: its
latest tag `2.29.1` holds only `MLXMNIST` and `StableDiffusion`, and its README
points to mlx-swift-lm ("containing _only_ reusable libraries",
[README](https://github.com/ml-explore/mlx-swift-examples/blob/main/README.md)). So the package
depends on **mlx-swift-lm 3.32.3**, not mlx-swift-examples. mlx-swift-lm 3.x
dropped its tokenizer dependency (`Package.swift` lists only mlx-swift and
swift-syntax); tokenizers plug in via the `MLXLMCommon.Tokenizer` protocol
(`Libraries/MLXLMCommon/Tokenizer.swift`), the README's default being
swift-transformers `Tokenizers`. mlx-swift 0.32.3 bundles MLX core **v0.32.2**
(submodule `Source/Cmlx/mlx` = `1f8e74e`, the `v0.32.2` tag); golden runs mlx 0.32.0.

## 1. Gemma 4 text (E2B PLE, 26B-A4B MoE) — present

- `Libraries/MLXLLM/Models/Gemma4Text.swift` (978 lines), registered as
  `gemma4_text`/`gemma4` in `LLMModelFactory.swift:63-65`. Added in
  [#180](https://github.com/ml-explore/mlx-swift-lm/pull/180) and
  [#185](https://github.com/ml-explore/mlx-swift-lm/pull/185) (2026-04-13); the first
  tag containing it is `3.31.3`.
- PLE: `hidden_size_per_layer_input`, `vocab_size_per_layer_input`,
  `per_layer_input_gate`/`per_layer_projection`/`post_per_layer_input_norm`
  (lines 57, 538-588). KV sharing: last `num_kv_shared_layers` reuse K/V (260-280).
  MoE: `enable_moe_block`, `Gemma4TextRouter` + `SwitchGLU` experts (430-510).
- **Gap for us:** the backbone forward is `func callAsFunction(_ inputs:, cache:)`
  over token ids only (line 748, internal); the backbone is exposed only as
  `@_spi(GemmaEncoder) public let model` (line 848). There is no public way to
  pass `inputsEmbeds`/`perLayerInputs`, nor to stop before `lm_head` and slice the
  readout row, which golden's split prefill needs (`mlx_runner.py` lines 1-60).
  Golden's reference is `mlx_lm/models/gemma4_text.py` (675 lines) + `gemma4.py`
  (92) in the golden `.venv` (mlx-lm 0.31.3).

## 2. Gemma 4 vision (tower + projector + merge) — present, but private

- `Libraries/MLXVLM/Models/Gemma4.swift` (3382 lines): `Gemma4VisionPatchEmbedder`,
  `Gemma4VisionPooler` (`pooling_kernel_size` 3), `Gemma4VisionModel`,
  `Gemma4MultimodalEmbedder` (`embed_vision`), `gemma4MaskedScatter` merge,
  image/boi/eoi ids 258880/255999 (lines 1754-2071, 603-607).
- Matches golden semantics: PLE ids zeroed at multimodal positions (lines
  1999-2012; golden story 6.3), per-image aspect-preserving resize, configurable
  `max_soft_tokens` default 280 (`Gemma4ProcessorConfiguration`, 2863-2941).
- **Differences from golden `src/mlx_classifiers/vision/`** (vendored from mlx-vlm 0.7.8):
  (a) every vision class is `private final class`, and `Gemma4TextLanguageModel`
  is internal (line 1364), so nothing is callable from outside the module;
  (b) resize is CoreImage `MediaProcessing.resampleBicubic` in sRGB-tone-curve
  float space (lines 2705-2722), golden is PIL `Image.BICUBIC` on uint8
  (`vision/processor.py:45`): not bit-identical, a likely >1e-3 probability miss
  on image fixtures; (c) `prepare` runs the whole multimodal prompt in one call
  (2106-2130), not golden's split prefill with image rows.

## 3. KV cache — the pieces exist; the exact rewind is ours to write

- `protocol KVCache` (`Libraries/MLXLMCommon/KVCache.swift:46`): `offset`,
  `update(keys:values:)`, `state`/`metaState` get+set, `isTrimmable`,
  `isTrimmable(after:)`, `trim(_:)`, `copy()`.
- `KVCacheSimple` (408): trims to any length (`isTrimmable` true, 488-491).
- `RotatingKVCache` (558): `isTrimmable(after:)` is `offset + positions < maxCacheSize`
  (865-871); past the wrap, `trim` is "merely consistent", not exact (873-884).
  Its fields are `private`, but `state`/`metaState` round-trip it.
- Gemma 4 `newCache` makes one cache per KV-owning layer via
  `makeHybridAttentionKVCache` (rotating for sliding layers, window 512 on E2B)
  (`Gemma4Text.swift:951-960`).
- Cache-only prefill of N-1 then an L=1 readout is just two forward calls on the
  same `[KVCache]`; the readout slice needs the vendored backbone (§1). Golden's
  exact rotating rewind (W-1 rows behind a placeholder, `mlx_runner.py:134-192`)
  must be ported, as a `KVCache` subclass or via `state`/`metaState`.
  `PromptCacheReusePolicy.swift` is generation-oriented and internal; not reusable.

## 4. Custom Metal kernels — API present; the golden kernels are M5-shaped

`MLXFast.metalKernel(name:inputNames:outputNames:source:header:ensureRowContiguous:atomicOutputs:)`
returns `MLXFastKernel`, called with `template:grid:threadGroup:outputShapes:outputDTypes:initValue:verbose:`
(`Source/MLX/MLXFastKernel.swift:98-108, 176-186`), the same as Python `mx.fast.metal_kernel`.

| golden kernel (file) | features |
|---|---|
| `int8_linear_epilogue_w8a8opt_nk` (`int8/wiring.py:151`) | MPP `matmul2d` cooperative tensors, int8×int8→int, half out; "needs Apple M5 / Metal 4" (line 9) |
| `fused_rowquant_int8_w8a8opt` (`int8/wiring.py:207`) | plain MSL, `simd_max`, half in |
| `mlx_classifiers_int8_gemv_nk` (`int8/wiring.py:275`) | plain MSL, simdgroup-per-rows GEMV |
| `mlx_classifiers_w4a8_affine_gather_sm{sm}` (`int8/experts.py:133`) | MPP `matmul2d` with `int4b_format`, bfloat out |
| `mlx_classifiers_w4a8_rowquant_rowsum` (`int8/experts.py:220`) | plain MSL, `simd_sum`, bfloat out |
| `mlx_classifiers_segmented_gather_qmm` (`moe.py:220`) | inlines MLX's `quantized_nax.h` + `steel/gemm/nax.h` (NAX tiles); `available()` refuses unless `mx.__version__` is `0.32.*` and arch `applegpu_g17+` (`moe.py:79-88`) |

A19: every A19 has a Neural Accelerator per GPU core
([MacRumors](https://www.macrumors.com/2025/09/09/iphone-17-a19-chip/)), the same
feature M5 has, so MPP `matmul2d` is plausible on iOS 26; unverified until it runs
on a phone (E5). The MoE kernel is pinned to MLX 0.32 headers; mlx-swift 0.32.3
ships 0.32.2 headers, so re-check that the inlined `quantized_nax.h` matches
(E5). `bfloat` needs MSL 3.1+; A19 runs Metal 4 (unverified on device). This is E5 only: no text
fixture needs a custom kernel (fixtures are the stock 4-bit path, `generator.mlx` 0.32.0).

## 5. Memory — E2B fits a phone; the 26B does not

- RAM: iPhone 17 (A19) 8 GB; iPhone Air and 17 Pro/Pro Max (A19 Pro) 12 GB
  ([MacRumors](https://www.macrumors.com/2025/09/09/iphone-17-a19-chip/),
  [gori.me](https://gori.me/iphone/161530)).
- Per-app limit: Apple publishes no number; use `os_proc_available_memory()`
  ([docs](https://developer.apple.com/documentation/os/os_proc_available_memory))
  and request `com.apple.developer.kernel.increased-memory-limit`
  ([docs](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit)).
  A forum report puts a flat ~6 GB cap on 8 and 12 GB iPhones even with the
  entitlement ([thread 834805](https://developer.apple.com/forums/thread/834805)); anecdotal, measure in E6.
- E2B qat-4bit on disk (snapshot `42f6273`, one 4.03 GiB `model.safetensors`,
  header sizes): decoder layers 1742 MiB, PLE table `embed_tokens_per_layer` 1260 MiB
  (weight + scales + biases), `embed_tokens` 216 MiB, **audio tower 581 MiB** (unused),
  vision tower 319 MiB, `embed_vision` 0.6 MiB. Text + vision without audio ≈ 3.45 GiB.
- Golden measurement on M5 (story 2.9, `docs/stories/2-9-memory-budget.md`): E2B
  base weights 3.38 GB MLX-active, +0.34 GB vision, +1.84 GB INT8 patch, +0.07 GB
  delta store; footprint 7.5 GB with INT8+vision+delta. So on iPhone: stock 4-bit
  + vision ≈ 3.7 GB active fits under ~6 GB; the INT8 patch (+1.84) does not fit
  comfortably on an 8 GB iPhone 17. Skipping the audio tensors at load is free memory.
- 26B-A4B: 15 GB on disk locally; golden measured 14.47 GB active at load, 20.06 GB
  peak with INT8+vision. That is 2.4x the 6 GB app cap and above the 12 GB of
  physical RAM of the biggest iPhone 17. Not a candidate.

## 6. Tokenizer — swift-transformers maps Gemma to BPE; parity untested

- swift-transformers 1.3.5 maps `"GemmaTokenizer": BPETokenizer.self`
  (`Sources/Tokenizers/Tokenizer.swift:166`). Gemma 4's `tokenizer.json`: model
  `BPE`, `byte_fallback: true`, normalizer `Replace`, pre-tokenizer `Split`,
  decoder `Sequence`, 24 added tokens; `tokenizer_config.json` class `GemmaTokenizer`.
- Golden renders with HF `apply_chat_template` (`template.py:115-134`, Jinja) and then
  `tokenizer.encode(text, add_special_tokens=False)` (`mlx_runner.py:361`). Swift's
  equivalent is swift-transformers' `applyChatTemplate` (swift-jinja). Two parity
  risks: Jinja rendering and BPE/byte-fallback merges. Story 1.2 tests both against
  `expected.tokens` and `option_token_ids` in all 13 fixtures; the label-binding
  rule (tokenize `prompt + label`, take ids after the prompt's own) is in the golden
  fixture-format reference.

## Bottom line

Everything the port needs exists in MLX Swift: Gemma 4 text with PLE and MoE,
the vision tower, KV caches, `metalKernel`. But the parts we must control
(embedding injection, hidden-state readout, split prefill, PIL-exact resize,
exact rotating rewind) are internal or private. Decision in ADR-0004.
