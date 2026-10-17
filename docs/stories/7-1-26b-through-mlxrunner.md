---
type: story
status: completed
title: 26B through MLXRunner — bidirectional image blocks, the five 26B fixtures pass
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 7.1 — 26B through MLXRunner — bidirectional image blocks, the five 26B fixtures pass

`mlx-community/gemma-4-26B-A4B-it-qat-4bit` (revision `0e3cbab3…`) now loads
through `MLXRunner.load(directory:modelId:expectedTokenizerSha256:)` on the stock
4-bit path and reproduces golden's `fixtures/26b/` (golden story 4.4, vendored
from golden `8b39076`, `fixtures/VENDORED.txt`) under the same per-fixture gate
rules as the E2B set.

**What broke, and the fix.** Only one thing: the loader threw
`notImplemented` on `use_bidirectional_attention: "vision"`. The vendored
`Gemma4Text.swift` already carried the MoE block (router, `SwitchGLU` experts,
the dense MLP beside it), `attention_k_eq_v` on the full layers, the
proportional partial RoPE and the per-layer quantisation (8-bit dense MLP and
router, 4-bit experts), and the vendored vision tower already had
`standardize`. The 26B has `num_kv_shared_layers: 0`, so there is no KV sharing
to handle, and the stock path holds no duplicated weights (golden story 2.9's
duplication concern is the INT8 swap, not ported here). Ported from golden
`vision/merge.py` / `mlx_runner.py`:

- `Gemma4TextModelInner.callAsFunction(_:cache:inputEmbeddings:blocks:)` —
  with `blocks` (`[L]` int32, image k's rows hold k, text rows -1) and more
  than one row, the per-layer-type masks are built as arrays and
  `Gemma4TextModelInner.overlayImageBlocks` lets every pair of rows in the same
  image block attend both ways (golden `_overlay`, the last N columns are the
  segment).
- `MLXRunner.visionBidirectional` (from the text config); `MLXRunner.rows`
  passes `blocks` on image segments; `MLXRunner.cacheOnly` moves chunk bounds
  off image runs with `boundsOffRuns` (golden `_bounds_off_runs`).

Diagnostic: with the overlay switched off, the vt140 action case answers
`west` (golden `south`) with a +3.1 logit delta, so the mask is load-bearing.

## Results

`Tests/MLXClassifiersTests/Gemma26BFixtureTests.swift::test26BFixtures`
(xcodebuild, M5 Mac, low-power mode on, `pmset -g | grep lowpowermode` → 1):

| fixture | gate | max \|Δp\| | answer | raw-logit deltas |
|---|---|---|---|---|
| noul-first-token | full | 2.3e-9 | yes ✓ | +0.25, +0.375 |
| noul-sequence | full | 2.8e-8 | no ✓ | −0.094, +0.063 |
| choice-sequence-multi-token-options | full | 3.9e-4 | refund ✓ | +0.125, +0.188, +0.062 |
| choice-first-token-image-gridworld-you-vt140 | full | 4.4e-6 | blue ✓ | −0.875, −0.711, −0.063, −0.438 |
| choice-first-token-image-gridworld-vt140 | answer-and-ids | 0.149 (not gated) | south ✓ | −0.469, +1.656, +1.594, +0.445, +0.469, +0.906 |

Token ids, image runs and answers are exact on all five. The logit drift is
larger than the E2B's (text ±0.125, image ±0.5): 30 MoE layers of bf16 with
Swift's stock `SwitchGLU` against golden's. Golden's own stock path
(`moe_kernel=False`) and MoE-kernel path both reproduce the fixtures with
Δ = 0, so the drift is Swift against Python, not kernel against stock.

## Load, memory and ms per decision — Swift against Python, same state

Same machine, same power state (low-power mode on), the same fixture
requests through `runCase` / golden `run_case`, cold prompt cache, median of 7
after a warm-up. "New frame" clears the vision soft-token memo before each
run (an unseen image); "memo hit" reuses it. Swift from
`test26BDecisionTiming` (`TIME26B`, `LOAD26B`, `MEM26B` lines), Python from a
scratch script over `MLXRunner.load(..., moe_kernel=...)`.

| | Swift (stock) | Python stock (`moe_kernel=False`) | Python default (`moe_kernel=True`) |
|---|---|---|---|
| load (weights + tokenizer + vision) | 7.43 s | 5.35 s | 5.58 s |
| active memory after load | 15.61 GB | 15.61 GB | 15.61 GB |
| peak memory | 16.05 GB | 16.32 GB | 16.32 GB |
| text first-token (`noul-first-token`) | 540 ms | 673 ms | 488 ms |
| image vt140, new frame | 1808 ms | 1687 ms | 1676 ms |
| image vt140, memo hit | 1060 ms | 1221 ms | 1203 ms |

The ~200 ms 26B decision quoted earlier is golden's INT8 + MoE kernel at full
power; it was not re-measured here (INT8 is not ported, and low-power mode was
on throughout). Swift's text decision beats golden's stock path and sits
within 11% of the MoE kernel; its vision encode is ~120 ms slower per new
frame than golden's fast tower.

## Acceptance Criteria
- **Given** the vendored `fixtures/26b/` **when** it is loaded **then** it is
  golden 4.4's five cases, all pinned to the 26B, and only the vt140 action
  case is `answer-and-ids` (`testVendored26BFixturesArePinnedToThe26B`).
- **Given** chunk bounds and an image run **when** they are moved **then** they
  match golden `_bounds_off_runs` (`testBoundsOffRunsMatchesGolden`), and the
  overlay opens exactly the image-block pairs (`testImageBlockOverlay`).
- **Given** the 26B snapshot at the fixtures' revision **when** the five
  fixtures run from their requests **then** ids, image runs and answers are
  exact and every full-gate case is within 1e-3 (`test26BFixtures`); without
  the snapshot (or the metallib) the tests skip.
- **Given** the 26B loaded **when** the timing test runs **then** it prints
  load, memory and ms per decision for text and image (`test26BDecisionTiming`).

## Related
- [Epic 7](../epics/epic-7.md)
- [Story 4.1 — vision tower and image fixtures](4-1-vision-tower-and-image-fixtures.md)
