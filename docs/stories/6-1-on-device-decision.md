---
type: story
status: in-progress
title: On-device load, memory budget and one decision from a camera frame
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 6.1 — On-device load, memory budget and one decision from a camera frame

An iOS app target (separate from the package) that loads E2B from app
storage, reports memory at each stage, and makes one decision from a camera
frame over the EV3 action set. Requests the increased-memory-limit entitlement.

Target Swift anchors: `Apps/EV3Demo/DecisionLoop.swift::DecisionLoop`.

## Acceptance Criteria
- **Given** an iPhone 17 (8 GB) **when** E2B with vision loads **then** the app stays inside its limit and the measured footprint is recorded.
- **Given** a camera frame **when** a decision is requested **then** the app returns one action with probabilities and `output_tokens: 0`.

## Progress (2026-10-17) — E2B with vision runs on the iPhone 17; 16/16 fixtures pass

First step: a measurement probe before the EV3 app (the camera-frame decision, the second
criterion, is still open). Built:

- `App/ClassifierProbe/` — iOS app (bundle `com.maddie.classifierprobe`, iOS 18, hand-written
  `.xcodeproj` like the exploration repo's DeployProbe) linking the local package, signed by the
  free Personal Team with `com.apple.developer.kernel.increased-memory-limit` (the free profile
  grants it). `ProbeRun.swift::ProbeRun.run` loads E2B from `Documents/gemma-4-E2B-it-qat-4bit/`,
  runs the vendored fixtures (a folder reference to `fixtures/`) through the unchanged `runCase`
  and the gate-aware `compareCase(_:result:)`, cold then warm, and logs per-fixture result,
  max|dp| and ms. `Memory.swift::MemorySample` logs `phys_footprint`, its lifetime peak,
  `os_proc_available_memory()` and MLX active/cache/peak at each stage. `--text-only` skips
  the vision tower; `--kernels` is story 5.1's probe.
- `MLXRunner.load(directory:modelId:expectedTokenizerSha256:includeVision:)` — the text-only
  flag; test `EndToEndFixtureTests.testTextOnlyLoad`. iOS build fix: the HF cache lookup no
  longer calls `homeDirectoryForCurrentUser`.
- `scripts/deploy.sh` and [the iPhone deploy runbook](../runbooks/iphone-deploy.md).

### Measurements (iPhone 17, A19, iOS 26.6.1, Release build, 2026-10-17)

Process limit with the entitlement: **6.00 GiB** (`phys_footprint + os_proc_available_memory`),
on 7.49 GiB physical; this confirms the recon's anecdotal ~6 GB cap.

| | text + vision | text only |
|---|---|---|
| load (`MLXRunner.load`) | 3.9–5.4 s | 4.3 s |
| footprint before load | 0.01 GiB | 0.01 GiB |
| footprint after load (MLX cache cleared) | 3.59 GiB | 3.27 GiB |
| MLX active after load | 3.46 GiB | 3.15 GiB |
| MLX peak (during load) | 4.03 GiB | 4.03 GiB |
| footprint lifetime peak | 4.21 GiB | 4.06 GiB |
| headroom at the end | 2.09 GiB | 2.46 GiB |

The load peak is the whole 4.03 GiB checkpoint: MLXLMCommon's `loadWeights` evaluates every
tensor (audio included) before `sanitize` drops audio, so text-only does not lower it.

**Jetsam, first attempt:** with MLX's default buffer cache the app was killed on the first
fixture (`JetsamEvent`, reason `vm-pageshortage`, 5.34 GiB resident, frontmost). This is system
memory pressure, not the 6 GiB per-app limit: right after load MLX's cache held 0.57 GiB of
freed buffers (the dropped audio tensors). The fix in the probe is `Memory.cacheLimit = 256 MiB`
plus `Memory.clearCache()` after load. With it both modes ran to the end with no kill.

Fixtures (all 16 pass; ms cold / warm; image budgets 280 unless named vt140 / vt70):

| fixture | readout | gate | max\|dp\| | cold ms | warm ms |
|---|---|---|---|---|---|
| choice-first-token-cached-list-state | first-token | full | 1.55e-4 | 627 | 134 |
| choice-first-token-multi-token-label-error | first-token (binding error) | full | n/a | 2 | 2 |
| choice-first-token-object-state | first-token | full | 8.9e-7 | 112 | 112 |
| choice-sequence-multi-token-options | sequence | full | 4.8e-6 | 179 | 178 |
| noul-first-token | first-token | full | 4.77e-4 | 89 | 97 |
| noul-sequence | sequence | full | 3.72e-4 | 113 | 111 |
| score-first-token-no-prefix | first-token | full | 2.49e-4 | 94 | 90 |
| score-sequence-object-state | sequence | full | 0 | 216 | 214 |
| choice-first-token-image-colour (280) | first-token | full | 1.99e-4 | 559 | 166 |
| choice-first-token-image-gridworld (280) | first-token | answer-and-ids | 7.77e-3 | 195 | 196 |
| choice-first-token-image-gridworld-vt140 | first-token | answer-and-ids | 3.48e-2 | 306 | 150 |
| choice-first-token-image-gridworld-vt70 | first-token | answer-and-ids | 3.90e-2 | 196 | 127 |
| choice-first-token-image-gridworld-you (280) | first-token | full | 1.76e-4 | 582 | 172 |
| choice-first-token-image-gridworld-you-vt140 | first-token | full | 4.45e-4 | 125 | 133 |
| choice-first-token-image-gridworld-you-vt70 | first-token | full | 6.55e-4 | 109 | 109 |
| noul-sequence-image | sequence | full | 0 | 170 | 167 |

The max|dp| values equal the M5 Mac's (story 4.1's table: 7.77e-3, 3.48e-2, 3.90e-2 for the
three gridworld cases), so the A19 reproduces the Mac's numerics. Text-only mode gives the same
eight text results and timings. The first cold fixture (~630 ms) and first image (~560 ms)
include Metal pipeline JIT.

Still open: the camera-frame decision over the EV3 action set (second criterion) and the
`Apps/EV3Demo/DecisionLoop.swift` target.
