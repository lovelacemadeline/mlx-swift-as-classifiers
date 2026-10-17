---
type: story
status: blocked
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

## Progress (2026-10-17) — probe app built and signed, install blocked

First step: a measurement probe before the EV3 app. Built:

- `App/ClassifierProbe/` — iOS app (bundle `com.maddie.classifierprobe`, iOS 18, hand-written
  `.xcodeproj` like the exploration repo's DeployProbe) linking the local package.
  `ProbeRun.swift::ProbeRun.run` loads E2B from `Documents/gemma-4-E2B-it-qat-4bit/`, runs
  the 13 vendored fixtures (bundled as a folder reference to `fixtures/`) through the
  unchanged `runCase` / `compareCase`, cold then warm, and logs per-fixture PASS/FAIL,
  max|dp| and ms. `Memory.swift::MemorySample` logs `phys_footprint`, its peak,
  `os_proc_available_memory()` and MLX's active/cache/peak at launch, before and after load
  and after each fixture. `--text-only` skips the vision tower.
- `MLXRunner.load(directory:modelId:expectedTokenizerSha256:includeVision:)` — new
  `includeVision` flag; test `EndToEndFixtureTests.testTextOnlyLoad`. iOS build fix: the HF
  cache lookup no longer calls `homeDirectoryForCurrentUser` (unavailable on iOS).
- `scripts/deploy.sh` and [the iPhone deploy runbook](../runbooks/iphone-deploy.md).

Findings so far: the app builds for the phone and signs with the free Personal Team, and the
profile grants `increased-memory-limit`. The model is one safetensors shard, so all 4.03 GiB
ship (audio cannot be left behind without rewriting the file). `xcodebuild test` on macOS
passes (62 tests, 2 skipped).

**Blocked:** install fails with the free-profile three-app cap (UTM, UTM SE, WDARunner are
installed). Removing one needs Maddie (see the runbook). After that, `./scripts/deploy.sh
--measure` does install, model push and both runs with no further input.
