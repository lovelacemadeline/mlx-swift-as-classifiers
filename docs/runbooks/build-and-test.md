---
type: runbook
status: active
title: Build and test — xcodebuild for the GPU tests, swift test for the rest
owner: maddie
updated: 2026-10-17
---

# Build and test — xcodebuild for the GPU tests, swift test for the rest

**The fact to remember:** command-line `swift build` does **not** compile MLX's
Metal shaders (no `mlx-swift_Cmlx.bundle` / `default.metallib` is produced), so
anything that runs a model on the GPU fails at runtime under `swift test`.
`xcodebuild` runs the Metal compiler and puts `mlx-swift_Cmlx.bundle` (with
`default.metallib`) beside the test bundle. Verified 2026-10-17 on an M5 Mac,
Xcode 26.6 (17F113), Swift 6.3.3, Metal toolchain 17.6.109.

## Prerequisites

- Xcode 26 with the Metal toolchain: `xcrun -f metal` must print a path. If it
  does not, `xcodebuild -downloadComponent MetalToolchain`.
- For the GPU tests: `mlx-community/gemma-4-E2B-it-qat-4bit` at revision
  `42f62737af7a9fd8c1d55d79666c1a217be4e2e2` in the local Hugging Face cache
  (`~/.cache/huggingface/hub`, or `$HF_HUB_CACHE` / `$HF_HOME/hub`). Tests never
  download. `MLXCLASSIFIERS_MODEL_DIR=/path/to/snapshot` overrides the lookup.

## Full suite, GPU included (the definition-of-done run)

```bash
xcodebuild test -scheme MLXClassifiers -destination 'platform=macOS' \
  -derivedDataPath .build/xcode -skipMacroValidation -skipPackagePluginValidation
```

- `-derivedDataPath .build/xcode` keeps the build inside the (gitignored) `.build/`.
- `-skipMacroValidation -skipPackagePluginValidation`: swift-syntax macros from
  the dependency graph otherwise need an interactive trust prompt.
- First build about 90 s on an M5 in low-power mode (MLX C++ plus the Metal
  shaders); incremental builds are seconds. Debug configuration.
- Filter one class: `-only-testing:MLXClassifiersTests/TextFixtureTests`.
- `TextFixtureTests` prints one `FIXTURE <id>: max|dp| = ...` line per case,
  `EndToEndFixtureTests` one `E2E <id>: ...` line, `DecideTests` `DECIDE` and
  `REPLAY` lines; grep for them.
- `ImageFixtureTests` prints `IMAGE <id>` (the five image fixtures), `TOWER`
  (soft-token parity with golden) and `ENCODE` (per-image encode time) lines;
  `ImagePreprocessTests` `PREPROCESS` lines. The golden vision reference
  (`Tests/MLXClassifiersTests/GoldenVision/`) is regenerated with
  `scripts/golden_vision_reference.py [DUMP_DIR]` (golden venv). Diagnostics:
  `TEST_RUNNER_MLXCLASSIFIERS_VISION_DUMP=<dir>` writes the tower's
  intermediates, `TEST_RUNNER_MLXCLASSIFIERS_GOLDEN_SOFT=<safetensors>` runs
  the image fixtures on injected soft tokens.
- `PromptCacheTests` prints one `DELTA <scenario>[step]` line per delta-store
  step and a `TIMING` line (full vs delta prefill).
- The golden decide reference (`Tests/MLXClassifiersTests/GoldenDecide/`) is
  regenerated with the golden venv:
  `../mlx-lm-as-classifiers/.venv/bin/python scripts/golden_decide_reference.py`
  (loads the model from the local HF cache; set `HF_HUB_OFFLINE=1`). The golden
  delta reference (`Tests/MLXClassifiersTests/GoldenDelta/`) the same way with
  `scripts/golden_delta_reference.py`.

## Without a GPU (CI, Linux-less sandboxes, quick loops)

```bash
swift build && swift test
```

The GPU tests (`TextFixtureTests`) skip themselves when the MLX metallib is not
beside the test bundle, which is always the case under `swift test`. To skip
them explicitly:

- `swift test`: `MLXCLASSIFIERS_SKIP_GPU=1 swift test`
- `xcodebuild test`: `TEST_RUNNER_MLXCLASSIFIERS_SKIP_GPU=1 xcodebuild test ...`
  (xcodebuild passes `TEST_RUNNER_`-prefixed variables to the test process with
  the prefix removed).

They also skip when the model snapshot is not in the cache.

## Failure modes

- `Failed to load the default metallib` at runtime: the binary was built by
  `swift build`; use `xcodebuild`.
- `xcodebuild` asks to trust macros or plugins: add the two `-skip...Validation` flags.
- A fixture misses by a bf16 step: logit deltas of 0.0625 or 0.125 at logits
  near 16 to 32 are one bf16 ulp; see story 2.1 for the measured baseline.
