---
type: prd
status: active
title: MLXClassifiers — a Swift port of the System One classifier, conformance-tested by the golden fixtures
owner: maddie
updated: 2026-10-17
---

# PRD — MLXClassifiers, the Swift port

## Problem

The golden Python implementation
([mlx-lm-as-classifiers](https://github.com/lovelacemadeline/mlx-lm-as-classifiers))
gives typed, calibrated decisions in one forward pass on a Mac. The first real
consumer is an iPhone app that drives an EV3 robot, and it cannot run Python.
It needs the same decisions, with the same numbers, in-process on an iPhone 17.

## What we build

A Swift package, `MLXClassifiers`, on MLX Swift:

1. The `/v1/systemone` wire types (ADR-0001), vendored from golden `schema/`,
   as Codable Swift types. No server: the app calls the library directly.
2. Prompt compilation and label binding with Gemma 4's chat template and
   tokenizer, producing the golden token ids.
3. A runner with the golden split prefill, first-token and sequence readouts,
   permutations and temperature calibration (ADR-0003), ported unchanged.
4. The delta prefix cache (longest common prefix, exact rewind).
5. The vision path: image blocks in a list state, the Gemma 4 vision tower,
   projector and merge, soft-token budgets 70/140/280.
6. Gemma 4 E2B (qat-4bit) as the target model; the INT8 kernels and MoE only
   where an A19 can run them.
7. An EV3 demo app as the first consumer.

How the model code is sourced is ADR-0004.

## Definition of done

Reproduce the vendored golden `fixtures/` (ADR-0002): `tokens`, `prefix_len`,
`readout_position`, `options`, `labels`, `option_token_ids`, image runs and the
`error` type exactly; `raw_probabilities` within 1e-3; the same `answer`. The
results file is the golden `mlx-classifiers-results/1` shape, so the golden
checker can judge it.

## Non-goals

- New readout semantics, new options, or any change to calibration. Changes
  land in the golden repo first, then arrive here as new fixtures.
- Any divergence from the wire schema. `schema/` is vendored, never extended.
- An HTTP server, the battery, the gridworld harness (they stay in Python).
- Training or fine-tuning.
- The 26B MoE on a phone (it does not fit; see the recon).

## Success criteria

- `swift test` passes every golden fixture on a Mac, and the text fixtures on an
  iPhone 17.
- The EV3 app makes a decision from a camera frame on device, with
  `output_tokens: 0`.

## Epics

See the [epic registry](../epics/index.md) and the
[recon](../research/swift-port-recon-2026-10.md).
