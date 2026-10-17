---
type: decision
status: stub
points-to: https://github.com/lovelacemadeline/mlx-lm-as-classifiers/blob/9b64a14/docs/adr/0002-golden-python-first-swift-second.md
title: Golden Python MLX-LM implementation first; Swift + vision port in a second repo, conformance-tested by fixtures
owner: maddie
updated: 2026-10-17
---

# ADR-0002 — Golden Python first, Swift second (upstream)

Upstream decision in the golden repo, mirrored here so local `ADR-0002`
citations resolve. Canonical text:
[mlx-lm-as-classifiers ADR-0002](https://github.com/lovelacemadeline/mlx-lm-as-classifiers/blob/9b64a14/docs/adr/0002-golden-python-first-swift-second.md) (golden commit 9b64a14).

What it means for this repo: this is the second repo. Its definition of done is reproducing the vendored `fixtures/`: token ids and readout positions exactly, raw probabilities within 1e-3, the same answer. Readout, masking and calibration are ported unchanged; only the runner is new.
