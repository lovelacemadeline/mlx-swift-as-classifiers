---
type: decision
status: stub
points-to: https://github.com/lovelacemadeline/mlx-lm-as-classifiers/blob/9b64a14/docs/adr/0003-first-token-readout-default.md
title: First-token readout over single-token labels is the default; sequence scoring is opt-in
owner: maddie
updated: 2026-10-17
---

# ADR-0003 — First-token readout by default (upstream)

Upstream decision in the golden repo, mirrored here so local `ADR-0003`
citations resolve. Canonical text:
[mlx-lm-as-classifiers ADR-0003](https://github.com/lovelacemadeline/mlx-lm-as-classifiers/blob/9b64a14/docs/adr/0003-first-token-readout-default.md) (golden commit 9b64a14).

What it means for this repo: the port implements both readouts with the golden defaults, including the LabelTokenizationError for multi-token first-token labels.
