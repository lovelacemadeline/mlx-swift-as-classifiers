---
type: decision
status: stub
points-to: https://github.com/lovelacemadeline/mlx-lm-as-classifiers/blob/9b64a14/docs/adr/0001-jev-compatible-wire-shape.md
title: Adopt TypeSafe Jev's /v1/systemone wire shape, with x-local extensions
owner: maddie
updated: 2026-10-17
---

# ADR-0001 — Jev-compatible wire shape (upstream)

Upstream decision in the golden repo, mirrored here so local `ADR-0001`
citations resolve. Canonical text:
[mlx-lm-as-classifiers ADR-0001](https://github.com/lovelacemadeline/mlx-lm-as-classifiers/blob/9b64a14/docs/adr/0001-jev-compatible-wire-shape.md) (golden commit 9b64a14).

What it means for this repo: the Swift types in `Sources/MLXClassifiers/Schema.swift` are the vendored `schema/` exactly, x-local extensions included; this repo never extends the schema.
