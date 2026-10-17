---
type: story
status: pending
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
