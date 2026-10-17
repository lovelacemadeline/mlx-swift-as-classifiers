---
type: story
status: completed
title: Per-fixture gate from golden — gridworld action cases on answer and ids, clear-winner image fixtures pass
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 4.2 — Per-fixture gate from golden, clear-winner image fixtures

Golden story 4.3 (mlx-lm-as-classifiers commit `4968020`) added an optional
top-level fixture field `gate`: absent or `"full"` is the old check;
`"answer-and-ids"` keeps every exact comparison (token ids, prefix length,
readout position, options, labels, option token ids, image runs, answer,
error type) and reports max |dp| and max |dlogit| without gating them. An
unknown value fails the case. Golden marks the three near-tied gridworld
action fixtures `answer-and-ids` (story 4.1 here: one bf16 step there is
several percent in probability, and MLX core 0.32.0 vs 0.32.2 round
differently). It adds three full-gate cases on the same frame at 280 / 140 /
70 whose answer has a clear winner (`blue` by 6.5 / 6.125 / 6.125 raw logits).
This story re-vendors those fixtures and honours the gate the same way.

Swift anchors: `Sources/MLXClassifiers/Fixtures.swift::FixtureGate`,
`Sources/MLXClassifiers/Fixtures.swift::GoldenFixture.gate`,
`Sources/MLXClassifiers/FixtureRunner.swift::compareCase`,
`Sources/MLXClassifiers/FixtureRunner.swift::CaseReport`,
`Tests/MLXClassifiersTests/ImageFixtureTests.swift::testImageFixtures`,
`Tests/MLXClassifiersTests/FixtureDecodingTests.swift::FixtureGateTests`,
`Tests/MLXClassifiersTests/FixtureDecodingTests.swift::testGateField`.

## As built

- `fixtures/` re-vendored from golden `4968020` (16 files; `VENDORED.txt`
  carries the commit, its date and a sha256 per file). `schema/` did not change
  between `9b64a14` and `4968020` and is not touched.
- `GoldenFixture.gate: String?` between `prefix_cached` and `question`, so the
  files still round-trip in key order. It stays a raw string so an unknown
  value reaches the comparison and fails the case, as in golden.
  `resolvedGate` gives `full` when the key is absent.
- `compareCase(expected:result:caseId:probTol:logitTol:gate:)` follows golden
  `compare`. `CaseReport.gate` records the gate. A new
  `compareCase(_ fixture:result:)` passes the fixture's id and gate, and every
  fixture test now uses it.
- `testImageFixtures`: the strict `XCTExpectFailure` around the three gridworld
  cases is gone. All eight image fixtures must pass under their own gate, and
  the gridworld three must carry `answer-and-ids`.
- Fixture counts updated to golden's 16 cases and 8 image cases.

## Results

`xcodebuild test` (runbook), M5, low-power mode on, Debug, 2026-10-17: 65
tests, 63 pass, 2 skipped (the environment-gated diagnostics), 0 failures,
exit 0.

| fixture | gate | image run | max \|Δp\| | answer |
|---|---|---|---|---|
| `choice-first-token-image-colour` | full | 36+256 | 1.99e-4 | blue |
| `noul-sequence-image` | full | 36+256 | 0 | no |
| `choice-first-token-image-gridworld-you` (280) | full | 128+270 | 1.76e-4 | blue |
| `choice-first-token-image-gridworld-you-vt140` | full | 128+120 | 4.45e-4 | blue |
| `choice-first-token-image-gridworld-you-vt70` | full | 128+63 | 6.55e-4 | blue |
| `choice-first-token-image-gridworld` (280) | answer-and-ids | 128+270 | 7.77e-3 (reported) | north |
| `choice-first-token-image-gridworld-vt140` | answer-and-ids | 128+120 | 3.48e-2 (reported) | north |
| `choice-first-token-image-gridworld-vt70` | answer-and-ids | 128+63 | 3.90e-2 (reported) | north |

Token ids, image runs and answers are exact in all eight. A first golden
candidate with a smaller margin (a Yes/No at about 4 logits, p 0.99) missed
here at 70 (1.6e-3) on a single 0.125 step. That is why golden settled on a gap
of at least 6 (golden story 4.3).

## Acceptance Criteria
- **Given** golden's fixtures at `4968020` **when** vendored **then** `VENDORED.txt` names the commit, date and per-file sha256, and every file decodes, validates and round-trips in key order. Met (`FixtureDecodingTests`).
- **Given** a fixture with `gate: "answer-and-ids"` **when** probabilities differ beyond 1e-3 but ids, images and answer match **then** the case passes and reports max |dp|; a wrong id, image hash or answer still fails; an unknown gate fails. Met (`FixtureGateTests`).
- **Given** the eight image fixtures **when** run from `request` **then** the five full-gate ones are within 1e-3 and the three gridworld action cases pass on answer and ids with max |dp| reported. Met (table above).
