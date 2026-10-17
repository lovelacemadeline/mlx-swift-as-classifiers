---
type: story
status: completed
title: Scaffold — package, docloom, vendored schema and fixtures
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 1.1 — Scaffold — package, docloom, vendored schema and fixtures

Create the repo: Swift package `MLXClassifiers` (library + XCTest target) pinned to
mlx-swift 0.32.3, mlx-swift-lm 3.32.3 and swift-transformers 1.3.5; docloom
scaffold; the golden `schema/` and `fixtures/` vendored at golden 9b64a14 with
sha256 provenance; Codable wire types and a typed fixture loader.

## As built

- `Sources/MLXClassifiers/Schema.swift`: `SystemOneRequest`, `SystemOneState`
  (text, object, blocks), `ImageBlock`, `Question` (`NoulQuestion`,
  `ChoiceQuestion`, `ScoreQuestion`), `RequestOptions` with `Readout`,
  `LabelStyle`, `VisionSoftTokens`; `SystemOneResponse`, `Usage`, `Timing`,
  `Answer` (`NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`), `Debug`. The schema's
  constraints (additionalProperties false, minProperties, minItems, ranges,
  const, enums) throw `SchemaError` on decode.
- `JSONValue.swift`, `JSONCoding.swift`: an order-preserving JSON tree and a
  Codable bridge (`SystemOneJSON`). Foundation's JSONDecoder loses object key
  order, and choice option order decides the letters.
- `Fixtures.swift`: `GoldenFixture` and `Fixtures.load(directory:)`, sorted by
  file name and validated like golden `load_fixtures` and `FixtureCase`.
- Tests (`Tests/MLXClassifiersTests/`): `FixtureDecodingTests` (all 13 decode
  and validate, option order kept, probabilities and answers consistent, image
  runs and PNG sha256, byte-order round trip of every file) and
  `SchemaDriftTests` / `SchemaValidationTests` (every schema property and enum
  has a Swift counterpart; constraints enforced; responses round-trip).
- Anchors are Swift, which docloom v1 cannot resolve; hence `anchor: none`.

## Acceptance Criteria
- **Given** the vendored fixtures **when** `swift test` runs **then** all 13 decode, validate and round-trip in key order.
- **Given** the vendored schema **when** a property or enum value is added upstream **then** `SchemaDriftTests` fails until the Swift types follow.
- **Given** a request that breaks a schema constraint **when** it is decoded **then** decoding throws.
