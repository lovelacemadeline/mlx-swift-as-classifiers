---
type: guide
status: active
title: Claude Code Project Instructions — mlx-swift-as-classifiers
---

# Claude Code Project Instructions

## Documentation — docloom

All documentation in this repo follows [`docs/doc-conventions.md`](docs/doc-conventions.md)
— read it before you create, move, rename, or re-purpose any `.md` file.
Enforced by `docloom check` (pre-commit + CI):

- Every doc carries `type:` (12 closed values) and `status:` (8 closed values);
  a doc's type determines its home directory. **Never invent a new kind,
  status word, or ad-hoc label.**
- Stories live in `docs/stories/` with a matching row in
  `docs/sprint-status.yaml` (bijection-checked) and an `anchor:` list grounding
  them in real symbols/tests/routes.
- To record finished or discussed work as docs, use the `docloom` skill
  (`.claude/skills/docloom/SKILL.md`).
- Run `docloom check` before committing doc changes.

## Swift port rules

- Definition of done is the vendored golden `fixtures/` (golden ADR-0002, stub
  in `docs/adr/`). `schema/` and `fixtures/` are vendored, never edited; re-vendor
  from the golden repo and bump their `VENDORED.txt`.
- `swift build` and `swift test` must pass before a commit. Decode wire JSON
  with `SystemOneJSON`, not Foundation's JSONDecoder (it loses key order).
- docloom's anchor resolver is Python-only in v1, so stories use
  `anchor: none  # <reason>` and name their Swift anchors in the body.
