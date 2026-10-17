---
type: register
status: active
register:
  owns-no-clause-ids: ADR numbers, not clause ids; every ADR-NNNN citation is resolved by globbing the ADR directory
title: ADR Ledger
---

# ADR Ledger

Allocate the next ADR number here before creating `NNNN-<slug>.md`. Upstream
decisions get a local `status: stub` pointer so every `ADR-NNNN` citation
resolves locally. ADR-0001 to ADR-0003 keep their golden numbers.

| ADR | Title | Status | Local/Upstream |
|---|---|---|---|
| ADR-0001 | [Jev-compatible wire shape](0001-jev-compatible-wire-shape.md) | stub | upstream (golden) |
| ADR-0002 | [Golden Python first, Swift second](0002-golden-python-first-swift-second.md) | stub | upstream (golden) |
| ADR-0003 | [First-token readout by default](0003-first-token-readout-default.md) | stub | upstream (golden) |
| ADR-0004 | [Vendor the Gemma 4 model code](0004-vendor-gemma4-model-code.md) | active | local |
