---
type: story
status: completed
title: HTTP shim and a gridworld comparison — Swift against Python on the 26B
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 7.2 — HTTP shim and a gridworld comparison — Swift against Python on the 26B

**The shim.** A new executable target and product, `mlx-classifiers-serve`
(`Sources/mlx-classifiers-serve/Serve.swift`), serves golden `server.py`'s
`POST /v1/systemone` on 127.0.0.1 over one loaded `MLXRunner`, with no chat
route. It uses Network.framework's `NWListener` (a system framework, so no new
package dependency) and minimal HTTP/1.1 with keep-alive and Content-Length
bodies. The body is the wire schema both ways (`SystemOneJSON`, key order kept).
Statuses follow golden:

- 200 with the response;
- 422 `request body is not valid JSON`;
- 422 `request does not match the SystemOneRequest schema` with an `errors`
  list (field paths are best effort, since Swift's `DecodingError` is not
  pydantic);
- 422 with the message on a `ClassifierError`;
- 404 otherwise.

The `x-prefix-cache-tokens` header reports the first prefill's cached tokens.
`x-prefix-cache` is always `none`, because there is no declared-prefix LRU:
the runner's delta store serves repeated prefixes. Decisions run one at a time
on one serial queue. Flags: `--model`, `--model-dir`, `--port`, and
`--vision-tokens {70,140,280}` (sets `MLXRunner.imageTokens.maxSoftTokens`,
the counterpart of golden `serve --vision-tokens`). Build it with xcodebuild
(Release) so the metallib sits beside the binary; the command is in the file
header. Library changes: `MLXRunner.imageTokens` is now settable, and
`MLXRunner.clearCallRecords()` keeps a long-lived server's call records from
growing (golden `del calls[:]`).

**The comparison.** `gridworld play --backend http --labels words --legal-only
--memory summary --mode image --think never --stuck-after 10 --episodes 10
--seed 0` (gridworld `a2bcfa8`, with the story 4.10 stuck rule) ran headless
against each server in turn, with one 26B in memory at a time:

- the Swift shim (`--vision-tokens 140`, Release);
- golden `mlx-classifiers serve --vision-tokens 140 --calibration <none>`
  (golden `8b39076`). This is the stock 4-bit path with `moe_kernel` at its
  default (on); story 7.1 measured it bit-identical to `moe_kernel=False` on
  every 26B fixture.

The M5 Mac had low-power mode on (`pmset lowpowermode 1`) before and after
both runs. Records are in the session scratchpad `swift-vs-python/`.

| | Swift shim | Python server |
|---|---|---|
| wins (goal) | 4 / 10 | 4 / 10 |
| endings | 4 goal, 1 dead, 5 stuck | 4 goal, 6 stuck |
| ticks | 146 | 155 |
| oracle agreement | 66.4% | 65.8% |
| decisions/s (end to end) | 0.49 | 0.49 |
| backend p50 per decision | 1747 ms | 1967 ms |
| wall time, 10 episodes | 297 s | 317 s |

**Per-tick action agreement while the trajectories were still identical**
(same observation on both sides, counted up to and including the first tick
whose action differs): **94 of 100** shared ticks. Seeds 3, 6, 7 and 8 match
for the whole episode (14, 13, 11 and 17 ticks). The other six each diverge
once on a near-tie, after 4 to 16 identical ticks. Once they diverge the
episodes play out differently: seed 1 is a Swift goal but a Python stuck, and
seed 2 is a Swift death but a Python goal. Over the whole run the totals come
out the same: 4 wins each and the same oracle agreement within a point.

The near-ties match story 7.1's fixture result. The Swift and Python 26B agree
on token ids and clear winners, but the bf16 MoE drift (up to ~1.7 logits on
the near-tied action case) can flip an argmax when two actions are close. On
the wire route, `decide`'s permuted readouts can do it too: the vt140 action
fixture request answers `east` on the shim, while the fixture's single readout
gives `south`.

## Acceptance Criteria
- **Given** the shim on the 26B **when** each `fixtures/26b` request is posted
  **then** it answers 200 with a wire `SystemOneResponse` (the four clear cases
  answer as the fixtures do). **When** the body is not JSON, or is missing
  `questions`, **then** it answers 422 with golden's `detail` (smoke test, this
  story).
- **Given** the same gridworld configuration and seeds **when** it runs against
  the shim and against golden's server **then** both complete headless, and
  the table above records wins, oracle agreement, decisions/s and per-tick
  action agreement under the same power state.

## Related
- [Story 7.1 — 26B through MLXRunner](7-1-26b-through-mlxrunner.md)
- [Epic 7](../epics/epic-7.md)
