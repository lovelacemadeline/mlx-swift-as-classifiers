---
type: story
status: completed
title: Exact KV rewind and the longest-prefix store
owner: maddie
updated: 2026-10-17
anchor: none  # Swift repo: docloom v1 resolves python symbols and tests only; the Swift anchors are named in the story body
---

# Story 3.1 — Exact KV rewind and the longest-prefix store

Port golden `prompt_cache.py` (`PromptCacheStore`, `common_prefix_len`) and
`mlx_runner.rewind_cache`: a `KVCacheSimple` trims to any length; a
`RotatingKVCache` rewinds exactly only under the golden W-1 rule, otherwise the
store tries the next entry. `RotatingKVCache.trim` is not exact past the wrap
(see the recon), so it is not used.

Swift anchors: `Sources/MLXClassifiers/PromptCache.swift::PromptCacheStore`,
`Sources/MLXClassifiers/PromptCache.swift::rewindCache`,
`Sources/MLXClassifiers/MLXRunner.swift::MLXRunner.start`,
`Tests/MLXClassifiersTests/PromptCacheTests.swift::testDeltaHitMatchesColdRun`,
`Tests/MLXClassifiersTests/PromptCacheTests.swift::KVRewindTests`,
`Tests/MLXClassifiersTests/PromptCacheTests.swift::testFixtureRunIgnoresTheStore`,
`Tests/MLXClassifiersTests/PromptCacheTests.swift::testDeltaPrefillTiming`.

## As built

- `PromptCache.swift`: `PromptCacheStore` follows golden line for line. Lookup
  is a linear scan of the entries by common-prefix length, capped at `maxLen`
  and sorted stably, so ties go to the least recently used entry, as with
  Python's `list.sort`. It stops at `minLen`. A refused rewind is counted in
  `unrepresentable` and the next entry is tried. Insert bumps an existing key,
  refuses an entry over `maxBytes`, and evicts LRU until both bounds hold.
  Defaults: 8 entries and 2 GiB, behind a lock. `withoutDelta` / `NoDeltaRunner`
  is golden's `without_delta` proxy over a `DeltaCachingRunner`
  (`cachePrefix(_:delta:)`, `prefill(...delta:)`).
- `rewindCache` is golden `rewind_cache`.
  - `KVCacheSimple`: copy, `trim`, copy (which compacts to the hit rows).
  - `RotatingKVCache`: its fields are private, so it is read and rebuilt
    through `state` / `metaState`. In temporal order it holds rows
    `[start, offset)`. With `start == 0` it is sliced to the hit. Otherwise a
    hit `h` is representable iff `h - start >= W - 1`. The rewound cache is
    those W - 1 rows behind one zero row, with `idx = W` and `wrapped = false`.
    Both upstream update paths drop or overwrite that row before attending.
  - A keep prefix, a wrapped ring or a rewind into evicted rows refuses the
    whole list.
- `MLXRunner`:
  - `promptCache` is on by default, as in golden; nil turns it off.
  - `start` is golden `_start`: the longer of the declared prefix and the
    store's hit, both copies.
  - `store` is golden `_store`: a copy keyed by `tokens[..<p0]`, taken after the
    cache-only rows when the call added rows.
  - `decide` runs a request with `options.delta_cache: false` through
    `withoutDelta`. `runCase` always does, as golden `run_case` does: a case
    pins its cache path.
  - The shared GPU test runner sets `promptCache = nil`, so the fixture and
    decide tests run cold.
- One difference from golden: an entry's `nbytes` counts the compacted copy
  (Swift's `copy()` slices to the offset), where golden counts the step-padded
  buffer. So the byte cap evicts slightly later in Swift. Entry counts, hits
  and refusals are equal.
- Golden reference: `scripts/golden_delta_reference.py` (golden venv) runs six
  scenarios through golden `decide` on a fresh store and writes
  `Tests/MLXClassifiersTests/GoldenDelta/delta-reference.json`. For each step it
  records the delta-on response, the prefill calls, the store stats and the
  `delta_cache: false` response.

## Results

Measured 2026-10-17 on an M5 Mac with low-power mode on (`pmset -g`:
`lowpowermode 1`), Debug build, `xcodebuild test`: 52 tests, 51 pass, 1
skipped (the image fixtures, epic 4).

**Cache-hit accounting equals golden on all 23 steps.** That covers
`input_tokens`, `cached_tokens`, every prefill's `(cached, computed)`, and the
store's entries, hits, misses and refused rewinds.

| scenario (entries) | step | usage in/cached | refused | Δp vs own cold | Δp vs golden |
|---|---|---|---|---|---|
| tail-change-1024 (8) | 50-token tail change | 98/1046 | 0 | 0 | 5.3e-6 |
| ticks (8) | tick 2, hit past the window | 76/851 | 0 | 0 | 1.8e-4 |
| ticks (8) | tick 3, from tick 1's retained entry | 118/809 | 0 | 0 | 1.3e-5 |
| ticks (1) | tick 3, tick 2's entry refused | 861/66 | 1 | 0 | 1.3e-5 |
| edits-1k (8) | 1-token edit near the end | 51/1069 | 0 | 0 | 6.8e-6 |
| edits-1k (8) | edit inside the last 512 rows | 204/916 | 0 | 0 | 0 |
| edits-1k (8) | edit before the window | 944/176 | 2 | 6.3e-4 | 3.3e-4 |
| edits-1k (1) | edit inside the window, entry refused | 1054/66 | 1 | 0 | 6.0e-6 |
| edits-1k (1) | edit before the window | 944/176 | 1 | 6.3e-4 | 3.3e-4 |
| fixture-requests (8) | cached-list-state, first | 112/20 | 0 | 0 | 1.6e-4 |
| fixture-requests (8) | cached-list-state, again | 1/131 | 0 | 0 | 1.6e-4 |
| fixture-requests (8) | permutations 3 (reported) | 89/307 | 0 | 4.9e-2 | 7.9e-2 |
| fixture-requests (8) | two questions, perm 2 (reported) | 24/337 | 0 | 7.4e-2 | 1.2e-1 |
| fixture-requests (8) | noul-sequence (reported) | 2/108 | 0 | 3.7e-4 | 3.7e-4 |
| fixture-requests (8) | choice-sequence (reported) | 4/336 | 0 | 2.1e-3 | 4.5e-3 |

- **Asserted steps.** The single-ordering first-token steps are within 1e-3 of
  both the cold run and golden. On 16 of the 23 steps the option probabilities
  are bit-identical to the cold run, including the edit near the end and the
  edit inside the sliding window. The edit before the window is not
  bit-identical (6.3e-4).
- **Bit identity is not reachable in general, and the store is not the
  cause.**
  - A diagnostic run, not kept as a test, compared each delta hit with a cold
    run whose cache-only rows were chunked at the hit. The two were
    bit-identical wherever the served entry's rows had been computed in the
    same chunk shape (ticks 2 and 3, hits 851 and 809, past the window).
  - With no store at all, a cold run chunked at 851 differs from the
    one-chunk cold run by up to 0.51 in a full-vocab logit. MLX's bf16 kernels
    give rows that depend on the chunk length: the sliding layers switch
    between a causal and an array mask with it.
  - So a hit, which changes the chunking, moves logits by bf16 noise. Golden
    story 3.4 records the same: golden's own delta-vs-cold distance on these
    requests is 7.9e-2, 1.2e-1, 3.7e-4 and 6.6e-3.
  - The rewind itself is exact. In `KVRewindTests` the rewound caches present
    the same K/V, bit for bit, as caches fed only the first h rows. This holds
    for both cache kinds and both update paths.
- **`cached-list-state` through `runCase` with a store on the runner.** The
  prefill calls are `[(20, 112)]` both times, as without a store, and the store
  is untouched.

**Timing** (`testDeltaPrefillTiming`, low-power mode on, Debug, median of 5
after one warm-up). The prompt has 1120 tokens and a 1-token edit near the end,
so the common prefix is 1069 tokens. The delta entry comes from the unedited
prompt.

| path | rows computed | median ms | min ms |
|---|---|---|---|
| full prefill (no store) | 1120 | 397.7 | 396.7 |
| delta prefill (hit 1069) | 51 | 102.3 | 100.9 |

The delta prefill is about 3.9x faster than the full prefill. Its cost is the
51 computed rows, plus copying the 1069-row cache in and the stored snapshot
out.

## Acceptance Criteria
- **Given** two prompts sharing a prefix longer than the sliding window **when** the second hits the store **then** its raw probabilities equal a cold run. Met for single-ordering first-token prompts: option probabilities are bit-identical on hits past the window (ticks, tail change, edits near the end and inside the window), and within 1e-3 everywhere (worst 6.3e-4, the edit before the window). On permuted and sequence prompts the differences are bf16 chunk-shape noise; they are reported above, not asserted.
- **Given** a rewind the window cannot represent **when** looked up **then** the store reports no hit from that entry. Met: `testRotatingCacheRefusesARewindTheWindowCannotRepresent` passes, and the refusal counts equal golden's (ticks-1, edits-1k).
- **Given** golden `decide` with delta caching on **when** the same requests run in the same order on a fresh store **then** `usage`, the prefill calls and the store stats are equal. Met on all 23 steps.
