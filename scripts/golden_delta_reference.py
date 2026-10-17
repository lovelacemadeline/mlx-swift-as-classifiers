"""Write the golden delta-prefix-cache reference the Swift PromptCacheTests compare to.

Run with the golden repo's venv (it imports `mlx_classifiers` and loads the real
model from the local HF cache, never downloading):

    HF_HUB_OFFLINE=1 ../mlx-lm-as-classifiers/.venv/bin/python scripts/golden_delta_reference.py

Each scenario runs its requests in order through golden `decide` on one runner
with a fresh `PromptCacheStore(max_entries=...)` (delta caching on, the golden
default). Per step it records the wire response with delta on (timing dropped),
the prefill calls `(cached, computed)` that step made, the store's stats after
the step, and the response of the same request with `delta_cache: false` (the
cold reference, which neither reads nor writes the store).

Scenarios: the golden story 3.4 tests (a 1024-token state with a 50-token tail
change; three ticks whose change points move earlier, with 8 entries and 1),
single-token edits of a ~1000-token state near the end, inside the last 512 rows
(the sliding window) and before it, with 8 entries and 1, and fixture requests
(cached system prefix, permutations, two questions, sequence mode) decided twice.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import mlx_classifiers
from mlx_classifiers.bench import PREFIX, QUESTION, delta_states, filler_state
from mlx_classifiers.decide import decide
from mlx_classifiers.mlx_runner import MLXRunner
from mlx_classifiers.prompt_cache import PromptCacheStore
from mlx_classifiers.schema import SystemOneRequest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"
OUT = ROOT / "Tests/MLXClassifiersTests/GoldenDelta/delta-reference.json"


def fixture(name: str) -> dict:
    return json.loads((FIX / f"{name}.json").read_text())


def bench_req(state: str) -> dict:
    return {"prefix": PREFIX, "state": state, "questions": {"q": QUESTION}, "options": {"debug": True}}


def fixture_req(name: str, **opts) -> dict:
    fx = fixture(name)
    r = copy.deepcopy(fx["request"])
    o = dict(r.get("options") or {})
    if fx["label_style"] != "letters":
        o["label_style"] = fx["label_style"]
    o.update({"debug": True}, **opts)
    r["options"] = o
    return r


def replace_word(text: str, index: int, word: str) -> str:
    words = text.split(" ")
    words[index] = word
    return " ".join(words)


def scenarios(tokenize) -> list[tuple[str, int, list[dict]]]:
    out = []
    s0, s1 = delta_states(tokenize, 1024, 50, 2)
    out.append(("tail-change-1024", 8, [bench_req(s0), bench_req(s1)]))

    head, _ = filler_state(tokenize, 800)
    words = head.split()
    ticks = [
        head + " The agent is at row one.",
        " ".join(words[:-20] + ["pending"] + words[-19:]) + " The agent is at row two.",
        " ".join(words[:-60] + ["blocked"] + words[-59:]) + " The agent is at row three.",
    ]
    for entries in (8, 1):
        out.append((f"ticks-{entries}", entries, [bench_req(t) for t in ticks]))

    base, _ = filler_state(tokenize, 1000)
    n_words = len(base.split(" "))
    edits = [
        base,
        replace_word(base, n_words - 3, "pending"),  # near the end
        replace_word(base, n_words - 150, "blocked"),  # inside the last 512 rows
        replace_word(base, 100, "refunded"),  # before the window
    ]
    for entries in (8, 1):
        out.append((f"edits-1k-{entries}", entries, [bench_req(e) for e in edits]))

    multi = fixture_req("choice-first-token-cached-list-state", permutations=2)
    multi["questions"]["escalate"] = {
        "type": "noul",
        "instructions": "Should this ticket be escalated to a human agent?",
    }
    fixture_steps = [
        fixture_req("choice-first-token-cached-list-state"),
        fixture_req("choice-first-token-cached-list-state"),
        fixture_req("choice-first-token-cached-list-state", permutations=3),
        multi,
        fixture_req("noul-sequence"),
        fixture_req("noul-sequence"),
        fixture_req("choice-sequence-multi-token-options"),
    ]
    out.append(("fixture-requests", 8, fixture_steps))
    return out


def wire(resp) -> dict:
    w = resp.to_wire()
    w.pop("timing_ms", None)
    return w


def main() -> int:
    runner = MLXRunner.load(fixture("noul-first-token")["model"]["repo"])
    results = []
    for sid, entries, reqs in scenarios(runner.tokenize):
        store = PromptCacheStore(max_entries=entries)
        runner.prompt_cache = store
        steps = []
        for body in reqs:
            n0 = len(runner.prefill_calls)
            resp = decide(SystemOneRequest.model_validate(body), runner)
            calls = [list(c) for c in runner.prefill_calls[n0:]]
            stats = {"entries": len(store), "hits": store.hits, "misses": store.misses,
                     "unrepresentable": store.unrepresentable, "nbytes": store.nbytes}
            cold_body = copy.deepcopy(body)
            cold_body.setdefault("options", {})["delta_cache"] = False
            cold = decide(SystemOneRequest.model_validate(cold_body), runner)
            steps.append({"request": body, "response": wire(resp), "prefill_calls": calls,
                          "store": stats, "cold_response": wire(cold)})
            print(sid, calls, stats, file=sys.stderr)
        results.append({"id": sid, "max_entries": entries, "steps": steps})
    runner.prompt_cache = PromptCacheStore()
    doc = {
        "format": "mlx-classifiers-swift-delta-reference/1",
        "generator": {
            "mlx_classifiers": mlx_classifiers.__version__,
            "golden_commit": subprocess.run(
                ["git", "-C", str(Path(mlx_classifiers.__file__).parent), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=False,
            ).stdout.strip(),
            "script": "scripts/golden_delta_reference.py",
        },
        "scenarios": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(results)} scenarios to {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
