"""Write the golden `decide` reference responses the Swift DecideTests compare to.

Run with the golden repo's venv (it imports `mlx_classifiers` and loads the real
model from the local HF cache, never downloading):

    ../mlx-lm-as-classifiers/.venv/bin/python scripts/golden_decide_reference.py

Each case is a wire request built from a vendored fixture's `request`, with
`delta_cache: false` (the Swift runner has no delta store yet, epic 3) and
`debug: true`, plus permutations / calibration variants. The output records the
golden `to_wire()` response (timing dropped: wall clock) or the raised error,
and `prefills`: every prefill decide made, keyed by the sha256 of its token ids
("1,2,3"), with the logits golden gathered from it ({token id: logit}). Swift's
`ReplayRunner` serves those to check decide's own arithmetic without bf16 noise.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import importlib

import mlx_classifiers
from mlx_classifiers.decide import decide
from mlx_classifiers.errors import ClassifierError
from mlx_classifiers.mlx_runner import MLXRunner
from mlx_classifiers.schema import SystemOneRequest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"
OUT = ROOT / "Tests/MLXClassifiersTests/GoldenDecide/decide-responses.json"


def fixture(name: str) -> dict:
    return json.loads((FIX / f"{name}.json").read_text())


def req(name: str, temperature=None, **opts) -> tuple[str, dict, float | None]:
    fx = fixture(name)
    r = copy.deepcopy(fx["request"])
    o = dict(r.get("options") or {})
    if fx["label_style"] != "letters":
        o["label_style"] = fx["label_style"]
    o.update({"delta_cache": False, "debug": True}, **opts)
    r["options"] = o
    return r, temperature


def cases() -> list[tuple[str, dict, float | None]]:
    out = []
    for name in (
        "noul-first-token",
        "choice-first-token-object-state",
        "choice-first-token-cached-list-state",
        "score-first-token-no-prefix",
        "choice-first-token-multi-token-label-error",
    ):
        out.append((name, *req(name)))
    out.append(("object-state-perm3-T1.7", *req("choice-first-token-object-state", 1.7, permutations=3)))
    out.append(("cached-list-perm5-T0.8", *req("choice-first-token-cached-list-state", 0.8, permutations=5)))
    out.append(("noul-uncalibrated-T2", *req("noul-first-token", 2.0, calibrated=False)))
    out.append(("score-calibrated-T1.3", *req("score-first-token-no-prefix", 1.3)))
    # Two questions over one state and prefix: the shared system-turn cache, answer
    # order and the usage sums.
    multi, _ = req("choice-first-token-cached-list-state", permutations=2)
    multi["questions"]["escalate"] = {
        "type": "noul",
        "instructions": "Should this ticket be escalated to a human agent?",
    }
    multi["questions"]["urgency"] = {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"],
    }
    out.append(("multi-question-perm2", multi, None))
    return out


def token_key(tokens) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


class Recorder:
    """Wraps a runner: remembers each prefill's tokens; `gather` (patched into
    decide) then records the logits read from that prefill."""

    def __init__(self, runner):
        self.runner, self.prefills, self.last = runner, {}, None

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def prefill(self, tokens, *args, **kw):
        self.last = token_key(tokens)
        return self.runner.prefill(tokens, *args, **kw)


def main() -> int:
    decide_mod = importlib.import_module("mlx_classifiers.decide")  # the package re-exports `decide`
    runner = Recorder(MLXRunner.load(fixture("noul-first-token")["model"]["repo"]))
    real_gather = decide_mod.gather

    def gather(row, token_ids):
        values = real_gather(row, token_ids)
        runner.prefills.setdefault(runner.last, {}).update(
            {str(t): v for t, v in zip(token_ids, values, strict=True)}
        )
        return values

    decide_mod.gather = gather
    results = []
    for cid, request, temperature in cases():
        runner.prefills = {}
        entry = {"id": cid, "request": request, "temperature": temperature}
        try:
            resp = decide(SystemOneRequest.model_validate(request), runner, temperature=temperature)
            wire = resp.to_wire()
            wire.pop("timing_ms", None)
            entry["response"] = wire
        except ClassifierError as e:
            entry["error"] = {"type": type(e).__name__, "message": str(e)}
        entry["prefills"] = runner.prefills
        results.append(entry)
        print(cid, "error" if "error" in entry else "ok", file=sys.stderr)
    doc = {
        "format": "mlx-classifiers-swift-decide-reference/1",
        "generator": {
            "mlx_classifiers": mlx_classifiers.__version__,
            "golden_commit": subprocess.run(
                ["git", "-C", str(Path(mlx_classifiers.__file__).parent), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=False,
            ).stdout.strip(),
            "script": "scripts/golden_decide_reference.py",
        },
        "cases": results,
    }
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(results)} cases to {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
