"""Write the golden vision reference the Swift ImageFixtureTests compare to.

Run with the golden repo's venv (loads the vision tower from the local HF cache,
never downloading):

    HF_HUB_OFFLINE=1 ../mlx-lm-as-classifiers/.venv/bin/python scripts/golden_vision_reference.py [DUMP_DIR]

For each distinct fixture PNG at each soft-token budget (280, 140, 70) it records
golden's preprocessing (`processor.resized_uint8`: the PIL-bicubic uint8 HWC
image) and the soft tokens (`VisionTower.encode_png`, bf16 `[N, 1536]`, the
runner's default fast path): shapes, the sha256 of the raw bytes, and the first
row's first 8 values. With DUMP_DIR, the full arrays are also saved there as
.npy (uint8 pixels, float32 copies of the soft tokens and of the tower output
before `embed_vision`) for diffing a mismatch.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_classifiers.mlx_runner import MLXRunner
from mlx_classifiers.vision.processor import preprocess_png, resized_uint8

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Tests/MLXClassifiersTests/GoldenVision/vision-reference.json"


def pngs() -> dict[str, bytes]:
    out = {}
    for f in sorted((ROOT / "fixtures").glob("*.json")):
        state = json.loads(f.read_text())["request"]["state"]
        for b in state if isinstance(state, list) else []:
            if isinstance(b, dict) and b.get("type") == "image/png":
                png = base64.b64decode(b["base64"])
                out[hashlib.sha256(png).hexdigest()] = png
    return out


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> int:
    dump = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
    runner = MLXRunner.load("mlx-community/gemma-4-E2B-it-qat-4bit", vision=True)
    tower = runner.vision
    vc = tower.vision_config
    rows = []
    for digest, png in pngs().items():
        for budget in (280, 140, 70):
            img = resized_uint8(png, vc.patch_size, budget, vc.pooling_kernel_size)
            feats = tower.encode_png(png, digest, budget)
            raw = tower.vision_tower(mx.array(preprocess_png(png, vc.patch_size, budget, vc.pooling_kernel_size)))[0]
            mx.eval(feats, raw)
            f16 = np.array(feats.view(mx.uint16))
            r16 = np.array(raw.view(mx.uint16))
            rows.append({
                "sha256": digest, "max_soft_tokens": budget,
                "resized": list(img.shape), "resized_sha256": sha(img),
                "tower_shape": list(raw.shape), "tower_bf16_sha256": sha(r16),
                "soft_tokens_shape": list(feats.shape), "soft_tokens_bf16_sha256": sha(f16),
                "soft_tokens_row0": [float(x) for x in np.array(feats[0, :8].astype(mx.float32))],
            })
            if dump:
                tag = f"{digest[:12]}-{budget}"
                np.save(dump / f"{tag}-resized.npy", img)
                np.save(dump / f"{tag}-tower.npy", np.array(raw.astype(mx.float32)))
                np.save(dump / f"{tag}-soft.npy", np.array(feats.astype(mx.float32)))
            print(digest[:12], budget, img.shape, feats.shape, file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"format": "mlx-classifiers-swift-vision-reference/1",
                               "script": "scripts/golden_vision_reference.py", "images": rows}, indent=2) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
