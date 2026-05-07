"""Kaggle notebook entry-point.

Place this in a Kaggle notebook cell. Adjust CKPT_GLOB to point at your dataset
of trained checkpoints (uploaded as a Kaggle dataset, mounted under /kaggle/input).
"""

import sys
from pathlib import Path

# Make src/ importable when uploaded as a dataset, e.g. /kaggle/input/pantanal-src
SRC_DIR = Path("/kaggle/input/pantanal-src")
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

CKPT_GLOB = "/kaggle/input/pantanal-ckpts/fold*_best.pt"

from src.infer import main as run_infer  # noqa: E402

ckpts = sorted(Path().glob(CKPT_GLOB))
if not ckpts:
    raise SystemExit(f"No checkpoints found at {CKPT_GLOB}")

sys.argv = [
    "infer",
    "--data_root", "/kaggle/input/birdclef-2026",
    "--ckpts", ",".join(str(p) for p in ckpts),
    "--out", "/kaggle/working/submission.csv",
    "--batch_size", "16",
]
run_infer()
