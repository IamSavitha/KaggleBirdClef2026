"""Inference / submission generator for the Kaggle Pantanal acoustic competition.

Run as a Kaggle notebook entry point:
    python -m src.infer --ckpts /kaggle/input/my-ckpts/fold0_best.pt,/kaggle/input/my-ckpts/fold1_best.pt
The script writes /kaggle/working/submission.csv.
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import sys
from pathlib import Path

# Allow loading checkpoints saved on Windows (which pickles pathlib.WindowsPath) on Linux.
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PurePosixPath  # type: ignore[misc, assignment]

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .config import Config
from .model import AudioCNN


def _segment_starts(total_seconds: float, clip_seconds: float) -> list[int]:
    """End-times in seconds for each 5-s segment of a 1-min clip: 5, 10, ..., 60."""
    n = int(round(total_seconds / clip_seconds))
    return [int((i + 1) * clip_seconds) for i in range(n)]


def load_resample(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    t = torch.from_numpy(wav.astype(np.float32, copy=False))
    if sr != target_sr:
        t = torchaudio.functional.resample(t.unsqueeze(0), sr, target_sr).squeeze(0)
    return t  # (T,)


def load_models(ckpt_paths: list[Path], num_classes: int, device: torch.device) -> list[AudioCNN]:
    models: list[AudioCNN] = []
    for cp in ckpt_paths:
        state = torch.load(str(cp), map_location="cpu", weights_only=False)
        cfg_dict = state.get("cfg", {})
        model = AudioCNN(
            backbone=cfg_dict.get("backbone", "tf_efficientnetv2_s.in21k_ft_in1k"),
            num_classes=num_classes,
            pretrained=False,
            drop_rate=cfg_dict.get("drop_rate", 0.2),
            drop_path_rate=cfg_dict.get("drop_path_rate", 0.2),
            in_channels=cfg_dict.get("in_channels", 1),
            sr=cfg_dict.get("sample_rate", 32000),
            n_fft=cfg_dict.get("n_fft", 2048),
            hop_length=cfg_dict.get("hop_length", 512),
            n_mels=cfg_dict.get("n_mels", 128),
            fmin=cfg_dict.get("fmin", 50),
            fmax=cfg_dict.get("fmax", 16000),
            top_db=cfg_dict.get("top_db", 80.0),
        )
        model.load_state_dict(state["model"])
        model.eval().to(device)
        models.append(model)
    return models


@torch.inference_mode()
def predict_soundscape(
    wav: torch.Tensor,
    models: list[AudioCNN],
    sr: int,
    clip_samples: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    total = wav.shape[0]
    n_segments = max(1, int(round(total / clip_samples)))
    # pad if too short
    if total < n_segments * clip_samples:
        wav = F.pad(wav, (0, n_segments * clip_samples - total))

    segments = wav.view(n_segments, clip_samples)  # (N, T)

    preds = np.zeros((n_segments, models[0].head[-1].out_features), dtype=np.float32)

    for start in range(0, n_segments, batch_size):
        chunk = segments[start : start + batch_size].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            ensemble = None
            for m in models:
                logits = m(chunk)
                p = torch.sigmoid(logits).float()
                ensemble = p if ensemble is None else ensemble + p
            ensemble = ensemble / len(models)
        preds[start : start + chunk.shape[0]] = ensemble.cpu().numpy()
    return preds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=Path("/kaggle/input/birdclef-2026"))
    p.add_argument("--ckpts", type=str, required=True, help="Comma-separated checkpoint paths")
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/submission.csv"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--input_dir", type=Path, default=None, help="Override test_soundscapes dir (e.g., train_soundscapes for local smoke runs)")
    p.add_argument("--limit_files", type=int, default=0, help="If >0, only process the first N soundscape files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(data_root=args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    taxonomy = pd.read_csv(cfg.taxonomy_csv)
    classes: list[str] = taxonomy["primary_label"].tolist()

    ckpt_paths = [Path(p.strip()) for p in args.ckpts.split(",") if p.strip()]
    models = load_models(ckpt_paths, num_classes=len(classes), device=device)

    sample_sub = pd.read_csv(cfg.sample_submission_csv)
    src_dir = args.input_dir if args.input_dir else cfg.test_soundscapes_dir
    soundscape_files = sorted(src_dir.glob("*.ogg"))
    if args.limit_files and args.limit_files > 0:
        soundscape_files = soundscape_files[: args.limit_files]
    print(f"Processing {len(soundscape_files)} soundscape file(s) from {src_dir}")

    rows: list[dict] = []
    for path in soundscape_files:
        wav = load_resample(path, cfg.sample_rate)
        preds = predict_soundscape(
            wav,
            models,
            cfg.sample_rate,
            cfg.clip_samples,
            device,
            args.batch_size,
        )
        stem = path.stem
        for i, p in enumerate(preds):
            end_s = int((i + 1) * cfg.clip_seconds)
            row = {"row_id": f"{stem}_{end_s}"}
            for cls, prob in zip(classes, p, strict=False):
                row[cls] = float(prob)
            rows.append(row)
        del wav, preds
        gc.collect()

    if rows:
        sub = pd.DataFrame(rows)
        sub = sub[sample_sub.columns]
    else:
        # No test soundscapes mounted (public dev run). Emit header-only CSV so the
        # kernel completes successfully; the hidden rerun will populate test files.
        sub = pd.DataFrame(columns=sample_sub.columns)
    sub.to_csv(args.out, index=False)
    print(f"Wrote submission: {args.out}  shape={sub.shape}")


if __name__ == "__main__":
    main()
