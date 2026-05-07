from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import crop_or_pad, load_audio


class TrainDataset(Dataset):
    """Multi-label dataset combining XC/iNat clips and labeled train_soundscapes segments."""

    def __init__(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        audio_root: Path,
        sample_rate: int,
        clip_samples: int,
        training: bool,
    ) -> None:
        super().__init__()
        assert len(df) == len(labels)
        self.df = df.reset_index(drop=True)
        self.labels = labels.astype(np.float32)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        self.training = training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        rng = np.random.default_rng(seed=None if self.training else 42 + idx)

        path = self.audio_root / row["filename"]
        wav = load_audio(str(path), self.sample_rate)

        # If a (start_s, end_s) is provided (labeled soundscape segments), crop to it first.
        if "start_s" in row and not pd.isna(row["start_s"]):
            s = int(float(row["start_s"]) * self.sample_rate)
            e = int(float(row["end_s"]) * self.sample_rate)
            wav = wav[s:e]

        wav = crop_or_pad(wav, self.clip_samples, training=self.training, rng=rng)

        if self.training:
            wav = self._augment_wave(wav, rng)

        return torch.from_numpy(wav), torch.from_numpy(self.labels[idx])

    def _augment_wave(self, wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Gaussian noise
        if rng.random() < 0.3:
            wav = wav + rng.normal(0, 0.005, size=wav.shape).astype(np.float32)
        # Random gain
        if rng.random() < 0.5:
            wav = wav * float(rng.uniform(0.7, 1.3))
        # Polarity flip
        if rng.random() < 0.5:
            wav = -wav
        return np.clip(wav, -1.0, 1.0)


def build_label_matrix(df: pd.DataFrame, classes: list[str]) -> np.ndarray:
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(df)
    y = np.zeros((n, len(classes)), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        primary = getattr(row, "primary_label")
        # primary_label can be a single label (train.csv) or semicolon-separated (soundscape labels)
        if isinstance(primary, str) and ";" in primary:
            labels = [s.strip() for s in primary.split(";") if s.strip()]
        else:
            labels = [primary] if isinstance(primary, str) and primary else []
        for lbl in labels:
            j = cls_to_idx.get(lbl)
            if j is not None:
                y[i, j] = 1.0
        # secondary labels (from train.csv only) — soft label 0.5
        if hasattr(row, "secondary_labels"):
            sec = getattr(row, "secondary_labels")
            if isinstance(sec, str) and sec and sec != "[]":
                # may be like "['code1','code2']" or "code1;code2"
                cleaned = sec.strip("[]").replace("'", "").replace('"', "")
                tokens = [t.strip() for t in cleaned.replace(";", ",").split(",") if t.strip()]
                for tok in tokens:
                    j = cls_to_idx.get(tok)
                    if j is not None and y[i, j] < 0.5:
                        y[i, j] = 0.5
    return y
