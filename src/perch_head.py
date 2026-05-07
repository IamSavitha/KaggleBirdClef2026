"""Lightweight MLP head trained inside the submission kernel on Perch embeddings.

Trains on the labeled segments in train_soundscapes_labels.csv. Captures species
that Perch can't predict directly (insect sonotypes etc.) by leveraging its
representations rather than its vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .perch_infer import N_WINDOWS, WINDOW_SAMPLES, read_60s


def _seconds_from_str(v: object) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        while len(parts) < 3:
            parts = [0.0] + parts
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    return float(s)


class PerchHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int, hidden: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_train_features(
    comp_root: Path,
    perch_session,
    inp_name: str,
    embed_idx: int,
    primary_labels: list[str],
    max_files: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Perch on labeled train_soundscapes; return (X, Y) for head training."""
    label_csv = comp_root / "train_soundscapes_labels.csv"
    if not label_csv.exists():
        return (
            np.zeros((0, 1536), np.float32),
            np.zeros((0, len(primary_labels)), np.float32),
        )

    df = pd.read_csv(label_csv)
    df["start_s"] = df["start"].map(_seconds_from_str)
    df["end_s"] = df["end"].map(_seconds_from_str)
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}

    Xs, Ys = [], []
    files = list(df["filename"].unique())[:max_files]
    print(f"perch_head: collecting features from {len(files)} train soundscape file(s)")

    for fname in files:
        fpath = comp_root / "train_soundscapes" / fname
        if not fpath.exists():
            continue
        try:
            y_audio = read_60s(fpath)
        except Exception as e:
            print(f"  skip {fname}: {e}")
            continue
        x = y_audio.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        outs = perch_session.run(None, {inp_name: x})
        embeddings = outs[embed_idx]  # (12, embed_dim)

        seg_df = df[df["filename"] == fname]
        for _, row in seg_df.iterrows():
            try:
                end_sec = int(row["end_s"])
            except (ValueError, TypeError):
                continue
            window_idx = (end_sec // 5) - 1
            if 0 <= window_idx < N_WINDOWS:
                lbl_str = str(row["primary_label"])
                lbls = [t.strip() for t in lbl_str.split(";") if t.strip()]
                y_vec = np.zeros(len(primary_labels), dtype=np.float32)
                for lbl in lbls:
                    if lbl in label_to_idx:
                        y_vec[label_to_idx[lbl]] = 1.0
                if y_vec.sum() > 0:  # only keep windows with at least one positive
                    Xs.append(embeddings[window_idx])
                    Ys.append(y_vec)

    if not Xs:
        return (
            np.zeros((0, 1536), np.float32),
            np.zeros((0, len(primary_labels)), np.float32),
        )
    return (
        np.stack(Xs).astype(np.float32),
        np.stack(Ys).astype(np.float32),
    )


def train_head(
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-3,
) -> nn.Module | None:
    if len(X) < 50:
        print(f"perch_head: only {len(X)} training samples, skipping")
        return None

    embed_dim = X.shape[1]
    n_classes = Y.shape[1]
    model = PerchHead(embed_dim=embed_dim, n_classes=n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 5 == 0:
            print(f"perch_head epoch {epoch + 1}/{epochs} loss={total / len(ds):.4f}")

    model.eval()
    return model


def infer_head(model: nn.Module | None, embeddings: np.ndarray) -> np.ndarray | None:
    if model is None:
        return None
    Xt = torch.from_numpy(embeddings.astype(np.float32))
    with torch.inference_mode():
        logits = model(Xt)
        probs = torch.sigmoid(logits).numpy().astype(np.float32)
    return probs
