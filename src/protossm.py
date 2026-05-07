"""ProtoSSM: a small bidirectional sequence model over the 12 windows of Perch embeddings per soundscape.

Captures temporal structure — calls that persist or repeat across consecutive 5-second
windows are reinforced; isolated noise is dampened. Trained inside the submission
kernel from train_soundscapes_labels.csv.

This is intentionally a "ProtoSSM-lite": a 2-layer bi-LSTM with a linear head, instead
of the full state-space + cross-attention design from the public 0.943 inspiration. The
goal is the temporal-context lift, not architectural fidelity.
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


class ProtoSSM(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, embed_dim)
        h = F.relu(self.proj(x))
        h, _ = self.lstm(h)
        h = self.dropout(h)
        return self.head(h)  # (B, T, n_classes)


def collect_train_sequences(
    comp_root: Path,
    perch_session,
    inp_name: str,
    embed_idx: int,
    primary_labels: list[str],
    max_files: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) where each row is one file's full 12-window sequence.

    X: (n_files, 12, embed_dim) — Perch embeddings
    Y: (n_files, 12, n_classes) — multi-label binary targets per window
    Windows without a corresponding label-CSV row are treated as all-negative.
    """
    label_csv = comp_root / "train_soundscapes_labels.csv"
    if not label_csv.exists():
        return np.zeros((0, N_WINDOWS, 1536), np.float32), np.zeros(
            (0, N_WINDOWS, len(primary_labels)), np.float32
        )

    df = pd.read_csv(label_csv)
    df["start_s"] = df["start"].map(_seconds_from_str)
    df["end_s"] = df["end"].map(_seconds_from_str)
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}
    n_classes = len(primary_labels)

    files = list(df["filename"].unique())[:max_files]
    print(f"protossm: collecting sequences from {len(files)} train soundscape file(s)")

    Xs, Ys = [], []
    for fname in files:
        fpath = comp_root / "train_soundscapes" / fname
        if not fpath.exists():
            continue
        try:
            y_audio = read_60s(fpath)
        except Exception as e:
            print(f"  skip {fname}: {e}")
            continue
        x_in = y_audio.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        outs = perch_session.run(None, {inp_name: x_in})
        embeddings = outs[embed_idx].astype(np.float32)  # (12, embed_dim)

        y_seq = np.zeros((N_WINDOWS, n_classes), dtype=np.float32)
        seg_df = df[df["filename"] == fname]
        for _, row in seg_df.iterrows():
            try:
                end_sec = int(row["end_s"])
            except (ValueError, TypeError):
                continue
            window_idx = (end_sec // 5) - 1
            if not (0 <= window_idx < N_WINDOWS):
                continue
            lbl_str = str(row["primary_label"])
            for lbl in (t.strip() for t in lbl_str.split(";") if t.strip()):
                if lbl in label_to_idx:
                    y_seq[window_idx, label_to_idx[lbl]] = 1.0
        Xs.append(embeddings)
        Ys.append(y_seq)

    if not Xs:
        return np.zeros((0, N_WINDOWS, 1536), np.float32), np.zeros(
            (0, N_WINDOWS, n_classes), np.float32
        )
    return np.stack(Xs).astype(np.float32), np.stack(Ys).astype(np.float32)


def train_protossm(
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 8,
    weight_decay: float = 1e-3,
    hidden: int = 256,
) -> nn.Module | None:
    if len(X) < 10:
        print(f"protossm: only {len(X)} files, skipping training")
        return None

    embed_dim = X.shape[2]
    n_classes = Y.shape[2]
    model = ProtoSSM(embed_dim=embed_dim, n_classes=n_classes, hidden=hidden)
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
            print(f"protossm epoch {epoch + 1}/{epochs} loss={total / len(ds):.4f}")

    model.eval()
    return model


def train_and_save(
    comp_root: Path,
    perch_onnx_path: Path,
    perch_labels_csv: Path,
    out_ckpt: Path,
    max_files: int = 200,
    epochs: int = 40,
    hidden: int = 256,
) -> None:
    """End-to-end ProtoSSM training pipeline used by the dedicated training kernel."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(perch_onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    inp_name = session.get_inputs()[0].name
    name_to_idx = {o.name: i for i, o in enumerate(session.get_outputs())}
    embed_idx = name_to_idx.get("embedding", 1 if len(name_to_idx) > 1 else 0)

    sample_sub = pd.read_csv(comp_root / "sample_submission.csv")
    primary_labels = sample_sub.columns[1:].tolist()

    print("[protossm-train] collecting Perch sequences from labeled train_soundscapes...")
    X, Y = collect_train_sequences(
        comp_root, session, inp_name, embed_idx, primary_labels, max_files=max_files
    )
    print(f"[protossm-train] X={X.shape} Y={Y.shape}")
    if len(X) < 10:
        raise SystemExit(f"too few training files ({len(X)}); aborting")

    print("[protossm-train] training ProtoSSM...")
    model = train_protossm(X, Y, epochs=epochs, hidden=hidden)
    if model is None:
        raise SystemExit("training returned None")

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "state_dict": model.state_dict(),
        "embed_dim": int(X.shape[2]),
        "n_classes": int(Y.shape[2]),
        "hidden": hidden,
    }
    torch.save(state, str(out_ckpt))
    print(f"[protossm-train] saved {out_ckpt}")


def train_and_save_from_cache(
    perch_meta_dir: Path,
    comp_root: Path,
    out_ckpt: Path,
    hidden: int = 128,
    dropout: float = 0.5,
    weight_decay: float = 1e-2,
    max_epochs: int = 30,
    val_frac: float = 0.2,
    patience: int = 5,
    lr: float = 5e-4,
    batch_size: int = 8,
    seed: int = 42,
) -> None:
    """Train ProtoSSM v2 from jaejohn/perch-meta cached Perch embeddings.

    Stronger regularization + train/val split + early stopping (best-val-loss ckpt) to
    fight the severe overfitting we saw in v1 (loss dropping to 0.05 in 5 epochs).
    """
    perch_meta_dir = Path(perch_meta_dir)
    comp_root = Path(comp_root)
    out_ckpt = Path(out_ckpt)

    npz = np.load(str(perch_meta_dir / "full_perch_arrays.npz"))
    emb_keys = ["emb_full", "embs", "emb", "embeddings", "features", "perch_embs", "arr_1"]
    emb = None
    for k in emb_keys:
        if k in npz.files:
            emb = npz[k]
            break
    if emb is None:
        raise SystemExit(f"could not locate embeddings in npz; keys={list(npz.files)}")
    print(f"[protossm-train-v2] perch-meta embeddings shape={emb.shape}")

    meta = pd.read_parquet(str(perch_meta_dir / "full_perch_meta.parquet"))
    if "row_id" not in meta.columns or "filename" not in meta.columns:
        raise SystemExit(f"unexpected parquet columns: {meta.columns.tolist()}")

    sample_sub = pd.read_csv(comp_root / "sample_submission.csv")
    primary_labels = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}
    n_classes = len(primary_labels)

    labels_df = pd.read_csv(comp_root / "train_soundscapes_labels.csv")
    labels_df["start_s"] = labels_df["start"].map(_seconds_from_str)
    labels_df["end_s"] = labels_df["end"].map(_seconds_from_str)

    # Build per-row label vector keyed by (filename, end_sec)
    seg_labels: dict[tuple[str, int], np.ndarray] = {}
    for _, r in labels_df.iterrows():
        try:
            end_sec = int(r["end_s"])
        except (ValueError, TypeError):
            continue
        key = (r["filename"], end_sec)
        v = seg_labels.setdefault(key, np.zeros(n_classes, dtype=np.float32))
        for lbl in (t.strip() for t in str(r["primary_label"]).split(";") if t.strip()):
            j = label_to_idx.get(lbl)
            if j is not None:
                v[j] = 1.0

    # Group meta rows by filename and build (T, D) sequences in window order
    meta = meta.copy()
    meta["end_sec"] = meta["row_id"].str.rsplit("_", n=1).str[-1].astype(int)
    meta["window_idx"] = (meta["end_sec"] // 5) - 1

    Xs: list[np.ndarray] = []
    Ys: list[np.ndarray] = []
    for fname, grp in meta.groupby("filename", sort=False):
        grp = grp.sort_values("window_idx")
        if len(grp) != N_WINDOWS:
            continue
        idx = grp.index.to_numpy()
        seq_emb = emb[idx]  # (12, D)
        seq_y = np.zeros((N_WINDOWS, n_classes), dtype=np.float32)
        for w_i, (_, row) in enumerate(grp.iterrows()):
            seq_y[w_i] = seg_labels.get((fname, int(row["end_sec"])), seq_y[w_i])
        Xs.append(seq_emb)
        Ys.append(seq_y)

    X = np.stack(Xs).astype(np.float32)
    Y = np.stack(Ys).astype(np.float32)
    print(f"[protossm-train-v2] X={X.shape} Y={Y.shape} positives_per_seq={Y.sum(axis=(1,2)).mean():.1f}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(round(len(X) * val_frac)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xva, Yva = X[val_idx], Y[val_idx]
    print(f"[protossm-train-v2] split: train={len(Xtr)} val={len(Xva)}")

    embed_dim = X.shape[2]
    model = ProtoSSM(embed_dim=embed_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    Xva_t = torch.from_numpy(Xva)
    Yva_t = torch.from_numpy(Yva)

    best_val = float("inf")
    best_state = None
    epochs_since_improvement = 0
    for epoch in range(max_epochs):
        model.train()
        train_total = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * xb.size(0)
        train_loss = train_total / len(train_ds)

        model.eval()
        with torch.inference_mode():
            v_logits = model(Xva_t)
            val_loss = F.binary_cross_entropy_with_logits(v_logits, Yva_t).item()

        marker = ""
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
            marker = " *"
        else:
            epochs_since_improvement += 1

        print(
            f"[protossm-train-v2] epoch {epoch + 1}/{max_epochs} "
            f"train={train_loss:.4f} val={val_loss:.4f}{marker}"
        )
        if epochs_since_improvement >= patience:
            print(f"[protossm-train-v2] early stop after {epoch + 1} epochs (best val={best_val:.4f})")
            break

    if best_state is None:
        raise SystemExit("no best state captured")

    model.load_state_dict(best_state)
    model.eval()
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "state_dict": model.state_dict(),
        "embed_dim": embed_dim,
        "n_classes": n_classes,
        "hidden": hidden,
        "best_val_loss": float(best_val),
    }
    torch.save(state, str(out_ckpt))
    print(f"[protossm-train-v2] saved {out_ckpt} (best val={best_val:.4f})")


def load_protossm(ckpt_path: Path | str) -> nn.Module:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = ProtoSSM(
        embed_dim=state["embed_dim"], n_classes=state["n_classes"], hidden=state.get("hidden", 256)
    )
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def infer_protossm(
    model: nn.Module | None, embeddings_flat: np.ndarray, n_files: int
) -> np.ndarray | None:
    """embeddings_flat: (n_files * N_WINDOWS, embed_dim) → probs (n_files * N_WINDOWS, n_classes)."""
    if model is None:
        return None
    embed_dim = embeddings_flat.shape[1]
    x = embeddings_flat.reshape(n_files, N_WINDOWS, embed_dim)
    Xt = torch.from_numpy(x.astype(np.float32))
    with torch.inference_mode():
        logits = model(Xt)
        probs = torch.sigmoid(logits).numpy().astype(np.float32)
    return probs.reshape(n_files * N_WINDOWS, -1)
