from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from .config import Config
from .dataset import TrainDataset, build_label_matrix
from .model import AudioCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assemble_train_df(cfg: Config, classes: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[Path]]:
    train = pd.read_csv(cfg.train_csv)
    train["audio_root"] = str(cfg.train_audio_dir)
    audio_roots = [cfg.train_audio_dir] * len(train)

    if cfg.train_soundscapes_labels_csv.exists():
        sc = pd.read_csv(cfg.train_soundscapes_labels_csv)

        def _to_seconds(v: object) -> float:
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip()
            if ":" in s:
                parts = s.split(":")
                parts = [float(p) for p in parts]
                while len(parts) < 3:
                    parts = [0.0] + parts
                h, m, sec = parts
                return h * 3600 + m * 60 + sec
            return float(s)

        sc["start_s"] = sc["start"].map(_to_seconds)
        sc["end_s"] = sc["end"].map(_to_seconds)
        sc = sc.drop(columns=["start", "end"])
        # filename in soundscape labels references a soundscape file in train_soundscapes
        sc["audio_root"] = str(cfg.train_soundscapes_dir)
        # only segments that have at least one species
        sc = sc[sc["primary_label"].astype(str).str.len() > 0].reset_index(drop=True)
        for col in ("secondary_labels", "rating", "collection"):
            if col not in sc.columns:
                sc[col] = ""
        common = ["filename", "primary_label", "secondary_labels", "start_s", "end_s", "audio_root"]
        train_part = train.assign(start_s=np.nan, end_s=np.nan)[
            ["filename", "primary_label", "secondary_labels", "start_s", "end_s", "audio_root"]
        ]
        sc_part = sc[["filename", "primary_label", "secondary_labels", "start_s", "end_s", "audio_root"]]
        df = pd.concat([train_part, sc_part], ignore_index=True)
    else:
        df = train.assign(start_s=np.nan, end_s=np.nan)[
            ["filename", "primary_label", "secondary_labels", "start_s", "end_s"]
        ]
        df["audio_root"] = str(cfg.train_audio_dir)

    y = build_label_matrix(df, classes)
    return df, y, audio_roots


class CombinedDataset(TrainDataset):
    """TrainDataset variant that uses the per-row audio_root."""

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        from .audio import crop_or_pad, load_audio

        rng = np.random.default_rng(seed=None if self.training else 42 + idx)
        path = Path(row["audio_root"]) / row["filename"]
        wav = load_audio(str(path), self.sample_rate)
        if not pd.isna(row.get("start_s")):
            s = int(float(row["start_s"]) * self.sample_rate)
            e = int(float(row["end_s"]) * self.sample_rate)
            wav = wav[s:e]
        wav = crop_or_pad(wav, self.clip_samples, training=self.training, rng=rng)
        if self.training:
            wav = self._augment_wave(wav, rng)
        return torch.from_numpy(wav), torch.from_numpy(self.labels[idx])


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    y_mix = torch.maximum(y * lam, y[perm] * (1 - lam))
    return x_mix, y_mix


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def train_one_fold(cfg: Config, max_samples: int = 0) -> None:
    set_seed(cfg.seed + cfg.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    taxonomy = pd.read_csv(cfg.taxonomy_csv)
    classes: list[str] = taxonomy["primary_label"].tolist()
    num_classes = len(classes)

    df, y, _ = assemble_train_df(cfg, classes)

    if max_samples and max_samples > 0 and len(df) > max_samples:
        sub_idx = np.random.RandomState(cfg.seed).permutation(len(df))[:max_samples]
        df = df.iloc[sub_idx].reset_index(drop=True)
        y = y[sub_idx]

    # Stratify by primary_label index where possible; segment rows may have multi-labels — stratify on first label.
    strata = df["primary_label"].astype(str).str.split(";").str[0].fillna("__none__")
    # collapse rare strata for the smoke run so KFold doesn't blow up on n<n_splits
    counts = strata.value_counts()
    rare = counts[counts < cfg.folds].index
    strata = strata.where(~strata.isin(rare), other="__rare__")
    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    splits = list(skf.split(df, strata))
    train_idx, val_idx = splits[cfg.fold]

    train_ds = CombinedDataset(
        df.iloc[train_idx], y[train_idx], cfg.train_audio_dir, cfg.sample_rate, cfg.clip_samples, training=True
    )
    val_ds = CombinedDataset(
        df.iloc[val_idx], y[val_idx], cfg.train_audio_dir, cfg.sample_rate, cfg.clip_samples, training=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    model = AudioCNN(
        backbone=cfg.backbone,
        num_classes=num_classes,
        pretrained=cfg.pretrained,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
        in_channels=cfg.in_channels,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        top_db=cfg.top_db,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.use_amp and device.type == "cuda"
    if use_amp and torch.cuda.get_device_capability(0)[0] < 7:
        # Pre-Volta GPUs (P100, sm_60) have unreliable fp16; skip AMP to avoid NaN.
        print("[amp] disabling AMP — GPU compute capability < 7.0 (no reliable fp16 / tensor cores)")
        use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = len(train_loader)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    best_score = -1.0
    best_path = cfg.ckpt_dir / f"fold{cfg.fold}_best.pt"
    pos_weight = None  # could compute from class frequency; left None for simplicity

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for wav, target in train_loader:
            wav = wav.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target = target * (1.0 - cfg.label_smoothing) + cfg.label_smoothing / target.shape[1]

            do_mixup = cfg.mixup_alpha > 0 and np.random.rand() < cfg.mixup_prob
            if do_mixup:
                wav, target = mixup(wav, target, cfg.mixup_alpha)

            lr = cosine_with_warmup(global_step, total_steps, warmup_steps, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(wav)
                loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            global_step += 1

        train_loss = running / max(1, steps_per_epoch)

        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for wav, target in val_loader:
                wav = wav.to(device, non_blocking=True)
                logits = model(wav)
                all_logits.append(torch.sigmoid(logits).float().cpu().numpy())
                all_targets.append(target.numpy())
        probs = np.concatenate(all_logits)
        targets = np.concatenate(all_targets)

        # Guard against NaN/inf that AMP overflows or upstream silent audio can produce.
        nan_count = int(np.isnan(probs).sum() + np.isinf(probs).sum())
        if nan_count:
            print(f"[val] warning: {nan_count} non-finite values in probs; replacing with 0")
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

        # macro mAP across classes that have at least one positive in val.
        # Binarize the targets (we use 0.5 soft labels for secondary species during training).
        targets_bin = (targets >= 0.5).astype(np.int32)
        active = targets_bin.sum(axis=0) > 0
        if active.any():
            score = average_precision_score(targets_bin[:, active], probs[:, active], average="macro")
        else:
            score = 0.0

        print(
            f"[fold {cfg.fold}] epoch {epoch + 1}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} val_macroAP={score:.4f} lr={lr:.2e}"
        )

        if score > best_score:
            best_score = score
            torch.save(
                {"model": model.state_dict(), "classes": classes, "cfg": cfg.__dict__},
                best_path,
            )

    print(f"[fold {cfg.fold}] best val macroAP = {best_score:.4f} -> {best_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=None)
    p.add_argument("--work_dir", type=Path, default=Path("./work"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone", type=str, default="tf_efficientnetv2_s.in21k_ft_in1k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", type=int, default=1, help="1=use timm pretrained weights, 0=random init")
    p.add_argument("--max_samples", type=int, default=0, help="if >0, subsample the training set to this many rows (debug/smoke)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        data_root=args.data_root if args.data_root else Path("/kaggle/input/birdclef-2026"),
        work_dir=args.work_dir,
        fold=args.fold,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        backbone=args.backbone,
        seed=args.seed,
        pretrained=bool(args.pretrained),
    )
    train_one_fold(cfg, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
