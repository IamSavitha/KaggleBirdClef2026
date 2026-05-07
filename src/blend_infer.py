"""Phase 2 (revised): 3-way rank-average blend of Perch + our EffNet + Tucker's distilled SED."""

from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .perch_infer import (
    FILE_SAMPLES,
    N_WINDOWS,
    SR,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    build_class_mapping,
    read_60s,
)

# ---- SED preprocessing (matches Tucker Arrants' distilled SED ONNX inputs) ----
SED_N_MELS = 256
SED_N_FFT = 2048
SED_HOP = 512
SED_FMIN = 20
SED_FMAX = 16000
SED_TOP_DB = 80


def _audio_to_sed_mel(chunks: np.ndarray) -> np.ndarray:
    """Take (12, WINDOW_SAMPLES) → (12, 1, n_mels, T) float32."""
    import librosa

    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x,
            sr=SR,
            n_fft=SED_N_FFT,
            hop_length=SED_HOP,
            n_mels=SED_N_MELS,
            fmin=SED_FMIN,
            fmax=SED_FMAX,
            power=2.0,
        )
        s = librosa.power_to_db(s, top_db=SED_TOP_DB)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


def _rank_pct(probs: np.ndarray) -> np.ndarray:
    """Per-class column-rank scaled to (0, 1]; rows are (file, window) entries."""
    return pd.DataFrame(probs).rank(axis=0, pct=True).to_numpy(dtype=np.float32)


def _make_session(path: Path | str, threads: int = 4):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


# ---- Per-branch inference over all soundscapes ----


def _run_perch_branch(
    perch_onnx: Path,
    perch_labels: Path,
    primary_labels: list[str],
    taxonomy_csv: Path,
    paths: list[Path],
    batch_files: int = 4,
    return_session: bool = False,
):
    """Run Perch over all paths.

    Returns (scores, embeddings) if return_session=False; (scores, embeddings, session, inp_name, embed_idx)
    if return_session=True so the caller can reuse the session for head training.
    """
    session = _make_session(perch_onnx)
    inp_name = session.get_inputs()[0].name
    name_to_idx = {o.name: i for i, o in enumerate(session.get_outputs())}
    score_idx = name_to_idx.get("label", 0)
    embed_idx = name_to_idx.get("embedding", 1 if len(name_to_idx) > 1 else 0)

    bc_indices, proxy_map, NO_LABEL = build_class_mapping(taxonomy_csv, perch_labels, primary_labels)
    mapped_pos = np.where(bc_indices != NO_LABEL)[0].astype(np.int32)
    mapped_bc = bc_indices[mapped_pos]
    n_classes = len(primary_labels)

    n_rows = len(paths) * N_WINDOWS
    scores_out = np.zeros((n_rows, n_classes), dtype=np.float32)
    embeds_out: list[np.ndarray] = []

    wr = 0
    for start in range(0, len(paths), batch_files):
        batch_paths = paths[start : start + batch_files]
        batch_audio = [read_60s(p) for p in batch_paths]
        x = (
            np.stack([y.reshape(N_WINDOWS, WINDOW_SAMPLES) for y in batch_audio])
            .reshape(-1, WINDOW_SAMPLES)
            .astype(np.float32)
        )
        outs = session.run(None, {inp_name: x})
        logits = outs[score_idx].astype(np.float32)
        embeddings = outs[embed_idx].astype(np.float32)

        scores = np.zeros((logits.shape[0], n_classes), dtype=np.float32)
        scores[:, mapped_pos] = logits[:, mapped_bc]
        for cls_idx, perch_idxs in proxy_map.items():
            scores[:, cls_idx] = logits[:, perch_idxs].max(axis=1)
        scores_out[wr : wr + scores.shape[0]] = _sigmoid(scores)
        embeds_out.append(embeddings)
        wr += scores.shape[0]

        del x, logits, scores, batch_audio
        gc.collect()

    embeds_arr = np.concatenate(embeds_out, axis=0)
    if return_session:
        return scores_out, embeds_arr, session, inp_name, embed_idx
    return scores_out, embeds_arr


def _run_sed_branch(
    sed_dir: Path,
    paths: list[Path],
    n_classes: int,
    max_folds: int = 5,
) -> np.ndarray:
    fold_paths = sorted(
        sed_dir.glob("sed_fold*.onnx"),
        key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)),
    )
    if not fold_paths:
        raise FileNotFoundError(f"no sed_fold*.onnx under {sed_dir}")
    fold_paths = fold_paths[:max_folds]
    sessions = [_make_session(p) for p in fold_paths]
    print(f"SED: {len(sessions)} folds loaded (cap={max_folds})")

    n_rows = len(paths) * N_WINDOWS
    out = np.zeros((n_rows, n_classes), dtype=np.float32)
    wr = 0

    for path in paths:
        y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr0 != SR:
            import librosa

            y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
        if len(y) < FILE_SAMPLES:
            y = np.pad(y, (0, FILE_SAMPLES - len(y)))
        else:
            y = y[:FILE_SAMPLES]
        chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
        mel = _audio_to_sed_mel(chunks)

        p_sum = np.zeros((N_WINDOWS, n_classes), dtype=np.float32)
        for sess in sessions:
            outs = sess.run(None, {sess.get_inputs()[0].name: mel})
            clip_logits = outs[0]
            frame_max = outs[1].max(axis=1)
            p_sum += 0.5 * _sigmoid(clip_logits) + 0.5 * _sigmoid(frame_max)
        p_mean = p_sum / len(sessions)

        out[wr : wr + N_WINDOWS] = p_mean
        wr += N_WINDOWS

    return out


def _run_effnet_branch(
    ckpt_path: Path,
    paths: list[Path],
    n_classes: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Reuse src.model.AudioCNN with the trained ckpt."""
    import sys

    import torch

    # Allow loading a Windows-saved ckpt on Linux
    import pathlib as _pathlib

    if sys.platform != "win32":
        _pathlib.WindowsPath = _pathlib.PurePosixPath  # type: ignore[misc, assignment]

    from .model import AudioCNN

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = state.get("cfg", {})
    model = AudioCNN(
        backbone=cfg.get("backbone", "tf_efficientnetv2_s.in21k_ft_in1k"),
        num_classes=n_classes,
        pretrained=False,
        drop_rate=cfg.get("drop_rate", 0.2),
        drop_path_rate=cfg.get("drop_path_rate", 0.2),
        in_channels=cfg.get("in_channels", 1),
        sr=cfg.get("sample_rate", SR),
        n_fft=cfg.get("n_fft", 2048),
        hop_length=cfg.get("hop_length", 512),
        n_mels=cfg.get("n_mels", 128),
        fmin=cfg.get("fmin", 50),
        fmax=cfg.get("fmax", 16000),
        top_db=cfg.get("top_db", 80.0),
    )
    model.load_state_dict(state["model"])
    model.eval()

    n_rows = len(paths) * N_WINDOWS
    out = np.zeros((n_rows, n_classes), dtype=np.float32)
    wr = 0

    with torch.inference_mode():
        for path in paths:
            y = read_60s(path)
            chunks = torch.from_numpy(y.reshape(N_WINDOWS, WINDOW_SAMPLES)).float()
            for s in range(0, N_WINDOWS, batch_size):
                batch = chunks[s : s + batch_size]
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
                out[wr : wr + probs.shape[0]] = probs
                wr += probs.shape[0]
    return out


# ---- Top-level driver ----


def _per_class_weights3(
    bc_indices: np.ndarray,
    proxy_map: dict[int, list[int]],
    NO_LABEL: int,
    base: tuple[float, float, float],
    unmapped: tuple[float, float, float],
    proxy: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3-branch per-class weights: (perch_direct, sed, protossm)."""
    n = len(bc_indices)
    wp = np.full(n, base[0], dtype=np.float32)
    ws = np.full(n, base[1], dtype=np.float32)
    wh = np.full(n, base[2], dtype=np.float32)
    proxy_idx = set(proxy_map.keys())
    for i in range(n):
        if bc_indices[i] == NO_LABEL and i not in proxy_idx:
            wp[i], ws[i], wh[i] = unmapped
        elif i in proxy_idx:
            wp[i], ws[i], wh[i] = proxy
    s = wp + ws + wh
    return wp / s, ws / s, wh / s


def _per_class_weights4(
    bc_indices: np.ndarray,
    proxy_map: dict[int, list[int]],
    NO_LABEL: int,
    base: tuple[float, float, float, float],
    unmapped: tuple[float, float, float, float],
    proxy: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """4-branch per-class weights: (perch_direct, sed, effnet, perch_head)."""
    n = len(bc_indices)
    wp = np.full(n, base[0], dtype=np.float32)
    ws = np.full(n, base[1], dtype=np.float32)
    we = np.full(n, base[2], dtype=np.float32)
    wh = np.full(n, base[3], dtype=np.float32)
    proxy_idx = set(proxy_map.keys())
    for i in range(n):
        if bc_indices[i] == NO_LABEL and i not in proxy_idx:
            wp[i], ws[i], we[i], wh[i] = unmapped
        elif i in proxy_idx:
            wp[i], ws[i], we[i], wh[i] = proxy
    s = wp + ws + we + wh
    return wp / s, ws / s, we / s, wh / s


def _per_class_weights(
    bc_indices: np.ndarray,
    proxy_map: dict[int, list[int]],
    NO_LABEL: int,
    base: tuple[float, float, float],
    unmapped: tuple[float, float, float],
    proxy: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class blend weight vectors.

    Mapped classes get `base`. Unmapped (Perch can't predict them at all) get `unmapped`
    so EffNet/SED carry the load. Genus-proxy classes get `proxy` (intermediate Perch trust).
    Returns (w_perch, w_sed, w_eff), each shape (n_classes,).
    """
    n = len(bc_indices)
    wp = np.full(n, base[0], dtype=np.float32)
    ws = np.full(n, base[1], dtype=np.float32)
    we = np.full(n, base[2], dtype=np.float32)

    proxy_idx = set(proxy_map.keys())
    for i in range(n):
        if bc_indices[i] == NO_LABEL and i not in proxy_idx:
            wp[i], ws[i], we[i] = unmapped
        elif i in proxy_idx:
            wp[i], ws[i], we[i] = proxy
    # Re-normalize per class so weights sum to 1
    s = wp + ws + we
    return wp / s, ws / s, we / s


FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")


def _parse_site_hour(filename: str) -> tuple[str | None, int | None]:
    m = FNAME_RE.match(filename)
    if not m:
        return None, None
    return m.group(2), int(m.group(4)[:2])


def _build_site_hour_prior(
    label_csv: Path, primary_labels: list[str]
) -> tuple[dict[tuple[str, int], np.ndarray], np.ndarray]:
    """Returns ({(site, hour) → prior_vec}, global_prior_vec). Each vec is per-class
    occurrence frequency over labeled segments matching that bucket.
    """
    if not label_csv.exists():
        return {}, np.zeros(len(primary_labels), dtype=np.float32)
    df = pd.read_csv(label_csv)
    label_to_idx = {c: i for i, c in enumerate(primary_labels)}

    bucket_counts: dict[tuple[str, int], np.ndarray] = {}
    bucket_total: dict[tuple[str, int], int] = {}
    global_count = np.zeros(len(primary_labels), dtype=np.float32)
    global_total = 0
    for _, r in df.iterrows():
        site, hour = _parse_site_hour(str(r["filename"]))
        if site is None:
            continue
        key = (site, hour)
        bucket_counts.setdefault(key, np.zeros(len(primary_labels), dtype=np.float32))
        bucket_total[key] = bucket_total.get(key, 0) + 1
        global_total += 1
        for lbl in (t.strip() for t in str(r["primary_label"]).split(";") if t.strip()):
            j = label_to_idx.get(lbl)
            if j is not None:
                bucket_counts[key][j] += 1
                global_count[j] += 1

    priors = {k: v / max(1, bucket_total[k]) for k, v in bucket_counts.items()}
    global_prior = global_count / max(1, global_total)
    return priors, global_prior


def _apply_site_hour_prior(
    blended: np.ndarray,
    paths: list[Path],
    label_csv: Path,
    primary_labels: list[str],
    weight: float,
) -> np.ndarray:
    """Mix per-(site, hour) class prior into the blended rank-percentile predictions.

    blended is in [0, 1] (rank space). The prior is per-class occurrence frequency in
    the same bucket. Mixing them moves predictions slightly toward classes that
    historically appear at this site+hour, dampens rare-bucket false positives.
    """
    if weight <= 0:
        return blended
    priors, global_prior = _build_site_hour_prior(label_csv, primary_labels)
    print(
        f"[postproc] site/hour prior: {len(priors)} buckets, "
        f"global_prior_mean={float(global_prior.mean()):.4f}, weight={weight}"
    )
    for fi, path in enumerate(paths):
        site, hour = _parse_site_hour(path.name)
        prior = priors.get((site, hour), global_prior)
        for w in range(N_WINDOWS):
            row_i = fi * N_WINDOWS + w
            blended[row_i] = (1 - weight) * blended[row_i] + weight * prior
    return blended


def _rank_aware_scaling(blended: np.ndarray, n_files: int, power: float) -> np.ndarray:
    """File-level confidence scaling: multiply each (file, class) row by (file_max for that class) ^ power.

    Cited as the 2025 Rank-3 technique. Suppresses isolated false positives in uncertain
    files and reinforces confident-class clusters. Power=0.5 is the inspiration's default.
    """
    n_rows, n_classes = blended.shape
    view = blended.reshape(n_files, N_WINDOWS, n_classes)
    file_max = view.max(axis=1, keepdims=True)
    scale = np.power(np.maximum(file_max, 1e-9), power)
    return (view * scale).reshape(n_rows, n_classes).astype(np.float32)


def _delta_shift_smooth(blended: np.ndarray, n_files: int, alpha: float) -> np.ndarray:
    """3-point neighbor exponential smoothing across the time axis.

    new[t] = (1 - alpha) * old[t] + 0.5 * alpha * (old[t-1] + old[t+1])
    Cited as the 2025 Rank-1 technique. Lighter than gaussian — only adjacent windows.
    """
    n_rows, n_classes = blended.shape
    view = blended.reshape(n_files, N_WINDOWS, n_classes)
    prev_v = np.concatenate([view[:, :1, :], view[:, :-1, :]], axis=1)
    next_v = np.concatenate([view[:, 1:, :], view[:, -1:, :]], axis=1)
    smoothed = (1 - alpha) * view + 0.5 * alpha * (prev_v + next_v)
    return smoothed.reshape(n_rows, n_classes).astype(np.float32)


def _smooth_per_file(blend: np.ndarray, n_files: int, sigma: float) -> np.ndarray:
    """Gaussian smooth along the 12-window time axis within each file."""
    from scipy.ndimage import gaussian_filter1d

    out = blend.reshape(n_files, N_WINDOWS, -1)
    out = gaussian_filter1d(out, sigma=sigma, axis=1, mode="nearest")
    return out.reshape(n_files * N_WINDOWS, -1).astype(np.float32)


def run_blend_inference(
    perch_onnx_path: Path | str,
    perch_labels_csv: Path | str,
    sed_dir: Path | str,
    effnet_ckpt: Path | str,
    comp_root: Path | str,
    out_csv: Path | str,
    # 4 weights: (perch, sed, effnet, protossm). Head=0 falls back to 3-way blend.
    weights: tuple[float, float, float, float] = (0.4, 0.25, 0.15, 0.20),
    unmapped_weights: tuple[float, float, float, float] = (0.0, 0.4, 0.3, 0.3),
    proxy_weights: tuple[float, float, float, float] = (0.25, 0.30, 0.20, 0.25),
    smoothing_sigma: float = 0.0,
    test_dir_override: Path | str | None = None,
    perch_batch_files: int = 8,
    train_perch_head: bool = True,
    head_max_files: int = 200,
    head_epochs: int = 25,
    protossm_ckpt: Path | str | None = None,  # if set, skip in-kernel training
    sed_max_folds: int = 5,
    site_hour_prior_weight: float = 0.0,  # mix in per-(site,hour) prior at this weight
    rank_aware_power: float = 0.0,  # 0=off; 0.5 is the 2025 Rank-3 default
    delta_shift_alpha: float = 0.0,  # 0=off; 0.15 is the 2025 Rank-1 default
) -> pd.DataFrame:
    comp_root = Path(comp_root)
    sample_sub = pd.read_csv(comp_root / "sample_submission.csv")
    primary_labels = sample_sub.columns[1:].tolist()
    n_classes = len(primary_labels)

    test_dir = Path(test_dir_override) if test_dir_override else comp_root / "test_soundscapes"
    paths = sorted(test_dir.glob("*.ogg"))
    print(f"blend: {len(paths)} soundscape file(s) from {test_dir}")

    if not paths:
        sub = pd.DataFrame(columns=sample_sub.columns)
        sub.to_csv(out_csv, index=False)
        print(f"No test files; wrote header-only CSV to {out_csv}")
        return sub

    bc_indices, proxy_map, NO_LABEL = build_class_mapping(
        comp_root / "taxonomy.csv", perch_labels_csv, primary_labels
    )
    wp, ws, we, wh = _per_class_weights4(
        bc_indices, proxy_map, NO_LABEL, weights, unmapped_weights, proxy_weights
    )
    print(
        f"per-class weights: "
        f"perch [{wp.min():.2f},{wp.max():.2f}] "
        f"sed [{ws.min():.2f},{ws.max():.2f}] "
        f"eff [{we.min():.2f},{we.max():.2f}] "
        f"head [{wh.min():.2f},{wh.max():.2f}]"
    )

    print("blend: running Perch (test files)...")
    perch, perch_emb_test, session, inp_name, embed_idx = _run_perch_branch(
        Path(perch_onnx_path),
        Path(perch_labels_csv),
        primary_labels,
        comp_root / "taxonomy.csv",
        paths,
        batch_files=perch_batch_files,
        return_session=True,
    )
    print(f"  perch shape={perch.shape} embeddings={perch_emb_test.shape}")

    import time

    head_probs = None
    if protossm_ckpt is not None:
        from .protossm import infer_protossm, load_protossm

        print(f"blend: loading pre-trained ProtoSSM from {protossm_ckpt}")
        t0 = time.time()
        seq_model = load_protossm(Path(protossm_ckpt))
        head_probs = infer_protossm(seq_model, perch_emb_test, n_files=len(paths))
        print(f"  ProtoSSM(ckpt) inference: {time.time() - t0:.1f}s shape={head_probs.shape}")
    elif train_perch_head:
        from .protossm import collect_train_sequences, infer_protossm, train_protossm

        print("blend: training ProtoSSM on labeled train_soundscapes...")
        X_train, Y_train = collect_train_sequences(
            comp_root,
            session,
            inp_name,
            embed_idx,
            primary_labels,
            max_files=head_max_files,
        )
        print(f"  collected {len(X_train)} sequences (each (12, {X_train.shape[2] if len(X_train) else 0}))")
        if len(X_train) >= 10:
            seq_model = train_protossm(X_train, Y_train, epochs=head_epochs)
            head_probs = infer_protossm(seq_model, perch_emb_test, n_files=len(paths))
            if head_probs is not None:
                print(f"  protossm probs shape={head_probs.shape}")
        else:
            print("  not enough training files; skipping ProtoSSM")

    print(f"blend: running SED ({sed_max_folds} fold(s))...")
    t0 = time.time()
    sed = _run_sed_branch(Path(sed_dir), paths, n_classes, max_folds=sed_max_folds)
    print(f"  sed shape={sed.shape} ({time.time() - t0:.1f}s)")

    print("blend: running EffNet...")
    t0 = time.time()
    eff = _run_effnet_branch(Path(effnet_ckpt), paths, n_classes)
    print(f"  effnet shape={eff.shape} ({time.time() - t0:.1f}s)")

    # Rank-average per-class with broadcast weights
    print("blend: rank-averaging with per-class weights...")
    rp = _rank_pct(perch)
    rs = _rank_pct(sed)
    re_ = _rank_pct(eff)
    if head_probs is not None:
        rh = _rank_pct(head_probs)
        blended = rp * wp[None, :] + rs * ws[None, :] + re_ * we[None, :] + rh * wh[None, :]
    else:
        # fall back to 3-way blend, redistributing head weight uniformly
        wp3 = wp + wh / 3
        ws3 = ws + wh / 3
        we3 = we + wh / 3
        blended = rp * wp3[None, :] + rs * ws3[None, :] + re_ * we3[None, :]

    if smoothing_sigma > 0:
        print(f"blend: gaussian-smoothing sigma={smoothing_sigma} across 12-window axis...")
        blended = _smooth_per_file(blended, len(paths), smoothing_sigma)

    if site_hour_prior_weight > 0:
        blended = _apply_site_hour_prior(
            blended,
            paths,
            comp_root / "train_soundscapes_labels.csv",
            primary_labels,
            site_hour_prior_weight,
        )

    if rank_aware_power > 0:
        print(f"[postproc] rank-aware scaling power={rank_aware_power}")
        blended = _rank_aware_scaling(blended, len(paths), rank_aware_power)

    if delta_shift_alpha > 0:
        print(f"[postproc] delta-shift smoothing alpha={delta_shift_alpha}")
        blended = _delta_shift_smooth(blended, len(paths), delta_shift_alpha)

    # row_ids
    rows: list[dict] = []
    for fi, path in enumerate(paths):
        stem = path.stem
        for w in range(N_WINDOWS):
            end_s = (w + 1) * WINDOW_SECONDS
            row = {"row_id": f"{stem}_{end_s}"}
            for cls, p in zip(primary_labels, blended[fi * N_WINDOWS + w], strict=False):
                row[cls] = float(p)
            rows.append(row)

    sub = pd.DataFrame(rows)[sample_sub.columns]
    sub.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} shape={sub.shape}")
    return sub


def run_lean_blend_inference(
    perch_onnx_path: Path | str,
    perch_labels_csv: Path | str,
    sed_dir: Path | str,
    protossm_ckpt: Path | str,
    comp_root: Path | str,
    out_csv: Path | str,
    weights: tuple[float, float, float] = (0.45, 0.30, 0.25),  # perch, sed, protossm
    unmapped_weights: tuple[float, float, float] = (0.0, 0.5, 0.5),
    proxy_weights: tuple[float, float, float] = (0.30, 0.40, 0.30),
    test_dir_override: Path | str | None = None,
    perch_batch_files: int = 8,
    sed_max_folds: int = 1,
) -> pd.DataFrame:
    """Lean 3-branch blend: Perch + SED (1 fold) + pre-trained ProtoSSM.

    Designed to fit comfortably under Kaggle's 90-min CPU runtime cap by dropping the
    EffNet branch and capping SED to 1 fold. Loads ProtoSSM from a checkpoint produced
    by an external training kernel (no in-kernel training).
    """
    import time

    from .protossm import infer_protossm, load_protossm

    t_total = time.time()

    comp_root = Path(comp_root)
    sample_sub = pd.read_csv(comp_root / "sample_submission.csv")
    primary_labels = sample_sub.columns[1:].tolist()
    n_classes = len(primary_labels)

    test_dir = Path(test_dir_override) if test_dir_override else comp_root / "test_soundscapes"
    paths = sorted(test_dir.glob("*.ogg"))
    print(f"[lean] {len(paths)} soundscape file(s) from {test_dir}")

    if not paths:
        sub = pd.DataFrame(columns=sample_sub.columns)
        sub.to_csv(out_csv, index=False)
        print(f"[lean] No test files; wrote header-only CSV to {out_csv}")
        return sub

    bc_indices, proxy_map, NO_LABEL = build_class_mapping(
        comp_root / "taxonomy.csv", perch_labels_csv, primary_labels
    )
    wp, ws, wh = _per_class_weights3(
        bc_indices, proxy_map, NO_LABEL, weights, unmapped_weights, proxy_weights
    )
    print(
        f"[lean] per-class weights perch [{wp.min():.2f},{wp.max():.2f}] "
        f"sed [{ws.min():.2f},{ws.max():.2f}] protossm [{wh.min():.2f},{wh.max():.2f}]"
    )

    print("[lean] loading ProtoSSM checkpoint...")
    t0 = time.time()
    proto_model = load_protossm(Path(protossm_ckpt))
    print(f"[lean][time] load ProtoSSM: {time.time() - t0:.1f}s")

    print("[lean] running Perch...")
    t0 = time.time()
    perch, perch_emb = _run_perch_branch(
        Path(perch_onnx_path),
        Path(perch_labels_csv),
        primary_labels,
        comp_root / "taxonomy.csv",
        paths,
        batch_files=perch_batch_files,
        return_session=False,
    )
    print(f"[lean][time] Perch: {time.time() - t0:.1f}s — perch={perch.shape} emb={perch_emb.shape}")

    print("[lean] running ProtoSSM...")
    t0 = time.time()
    proto_probs = infer_protossm(proto_model, perch_emb, n_files=len(paths))
    print(f"[lean][time] ProtoSSM: {time.time() - t0:.1f}s — proto={proto_probs.shape}")

    print(f"[lean] running SED ({sed_max_folds} fold)...")
    t0 = time.time()
    sed = _run_sed_branch(Path(sed_dir), paths, n_classes, max_folds=sed_max_folds)
    print(f"[lean][time] SED: {time.time() - t0:.1f}s — sed={sed.shape}")

    print("[lean] rank-blending...")
    t0 = time.time()
    rp = _rank_pct(perch)
    rs = _rank_pct(sed)
    rh = _rank_pct(proto_probs)
    blended = rp * wp[None, :] + rs * ws[None, :] + rh * wh[None, :]
    print(f"[lean][time] rank-blend: {time.time() - t0:.1f}s")

    rows: list[dict] = []
    for fi, path in enumerate(paths):
        stem = path.stem
        for w in range(N_WINDOWS):
            end_s = (w + 1) * WINDOW_SECONDS
            row = {"row_id": f"{stem}_{end_s}"}
            for cls, p in zip(primary_labels, blended[fi * N_WINDOWS + w], strict=False):
                row[cls] = float(p)
            rows.append(row)

    sub = pd.DataFrame(rows)[sample_sub.columns]
    sub.to_csv(out_csv, index=False)
    print(f"[lean] Wrote {out_csv} shape={sub.shape}")
    print(f"[lean][time] total: {time.time() - t_total:.1f}s")
    return sub
