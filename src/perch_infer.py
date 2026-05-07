"""Perch v2 ONNX inference for BirdCLEF 2026.

Maps the 234 competition classes to Perch's vocabulary via scientific name.
For competition classes whose scientific name has no exact Perch match, a
genus-level proxy logit is computed (max over same-genus Perch entries).
Insect sonotypes and other classes with no genus-level signal are emitted
as zeros — handled later by a learned head trained on Perch embeddings.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SR = 32_000
WINDOW_SECONDS = 5
WINDOW_SAMPLES = SR * WINDOW_SECONDS
N_WINDOWS = 12
FILE_SAMPLES = SR * 60

PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}


def read_60s(path: Path | str) -> np.ndarray:
    y, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:
        y = y[:FILE_SAMPLES]
    return y


def build_class_mapping(
    taxonomy_csv: Path | str,
    perch_labels_csv: Path | str,
    primary_labels: list[str],
) -> tuple[np.ndarray, dict[int, list[int]], int]:
    """Return (bc_indices, proxy_map, NO_LABEL).

    bc_indices: shape (n_target,), Perch index for each competition class or NO_LABEL sentinel.
    proxy_map: {target_idx -> [perch_indices]} for genus-level proxies.
    """
    taxonomy = pd.read_csv(taxonomy_csv)
    bc = pd.read_csv(perch_labels_csv).reset_index().rename(columns={"index": "bc_index"})
    if "scientific_name" not in bc.columns and "inat2024_fsd50k" in bc.columns:
        bc = bc.rename(columns={"inat2024_fsd50k": "scientific_name"})
    NO_LABEL = len(bc)

    direct = taxonomy.merge(bc[["bc_index", "scientific_name"]], on="scientific_name", how="left")
    direct["bc_index"] = direct["bc_index"].fillna(NO_LABEL).astype(int)
    lbl2bc = direct.set_index("primary_label")["bc_index"].to_dict()
    bc_indices = np.array([int(lbl2bc.get(c, NO_LABEL)) for c in primary_labels], dtype=np.int32)

    label_to_idx = {c: i for i, c in enumerate(primary_labels)}
    class_name_map = taxonomy.set_index("primary_label")["class_name"].to_dict()

    proxy_map: dict[int, list[int]] = {}
    unmapped_targets = [primary_labels[i] for i in np.where(bc_indices == NO_LABEL)[0]]
    unmapped_taxa = taxonomy[taxonomy["primary_label"].isin(unmapped_targets)]

    for _, row in unmapped_taxa.iterrows():
        target = row["primary_label"]
        sci = row.get("scientific_name")
        if not isinstance(sci, str) or not sci.strip():
            continue
        genus = sci.split()[0]
        if not genus:
            continue
        hits = bc[bc["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)]
        if len(hits) and class_name_map.get(target) in PROXY_TAXA:
            proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()

    return bc_indices, proxy_map, NO_LABEL


def _resolve_session(perch_onnx_path: Path | str, intra_op_threads: int = 4):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_op_threads
    session = ort.InferenceSession(
        str(perch_onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    inp_name = session.get_inputs()[0].name
    name_to_idx = {o.name: i for i, o in enumerate(session.get_outputs())}
    score_idx = name_to_idx.get("label", 0)
    embed_idx = name_to_idx.get("embedding", 1 if len(name_to_idx) > 1 else 0)
    return session, inp_name, score_idx, embed_idx


def run_perch_inference(
    perch_onnx_path: Path | str,
    perch_labels_csv: Path | str,
    comp_root: Path | str,
    out_csv: Path | str,
    batch_files: int = 4,
    test_dir_override: Path | str | None = None,
) -> pd.DataFrame:
    session, inp_name, score_idx, _ = _resolve_session(perch_onnx_path)

    comp_root = Path(comp_root)
    sample_sub = pd.read_csv(comp_root / "sample_submission.csv")
    primary_labels = sample_sub.columns[1:].tolist()

    bc_indices, proxy_map, NO_LABEL = build_class_mapping(
        comp_root / "taxonomy.csv", perch_labels_csv, primary_labels
    )
    n_mapped = int((bc_indices != NO_LABEL).sum())
    print(
        f"classes total={len(primary_labels)} direct={n_mapped} "
        f"proxy={len(proxy_map)} unmapped={len(primary_labels) - n_mapped - len(proxy_map)}"
    )

    test_dir = Path(test_dir_override) if test_dir_override else comp_root / "test_soundscapes"
    paths = sorted(test_dir.glob("*.ogg"))
    print(f"Processing {len(paths)} soundscape file(s) from {test_dir}")

    if not paths:
        sub = pd.DataFrame(columns=sample_sub.columns)
        sub.to_csv(out_csv, index=False)
        print(f"No test files; wrote header-only CSV to {out_csv}")
        return sub

    mapped_pos = np.where(bc_indices != NO_LABEL)[0].astype(np.int32)
    mapped_bc = bc_indices[mapped_pos]

    rows: list[dict] = []
    for start in range(0, len(paths), batch_files):
        batch_paths = paths[start : start + batch_files]
        batch_audio = [read_60s(p) for p in batch_paths]
        x = np.stack(
            [y.reshape(N_WINDOWS, WINDOW_SAMPLES) for y in batch_audio]
        ).reshape(-1, WINDOW_SAMPLES).astype(np.float32)
        outs = session.run(None, {inp_name: x})
        logits = outs[score_idx].astype(np.float32)

        scores = np.zeros((logits.shape[0], len(primary_labels)), dtype=np.float32)
        scores[:, mapped_pos] = logits[:, mapped_bc]
        for cls_idx, perch_idxs in proxy_map.items():
            scores[:, cls_idx] = logits[:, perch_idxs].max(axis=1)

        # Sigmoid: score in (0, 1). ROC-AUC is rank-based so monotonic shape is what matters.
        probs = 1.0 / (1.0 + np.exp(-scores))

        for fi, path in enumerate(batch_paths):
            stem = path.stem
            for w in range(N_WINDOWS):
                end_s = (w + 1) * WINDOW_SECONDS
                row = {"row_id": f"{stem}_{end_s}"}
                for cls, p in zip(primary_labels, probs[fi * N_WINDOWS + w], strict=False):
                    row[cls] = float(p)
                rows.append(row)

        del x, logits, scores, probs, batch_audio
        gc.collect()

    sub = pd.DataFrame(rows)[sample_sub.columns]
    sub.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} shape={sub.shape}")
    return sub
