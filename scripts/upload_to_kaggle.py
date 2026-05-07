"""Automate Kaggle uploads: src dataset, checkpoints dataset, and the submission notebook.

Prereqs:
  pip install kaggle
  Place kaggle.json at %USERPROFILE%\\.kaggle\\kaggle.json (or set KAGGLE_USERNAME / KAGGLE_KEY).

Examples:
  # First-time create + push everything
  python scripts/upload_to_kaggle.py --kaggle_user yourname --create --push_notebook

  # Subsequent runs (new versions of existing resources)
  python scripts/upload_to_kaggle.py --kaggle_user yourname --push_notebook
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    print(f"+ {' '.join(cmd)} (cwd={cwd or '.'})")
    if dry_run:
        print("  [dry-run, skipped]")
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def stage_src_dataset(stage_dir: Path, kaggle_user: str, slug: str, title: str) -> None:
    """Build a self-extracting src.zip preserving the src/ package layout.

    Kaggle's `datasets create -r zip` flattens directory structure, so we instead
    ship a single zip file the notebook will extract at runtime.
    """
    import zipfile

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    src_root = REPO_ROOT / "src"
    zip_path = stage_dir / "src.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_root.rglob("*"):
            if "__pycache__" in p.parts:
                continue
            if p.is_file():
                # arcname keeps the leading "src/" so unzipping recreates the package
                arcname = Path("src") / p.relative_to(src_root)
                zf.write(p, arcname=str(arcname))

    meta = {
        "title": title,
        "id": f"{kaggle_user}/{slug}",
        "licenses": [{"name": "MIT"}],
    }
    (stage_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


def stage_ckpt_dataset(stage_dir: Path, ckpt_glob: str, kaggle_user: str, slug: str, title: str) -> int:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    n = 0
    for cp in sorted(Path().glob(ckpt_glob)):
        shutil.copy(cp, stage_dir / cp.name)
        n += 1
    if n == 0:
        raise SystemExit(f"No checkpoints matched: {ckpt_glob}")
    meta = {
        "title": title,
        "id": f"{kaggle_user}/{slug}",
        "licenses": [{"name": "MIT"}],
    }
    (stage_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    return n


def stage_notebook(
    stage_dir: Path,
    kaggle_user: str,
    notebook_slug: str,
    title: str,
    competition_slug: str,
    src_dataset: str,
    ckpt_dataset: str,
    competition_data_root: str,
    enable_gpu: bool,
    training_kernel: str | None = None,
) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    # Build the notebook source. Single-cell Python script that does the inference.
    src_slug = src_dataset.split('/')[-1]
    ckpt_slug = ckpt_dataset.split('/')[-1]
    competition_slug_only = competition_slug.split('/')[-1]
    code_lines = [
        "import sys, glob, os, shutil",
        "# Walk /kaggle/input to find what we need (mount layout varies)",
        "INPUT = '/kaggle/input'",
        "src_init = None       # path to <something>/src/__init__.py",
        "ckpt_paths = []        # *.pt files anywhere",
        "competition_root = None  # dir containing taxonomy.csv",
        "for dirpath, dirnames, filenames in os.walk(INPUT):",
        "    for f in filenames:",
        "        full = os.path.join(dirpath, f)",
        "        if f == '__init__.py' and os.path.basename(dirpath) == 'src':",
        "            src_init = full",
        "        elif f.endswith('.pt'):",
        "            ckpt_paths.append(full)",
        "        elif f == 'taxonomy.csv':",
        "            competition_root = dirpath",
        "if not src_init:",
        "    print('=== /kaggle/input tree ===')",
        "    for dp,_,fs in os.walk(INPUT):",
        "        for f in fs: print(os.path.join(dp,f))",
        "    raise SystemExit('src package not found')",
        "if not ckpt_paths:",
        "    raise SystemExit('no .pt checkpoints found')",
        "if not competition_root:",
        "    raise SystemExit('competition data (taxonomy.csv) not found')",
        "src_parent = os.path.dirname(os.path.dirname(src_init))",
        "sys.path.insert(0, src_parent)",
        "print(f'src package: {os.path.dirname(src_init)}')",
        "print(f'ckpts: {sorted(ckpt_paths)}')",
        "print(f'competition_root: {competition_root}')",
        "from src.infer import main as run_infer",
        "sys.argv = [",
        "    'infer',",
        "    '--data_root', competition_root,",
        "    '--ckpts', ','.join(sorted(ckpt_paths)),",
        "    '--out', '/kaggle/working/submission.csv',",
        "    '--batch_size', '16',",
        "]",
        "run_infer()",
    ]
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code_lines],
            }
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (stage_dir / f"{notebook_slug}.ipynb").write_text(json.dumps(nb, indent=2))

    if training_kernel:
        dataset_sources = [src_dataset]
        kernel_sources = [training_kernel]
    else:
        dataset_sources = [src_dataset, ckpt_dataset]
        kernel_sources = []
    if training_kernel and "protossm" in training_kernel:
        # already covered
        pass
    meta = {
        "id": f"{kaggle_user}/{notebook_slug}",
        "title": title,
        "code_file": f"{notebook_slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": False,
        "dataset_sources": dataset_sources,
        "competition_sources": [competition_slug],
        "kernel_sources": kernel_sources,
    }
    (stage_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def stage_training_notebook(
    stage_dir: Path,
    kaggle_user: str,
    notebook_slug: str,
    title: str,
    competition_slug: str,
    src_dataset: str,
    epochs: int,
    batch_size: int,
    backbone: str,
    fold: int,
    folds: int,
) -> None:
    """Build a training notebook (GPU + internet) that runs src.train and saves checkpoints to /kaggle/working."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    code_lines = [
        "import subprocess, sys, os, glob",
        "# Detect GPU via nvidia-smi BEFORE importing torch — once torch loads its C",
        "# extensions, reimporting after a pip-upgrade fails. So we shell out first.",
        "needs_compat = False",
        "try:",
        "    gpu_info = subprocess.check_output(",
        "        ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader']",
        "    ).decode().strip()",
        "    print('detected GPU(s):', gpu_info)",
        "    # P100 = sm_60, V100 = sm_70, T4 = sm_75. Default Kaggle PyTorch needs >= sm_70.",
        "    needs_compat = ('6.0' in gpu_info) or ('6.1' in gpu_info)",
        "except Exception as e:",
        "    print('nvidia-smi failed:', e)",
        "if needs_compat:",
        "    print('Installing PyTorch 2.5.1+cu118 (sm_60 supported)...')",
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',",
        "        'torch==2.5.1', 'torchaudio==2.5.1', 'torchvision==0.20.1',",
        "        '--extra-index-url', 'https://download.pytorch.org/whl/cu118'])",
        "    print('install complete')",
        "INPUT = '/kaggle/input'",
        "src_init = None",
        "comp_root = None",
        "for dp,_,fs in os.walk(INPUT):",
        "    for f in fs:",
        "        if f == '__init__.py' and os.path.basename(dp) == 'src':",
        "            src_init = os.path.join(dp, f)",
        "        elif f == 'taxonomy.csv':",
        "            comp_root = dp",
        "assert src_init, 'src package not found'",
        "assert comp_root, 'competition data not found'",
        "sys.path.insert(0, os.path.dirname(os.path.dirname(src_init)))",
        "print(f'src package: {os.path.dirname(src_init)}')",
        "print(f'comp_root: {comp_root}')",
        "import torch",
        "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())",
        "if torch.cuda.is_available():",
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')",
        "from src.train import main as run_train",
        "sys.argv = [",
        "    'train',",
        "    '--data_root', comp_root,",
        "    '--work_dir', '/kaggle/working/work',",
        f"    '--backbone', '{backbone}',",
        f"    '--epochs', '{epochs}',",
        f"    '--batch_size', '{batch_size}',",
        "    '--num_workers', '4',",
        f"    '--folds', '{folds}',",
        f"    '--fold', '{fold}',",
        "    '--lr', '1e-3',",
        "    '--pretrained', '1',",
        "]",
        "run_train()",
        "# Surface the trained checkpoint at the kernel-output root for easy consumption",
        "import shutil",
        "for cp in glob.glob('/kaggle/working/work/ckpt/fold*_best.pt'):",
        "    dst = os.path.join('/kaggle/working', os.path.basename(cp))",
        "    if cp != dst:",
        "        shutil.copy(cp, dst)",
        "        print('exported', dst)",
    ]
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code_lines],
            }
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (stage_dir / f"{notebook_slug}.ipynb").write_text(json.dumps(nb, indent=2))
    meta = {
        "id": f"{kaggle_user}/{notebook_slug}",
        "title": title,
        "code_file": f"{notebook_slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        # Request T4 specifically. P100 (sm_60) is too old for current Kaggle PyTorch builds.
        "accelerator": "gpu-t4-x2",
        "enable_internet": True,
        "dataset_sources": [src_dataset],
        "competition_sources": [competition_slug],
        "kernel_sources": [],
    }
    (stage_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def stage_perch_notebook(
    stage_dir: Path,
    kaggle_user: str,
    notebook_slug: str,
    title: str,
    competition_slug: str,
    src_dataset: str,
    perch_onnx_dataset: str = "rishikeshjani/perch-onnx-for-birdclef-2026",
) -> None:
    """Phase 1: Perch-only inference. Attaches the ONNX Perch model and runs src.perch_infer."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    code_lines = [
        "import sys, os, glob, subprocess, gc",
        "INPUT = '/kaggle/input'",
        "src_init = perch_onnx = perch_labels = onnx_wheel = comp_root = None",
        "for dp, _, fs in os.walk(INPUT):",
        "    for f in fs:",
        "        full = os.path.join(dp, f)",
        "        if f == '__init__.py' and os.path.basename(dp) == 'src':",
        "            src_init = full",
        "        elif f == 'perch_v2.onnx':",
        "            perch_onnx = full",
        "        elif f == 'labels.csv' and 'perch' in full.lower():",
        "            perch_labels = full",
        "        elif f.startswith('onnxruntime-') and f.endswith('.whl'):",
        "            onnx_wheel = full",
        "        elif f == 'taxonomy.csv':",
        "            comp_root = dp",
        "assert src_init, 'src package not found'",
        "assert perch_onnx, 'perch_v2.onnx not found'",
        "assert perch_labels, 'perch labels.csv not found'",
        "assert onnx_wheel, 'onnxruntime wheel not found'",
        "assert comp_root, 'competition data not found'",
        "print(f'src: {os.path.dirname(src_init)}')",
        "print(f'perch: {perch_onnx}')",
        "print(f'labels: {perch_labels}')",
        "print(f'wheel: {onnx_wheel}')",
        "print(f'comp_root: {comp_root}')",
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', onnx_wheel])",
        "sys.path.insert(0, os.path.dirname(os.path.dirname(src_init)))",
        "from src.perch_infer import run_perch_inference",
        "run_perch_inference(",
        "    perch_onnx_path=perch_onnx,",
        "    perch_labels_csv=perch_labels,",
        "    comp_root=comp_root,",
        "    out_csv='/kaggle/working/submission.csv',",
        "    batch_files=4,",
        ")",
    ]
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code_lines],
            }
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (stage_dir / f"{notebook_slug}.ipynb").write_text(json.dumps(nb, indent=2))
    meta = {
        "id": f"{kaggle_user}/{notebook_slug}",
        "title": title,
        "code_file": f"{notebook_slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [src_dataset, perch_onnx_dataset],
        "competition_sources": [competition_slug],
        "kernel_sources": [],
    }
    (stage_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def stage_lean_blend_notebook(
    stage_dir: Path,
    kaggle_user: str,
    notebook_slug: str,
    title: str,
    competition_slug: str,
    src_dataset: str,
    protossm_kernel: str,
    perch_onnx_dataset: str = "rishikeshjani/perch-onnx-for-birdclef-2026",
    sed_dataset: str = "tuckerarrants/bc2026-distilled-sed-public",
) -> None:
    """Lean 3-branch submission: Perch + SED 1 fold + pre-trained ProtoSSM."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    code_lines = [
        "import sys, os, glob, subprocess",
        "INPUT = '/kaggle/input'",
        "src_init = perch_onnx = perch_labels = onnx_wheel = comp_root = sed_dir = protossm_ckpt = None",
        "for dp, _, fs in os.walk(INPUT):",
        "    for f in fs:",
        "        full = os.path.join(dp, f)",
        "        if f == '__init__.py' and os.path.basename(dp) == 'src':",
        "            src_init = full",
        "        elif f == 'perch_v2.onnx':",
        "            perch_onnx = full",
        "        elif f == 'labels.csv' and 'perch' in full.lower():",
        "            perch_labels = full",
        "        elif f.startswith('onnxruntime-') and f.endswith('.whl'):",
        "            onnx_wheel = full",
        "        elif f == 'taxonomy.csv':",
        "            comp_root = dp",
        "        elif f == 'sed_fold0.onnx':",
        "            sed_dir = dp",
        "        elif f == 'protossm_best.pt':",
        "            protossm_ckpt = full",
        "for v in ['src_init','perch_onnx','perch_labels','onnx_wheel','comp_root','sed_dir','protossm_ckpt']:",
        "    print(f'  {v}={eval(v)}')",
        "assert all([src_init, perch_onnx, perch_labels, onnx_wheel, comp_root, sed_dir, protossm_ckpt]), 'missing input'",
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', onnx_wheel])",
        "sys.path.insert(0, os.path.dirname(os.path.dirname(src_init)))",
        "from src.blend_infer import run_lean_blend_inference",
        "run_lean_blend_inference(",
        "    perch_onnx_path=perch_onnx,",
        "    perch_labels_csv=perch_labels,",
        "    sed_dir=sed_dir,",
        "    protossm_ckpt=protossm_ckpt,",
        "    comp_root=comp_root,",
        "    out_csv='/kaggle/working/submission.csv',",
        "    weights=(0.45, 0.30, 0.25),",
        "    perch_batch_files=8,",
        "    sed_max_folds=1,",
        ")",
    ]
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code_lines],
            }
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (stage_dir / f"{notebook_slug}.ipynb").write_text(json.dumps(nb, indent=2))
    meta = {
        "id": f"{kaggle_user}/{notebook_slug}",
        "title": title,
        "code_file": f"{notebook_slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [src_dataset, perch_onnx_dataset, sed_dataset],
        "competition_sources": [competition_slug],
        "kernel_sources": [protossm_kernel],
    }
    (stage_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def stage_protossm_training_notebook(
    stage_dir: Path,
    kaggle_user: str,
    notebook_slug: str,
    title: str,
    competition_slug: str,
    src_dataset: str,
    perch_onnx_dataset: str = "rishikeshjani/perch-onnx-for-birdclef-2026",
    perch_meta_dataset: str = "jaejohn/perch-meta",
    epochs: int = 30,
    hidden: int = 128,
    dropout: float = 0.5,
    weight_decay: float = 1e-2,
    max_files: int = 200,
) -> None:
    """CPU+internet kernel: trains ProtoSSM v2 on cached Perch embeddings (perch-meta) with strong regularization + early stopping."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    code_lines = [
        "import sys, os, glob, subprocess",
        "INPUT = '/kaggle/input'",
        "src_init = comp_root = perch_meta_dir = None",
        "for dp, _, fs in os.walk(INPUT):",
        "    for f in fs:",
        "        full = os.path.join(dp, f)",
        "        if f == '__init__.py' and os.path.basename(dp) == 'src':",
        "            src_init = full",
        "        elif f == 'taxonomy.csv':",
        "            comp_root = dp",
        "        elif f == 'full_perch_arrays.npz':",
        "            perch_meta_dir = dp",
        "for v in ['src_init','comp_root','perch_meta_dir']:",
        "    print(f'  {v}={eval(v)}')",
        "assert all([src_init, comp_root, perch_meta_dir]), 'missing input'",
        "# pyarrow needed for parquet read",
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyarrow'])",
        "sys.path.insert(0, os.path.dirname(os.path.dirname(src_init)))",
        "from pathlib import Path",
        "from src.protossm import train_and_save_from_cache",
        "train_and_save_from_cache(",
        "    perch_meta_dir=Path(perch_meta_dir),",
        "    comp_root=Path(comp_root),",
        "    out_ckpt=Path('/kaggle/working/protossm_best.pt'),",
        f"    hidden={hidden},",
        f"    dropout={dropout},",
        f"    weight_decay={weight_decay},",
        f"    max_epochs={epochs},",
        "    val_frac=0.20,",
        "    patience=5,",
        ")",
    ]
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code_lines],
            }
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (stage_dir / f"{notebook_slug}.ipynb").write_text(json.dumps(nb, indent=2))
    meta = {
        "id": f"{kaggle_user}/{notebook_slug}",
        "title": title,
        "code_file": f"{notebook_slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [src_dataset, perch_meta_dataset],
        "competition_sources": [competition_slug],
        "kernel_sources": [],
    }
    (stage_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def stage_blend_notebook(
    stage_dir: Path,
    kaggle_user: str,
    notebook_slug: str,
    title: str,
    competition_slug: str,
    src_dataset: str,
    training_kernel: str,
    protossm_kernel: str | None = None,
    perch_onnx_dataset: str = "rishikeshjani/perch-onnx-for-birdclef-2026",
    sed_dataset: str = "tuckerarrants/bc2026-distilled-sed-public",
) -> None:
    """Phase 2: 3-way rank-average blend of Perch + EffNet + distilled SED."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    code_lines = [
        "import sys, os, glob, subprocess",
        "INPUT = '/kaggle/input'",
        "src_init = perch_onnx = perch_labels = onnx_wheel = comp_root = sed_dir = ckpt = protossm_ckpt = None",
        "for dp, _, fs in os.walk(INPUT):",
        "    for f in fs:",
        "        full = os.path.join(dp, f)",
        "        if f == '__init__.py' and os.path.basename(dp) == 'src':",
        "            src_init = full",
        "        elif f == 'perch_v2.onnx':",
        "            perch_onnx = full",
        "        elif f == 'labels.csv' and 'perch' in full.lower():",
        "            perch_labels = full",
        "        elif f.startswith('onnxruntime-') and f.endswith('.whl'):",
        "            onnx_wheel = full",
        "        elif f == 'taxonomy.csv':",
        "            comp_root = dp",
        "        elif f == 'sed_fold0.onnx':",
        "            sed_dir = dp",
        "        elif f == 'fold0_best.pt':",
        "            ckpt = full",
        "        elif f == 'protossm_best.pt':",
        "            protossm_ckpt = full",
        "for v in ['src_init','perch_onnx','perch_labels','onnx_wheel','comp_root','sed_dir','ckpt','protossm_ckpt']:",
        "    print(f'  {v}={eval(v)}')",
        "assert all([src_init, perch_onnx, perch_labels, onnx_wheel, comp_root, sed_dir, ckpt]), 'missing input'",
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', onnx_wheel])",
        "sys.path.insert(0, os.path.dirname(os.path.dirname(src_init)))",
        "from src.blend_infer import run_blend_inference",
        "run_blend_inference(",
        "    perch_onnx_path=perch_onnx,",
        "    perch_labels_csv=perch_labels,",
        "    sed_dir=sed_dir,",
        "    effnet_ckpt=ckpt,",
        "    comp_root=comp_root,",
        "    out_csv='/kaggle/working/submission.csv',",
        "    # Phase 11 best: 3-way uniform + prior 0.05 + rank-aware 0.5 + delta-shift 0.15 (LB 0.932)",
        "    # Phase 12 (power=0.7) and Phase 13 (prior=0.10) were flat; reverted.",
        "    weights=(0.50, 0.30, 0.20, 0.0),",
        "    unmapped_weights=(0.50, 0.30, 0.20, 0.0),",
        "    proxy_weights=(0.50, 0.30, 0.20, 0.0),",
        "    smoothing_sigma=0.0,",
        "    site_hour_prior_weight=0.05,",
        "    rank_aware_power=0.5,",
        "    delta_shift_alpha=0.15,",
        "    perch_batch_files=8,",
        "    sed_max_folds=5,",
        "    protossm_ckpt=None,  # explicitly skip ProtoSSM in this run",
        ")",
    ]
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code_lines],
            }
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (stage_dir / f"{notebook_slug}.ipynb").write_text(json.dumps(nb, indent=2))
    meta = {
        "id": f"{kaggle_user}/{notebook_slug}",
        "title": title,
        "code_file": f"{notebook_slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [src_dataset, perch_onnx_dataset, sed_dataset],
        "competition_sources": [competition_slug],
        "kernel_sources": [training_kernel] + ([protossm_kernel] if protossm_kernel else []),
    }
    (stage_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kaggle_user", required=True, help="Your Kaggle username")
    p.add_argument("--src_slug", default="pantanal-src")
    p.add_argument("--ckpt_slug", default="pantanal-ckpts")
    p.add_argument("--notebook_slug", default="pantanal-submission")
    p.add_argument("--training_slug", default="pantanal-training")
    p.add_argument("--ckpt_glob", default="work/ckpt/fold*_best.pt")
    p.add_argument("--competition", default="birdclef-2026")
    p.add_argument("--competition_data_root", default="/kaggle/input/birdclef-2026")
    p.add_argument("--stage_root", type=Path, default=REPO_ROOT / "work" / "kaggle_stage")
    p.add_argument("--create", action="store_true", help="Use 'datasets create' on first push (otherwise 'version')")
    p.add_argument("--push_notebook", action="store_true", help="Push the submission notebook")
    p.add_argument("--push_training", action="store_true", help="Push the training notebook (GPU+internet)")
    p.add_argument("--push_perch", action="store_true", help="Push a Phase-1 Perch-only submission notebook")
    p.add_argument("--push_blend", action="store_true", help="Push a Phase-2 3-way blend submission notebook (Perch+EffNet+SED)")
    p.add_argument("--push_protossm_train", action="store_true", help="Push a kernel that trains ProtoSSM and exports protossm_best.pt")
    p.add_argument("--push_lean", action="store_true", help="Push lean 3-branch submission notebook (Perch+SED1+ProtoSSM)")
    p.add_argument("--protossm_slug", default="pantanal-protossm-train")
    p.add_argument("--enable_gpu", action="store_true", help="Request GPU for the submission notebook")
    p.add_argument("--use_training_kernel", action="store_true", help="Submission notebook reads checkpoints from the training kernel output instead of pantanal-ckpts dataset")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs (training notebook only)")
    p.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    p.add_argument("--backbone", default="tf_efficientnetv2_s.in21k_ft_in1k")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--skip_src", action="store_true")
    p.add_argument("--skip_ckpt", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Stage files only; skip kaggle CLI calls")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dry_run:
        cred = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".kaggle" / "kaggle.json"
        if not cred.exists() and not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            raise SystemExit(f"Missing Kaggle credentials. Place kaggle.json at {cred} or set KAGGLE_USERNAME/KAGGLE_KEY.")

    src_dataset = f"{args.kaggle_user}/{args.src_slug}"
    ckpt_dataset = f"{args.kaggle_user}/{args.ckpt_slug}"

    if not args.skip_src:
        stage = args.stage_root / "src_dataset"
        stage_src_dataset(stage, args.kaggle_user, args.src_slug, "Pantanal Src")
        if args.create:
            run(["kaggle", "datasets", "create", "-p", str(stage)], dry_run=args.dry_run)
        else:
            run(["kaggle", "datasets", "version", "-p", str(stage), "-m", "auto: src update"], dry_run=args.dry_run)

    if not args.skip_ckpt:
        stage = args.stage_root / "ckpt_dataset"
        n = stage_ckpt_dataset(stage, args.ckpt_glob, args.kaggle_user, args.ckpt_slug, "Pantanal Checkpoints")
        print(f"  staged {n} checkpoint(s)")
        if args.create:
            run(["kaggle", "datasets", "create", "-p", str(stage)], dry_run=args.dry_run)
        else:
            run(["kaggle", "datasets", "version", "-p", str(stage), "-m", "auto: ckpt update"], dry_run=args.dry_run)

    if args.push_lean:
        stage = args.stage_root / "lean_notebook"
        protossm_kernel = f"{args.kaggle_user}/{args.protossm_slug}"
        stage_lean_blend_notebook(
            stage,
            args.kaggle_user,
            args.notebook_slug,
            "Pantanal Submission",
            args.competition,
            src_dataset,
            protossm_kernel,
        )
        run(["kaggle", "kernels", "push", "-p", str(stage)], dry_run=args.dry_run)
        print(
            f"Lean submission notebook pushed: https://www.kaggle.com/code/{args.kaggle_user}/{args.notebook_slug}"
        )

    if args.push_protossm_train:
        stage = args.stage_root / "protossm_training_notebook"
        stage_protossm_training_notebook(
            stage,
            args.kaggle_user,
            args.protossm_slug,
            "Pantanal ProtoSSM Training",
            args.competition,
            src_dataset,
        )
        run(["kaggle", "kernels", "push", "-p", str(stage)], dry_run=args.dry_run)
        print(
            f"ProtoSSM training notebook pushed: https://www.kaggle.com/code/{args.kaggle_user}/{args.protossm_slug}\n"
            f"Status: kaggle kernels status {args.kaggle_user}/{args.protossm_slug}"
        )

    if args.push_blend:
        stage = args.stage_root / "blend_notebook"
        training_kernel = f"{args.kaggle_user}/{args.training_slug}"
        protossm_kernel = f"{args.kaggle_user}/{args.protossm_slug}" if args.protossm_slug else None
        stage_blend_notebook(
            stage,
            args.kaggle_user,
            args.notebook_slug,
            "Pantanal Submission",
            args.competition,
            src_dataset,
            training_kernel,
            protossm_kernel=protossm_kernel,
        )
        run(["kaggle", "kernels", "push", "-p", str(stage)], dry_run=args.dry_run)
        print(
            "Blend submission notebook pushed. Monitor at\n"
            f"  https://www.kaggle.com/code/{args.kaggle_user}/{args.notebook_slug}\n"
            f"To submit after it completes:\n"
            f"  kaggle competitions submit -c {args.competition} -f submission.csv "
            f"-k {args.kaggle_user}/{args.notebook_slug} -v <V> -m 'Phase 2 blend'"
        )

    if args.push_perch:
        stage = args.stage_root / "perch_notebook"
        stage_perch_notebook(
            stage,
            args.kaggle_user,
            args.notebook_slug,
            "Pantanal Submission",
            args.competition,
            src_dataset,
        )
        run(["kaggle", "kernels", "push", "-p", str(stage)], dry_run=args.dry_run)
        print(
            "Perch submission notebook pushed. Monitor at\n"
            f"  https://www.kaggle.com/code/{args.kaggle_user}/{args.notebook_slug}\n"
            f"To submit after it completes:\n"
            f"  kaggle competitions submit -c {args.competition} -f submission.csv "
            f"-k {args.kaggle_user}/{args.notebook_slug} -v <V> -m 'Phase 1 Perch'"
        )

    if args.push_training:
        stage = args.stage_root / "training_notebook"
        training_kernel = f"{args.kaggle_user}/{args.training_slug}"
        stage_training_notebook(
            stage,
            args.kaggle_user,
            args.training_slug,
            "Pantanal Training",
            args.competition,
            src_dataset,
            args.epochs,
            args.batch_size,
            args.backbone,
            args.fold,
            args.folds,
        )
        run(["kaggle", "kernels", "push", "-p", str(stage)], dry_run=args.dry_run)
        print(
            "Training notebook pushed. Monitor at\n"
            f"  https://www.kaggle.com/code/{args.kaggle_user}/{args.training_slug}\n"
            f"or `kaggle kernels status {training_kernel}`"
        )

    if args.push_notebook:
        stage = args.stage_root / "notebook"
        training_kernel = f"{args.kaggle_user}/{args.training_slug}" if args.use_training_kernel else None
        stage_notebook(
            stage,
            args.kaggle_user,
            args.notebook_slug,
            "Pantanal Submission",
            args.competition,
            src_dataset,
            ckpt_dataset,
            args.competition_data_root,
            args.enable_gpu,
            training_kernel=training_kernel,
        )
        run(["kaggle", "kernels", "push", "-p", str(stage)], dry_run=args.dry_run)
        print(
            "Notebook pushed. To submit to the competition, run:\n"
            f"  kaggle competitions submit -c {args.competition} -f submission.csv -m 'auto'\n"
            f"or open https://www.kaggle.com/code/{args.kaggle_user}/{args.notebook_slug} and click Submit."
        )


if __name__ == "__main__":
    main()
