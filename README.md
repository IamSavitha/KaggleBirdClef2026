# KaggleBirdClef2026

End-to-end pipeline for the [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)
competition (acoustic species ID across 234 classes in the Brazilian Pantanal).

**Public LB best: 0.932** — 3-way rank-average blend (Google Perch ONNX + distilled SED 5-fold + EffNetV2-S trained on competition data) with a post-processing stack: per-(site, hour) prior, file-level rank-aware scaling, and 3-point delta-shift smoothing.

For the academic write-up + slides, see [`docs/REPORT.md`](docs/REPORT.md) and [`docs/Presentation.pptx`](docs/Presentation.pptx).

---

## Architecture (0.932 — Phase 11)

```
                            soundscape (60s, 32 kHz)
                                       │
                       split into 12 × 5-second windows
                                       │
   ┌───────────────────────────────────┼───────────────────────────────────┐
   ▼                                   ▼                                   ▼
┌────────────────────┐    ┌────────────────────┐                ┌────────────────────┐
│   Perch v2 ONNX    │    │  Distilled SED     │                │  EfficientNetV2-S  │
│   (Google) →       │    │  (5 ONNX folds) →  │                │  trained on comp → │
│   logits + 1536-d  │    │  clip + frame_max  │                │  fold0 ckpt        │
│   embeddings       │    │  sigmoid avg       │                │                    │
└────────┬───────────┘    └────────┬───────────┘                └────────┬───────────┘
         │ map 234 classes via      │                                    │
         │ scientific_name; genus   │                                    │
         │ proxy for unmapped       │                                    │
         ▼                          ▼                                    ▼
       (12,234)                 (12,234)                              (12,234)
         │                          │                                    │
         └─────────── per-class column rank → percentiles ────────────────┘
                                       │
                w_perch · rank_p   +   w_sed · rank_s   +   w_eff · rank_e
                                       │
       ┌───────────── post-processing (Phase 7+) ──────────────────────┐
       │  (a) site/hour prior mix (weight 0.05)                        │
       │  (b) rank-aware scaling (file_max ^ 0.5)                      │
       │  (c) delta-shift smoothing (α = 0.15)                         │
       └────────────────────────────────────────────────────────────────┘
                                       │
                       submission.csv  (row_id, 234 species cols)
```

### Class coverage (234 total)

| Bucket | Count | How it's handled |
|---|---:|---|
| Direct match in Perch vocabulary | 203 | `scientific_name` join with Perch labels.csv → use Perch's own logit |
| Genus-level proxy | 3 | max-pool over same-genus Perch logits |
| No match (insect sonotypes etc.) | 28 | EffNet covers these (trained on the comp's own labels) |

### Why this works

- **Perch** is a foundation model trained by Google on a massive bird/animal vocalization corpus; it directly produces good logits for ~206 of the 234 target species
- **EffNetV2-S** trained on the competition data covers the 28 unmapped sonotype classes that Perch can't predict
- **Distilled SED ONNX** (5 folds from Tucker Arrants' public dataset) adds spectrogram-local event evidence that's complementary to Perch's embedding-style predictions
- **Rank averaging** is robust against the very different score distributions of these heterogeneous models — raw probability averaging would be dominated by whichever branch happens to output the largest scores

### Leaderboard progression

| Date | Phase | Submission | Public LB |
|------|------|-----------|----------:|
| 2026-05-02 | 0 | EffNet fold 0 (10 epochs P100 no-AMP) | 0.856 |
| 2026-05-03 | 1 | Perch ONNX only | 0.835 |
| 2026-05-03 | **2** | **3-way blend (Perch + SED 5 + EffNet, rank-avg 0.5/0.3/0.2)** | **0.927** |
| 2026-05-03 | 2.1 | + per-class weights + gaussian smoothing σ=0.65 | 0.924 |
| 2026-05-03 | 4 | 4-way thick blend (in-kernel ProtoSSM training) | failed (90-min cap) |
| 2026-05-03 | 4-lean | Perch + SED 1 + pre-trained ProtoSSM (no EffNet) | 0.873 |
| 2026-05-04 | 5 | 4-way thick + ProtoSSM-v1-ckpt (overfit) | 0.908 |
| 2026-05-04 | 6 | 4-way thick + ProtoSSM-v2 regularized | 0.909 |
| 2026-05-04 | **7** | + (site, hour) prior weight 0.05 | **0.928** |
| 2026-05-04 | 8 | + gaussian smoothing σ=0.30 | 0.928 |
| 2026-05-04 | 9 | + per-class weights (Perch=0 on 28 unmapped) | 0.926 |
| 2026-05-05 | **10** | + rank-aware scaling power 0.5 | **0.931** |
| 2026-05-05 | **11** | **+ delta-shift smoothing α=0.15  ←  best** | **0.932** |
| 2026-05-05 | 12 | rank-aware power tuned to 0.7 | 0.932 (flat — reverted) |
| 2026-05-05 | 13 | (site, hour) prior weight tuned to 0.10 | 0.932 (flat — reverted) |

The dominant gain (+0.092) came from the 3-way rank-average blend itself — no individual model in the toolkit covers all 234 classes well. Subsequent post-processing added a further +0.005 to land at 0.932.

---

## Repo layout

```
src/
  config.py           # paths + hyperparameters
  audio.py            # soundfile loader, mel-spec, SpecAugment, NaN-safe
  dataset.py          # multi-label dataset (XC/iNat clips + soundscape segments)
  model.py            # timm CNN backbone + multi-label head
  train.py            # 5-fold CV trainer (mixup, cosine LR, AMP-on-cuda)
  infer.py            # standalone EffNet-only submission writer
  perch_infer.py      # Perch ONNX inference + class mapping + genus proxies
  protossm.py         # bi-LSTM over Perch embeddings (Phase 4 sequence head)
  perch_head.py       # simpler MLP probe variant
  blend_infer.py      # 3-/4-branch rank-average orchestrator
scripts/
  upload_to_kaggle.py    # automate dataset/notebook uploads + submissions
  build_presentation.py  # generates docs/Presentation.pptx
notebooks/
  submission.py          # local notebook entry-point (Kaggle re-uses src.infer/blend_infer)
docs/
  REPORT.md              # IEEE-format final report (architecture, ablations, failed experiments)
  Presentation.pptx      # 15-slide deck for milestone presentation
.claude/
  settings.json       # Powershell allow rule for the harness
.gitignore            # keeps secrets, data/, work/, *.pt out of git
README.md             # this file
requirements.txt      # torch (CPU), torchaudio, timm, pandas, sklearn, soundfile
```

---

## How to reproduce 0.927

Prerequisites:
- Kaggle account joined to the BirdCLEF 2026 competition (rules accepted)
- `kaggle.json` API token at `~/.kaggle/kaggle.json` (or env `KAGGLE_USERNAME`/`KAGGLE_KEY`)
- Python 3.12 + `pip install -r requirements.txt`

### 1. Train EffNetV2-S fold 0 on Kaggle GPU

```
python scripts/upload_to_kaggle.py --kaggle_user <yourname> --create --skip_ckpt --push_training \
       --epochs 10 --batch_size 32
```

Kaggle assigns a GPU (P100 or T4); the notebook auto-installs `torch==2.5.1+cu118`
on P100 (sm_60 needs older PyTorch). Training emits `fold0_best.pt` to the
kernel output. Roughly 3 hours of P100 wall time.

### 2. Push the submission notebook (3-way blend)

```
python scripts/upload_to_kaggle.py --kaggle_user <yourname> --skip_ckpt --push_blend
```

Attaches:
- `<yourname>/pantanal-src` (this repo's `src/` package, zipped & uploaded)
- `rishikeshjani/perch-onnx-for-birdclef-2026` (Perch ONNX + onnxruntime wheel)
- `tuckerarrants/bc2026-distilled-sed-public` (5 SED ONNX folds)
- `birdclef-2026` competition
- `<yourname>/pantanal-training` (output of step 1, contains `fold0_best.pt`)

The notebook walks `/kaggle/input` to discover all artifacts, installs the
onnxruntime wheel offline, and runs `src.blend_infer.run_blend_inference`.

### 3. Submit

```
kaggle competitions submit -c birdclef-2026 -f submission.csv \
       -k <yourname>/pantanal-submission -v <kernel-version> \
       -m "3-way blend Perch+SED+EffNet"
```

(Replace `<kernel-version>` with the integer printed in step 2.)

Verify with `kaggle competitions submissions birdclef-2026 | head`.

---

## Hard-won lessons (in `~/.claude/skills/kaggle-competition-iterator/SKILL.md`)

1. **`/kaggle/input/` layout varies** — never hard-code paths; walk for
   `taxonomy.csv`, `*.onnx`, `__init__.py`, `*.pt`.
2. **Kaggle dataset versioning is async** — when you `datasets version` then
   immediately push a kernel, the kernel can pin to the previous version.
3. **WindowsPath pickle fails on Linux** — ckpts saved on Windows pickle
   `pathlib.WindowsPath`; shim it before `torch.load` on Linux.
4. **P100 + recent PyTorch** — Kaggle's default torch dropped sm_60; install
   `torch==2.5.1+cu118` for P100 by detecting via `nvidia-smi` BEFORE any
   `import torch`.
5. **Code-competition rules** — submission notebook is CPU + no internet +
   ≤90 min runtime. The hidden test set populates `/kaggle/input/competitions/<comp>/test_*`
   only during the official scoring re-run.
6. **PowerShell silent CLI return ≠ failure** — the Kaggle CLI uses tqdm/rich
   that don't render on PowerShell. Verify with `kaggle competitions submissions`
   before retrying — duplicate submissions burn the 5/day quota.
7. **Pretrained foundation models are the highest-leverage win** — explicitly
   allowed by the rules. Perch (Google) handles 87% of our classes; EffNet
   trained from scratch handles the remaining 13%.
8. **Rank average for heterogeneous blends** — three models with different
   score distributions can't be averaged in raw-probability space.

---

## Acknowledgements

- Google Research + Cornell Lab of Ornithology for the Perch v2 model
- @rishikeshjani for the public Perch ONNX export
- @tuckerarrants for the distilled SED ONNX folds
- @mattiaangeli for the public ProtoSSM+SED reference implementation that
  taught us the blend recipe

---

## License

Code in this repository is MIT. Competition data and Kaggle-hosted models
remain under their respective licenses (CC BY-NC-SA 4.0 and Apache 2.0).
# KaggleBirdClef2026
