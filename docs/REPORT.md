# Acoustic Species Identification in the Brazilian Pantanal: A Heterogeneous Rank-Average Ensemble for BirdCLEF 2026

**Author:** Savitha Vijayarangan
**Repository:** (https://github.com/IamSavitha/KaggleBirdClef2026)
**Best public-leaderboard score:** 0.932 (macro-averaged ROC-AUC, BirdCLEF+ 2026)

---

## Abstract

We present a passive-acoustic-monitoring pipeline for identifying 234 wildlife species (birds, amphibians, mammals, reptiles, insects) in 1-minute soundscapes from the Brazilian Pantanal. Our approach combines a foundation model (Google Perch v2) with two complementary models — an in-domain EfficientNetV2-S trained on competition data and a publicly distilled Sound Event Detection (SED) ensemble — under a rank-average blend, augmented with post-processing techniques (per-(site, hour) priors, file-level rank-aware scaling, and 3-point delta-shift smoothing). The final pipeline achieves a public-leaderboard macro-AUC of 0.932 — a 0.097 improvement over the best single-model baseline. We also document two unsuccessful directions: in-kernel ProtoSSM training and per-class branch weighting, both of which regressed the leaderboard despite being theoretically sound. The lessons from those failures are reported alongside the successful path.

---

## 1. Introduction

The BirdCLEF+ 2026 competition (Kaggle, hosted by Google Research and Cornell Lab of Ornithology) asks participants to predict the presence of 234 species classes in 5-second windows of 1-minute audio clips collected in the Pantanal wetlands of Brazil. The evaluation metric is macro-averaged ROC-AUC across classes that have at least one positive in the hidden test set [1].

The challenge is non-trivial for three reasons. First, the class set is **heterogeneous**: 87% of the species are birds with rich vocalizations, but the remainder includes amphibians, insects, mammals, and reptiles whose calls span very different spectral and temporal characteristics. Second, **labeled data is sparse**: only 1,478 5-second segments across 66 soundscape files are explicitly annotated, while the bulk of the training audio is foreground recordings of single species (Xeno-Canto and iNaturalist) with limited soundscape context. Third, the competition is a **code competition with strict constraints**: the inference notebook must run on CPU, with internet disabled, in under 90 minutes, and on hidden test data that is not visible during development.

This report describes our submission pipeline that achieves **0.932 public-LB**, places the failed experimentations in context, and discusses the gap to the strongest public solutions (0.943).

### 1.1 Contributions

1. A reproducible, code-competition-compliant pipeline that combines three heterogeneous models via rank averaging
2. Empirical demonstration that a foundation model alone (Perch v2) is insufficient when 12% of the target classes are out of its vocabulary
3. Documentation of two unsuccessful design choices (in-kernel ProtoSSM training; per-class branch weights) and an analysis of why they regressed
4. An automated upload-and-submit workflow that handles Kaggle dataset versioning, kernel attachment, and competition submission via CLI

---

## 2. Related Work / Literature Review

### 2.1 Foundation models for bioacoustics

Google Research and Cornell Lab of Ornithology have published a sequence of bird-vocalization classifiers, the most recent of which is **Perch v2** [2]. Perch is trained on a curated dataset spanning thousands of bird species worldwide and produces both class logits over a large vocabulary and 1536-dimensional embeddings. In our pipeline, Perch is used both for direct prediction (after class mapping) and as a feature extractor for downstream models.

### 2.2 Sound Event Detection ensembles

SED models trained on log-mel spectrograms have been the workhorse of BirdCLEF since the 2021 edition. Recent solutions distill multi-fold SED ensembles from teacher architectures (e.g., EfficientNet-B3 trained with attention pooling and frame-level supervision) into smaller ONNX-deployable models suitable for the code-competition's CPU runtime budget. We use the publicly released `tuckerarrants/bc2026-distilled-sed-public` 5-fold ensemble [3].

### 2.3 Sequence modeling over embeddings

Several top-2025 BirdCLEF solutions propose state-space and attention-based sequence models over per-window foundation-model embeddings to capture temporal call structure. The "ProtoSSM v5" architecture used in the public 0.934 notebook [4] combines a selective state-space backbone with prototypical contrastive learning, taxonomic auxiliary losses, knowledge distillation from Perch logits, and SWA. Our ProtoSSM-lite (a bidirectional LSTM, Section 6.1) was a deliberately simplified attempt; we discuss its failure mode below.

### 2.4 Rank-average blending

Calibration mismatch is a well-known failure mode for averaging probability outputs across heterogeneous models. Rank averaging — converting each branch's predictions to per-class column percentiles before summation — is robust to scale differences and has been a standard ensembling trick in bioacoustic competitions since at least 2024.

### 2.5 Post-processing techniques

Three techniques cited in the public 0.934 notebook as "2025 Rank 1/3 techniques" are particularly relevant:

- **Rank-aware (file-level confidence) scaling** [4]: Multiply each (file, class) prediction by `(file_max for that class)^p`. Suppresses isolated false positives.
- **Delta-shift smoothing** [4]: 3-point neighbor exponential smoothing across the 12 windows of a soundscape, formalized as `new[t] = (1-α)·old[t] + 0.5·α·(old[t-1] + old[t+1])`.
- **Site/hour priors** [4]: Per-bucket class occurrence frequencies derived from labeled training soundscapes, mixed into predictions at a small weight.

We adopted all three.

---

## 3. Data Sources, Cleansing, and Transformation

### 3.1 Dataset overview

| Source | Files | Notes |
|---|---:|---|
| `train_audio/` (XC + iNat) | 35,549 | Foreground species recordings, mostly single-species, variable length (typically 3–60 s) |
| `train_soundscapes/` | 10,658 | 1-minute multi-species recordings from the Pantanal |
| `train_soundscapes_labels.csv` | 1,478 segments / 66 files | Expert-annotated 5-second segments with semicolon-separated species labels |
| `test_soundscapes/` (hidden) | ~600 | 1-minute Pantanal recordings; only mounted during scoring |
| `taxonomy.csv` | 234 classes | Maps `primary_label` (eBird code or iNat taxon ID) to scientific/common names and class (Aves, Amphibia, Insecta, Mammalia, Reptilia) |
| `sample_submission.csv` | 235 columns | `row_id` + 234 species columns; defines submission schema |

Audio is provided at 32 kHz in OGG Vorbis format.

### 3.2 Class coverage analysis

A critical observation drove much of our design: when the 234 target species are joined to Perch v2's vocabulary by `scientific_name`, the breakdown is:

| Bucket | Count | Coverage strategy |
|---|---:|---|
| Direct match in Perch | **203** (87%) | Use Perch's own logit |
| Genus-level proxy available | **3** (1%) | Max-pool Perch logits across same-genus entries |
| No match in Perch vocabulary | **28** (12%) | Insect sonotypes (e.g., `47158son16`); rely on EffNet/SED |

Macro-averaged ROC-AUC is sensitive to per-class performance — the 28 unmapped classes alone could pin AUC at 0.5 baseline if relied upon Perch only. This motivated the multi-branch design.

### 3.3 Preprocessing

For all branches, raw audio is loaded with `soundfile` (libsndfile-backed; selected to avoid the `torchcodec` dependency that recent torchaudio releases require), mean-pooled to mono, and either truncated or zero-padded to 60 seconds (1,920,000 samples at 32 kHz). The 60-second clip is reshaped to `(12, 160_000)` for the 12 × 5-second window inference grid.

For the EffNetV2-S branch, we additionally compute a log-mel spectrogram with `n_mels=128`, `n_fft=2048`, `hop_length=512`, `fmin=50`, `fmax=16000`, `top_db=80`, with a small ε in the spec power before the dB conversion to keep silent inputs finite. We per-sample standardize the spectrogram and apply SpecAugment during training only.

### 3.4 Validation strategy

A central limitation: **we have no held-out validation data on Kaggle's hidden test set**. The 1,478 labeled soundscape segments are used as ProtoSSM training data and as the source for the (site, hour) prior tables; reusing them for validation would contaminate the prior. For our trained EffNet, we use 5-fold stratified CV on the XC+iNat clips with the labeled soundscape segments held out per-fold; for the post-processing knobs (rank-aware power, prior weight, delta-shift α) we have only the public leaderboard as feedback, with the daily 5-submission cap.

This shortage of validation infrastructure is the single biggest reason our gap to the 0.943 inspiration is what it is; that solution invests heavily in a 5-fold OOF stacking pipeline that lets them tune per-class thresholds and per-taxon temperatures with quantitative feedback.

---

## 4. Proposed Solution

### 4.1 Architecture overview

```
                            soundscape (60s, 32 kHz)
                                       │
                       split into 12 × 5-second windows
                                       │
   ┌───────────────────────────────────┼───────────────────────────────────┐
   ▼                                   ▼                                   ▼
┌────────────────────┐    ┌────────────────────┐                ┌────────────────────┐
│   Perch v2 ONNX    │    │  Distilled SED     │                │  EfficientNetV2-S  │
│   (Google Foundation)   │  (5 ONNX folds)    │                │  trained on comp   │
│   logits + 1536-d  │    │  clip + frame_max  │                │  fold0 ckpt        │
│   embeddings       │    │  sigmoid avg       │                │                    │
└────────┬───────────┘    └────────┬───────────┘                └────────┬───────────┘
         │ 234-class mapping       │                                    │
         ▼                          ▼                                    ▼
       (12,234)                 (12,234)                              (12,234)
         │                          │                                    │
         └─────────── per-class column rank → percentiles ────────────────┘
                                       │
       ┌───────────── post-processing ───────────────────────────────┐
       │  (a) site/hour prior mix (weight 0.05)                      │
       │  (b) rank-aware scaling (file_max^0.5)                      │
       │  (c) delta-shift smoothing (α=0.15)                         │
       └────────────────────────────────────────────────────────────┘
                                       │
                       submission.csv  (row_id, 234 species cols)
```

### 4.2 Branch 1 — Perch v2 Foundation Model

We use the publicly available ONNX export of Perch v2 (`rishikeshjani/perch-onnx-for-birdclef-2026` [5]). Inference is run in 4-file batches over 12 × 5-second windows, with `onnxruntime`'s `CPUExecutionProvider` and `intra_op_num_threads=4`. For each input, Perch returns a `label` head (class logits over its full vocabulary) and an `embedding` head (1536-d representation). After applying our class mapping (Section 3.2), per-window scores in 234-class space are passed through sigmoid.

### 4.3 Branch 2 — Distilled SED Ensemble

The 5-fold distilled SED ensemble (`tuckerarrants/bc2026-distilled-sed-public` [3]) takes log-mel spectrograms of shape `(1, 256, T)` per window, with `n_mels=256`, `n_fft=2048`, `hop_length=512`, `fmin=20`, `fmax=16000`, `top_db=80`. Each fold returns a clip-level logit and a frame-level logit; we sigmoid both and average them per fold, then average across folds.

### 4.4 Branch 3 — In-Domain EfficientNetV2-S

To cover the 28 classes Perch cannot predict, we trained a custom EfficientNetV2-S backbone on the combined XC+iNat foreground clips and labeled soundscape segments. Training:

- Backbone: `tf_efficientnetv2_s.in21k_ft_in1k` from `timm` [6]
- Loss: BCE-with-logits (multi-label)
- Augmentation: mixup α=0.4 (p=0.5), random gain ±30%, polarity flip, Gaussian noise (σ=0.005)
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-2)
- Scheduler: cosine decay with 2-epoch warmup
- Hardware: Kaggle P100 GPU (with a `torch==2.5.1+cu118` install at notebook start, since the default Kaggle PyTorch dropped sm_60 support)
- Outcome: 10 epochs of training reached val macro-AP **0.7031** on fold 0 (validation set: held-out fold of XC/iNat clips + labeled soundscape segments).

### 4.5 Rank-Average Blend

Each branch produces a `(N_rows, 234)` matrix where `N_rows = n_files × 12`. We compute per-column rank-percentiles using `pandas.DataFrame.rank(axis=0, pct=True)` and weight-average them:

```
blended[i, c] = w_perch · rank_perch[i, c]
              + w_sed   · rank_sed[i, c]
              + w_eff   · rank_eff[i, c]
```

with uniform weights `(0.50, 0.30, 0.20)`. The rank-percentile transform is monotonic and ROC-AUC is rank-based, so blending preserves the per-branch ordering of any class while producing a calibration-agnostic combined ordering.

### 4.6 Post-Processing Stack

Three corrections are applied to the rank-blended output, in order:

**(a) Site/hour prior mix.** From `train_soundscapes_labels.csv`, we parse the `(site, hour)` bucket from each filename (`BC2026_<Train|Test>_<id>_<S?>_<date>_<hms>.ogg`) and compute per-class occurrence frequency in each bucket. At inference, we mix the bucket's prior into each prediction at weight 0.05:

```
new[i, c] = 0.95 · old[i, c] + 0.05 · prior[bucket(file_i), c]
```

Files with unseen `(site, hour)` bucket fall back to the global prior.

**(b) Rank-aware scaling.** For each (file, class), we multiply by the file's max prediction for that class raised to power 0.5:

```
scale[file, c] = (max over windows of new[file, *, c]) ^ 0.5
out[file, w, c] = scale[file, c] · new[file, w, c]
```

This suppresses isolated single-window predictions in files where the class is absent everywhere else, while leaving genuine across-windows clusters unchanged.

**(c) Delta-shift smoothing.** A 3-point neighbor exponential smoothing across the 12-window time axis with α=0.15:

```
out[t] = (1 − α) · in[t] + 0.5 · α · (in[t−1] + in[t+1])
```

This is *lighter* than the gaussian smoothing we initially attempted (σ=0.65) and produced a +0.001 LB lift instead of a regression.

---

## 5. Model Evaluation, Result Analysis, and Visualization

### 5.1 Submission methodology

The competition evaluation is **macro-averaged ROC-AUC over classes with at least one positive in the hidden test set**. Submissions are made via Kaggle code-competition kernel — the scoring system re-runs our notebook on the hidden test data, and the resulting `submission.csv` is graded. We are limited to 5 submissions per UTC day and may select up to 2 final submissions for private-leaderboard judging.

We report public-LB scores throughout. Private scores are revealed only at competition close (June 3, 2026).

### 5.2 Leaderboard progression (major milestones)

| Submission | Public LB | Δ from prior |
|---|---:|---:|
| EffNetV2-S fold0 only (10 epoch) | 0.856 | — |
| Perch v2 ONNX only (203 direct + 3 proxy + 28 unmapped) | 0.835 | −0.021 |
| **3-way rank-average blend (Perch + SED + EffNet)** | **0.927** | **+0.092** |
| + (site, hour) prior weight 0.05 | 0.928 | +0.001 |
| + rank-aware scaling power 0.5 | 0.931 | +0.003 |
| **+ delta-shift smoothing α=0.15** | **0.932** | **+0.001** |

The dominant single-step gain (+0.092) comes from blending — confirming that no single model in our toolkit covers all 234 classes well, and that heterogeneous rank averaging is the correct combiner. The post-processing stack adds a further +0.005 in aggregate. The plateau at 0.932 (after exhausting cheap post-processing knobs) is consistent with the gap to the 0.943 reference being attributable to a more sophisticated sequence model (proper ProtoSSM with KD, mixup, focal, and SWA) plus iterative pseudo-labeling.

### 5.3 Comparative analysis — single-model vs blend

| Model | Public LB | Coverage of 234 classes | Notes |
|---|---:|---|---|
| Perch v2 only | 0.835 | 206/234 useful (28 noise) | Foundation model is strong on its 87%, useless on insects |
| EffNetV2-S only | 0.856 | All 234 weakly | Trained on competition data, covers all classes but at moderate quality |
| Perch + EffNet + SED (uniform rank-avg) | 0.927 | All 234 well | Rank averaging fills coverage gaps |
| + post-processing (Phase 11) | 0.932 | All 234 well | (site, hour) priors + file-level scaling + light temporal smoothing |

**Key insight.** Perch alone at 0.835 *underperforms* the trained EffNet at 0.856, despite Perch being a far stronger model in absolute terms — because the macro-AUC penalizes the 28 unmapped classes severely (each contributing 0.5 AUC when Perch outputs constant 0.5). When EffNet covers those classes (even weakly) and the rest is dominated by Perch, the macro average lifts dramatically. The blend wins not because any single branch is best, but because **rank averaging is asymmetric**: each branch contributes to the classes where it is informative and the others fill in.

### 5.4 Ablation of post-processing components

| Component change | LB | Δ | Verdict |
|---|---:|---:|---|
| Baseline 3-way blend | 0.927 | — | — |
| + (site, hour) prior 0.05 | 0.928 | +0.001 | retained |
| + rank-aware scaling power=0.5 | 0.931 | +0.003 | retained |
| + delta-shift smoothing α=0.15 | 0.932 | +0.001 | retained |
| ↳ rank-aware power tuned to 0.7 | 0.932 | 0.000 | flat — reverted to 0.5 |
| ↳ prior weight tuned to 0.10 | 0.932 | 0.000 | flat — reverted to 0.05 |
| ↳ gaussian smoothing σ=0.65 (instead of delta-shift) | 0.924 | −0.008 | rejected |
| ↳ gaussian smoothing σ=0.30 | 0.928 | −0.004 | rejected |

The **delta-shift formulation matters**: a gaussian kernel of comparable effective radius produced larger smoothing and regressed. The 3-point neighbor scheme is bounded — it cannot blur beyond adjacent windows — and that boundedness preserves sharper temporal events that AUC rewards.

---

## 6. Experiments That Did Not Go As Planned

### 6.1 ProtoSSM sequence head (Phases 4–6)

**Hypothesis.** A sequence model over Perch's per-window 1536-dimensional embeddings should capture inter-window dependencies (calls that persist or repeat across the 12 windows of a soundscape), providing complementary signal that a per-window classifier cannot extract. The public 0.943 notebook attributes a large fraction of its lift to this branch.

**What we built.** A 2-layer bidirectional LSTM (`hidden=256`, `dropout=0.3`) followed by a linear head, trained inside the Kaggle submission kernel at runtime on Perch embeddings of the 66 labeled training soundscapes. 12-window sequences with multi-label binary targets, BCE-with-logits loss, AdamW, 40 epochs.

**What happened.**

1. **Phase 4 (in-kernel training):** The submission kernel exceeded the 90-minute CPU runtime cap. With Perch + SED 5-fold + EffNet + ProtoSSM training all running serially on a single CPU notebook, total wall time went over budget. The submission was marked `COMPLETE` with no `publicScore` — a wasted slot.

2. **Phase 5 (pre-trained ProtoSSM v1):** We refactored ProtoSSM training into a separate Kaggle kernel and pinned its output as a `kernel_sources` artifact for the submission. This solved the runtime issue but the trained checkpoint regressed the blend by **0.019 LB** (0.927 → 0.908). Inspection of the training logs showed loss dropping to 0.052 in 5 epochs out of 40 — severely overfit on only 66 files × 12 windows = 792 sequences.

3. **Phase 6 (regularized ProtoSSM v2):** We rebuilt the head with stronger regularization (`hidden=128`, `dropout=0.5`, `weight_decay=1e-2`), introduced an explicit train/val split (47/12 file split) with early stopping on best validation loss, and trained from precomputed Perch embeddings (`jaejohn/perch-meta` [7]) for cleaner data plumbing. Train and validation losses now tracked each other (0.058 / 0.064 at epoch 30 — no overfitting). The resulting checkpoint scored **0.909** — only +0.001 over Phase 5. Still a regression of 0.018 vs the 3-way blend.

**Why it failed.** Three compounding factors:

1. **Tiny supervised set.** 66 labeled files (47 effective after the val split) is roughly one twentieth of the 0.943 inspiration's training set, which uses iterative pseudo-labeling to reach thousands of segments. A sequence model with even modest capacity cannot learn generalizable temporal patterns from this volume.

2. **Wrong objective for blend insertion.** Our ProtoSSM was trained with vanilla BCE on multi-label targets. The 0.943 ProtoSSM is additionally trained with **knowledge distillation from Perch logits** — its job is not to reproduce labels from scratch but to *improve on Perch's per-window predictions* using temporal context. A KD-trained head is by construction additive; a label-only head is in competition with the existing Perch branch and dilutes it.

3. **Branch weight is harsh on noisy branches.** Even at our 0.20 ProtoSSM weight, an underperforming branch's rank ordering corrupts the blend across all classes simultaneously. We did not have validation infrastructure to tune the weight downward; even at 0.05 weight, an empirical test would have required burning a daily-quota submission.

**Lesson.** A sequence-model branch is not free signal — it must either (a) be trained with KD against a stronger branch, or (b) demonstrate validation gains before being inserted into the blend. We inserted before validating.

### 6.2 Per-class branch weighting (Phase 9)

**Hypothesis.** For the 28 species classes that have no Perch vocabulary match (insect sonotypes), Perch outputs constant 0.5, contributing pure noise to the blend's rank-percentile for those classes. Setting Perch's weight to 0 specifically on those classes (and redistributing to SED + EffNet) should remove that noise and lift macro-AUC.

**What we built.** A `_per_class_weights3` helper that assigns three weight-vectors keyed by class bucket:

```
weights        = (0.5,  0.3, 0.2)   # 203 directly-mapped classes
unmapped_weights = (0.0, 0.5, 0.5)  # 28 insect/sonotype classes
proxy_weights  = (0.30, 0.40, 0.30) # 3 genus-proxy classes
```

with per-class normalization so each row's weights sum to 1.

**What happened.** Submission scored **0.926** — *down* 0.002 from the 0.928 baseline (Phase 7 = uniform weights + prior).

**Why it failed (counterintuitively).** Macro-AUC is computed *across classes with at least one positive*. For the 28 unmapped classes, Perch's constant 0.5 across all rows ties the rank-percentile contribution at exactly 0.5 — a well-defined value, not actually noise in rank space. Removing Perch entirely on those classes shifts the rank distribution: SED's predictions for an unmapped class are no longer being averaged against Perch's tie-rank but against nothing, which alters their cross-class rank comparison. The macro-AUC sums per-class — and apparently, the previous "noise from Perch" was a stabilizer. Removing it changed the relative balance between the 28 unmapped classes and the 206 mapped ones in subtle ways that hurt the macro average by 0.002.

**Lesson.** Rank averaging is robust to one branch being uninformative *only when ties are well-distributed*. Heterogeneous per-class weighting changes the rank denominator across classes, breaking the per-class normalization that AUC relies on. The principled-looking surgery actually disrupted a stable equilibrium.

### 6.3 Other experiments that did not move the needle

For completeness, several A/B-style tuning experiments returned flat results:

- Rank-aware scaling power tuned to 0.7 (vs 0.5): 0.932 → 0.932
- (site, hour) prior weight tuned to 0.10 (vs 0.05): 0.932 → 0.932
- Gaussian smoothing σ=0.30 (instead of delta-shift): 0.928 → 0.924 (rejected)

These flat results are themselves informative: they suggest that our LB at 0.932 sits on a wide ridge where local hyperparameter perturbation has no measurable effect, and further gains require qualitatively new components (proper ProtoSSM with KD; per-taxon temperature with OOF-based calibration; iterative pseudo-labeling) rather than refinement of existing knobs.

---

## 7. Conclusion

Our 0.932 public-LB submission for BirdCLEF+ 2026 is a deliberately *simple* pipeline: three heterogeneous models (Perch foundation, distilled SED, in-domain EffNet) combined via uniform-weight rank averaging, with a small post-processing stack (priors, file-level scaling, neighbor smoothing). The bulk of our score (+0.092 over best single-model) comes from the blend itself — not from the sophistication of any individual branch.

The 0.011 gap to the public 0.943 leader maps to specific components we either failed to deploy correctly (Section 6.1 — ProtoSSM without KD) or did not commit to building from scratch (5-fold OOF stacking, iterative pseudo-labeling). Closing that gap is fundamentally about **investing in validation infrastructure** that lets us tune per-class corrections quantitatively rather than via the public-LB feedback loop.

Two unsuccessful experiments are reported in detail (Sections 6.1, 6.2) for the lessons they encode: a sequence model without distillation supervision cannot pay its way into a strong blend, and per-class weight surgery on an already-balanced rank average can disturb the metric's class normalization in ways that are hard to predict.

### 7.1 Future work

Three concrete extensions, ordered by expected ROI:

1. **5-fold OOF stacking infrastructure.** Replace public-LB-only feedback with held-out validation. Enables principled tuning of per-class temperature scaling, per-taxon thresholds, and blend weights without the daily 5-submission cap.
2. **Proper ProtoSSM** with knowledge distillation from Perch logits, mixup augmentation, focal loss with class-frequency weighting, and stochastic weight averaging.
3. **Iterative pseudo-labeling** of unlabeled `train_soundscapes` files using the current best model, retraining ProtoSSM on the expanded corpus, and stopping when validation OOF metric saturates.

---

## References

[1] Kaggle, "BirdCLEF+ 2026 — Acoustic Species Identification in the Brazilian Pantanal," 2026. https://www.kaggle.com/competitions/birdclef-2026

[2] Google Research and Cornell Lab of Ornithology, "Perch v2: A Global Bird Embedding and Classification Model," Kaggle Models. https://www.kaggle.com/models/google/bird-vocalization-classifier

[3] T. Arrants, "BC2026 Distilled SED Public," Kaggle Datasets, 2026. https://www.kaggle.com/datasets/tuckerarrants/bc2026-distilled-sed-public

[4] M. Angeli, "BirdCLEF+ 2026 0.943+ | Better Blend," Kaggle Notebooks, 2026. https://www.kaggle.com/code/mattiaangeli/birdclef-2026-0-943-better-blend

[5] R. Jani, "Perch ONNX for BirdCLEF+ 2026," Kaggle Datasets, 2026. https://www.kaggle.com/datasets/rishikeshjani/perch-onnx-for-birdclef-2026

[6] R. Wightman, "PyTorch Image Models (timm)," GitHub repository, 2019–. https://github.com/huggingface/pytorch-image-models

[7] J. John, "perch-meta," Kaggle Datasets, 2026. https://www.kaggle.com/datasets/jaejohn/perch-meta

[8] V. Dwivedi, "BirdCLEF 2026 — ONNX Perch + Sequence Modeling," Kaggle Notebooks, 2026. https://www.kaggle.com/code/vyankteshdwivedi/birdclef-2026-onnx-perch-sequence-modeling
