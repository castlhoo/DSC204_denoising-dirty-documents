# Denoising Dirty Documents

A deep-learning pipeline that removes noise — coffee stains, discoloration, uneven lighting,
wrinkles, fold shadows, and other scanning artifacts — from scanned document images, restoring
clean pages suitable for OCR (Optical Character Recognition). Built as an **MLE-bench** challenge
on [AI Coding Gym](https://aicodinggym.com), based on the Kaggle *Denoising Dirty Documents* competition.

| Item | Value |
|---|---|
| **Final score** | **0.01514 RMSE** |
| **Rank** | **5 / 162 (top 3.1%) — Gold tier** |
| AI baseline | 0.01649 RMSE |
| Improvement | **+8.2% relative** over the AI baseline |
| Model | ImprovedUNet (base channels 32, ~1.95M parameters) |
| Best epoch | 67 / 80 (val RMSE 0.01457) |

> A blow-by-blow analysis of every experiment and design decision lives in
> [REPORT_README.md](REPORT_README.md). **This document** is the practical entry point:
> what the problem is, how the solution works, and how to run and reproduce it.

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Why the Baseline Failed](#2-why-the-baseline-failed)
3. [Solution Overview](#3-solution-overview)
4. [How It Works (Deep Dive)](#4-how-it-works-deep-dive)
5. [Directory Structure](#5-directory-structure)
6. [Core Modules](#6-core-modules)
7. [Setup](#7-setup)
8. [Reproducing the Final Result](#8-reproducing-the-final-result)
9. [Hyperparameters](#9-hyperparameters)
10. [Experiment Log & Results](#10-experiment-log--results)
11. [Lessons Learned](#11-lessons-learned)
12. [References](#12-references)

---

## 1. Problem Overview

OCR turns static scans into searchable, editable text — but real-world documents are messy.
Coffee rings, sun-faded spots, dog-eared corners, and wrinkles all interfere with recognition.
The task is to take a noisy scan and predict the clean intensity of every pixel.

- **Input:** Noisy grayscale document images (PNG)
- **Output:** A CSV giving the restored intensity of each pixel (`0.0` = black text, `1.0` = white paper)
- **Metric:** **Root Mean Squared Error (RMSE)** between predicted and ground-truth pixel intensities — **lower is better**
- **Data:** 115 training pairs (a noisy image + its clean counterpart) and 29 test images (noisy only)
- **Leaderboard context:** AI baseline = 0.01649, gold-tier top = 0.01021

The submission is "melted" so each row is a single pixel, identified by `image_row_col`:

```
id,value
1_1_1,1.000000      # image 1, row 1, col 1  → white paper
1_2_1,0.984314
1_3_1,0.015686      # a near-black pixel → part of a letter
...
```

The key domain insight that drives the whole solution: **the dominant form of "noise" here is not
random pixel speckle — it is a smoothly varying, unevenly bright background.** A stain darkens a
region; yellowing tints the whole page; a fold casts a soft shadow. The text itself is nearly binary
(very dark strokes on light paper). So the real job is *separating a clean bimodal signal from a
slowly-varying background field* — and that framing motivates every design choice below.

---

## 2. Why the Baseline Failed

The original AI agent built a `ResidualUNet` and scored only 0.0969 RMSE — far behind the leaderboard.
Inspection revealed five compounding problems:

| Issue | What happened | Consequence |
|---|---|---|
| **Premature early stopping** | `patience=4` with a flat early-loss curve | Training halted at **epoch 2** — the model was essentially untrained |
| **Too little capacity** | `base_channels=16` (~488K params) | Cannot represent stroke shape + stain gradients simultaneously |
| **No domain preprocessing** | Plain `/255` normalization | A stained page (background ~0.7) and a clean page (~0.95) look identical to the model |
| **Residual cap** | Output limited to `0.1·tanh(x)`, i.e. ±0.1 | A background pixel sitting at 0.6 can never be pushed to the correct 1.0 |
| **Low data utilization** | Only 2 patches/image (~230 patches/epoch) | Far too little diversity for a CNN |

The takeaway: the baseline applied a *generic* ML recipe without reasoning about what document
noise actually is. We fixed the root causes first, then layered on improvements.

---

## 3. Solution Overview

`train_sean.py` was written from scratch to address all five weaknesses with principled changes:

| Area | Baseline | This solution | Rationale |
|---|---|---|---|
| **Architecture** | ResidualUNet, 3 levels, 16 ch | 4-level **ImprovedUNet** + BatchNorm, 32 ch | Larger receptive field captures global background structure; BatchNorm stabilizes training on a 115-image set |
| **Preprocessing** | `/255` | **Otsu background normalization** (from scratch) | Normalizes every page so background ≈ 1.0, regardless of staining |
| **Output** | `0.1·tanh` residual cap | **Direct prediction**, clamped to [0,1] | Allows full-range corrections |
| **Training length** | 25 epochs, patience 4 | 80 epochs, **patience 15** | Lets the model train through early plateaus |
| **LR schedule** | ReduceLROnPlateau | **CosineAnnealingLR** (1e-3 → 1e-6) | Smooth decay, no scheduler stagnation |
| **Augmentation** | flip, rot90 | flip, rot90, **brightness jitter (±10%)** | Simulates lighting variation |
| **Data per epoch** | 2 patches/image | **8–10 patches/image** | 4–5× more diversity |
| **Inference** | single forward pass | **Test-time augmentation (TTA)** | Averages 4 flipped views for free accuracy |

Two ideas were tried and **rejected on evidence** (gamma post-processing and a larger-model
ensemble) — see [Section 10](#10-experiment-log--results).

---

## 4. How It Works (Deep Dive)

### 4.1 Otsu Background Normalization — the most important step

Standard `/255` scaling treats a coffee-stained page and a pristine page identically, forcing the
network to *simultaneously* answer "what color is this page's background?" and "where is the text?".
We answer the first question with classical image processing so the network can focus on the second.

For each image, `otsu_background_normalize()`:

1. Builds the 256-bin intensity histogram.
2. Finds the threshold `t` that **maximizes between-class variance** (Otsu's 1979 criterion) — the
   value that best splits dark "text" pixels from bright "background" pixels.
3. Computes the **mean brightness of the background pixels** (those above `t`).
4. **Divides the whole image by that background mean** and clips to [0, 1].

After this, the paper is ≈ 1.0 (white) on every image — stained, yellowed, or clean — and text is
proportionally darker. The network always sees a consistent input distribution. The Otsu threshold
is implemented directly (no OpenCV/scikit-image dependency) in a single histogram pass.

### 4.2 ImprovedUNet architecture

A 4-level encoder–decoder U-Net. Each `conv_block` is `Conv 3×3 → BatchNorm → ReLU → Conv 3×3 →
BatchNorm → ReLU`. Skip connections concatenate encoder features into the matching decoder level so
fine text edges survive the down/up-sampling.

```
Input (1×H×W, grayscale)
 ├─ Enc1: conv_block(1→32)          ─┐ skip
 │   ↓ MaxPool 2×2                   │
 ├─ Enc2: conv_block(32→64)         ─┼┐ skip
 │   ↓ MaxPool 2×2                   ││
 ├─ Enc3: conv_block(64→128)        ─┼┼┐ skip
 │   ↓ MaxPool 2×2                   │││
 ├─ Bottleneck: conv_block(128→256)  │││   ← sees an 8× downsampled view (global context)
 │   ↑ Upsample ──── cat(Enc3) ──────┘││
 ├─ Dec3: conv_block(256+128→128)     ││
 │   ↑ Upsample ──── cat(Enc2) ───────┘│
 ├─ Dec2: conv_block(128+64→64)        │
 │   ↑ Upsample ──── cat(Enc1) ────────┘
 ├─ Dec1: conv_block(64+32→32)
 └─ Out: Conv 1×1 (32→1) → clamp(0,1)
```

- **Why 4 levels?** Each downsampling doubles the receptive field. The bottleneck "sees" an 8×
  downsampled page, so it can model large-scale background gradients (stains, shadows) while skip
  connections preserve pixel-accurate text edges.
- **Why BatchNorm everywhere?** With only 115 images, deep nets are prone to unstable gradients.
  BatchNorm keeps activation distributions in check and lets training converge faster and more reliably.
- **Capacity:** ~1.95M parameters at `base_channels=32` — 4× the baseline, still fast enough on CPU.

### 4.3 Patch-based training

Full pages are large and vary in size, so training samples random crops. `RandomPatchDataset`
draws `patches_per_image` random patches per page each epoch, then augments each with random
horizontal/vertical flips, a random 90° rotation, and an optional ±10% brightness jitter applied to
the *noisy input only* (the target stays fixed). This multiplies effective data volume and teaches
invariance to orientation and lighting.

### 4.4 Test-time augmentation (TTA)

At inference, `predict_single()` runs the model on four views of each test image — original,
horizontal flip, vertical flip, and both — un-flips each prediction back to the original frame, and
**averages the four**. Because a CNN responds slightly differently to each orientation, averaging
cancels random errors. This cut validation RMSE from ~0.0175 to ~0.0146 (~17%) at zero training cost.

> Implementation note: NumPy's `fliplr`/`flipud` return negative-stride views that PyTorch rejects,
> so the code calls `.copy()` after each flip to produce contiguous arrays.

### 4.5 Inference & submission

Before the forward pass, images are reflect-padded so H and W are multiples of 8 (the network needs
sizes divisible by its three 2× poolings); the prediction is then cropped back to the original size.
`write_predictions_csv()` walks pixels in row-major order and emits `image_row_col,value` rows,
matching `sampleSubmission.csv` exactly.

---

## 5. Directory Structure

```
denoising-dirty-documents/
├── README.md                     # (this file) project overview & run guide
├── REPORT_README.md              # detailed analysis report (experiment timeline, rationale)
│
├── train_sean.py                 # ⭐ main training/inference pipeline
├── postprocess_predict.py        # gamma post-processing tuning experiment (no improvement)
├── ensemble_predict.py           # multi-model ensemble inference (no improvement)
├── colab_train_ensemble.ipynb    # Colab GPU ensemble training notebook
│
├── unet_sean_best.pth            # ⭐ best model weights (epoch 67, val RMSE=0.01457)
│
├── predictions_sean.csv          # ⭐ final submission (test RMSE 0.01514)
├── predictions_postprocessed.csv # gamma post-processing result
├── predictions_ensemble.csv      # ensemble result (RMSE 0.01539, worse)
│
├── data/
│   ├── description.md            # competition description
│   ├── sampleSubmission.csv      # submission format + pixel ordering definition
│   ├── train/                    # 115 noisy training images
│   ├── train_cleaned/            # 115 clean (ground-truth) training images
│   └── test/                     # 29 test images
│
├── .log/                         # session logs (full decision history)
├── AGENTS.md                     # AI Coding Gym challenge instructions
└── CLAUDE.md / GEMINI.md         # per-agent settings
```

⭐ = core files needed to reproduce the final result.

> **Note on large files:** the three `predictions_*.csv` files (~122 MB each) and the `data/` folder
> are **not** tracked in git — they exceed GitHub's 100 MB limit and/or are re-downloadable via the
> `aicodinggym` CLI. Only source, docs, and the 7.8 MB model checkpoint are versioned.

---

## 6. Core Modules

### `train_sean.py` — main pipeline
The single source of truth for the model and data handling; the other scripts import its functions.

| Function / class | Role |
|---|---|
| `ImprovedUNet` | 4-level U-Net with BatchNorm (see §4.2) |
| `otsu_background_normalize()` | From-scratch Otsu thresholding + background division (see §4.1) |
| `load_pairs()` | Loads noisy/clean PNGs, applies normalization, returns paired arrays |
| `RandomPatchDataset` | Random patch sampling + augmentation (flip, rot90, brightness jitter) |
| `pad_to_multiple()` / `unpad_to_original()` | Reflect-pad to a multiple of 8, then crop back |
| `predict_single()` | Single-image inference with 4-way TTA averaging |
| `evaluate_images()` | Full-image validation RMSE |
| `write_predictions_csv()` | Emits the melted `id,value` submission CSV |
| `main()` | Training loop: CosineAnnealingLR, early stopping, best-checkpoint saving |

### `postprocess_predict.py` — gamma post-processing (experiment)
Tests whether `output^gamma` (γ>1 pushes background→1, text→0) reduces RMSE. A search over
γ ∈ [0.7, 2.0] on the validation set found **γ=1.0 (no change) optimal** — evidence the model is
already well-calibrated rather than hedging toward gray.

### `ensemble_predict.py` — ensemble inference (experiment)
Averages predictions from several checkpoints trained with different seeds. An ensemble of three
`base_channels=48` models scored 0.01539 — **worse** than the single 32-channel model (0.01514),
consistent with the larger models overfitting on only 115 images.

### `colab_train_ensemble.ipynb` — GPU training notebook
End-to-end notebook for a Colab T4 GPU: extract data, train seeds 42/123/456, then build the ensemble.

---

## 7. Setup

```bash
pip install torch torchvision numpy pillow
```

The code auto-detects CUDA and falls back to CPU. To fetch the dataset:

```bash
aicodinggym mle download denoising-dirty-documents
```

This populates `data/` with `train/`, `train_cleaned/`, `test/`, and `sampleSubmission.csv`.

---

## 8. Reproducing the Final Result

```bash
# 1) Train (CPU: ~3 hours / GPU: ~20 minutes)
python train_sean.py \
    --data-dir data \
    --epochs 80 \
    --batch-size 16 \
    --patches-per-image 8 \
    --patch-size 64 \
    --base-channels 32 \
    --patience 15 \
    --seed 42 \
    --model-path unet_sean_best.pth \
    --output-csv predictions_sean.csv

# 2) Submit
aicodinggym mle submit denoising-dirty-documents -F predictions_sean.csv
```

Training prints per-epoch train/val RMSE and saves the best checkpoint to `--model-path`. After
training it reloads the best weights, reports final train/val RMSE, and writes the test predictions.

> To skip training and run inference with the bundled `unet_sean_best.pth`, use
> `postprocess_predict.py` (with γ=1.0 it applies no post-processing and behaves as plain inference).

Expected result: **RMSE ≈ 0.015, gold tier, top ~5%**.

> **Defaults vs. reproduction command:** the `argparse` defaults in `train_sean.py` are
> `patches-per-image=10` and `patch-size=128`, whereas the command above uses `8` / `64` to match the
> reproduction report. Both reach gold-tier scores; the report settings train faster on CPU.

---

## 9. Hyperparameters

All configurable via `train_sean.py` CLI flags:

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | (required) | Folder with `train/`, `train_cleaned/`, `test/`, `sampleSubmission.csv` |
| `--epochs` | 80 | Maximum training epochs |
| `--batch-size` | 16 | Batch size |
| `--lr` | 1e-3 | Initial learning rate (cosine-annealed to 1e-6) |
| `--patch-size` | 128 | Training patch size (px) |
| `--patches-per-image` | 10 | Random patches sampled per image per epoch |
| `--val-split` | 0.1 | Fraction of images held out for validation |
| `--patience` | 15 | Early-stopping patience (epochs without val improvement) |
| `--base-channels` | 32 | U-Net base channel width (model capacity) |
| `--seed` | 42 | Random seed (Python/NumPy/PyTorch) |
| `--model-path` | `unet_sean_best.pth` | Where to save/load the best checkpoint |
| `--output-csv` | `predictions_sean.csv` | Submission output path |

---

## 10. Experiment Log & Results

| Run | Description | Val RMSE | Test RMSE | Outcome |
|---|---|---|---|---|
| Baseline (AI) | ResidualUNet, base 16, stopped @ epoch 2 | 0.0969 | 0.0969 | reference |
| Run 1 | ImprovedUNet, base 32, Otsu, 16 ep (interrupted) | 0.0204 | — | promising |
| Run 2 | Same, 40 epochs complete | 0.01734 | — | better |
| **Run 3** | **+ TTA + CosineAnnealingLR, 80 epochs** | **0.01457** | **0.01514** | **Gold, 5/162 ✅** |
| Run 4 | Gamma post-processing on Run 3 | 0.01457 (γ=1.0) | — | no change → rejected |
| Run 5 | Ensemble of 3× base-48 models (Colab GPU) | — | 0.01539 | worse → rejected |

**Why gamma failed:** if the model were systematically hedging toward 0.5, a γ>1 transform would
help — the fact that γ=1.0 was optimal is actually *good news*: the predictions are already well-calibrated.

**Why the ensemble failed:** larger (base-48, ~4.3M-param) models overfit the 115-image set, and three
seeds didn't provide enough complementary diversity to outweigh that. The single well-regularized
32-channel model with TTA was already at the sweet spot.

---

## 11. Lessons Learned

- **Domain analysis beats generic recipes.** Recognizing that the noise is *background variation*,
  not random speckle, led directly to Otsu normalization — the single biggest win.
- **Fix root causes before adding features.** The baseline's real bug was premature early stopping;
  no amount of architecture tweaking helps a model that stops at epoch 2.
- **Skip connections are essential for pixel-accurate restoration** — they carry high-resolution
  edge information past the bottleneck.
- **TTA is a free lunch** for inference variance reduction.
- **More capacity is not always better.** On 115 images, base-48 overfit where base-32 generalized.
- **Negative results are results.** Measuring gamma and ensembling — and *trusting* the numbers
  enough to drop them — kept the final pipeline lean.

---

## 12. References

- Detailed experiment analysis & retrospective: [REPORT_README.md](REPORT_README.md)
- Original competition description: [data/description.md](data/description.md)
- Session logs: [.log/](.log/)
- Challenge instructions: [AGENTS.md](AGENTS.md)
- Otsu, N. (1979). *A threshold selection method from gray-level histograms.* IEEE TSMC.
```
