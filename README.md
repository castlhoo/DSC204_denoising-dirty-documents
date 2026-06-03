# Denoising Dirty Documents

A project that removes noise (coffee stains, discoloration, uneven lighting, wrinkles, etc.)
from scanned document images to restore clean images suitable for OCR.
Built as an **MLE-bench** challenge on [AI Coding Gym](https://aicodinggym.com)
(the Kaggle *Denoising Dirty Documents* competition).

| Item | Value |
|---|---|
| **Final score** | **0.01514 RMSE** |
| **Rank** | **5 / 162 (top 3.1%) — Gold tier** |
| AI baseline | 0.01649 RMSE |
| Improvement | **+8.2% relative** over the AI baseline |
| Model | ImprovedUNet (base channels 32, ~1.95M parameters) |

> A detailed analysis of the experiments and decisions is in [REPORT_README.md](REPORT_README.md).
> This document is the entry point for quickly understanding the project structure and how to run it.

---

## 1. Problem Overview

- **Input:** Noisy grayscale document images (PNG)
- **Output:** A CSV of the restored intensity of each pixel (`0.0` = black, `1.0` = white)
- **Metric:** **RMSE** between predicted and ground-truth pixels (lower is better)
- **Data:** 115 training pairs (noisy + clean), 29 test images (noisy only)

Submission CSV format (melted to one row per pixel):

```
id,value
1_1_1,1.000000      # image 1, row 1, col 1
1_2_1,0.984314
...
```

---

## 2. Approach Summary

We analyzed five weaknesses of the original AI agent (`ResidualUNet`, base channels 16,
early-stopped at epoch 2) and applied domain-specific improvements.

| Area | Key change | Why |
|---|---|---|
| **Architecture** | 4-level ImprovedUNet + BatchNorm | Larger receptive field + stable training (115-image dataset) |
| **Preprocessing** | Otsu background normalization (implemented from scratch) | Document noise is essentially "uneven background brightness" → normalize background to ~1.0 |
| **Output** | Removed residual cap (±0.1), direct prediction | Allows full-range correction even for heavily stained pixels |
| **Training** | 80 epochs, patience 15, CosineAnnealingLR | Prevents under-training from premature early stopping |
| **Data** | Patches per image 2→8, brightness jitter augmentation | 4× more training diversity per epoch |
| **Inference** | Test-time augmentation (TTA, 4-way flip averaging) | ~17% validation RMSE improvement with no extra training |

Only validated results were adopted — both **gamma post-processing** (Run 4) and the
**ensemble** (Run 5) showed no improvement and were excluded from the final submission
(see Section 4 of the report for details).

---

## 3. Directory Structure

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

⭐ = core files needed to reproduce the final result

---

## 4. Core Modules

### `train_sean.py` — main pipeline
Handles everything from training to inference and CSV generation. Other scripts import its functions.

- `ImprovedUNet` — 4-level U-Net (3 encoder stages + bottleneck + 3 decoder stages), with BatchNorm
- `otsu_background_normalize()` — computes the Otsu threshold from the histogram (from scratch) and divides by background brightness
- `RandomPatchDataset` — random patch extraction + augmentation (horizontal/vertical flip, 90° rotation, brightness jitter)
- `predict_single()` — reflect padding → inference → 4-way TTA averaging
- `write_predictions_csv()` — writes per-pixel output in submission format (`id,value`)

### `postprocess_predict.py` — gamma post-processing (experiment)
Attempts to strengthen background↔text contrast by applying `output^gamma` to predictions.
Searching gamma 0.7–2.0 on the validation set found **gamma=1.0 (no change) optimal** → confirms the model output is already well-calibrated.

### `ensemble_predict.py` — ensemble inference (experiment)
Averages predictions from several checkpoints trained with different seeds. An ensemble of three
base-channels-48 models scored 0.01539, **worse** than the single model (0.01514) — likely overfitting of the larger model on the small dataset.

### `colab_train_ensemble.ipynb` — GPU training notebook
A notebook that trains three models (seeds 42/123/456) on a Colab T4 GPU and generates ensemble predictions.

---

## 5. Setup

```bash
pip install torch torchvision numpy pillow
```

Uses GPU (CUDA) automatically if available, otherwise runs on CPU.
If you don't have the data:

```bash
aicodinggym mle download denoising-dirty-documents
```

---

## 6. Reproducing the Final Result

```bash
# 1) Train the model (CPU: ~3 hours / GPU: ~20 minutes)
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

> To skip training and run inference only with the bundled `unet_sean_best.pth`,
> you can use `postprocess_predict.py` for inference (with gamma=1.0 it applies no post-processing).

Expected result: **RMSE ≈ 0.015, Gold tier, top 5%**.

For default hyperparameters, see the `argparse` definitions in `train_sean.py`
(script defaults are epochs=80, patches=10, patience=15; the command above uses the settings from the reproduction report).

---

## 7. Key Hyperparameters (`train_sean.py`)

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | (required) | Folder containing `train/`, `train_cleaned/`, `test/`, `sampleSubmission.csv` |
| `--epochs` | 80 | Maximum training epochs |
| `--batch-size` | 16 | Batch size |
| `--lr` | 1e-3 | Initial learning rate (decays to 1e-6 via cosine annealing) |
| `--patch-size` | 128 | Training patch size |
| `--patches-per-image` | 10 | Patches extracted per image |
| `--val-split` | 0.1 | Validation split fraction |
| `--patience` | 15 | Early-stopping patience (epochs) |
| `--base-channels` | 32 | U-Net base channel count (capacity) |
| `--seed` | 42 | Random seed |

---

## 8. References

- Detailed experiment analysis & retrospective: [REPORT_README.md](REPORT_README.md)
- Original competition description: [data/description.md](data/description.md)
- Session logs: [.log/](.log/)
- Challenge instructions: [AGENTS.md](AGENTS.md)
```
