# Denoising Dirty Documents — Project Report Reference

> **Author:** Sean  
> **Competition:** Denoising Dirty Documents (MLE-bench / Kaggle)  
> **Final Score:** 0.01514 RMSE — **Rank 5 / 162 (Top 3.1%) — Gold Tier**  
> **AI Baseline Score:** 0.01649 RMSE  
> **Date:** May 16, 2026

---

## 1. Competition Overview

The task is to remove noise from scanned document images. Real-world documents suffer from coffee stains, yellowing, uneven lighting, wrinkles, and other artifacts that interfere with OCR (Optical Character Recognition). The goal is to output clean pixel intensities for each pixel in the test images.

- **Input:** Noisy grayscale document images (PNG)
- **Output:** CSV file with predicted clean pixel intensities (0.0 = black, 1.0 = white)
- **Metric:** Root Mean Squared Error (RMSE) between predicted and true clean pixels — **lower is better**
- **Dataset:** 115 training pairs (noisy + clean), 29 test images (noisy only)
- **Leaderboard:** AI Baseline (Disarray) = 0.01649, Gold tier top = 0.01021

---

## 2. AI Agent Inspection — What the Existing AI Did and Why It Failed

The AI agent built a `ResidualUNet` model in `train_unet.py`. Upon careful inspection, **5 critical weaknesses** were identified:

### AI Agent Configuration

| Setting | AI Agent Value | Problem |
|---|---|---|
| Architecture | ResidualUNet, 3 encoder levels | Too shallow |
| Base channels | 16 (~488K parameters) | Too small for complex patterns |
| Patch size | 64×64 | OK |
| Patches per image | 2 | Too few — only 230 patches/epoch |
| Max epochs | 25 | Fine |
| Early stopping patience | 4 | **Too aggressive — fired at epoch 2** |
| LR scheduler | ReduceLROnPlateau | OK |
| Preprocessing | Raw `/255` normalization | No background correction |
| Residual output cap | `0.1 * tanh(output)` | **Limits correction to ±0.1** |

### Why the AI Failed

**Root cause #1 — Premature early stopping:** With `patience=4` and a flat loss landscape at the start of training (epoch 1 val RMSE = 0.148, epoch 2 = 0.097), the model stopped after epoch 2. It never had time to learn meaningful denoising. The AI's val RMSE of 0.097 represents an almost-untrained model.

**Root cause #2 — Model capacity too small:** `base_channels=16` gives ~488K parameters. Separating text pixels from background noise requires learning complex spatial patterns (stroke shape, texture variation, stain gradients). 488K parameters cannot represent this complexity.

**Root cause #3 — No domain-specific preprocessing:** Dirty document images have highly variable background brightness — some images are bright white, others are yellowed or stained brown. Dividing all images by 255 treats them identically. A stained image (background ~0.7) and a clean image (background ~0.95) require very different corrections, but the model sees the same input range.

**Root cause #4 — Residual cap:** The `0.1 * tanh(x)` cap limits each pixel's correction to a maximum of ±0.1. For a heavily stained document where background pixels might be at 0.6 (should be 1.0), a correction of only +0.1 is far too small.

**Root cause #5 — Low data utilization:** With only 2 patches per image, the model sees ~230 training patches per epoch. This is extremely low diversity for a neural network with 488K parameters.

### AI Agent Performance

- Median Filter baseline: RMSE = 0.1897
- AI Agent (2 epochs): RMSE = **0.0969**
- AI Agent did NOT beat the leaderboard baseline (0.01649)

---

## 3. Our Strategy — What We Changed and Why

We designed `train_sean.py` from scratch, addressing all 5 identified weaknesses with principled improvements.

### 3.1 Architecture: ImprovedUNet

We replaced the 3-level ResidualUNet with a 4-level ImprovedUNet:

```
Input (1 channel, grayscale)
  → Encoder 1: Conv(1→32) → BatchNorm → ReLU → Conv(32→32) → BatchNorm → ReLU
  → MaxPool 2×2
  → Encoder 2: Conv(32→64) → BatchNorm → ReLU → Conv(64→64) → BatchNorm → ReLU
  → MaxPool 2×2
  → Encoder 3: Conv(64→128) → BatchNorm → ReLU → Conv(128→128) → BatchNorm → ReLU
  → MaxPool 2×2
  → Bottleneck: Conv(128→256) → BatchNorm → ReLU → Conv(256→256) → BatchNorm → ReLU
  ↑ Decoder 3: Upsample + Skip from Enc3 → Conv(384→128)
  ↑ Decoder 2: Upsample + Skip from Enc2 → Conv(192→64)
  ↑ Decoder 1: Upsample + Skip from Enc1 → Conv(96→32)
  → Output: Conv(32→1) → clamp(0, 1)
```

**Why 4 levels instead of 3?** Each additional encoder level doubles the receptive field. With 4 levels, the bottleneck "sees" an 8× downsampled version of the input, capturing global document structure (background color regions, stain patterns) alongside fine-grained text details.

**Why BatchNorm after every Conv?** Without normalization, deep networks suffer from vanishing/exploding gradients. BatchNorm stabilizes the activation distribution, allowing faster and more reliable convergence — critical when training on only 115 images.

**Why remove the residual cap?** The `0.1 * tanh` cap was a design error. Removing it allows the model to make full corrections (e.g., turn a gray background pixel fully white).

**Result:** 1,949,121 parameters — 4× the AI baseline.

### 3.2 Preprocessing: Otsu Background Normalization

**Core insight:** The primary noise in dirty documents is uneven background brightness, not random pixel noise. A coffee stain makes the background dark brown. Yellowing makes it yellow-gray. Standard `/255` normalization ignores this completely.

**Solution — Otsu's method (implemented from scratch):**
1. Compute the pixel intensity histogram of the input image
2. Find the threshold that maximizes inter-class variance (Otsu's criterion) — this separates "background" pixels from "text" pixels
3. Compute the mean brightness of all background pixels (those above threshold)
4. Divide the entire image by that background mean

**Result:** After normalization, background pixels are approximately 1.0 (white) regardless of staining. Text pixels are proportionally darker. The model now always sees a consistent input range, making its job easier.

**Why this matters:** Without this, the model must simultaneously learn "what is the background color for this image" AND "where is the text." With Otsu normalization, the first question is already answered, and the model can focus entirely on the second.

### 3.3 Training Configuration Improvements

| Setting | AI Agent | Sean's Version | Decision Reasoning |
|---|---|---|---|
| Base channels | 16 | **32** | 4× capacity; still fast on CPU |
| Patches per image | 2 | **8** | 4× more training data per epoch; reduces overfitting risk |
| Patch size | 64×64 | **64×64** | Kept same; larger patches would slow CPU too much |
| Max epochs | 25 | **80** | Model kept improving at epoch 40; needed more room |
| Early stopping patience | 4 | **15** | Prevents stopping during plateau; model improved from epoch 39 to 67 |
| LR scheduler | ReduceLROnPlateau | **CosineAnnealingLR** | Cosine annealing smoothly reduces LR from 1e-3 to 1e-6 over all epochs; avoids LR stagnation |
| Augmentation | H-flip, rot90 | **H-flip, rot90, brightness jitter** | Brightness jitter (±10%) simulates lighting variation |
| Preprocessing | /255 | **Otsu normalization** | See Section 3.2 |
| Residual cap | 0.1 × tanh | **None (direct prediction)** | Allows full-range corrections |

### 3.4 Test-Time Augmentation (TTA)

**Concept:** At prediction time (not training time), we run the model 4 times on each test image:
- Original orientation
- Horizontally flipped (then flip prediction back)
- Vertically flipped (then flip prediction back)
- Both flips combined (then restore)

We average the 4 predictions. Each flip produces slightly different predictions due to the CNN's spatial behavior. Averaging reduces random errors with zero additional training cost.

**Implementation note:** NumPy's `np.fliplr()` and `np.flipud()` create views with negative strides, which PyTorch cannot handle. We added `.copy()` after each flip to create contiguous arrays.

**Effect:** Reduced val RMSE from ~0.0175 (without TTA) to ~0.0146 (with TTA) — approximately 17% improvement at inference for free.

---

## 4. Experiments and Results

### Experiment Timeline

| Run | Description | Val RMSE | Test RMSE | Leaderboard Rank |
|---|---|---|---|---|
| AI Agent | ResidualUNet, base=16, 2 epochs | 0.0969 | 0.0969 | — (baseline) |
| Run 1 | ImprovedUNet, base=32, Otsu, 16 epochs (interrupted) | 0.0204 | — | — |
| Run 2 | Same, 40 epochs complete | 0.01734 | — | — |
| **Run 3** | **+ TTA + CosineAnnealingLR, 80 epochs** | **0.01457** | **0.01514** | **5/162 Gold ✅** |
| Run 4 | Gamma post-processing on Run 3 | 0.01457 (no change) | — | — |
| Run 5 | Ensemble: 3× base=48 models on Colab GPU | — | 0.01539 | Worse than Run 3 |

### Why Post-Processing Failed (Run 4)

We hypothesized that applying a power transform `output^gamma` (gamma > 1) would push background pixels toward 1.0 and text pixels toward 0.0, reducing RMSE. After searching gamma values from 0.7 to 2.0, gamma=1.0 (no change) was the best. **This is actually a good sign** — it means the model is already outputting well-calibrated values and is not making the systematic error we hypothesized.

### Why Ensemble Was Worse (Run 5)

We trained 3 larger models (base_channels=48, ~4.3M parameters) on Colab GPU and averaged their predictions. The ensemble scored 0.01539 vs 0.01514 for the single model. Likely reasons:
1. The larger model overfits more on only 115 training images
2. The single base=32 model was already well-optimized with 80 epochs + TTA
3. Training variance across 3 seeds was insufficient to provide complementary diversity

### Final Best Result

| | Value |
|---|---|
| Model | ImprovedUNet (base_channels=32, 1.95M params) |
| Preprocessing | Otsu background normalization |
| Training | 80 epochs, CosineAnnealingLR, patches=8, batch=16 |
| Inference | TTA (4-way flip averaging) |
| Best epoch | 67 / 80 |
| Val RMSE | 0.01457 |
| **Test RMSE (leaderboard)** | **0.01514** |
| **Rank** | **5 / 162 (top 3.1%) — Gold tier** |
| AI baseline | 0.01649 |
| **Improvement over AI** | **+8.2% relative** |

---

## 5. Reflection

### Where the AI Agent Fell Short

The AI agent's most critical failure was **premature early stopping** — the model stopped at epoch 2 without any meaningful learning. This was caused by setting `patience=4` with a `ReduceLROnPlateau` scheduler that couldn't react fast enough in the early training phase. The second major failure was **no domain-specific preprocessing**: dirty documents require background normalization, which the AI agent completely missed.

These failures stem from the AI agent applying generic machine learning recipes without domain understanding. It used a standard U-Net, standard normalization, and standard hyperparameters without considering the specific characteristics of document noise (non-uniform background, binary text structure).

### What We Did Differently

1. **Domain analysis first:** We analyzed the images before choosing a model. Recognizing that document noise is primarily background variation (not pixel noise) led directly to the Otsu preprocessing solution.

2. **Root cause fixing:** Instead of adding complexity, we fixed the root cause of the AI's failure (patience, preprocessing) before adding new features.

3. **Empirical iteration:** We ran multiple experiments (Runs 1–5), measured results, and made data-driven decisions about what to keep (TTA worked) and what to discard (post-processing didn't, ensemble didn't).

4. **Recognizing when to stop:** After Run 3 achieved a strong result, we explored further improvements but accepted that the single-model result was already optimal for this dataset size. We did not over-engineer.

### Key Concepts Learned

- **U-Net skip connections:** Preserving high-resolution feature maps from encoder to decoder is critical for pixel-accurate restoration tasks. Without skip connections, the decoder cannot recover fine-grained text edge information.
- **BatchNorm in practice:** Adding BatchNorm after every Conv significantly stabilized training on this small dataset (115 images).
- **Otsu thresholding:** A classical image processing technique (1979) that remains highly effective as a preprocessing step for document images, even alongside modern deep learning.
- **TTA:** A simple, free technique that consistently improves results by reducing prediction variance.
- **Small dataset regimes:** With only 115 images, model capacity must be carefully controlled. A model that is too large (base=48) can overfit and perform worse than a well-regularized smaller model.

---

## 6. File Reference

| File | Description |
|---|---|
| `train_sean.py` | Main training script — ImprovedUNet + Otsu + TTA + CosineAnnealingLR |
| `ensemble_predict.py` | Ensemble prediction from multiple saved model checkpoints |
| `postprocess_predict.py` | Gamma post-processing tuning script (negative result) |
| `colab_train_ensemble.ipynb` | Colab notebook for GPU-accelerated ensemble training |
| `unet_sean_best.pth` | Saved weights of the best model (epoch 67, val RMSE=0.01457) |
| `predictions_sean.csv` | **Best submission** — test predictions, RMSE 0.01514 |
| `predictions_ensemble.csv` | Ensemble submission — RMSE 0.01539 (worse) |
| `.log/windsurf-20260516-210200.md` | Full session log with all decisions and experiments |

---

## 7. Reproduction

To reproduce the best result from scratch:

```bash
# 1. Install dependencies
pip install torch torchvision numpy pillow

# 2. Download data
aicodinggym mle download denoising-dirty-documents

# 3. Train the model (CPU: ~3 hours, GPU: ~20 minutes)
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

# 4. Submit
aicodinggym mle submit denoising-dirty-documents -F predictions_sean.csv
```

Expected output: RMSE ≈ 0.015, Gold tier, Top 5% on leaderboard.
