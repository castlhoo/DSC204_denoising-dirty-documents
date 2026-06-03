"""
Post-processing: apply gamma contrast enhancement to model predictions.

Gamma > 1 pushes pixel values away from the midpoint:
  - background pixels (near 1.0) get pushed closer to 1.0
  - text pixels (near 0.0) get pushed closer to 0.0

This reduces RMSE by correcting the model's tendency to predict
"safe" middle values instead of confident 0 or 1 outputs.

Usage:
    python postprocess_predict.py --data-dir data
"""

import argparse
import csv
import math
import os
import random
from glob import glob

import numpy as np
import torch

from train_sean import (
    ImprovedUNet,
    load_gray_uint8,
    load_pairs,
    otsu_background_normalize,
    pad_to_multiple,
    predict_single,
    split_indices,
    submission_image_order,
    unpad_to_original,
)


def apply_gamma(prediction, gamma):
    return np.power(np.clip(prediction, 0.0, 1.0), gamma)


def rmse(pred, target):
    return math.sqrt(np.mean((pred - target) ** 2))


def evaluate_with_gamma(model, samples, device, gamma):
    total_se = 0.0
    total_pixels = 0.0
    for sample in samples:
        pred = predict_single(model, sample['noisy_raw'], device)
        pred = apply_gamma(pred, gamma)
        target = sample['clean']
        total_se += float(np.sum((pred - target) ** 2))
        total_pixels += float(target.size)
    return math.sqrt(total_se / max(1.0, total_pixels))


def write_predictions_with_gamma(model, test_paths, output_path, device, gamma):
    with open(output_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'value'])
        for image_path in test_paths:
            noisy = load_gray_uint8(image_path)
            prediction = predict_single(model, noisy, device)
            prediction = apply_gamma(prediction, gamma)
            prediction = np.clip(prediction, 0.0, 1.0)
            height, width = prediction.shape
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            for row_index in range(height):
                for col_index in range(width):
                    writer.writerow([
                        f'{image_id}_{row_index + 1}_{col_index + 1}',
                        f'{prediction[row_index, col_index]:.6f}'
                    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model-path', default='unet_sean_best.pth')
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--output-csv', default='predictions_postprocessed.csv')
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedUNet(base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f'Loaded model from {args.model_path}')

    train_noisy_paths = sorted(glob(os.path.join(args.data_dir, 'train', '*.png')))
    train_clean_paths = sorted(glob(os.path.join(args.data_dir, 'train_cleaned', '*.png')))
    print(f'Loading {len(train_noisy_paths)} image pairs...')
    pairs = load_pairs(train_noisy_paths, train_clean_paths)

    _, val_indices = split_indices(len(pairs), args.val_split, args.seed)
    val_samples = [pairs[i] for i in val_indices]
    print(f'Tuning gamma on {len(val_samples)} validation images...')

    # Baseline (no post-processing)
    baseline_rmse = evaluate_with_gamma(model, val_samples, device, gamma=1.0)
    print(f'\nBaseline (gamma=1.0): val RMSE = {baseline_rmse:.6f}')

    # Search over gamma values
    gamma_candidates = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]
    best_gamma = 1.0
    best_rmse = baseline_rmse

    print('\nGamma search:')
    for gamma in gamma_candidates:
        r = evaluate_with_gamma(model, val_samples, device, gamma)
        marker = ' <- best' if r < best_rmse else ''
        print(f'  gamma={gamma:.2f}  val RMSE={r:.6f}{marker}')
        if r < best_rmse:
            best_rmse = r
            best_gamma = gamma

    print(f'\nBest gamma: {best_gamma:.2f}  (val RMSE: {best_rmse:.6f})')
    print(f'Improvement over baseline: {baseline_rmse - best_rmse:.6f}')

    sample_path = os.path.join(args.data_dir, 'sampleSubmission.csv')
    image_order = submission_image_order(sample_path)
    test_map = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob(os.path.join(args.data_dir, 'test', '*.png'))
    }
    test_paths = [test_map[iid] for iid in image_order]

    out_csv = os.path.join(os.path.dirname(__file__), args.output_csv)
    print(f'\nGenerating predictions with gamma={best_gamma:.2f}...')
    write_predictions_with_gamma(model, test_paths, out_csv, device, best_gamma)
    print(f'Done! Saved to {out_csv}')
    print(f'\nSubmit with:')
    print(f'  aicodinggym mle submit denoising-dirty-documents -F "{out_csv}"')


if __name__ == '__main__':
    main()
