"""
Ensemble prediction: average predictions from multiple trained models.

Usage (after training 3 models with different seeds):
    python ensemble_predict.py --data-dir data \
        --model-paths model_seed42.pth model_seed123.pth model_seed456.pth \
        --base-channels 48 \
        --output-csv predictions_ensemble.csv
"""

import argparse
import csv
import os
from glob import glob

import numpy as np
import torch

from train_sean import (
    ImprovedUNet,
    load_gray_uint8,
    predict_single,
    submission_image_order,
)


def ensemble_predict(models, noisy_array, device):
    predictions = [predict_single(m, noisy_array, device) for m in models]
    return np.mean(predictions, axis=0)


def write_ensemble_csv(models, test_paths, output_path, device):
    with open(output_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'value'])
        for i, image_path in enumerate(test_paths, 1):
            noisy = load_gray_uint8(image_path)
            prediction = ensemble_predict(models, noisy, device)
            prediction = np.clip(prediction, 0.0, 1.0)
            height, width = prediction.shape
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            print(f'  [{i}/{len(test_paths)}] {image_id}  shape={height}x{width}')
            for row_index in range(height):
                for col_index in range(width):
                    writer.writerow([
                        f'{image_id}_{row_index + 1}_{col_index + 1}',
                        f'{prediction[row_index, col_index]:.6f}'
                    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model-paths', nargs='+', required=True,
                        help='Paths to trained model .pth files')
    parser.add_argument('--base-channels', type=int, default=48)
    parser.add_argument('--output-csv', default='predictions_ensemble.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    models = []
    for path in args.model_paths:
        m = ImprovedUNet(base_channels=args.base_channels).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
        print(f'Loaded: {path}')

    print(f'\nEnsemble of {len(models)} models')

    sample_path = os.path.join(args.data_dir, 'sampleSubmission.csv')
    image_order = submission_image_order(sample_path)
    test_map = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob(os.path.join(args.data_dir, 'test', '*.png'))
    }
    test_paths = [test_map[iid] for iid in image_order]

    out_csv = os.path.join(os.path.dirname(__file__), args.output_csv)
    print(f'\nGenerating ensemble predictions for {len(test_paths)} test images...')
    write_ensemble_csv(models, test_paths, out_csv, device)
    print(f'\nDone! Saved to {out_csv}')
    print(f'\nSubmit with:')
    print(f'  aicodinggym mle submit denoising-dirty-documents -F "{out_csv}"')


if __name__ == '__main__':
    main()
