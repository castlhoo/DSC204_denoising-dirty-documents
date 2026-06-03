"""
Sean's improved U-Net denoising pipeline.

Improvements over AI baseline:
1. Larger U-Net (base_channels=32 vs 16, + BatchNorm)
2. Otsu-based background normalization preprocessing
3. More patches per image (10 vs 2) and larger patch size (128 vs 64)
4. More training epochs with better patience (60 epochs, patience=8)
5. Stronger augmentation (brightness jitter added)
"""

import argparse
import csv
import math
import os
import random
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sorted_numeric_filenames(dirpath):
    files = [f for f in os.listdir(dirpath) if f.lower().endswith('.png')]

    def keyfn(fn):
        name = os.path.splitext(fn)[0]
        try:
            return int(name)
        except Exception:
            return name

    return sorted(files, key=keyfn)


def submission_image_order(template_path):
    order = []
    seen = set()
    with open(template_path, newline='') as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            image_id = row[0].split('_', 1)[0]
            if image_id not in seen:
                seen.add(image_id)
                order.append(image_id)
    return order


def load_gray_uint8(path):
    return np.array(Image.open(path).convert('L'), dtype=np.uint8)


def otsu_background_normalize(image_uint8):
    """
    Preprocessing: estimate background using Otsu threshold.
    Divides image by estimated background brightness so that
    background becomes ~1.0 (white) and text stands out clearly.
    """
    img = image_uint8.astype(np.float32)
    # Compute Otsu threshold manually
    hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 256))
    total = img.size
    sum_all = np.dot(np.arange(256), hist)
    sum_bg, w_bg, w_fg = 0.0, 0.0, 0.0
    max_var, threshold = 0.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var > max_var:
            max_var = var
            threshold = t

    # Background pixels = above threshold; estimate background brightness
    bg_mask = img > threshold
    if bg_mask.sum() > 0:
        bg_mean = img[bg_mask].mean()
    else:
        bg_mean = 255.0

    # Normalize so background ~ 1.0
    normalized = img / max(bg_mean, 1.0)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def load_pairs(noisy_paths, clean_paths):
    pairs = []
    for noisy_path, clean_path in zip(noisy_paths, clean_paths):
        noisy_raw = load_gray_uint8(noisy_path)
        clean_raw = load_gray_uint8(clean_path)
        pairs.append({
            'name': os.path.basename(noisy_path),
            'noisy_raw': noisy_raw,
            'clean_raw': clean_raw,
            'noisy': otsu_background_normalize(noisy_raw),
            'clean': clean_raw.astype(np.float32) / 255.0,
        })
    return pairs


def split_indices(num_items, val_fraction, seed):
    indices = list(range(num_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(num_items * val_fraction)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def pad_to_multiple(tensor, multiple):
    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    padded = nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return padded, (height, width)


def unpad_to_original(tensor, original_shape):
    height, width = original_shape
    return tensor[..., :height, :width]


class ImprovedUNet(nn.Module):
    """
    Larger U-Net with BatchNorm for training stability.
    base_channels=32 gives 4x more parameters than the AI baseline (16).
    """
    def __init__(self, base_channels=32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(1, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(c3, c4)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = conv_block(c4 + c3, c3)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = conv_block(c3 + c2, c2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = conv_block(c2 + c1, c1)

        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.clamp(self.out(d1), 0.0, 1.0)


class RandomPatchDataset(Dataset):
    def __init__(self, pairs, indices, patch_size, patches_per_image, augment=True):
        self.pairs = [pairs[i] for i in indices]
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.length = max(1, len(self.pairs) * patches_per_image)

    def __len__(self):
        return self.length

    def _augment(self, noisy, clean):
        if random.random() < 0.5:
            noisy = np.fliplr(noisy)
            clean = np.fliplr(clean)
        if random.random() < 0.5:
            noisy = np.flipud(noisy)
            clean = np.flipud(clean)
        rot_k = random.randint(0, 3)
        if rot_k:
            noisy = np.rot90(noisy, rot_k)
            clean = np.rot90(clean, rot_k)
        # Brightness jitter on noisy input only
        if random.random() < 0.3:
            factor = random.uniform(0.9, 1.1)
            noisy = np.clip(noisy * factor, 0.0, 1.0)
        return noisy.copy(), clean.copy()

    def __getitem__(self, idx):
        pair = self.pairs[idx % len(self.pairs)]
        noisy = pair['noisy']
        clean = pair['clean']
        height, width = noisy.shape
        patch_size = self.patch_size
        max_y = height - patch_size
        max_x = width - patch_size
        top = random.randint(0, max_y) if max_y > 0 else 0
        left = random.randint(0, max_x) if max_x > 0 else 0
        noisy_patch = noisy[top:top + patch_size, left:left + patch_size]
        clean_patch = clean[top:top + patch_size, left:left + patch_size]
        if self.augment:
            noisy_patch, clean_patch = self._augment(noisy_patch, clean_patch)
        noisy_tensor = torch.from_numpy(noisy_patch.astype(np.float32)).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_patch.astype(np.float32)).unsqueeze(0)
        return noisy_tensor, clean_tensor


def rmse_from_sse(sum_squared_error, pixel_count):
    return math.sqrt(sum_squared_error / max(1.0, pixel_count))


@torch.no_grad()
def predict_single(model, noisy_array, device, multiple=8, tta=True):
    model.eval()
    noisy_float = otsu_background_normalize(noisy_array)

    def _infer(arr):
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
        padded, orig_shape = pad_to_multiple(t, multiple)
        out = model(padded)
        out = unpad_to_original(out, orig_shape)
        return out.squeeze(0).squeeze(0).cpu().numpy()

    pred = _infer(noisy_float)
    if tta:
        pred_h = np.fliplr(_infer(np.fliplr(noisy_float).copy())).copy()
        pred_v = np.flipud(_infer(np.flipud(noisy_float).copy())).copy()
        pred_hv = np.fliplr(np.flipud(_infer(np.fliplr(np.flipud(noisy_float).copy()).copy()))).copy()
        pred = (pred + pred_h + pred_v + pred_hv) / 4.0
    return pred


@torch.no_grad()
def evaluate_images(model, samples, device):
    total_se = 0.0
    total_pixels = 0.0
    for sample in samples:
        prediction = predict_single(model, sample['noisy_raw'], device)
        target = sample['clean']
        se = float(np.sum((prediction - target) ** 2))
        total_se += se
        total_pixels += float(target.size)
    return rmse_from_sse(total_se, total_pixels)


def write_predictions_csv(model, test_paths, output_path, device):
    with open(output_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'value'])
        for image_path in test_paths:
            noisy = load_gray_uint8(image_path)
            prediction = predict_single(model, noisy, device)
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
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--patches-per-image', type=int, default=10)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--output-csv', default='predictions_sean.csv')
    parser.add_argument('--model-path', default='unet_sean_best.pth')
    args = parser.parse_args()

    set_seed(args.seed)

    train_noisy_paths = sorted(glob(os.path.join(args.data_dir, 'train', '*.png')))
    train_clean_paths = sorted(glob(os.path.join(args.data_dir, 'train_cleaned', '*.png')))
    sample_submission_path = os.path.join(args.data_dir, 'sampleSubmission.csv')
    test_dir = os.path.join(args.data_dir, 'test')

    image_order = submission_image_order(sample_submission_path)
    test_map = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob(os.path.join(test_dir, '*.png'))
    }
    test_paths = [test_map[iid] for iid in image_order]

    assert len(train_noisy_paths) == len(train_clean_paths), 'train/ and train_cleaned/ must match'

    print(f'Loading {len(train_noisy_paths)} image pairs with Otsu background normalization...')
    pairs = load_pairs(train_noisy_paths, train_clean_paths)
    train_indices, val_indices = split_indices(len(pairs), args.val_split, args.seed)
    train_samples = [pairs[i] for i in train_indices]
    val_samples = [pairs[i] for i in val_indices]

    print(f'Train: {len(train_samples)} images, Val: {len(val_samples)} images')
    print(f'Patch size: {args.patch_size}x{args.patch_size}, patches/image: {args.patches_per_image}')
    print(f'Model: ImprovedUNet(base_channels={args.base_channels})')

    train_dataset = RandomPatchDataset(
        pairs, train_indices,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        augment=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = ImprovedUNet(base_channels=args.base_channels).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {param_count:,}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_rmse = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_mse = 0.0
        total_pixels = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_pixels = float(xb.size(0) * xb.size(2) * xb.size(3))
            total_mse += loss.item() * batch_pixels
            total_pixels += batch_pixels

        train_rmse = math.sqrt(total_mse / max(1.0, total_pixels))
        val_rmse = evaluate_images(model, val_samples, device)
        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f'Epoch {epoch:02d}/{args.epochs} '
            f'train_RMSE={train_rmse:.6f} '
            f'val_RMSE={val_rmse:.6f} '
            f'lr={current_lr:.2e}'
        )

        if val_rmse + 1e-6 < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), args.model_path)
            epochs_without_improvement = 0
            print(f'  -> Best model saved (val_RMSE={best_val_rmse:.6f})')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    final_train_rmse = evaluate_images(model, train_samples, device)
    final_val_rmse = evaluate_images(model, val_samples, device)

    print(f'\n=== Final Results ===')
    print(f'Best epoch: {best_epoch}')
    print(f'Train RMSE: {final_train_rmse:.6f}')
    print(f'Val RMSE:   {final_val_rmse:.6f}')

    out_csv = os.path.join(os.path.dirname(__file__), args.output_csv)
    print(f'\nWriting predictions to {out_csv}...')
    write_predictions_csv(model, test_paths, out_csv, device)
    print('Done!')


if __name__ == '__main__':
    main()
