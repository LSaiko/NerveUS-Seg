import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import NerveDataset, get_val_transform
from metrics import dice_score
from train import build_model, find_pairs


@torch.no_grad()
def predict_tta(model, img_tensor, device):
    """img_tensor: normalized CHW tensor (no batch dim). Averages predictions
    over original/h-flip/v-flip, each restored to original orientation."""
    flips = [None, [2], [1]]  # dims of the unbatched (C,H,W) tensor: none, width, height
    probs = torch.zeros(1, *img_tensor.shape[1:])
    for dims in flips:
        x = img_tensor if dims is None else torch.flip(img_tensor, dims)
        logits = model(x.unsqueeze(0).to(device))
        p = torch.sigmoid(logits).cpu()[0]
        if dims is not None:
            p = torch.flip(p, dims)
        probs += p
    return probs / len(flips)


def _elastic_field(h, w, alpha, sigma, rng):
    """Smooth random displacement field, same construction as Albumentations'
    ElasticTransform: blur noise, scale by alpha."""
    dx = rng.uniform(-1, 1, size=(h, w)).astype(np.float32)
    dy = rng.uniform(-1, 1, size=(h, w)).astype(np.float32)
    k = int(sigma) | 1  # odd kernel size
    dx = cv2.GaussianBlur(dx, (k, k), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (k, k), sigma) * alpha
    return dx, dy


def _remap_chw(tensor, dx, dy):
    h, w = dx.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)
    arr = tensor.permute(1, 2, 0).numpy()
    warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if warped.ndim == 2:
        warped = warped[..., None]
    return torch.from_numpy(warped).permute(2, 0, 1)


@torch.no_grad()
def predict_tta_elastic(model, img_tensor, device, n_samples=4, alpha=15.0, sigma=8.0, seed=0):
    """Averages the identity prediction with n_samples elastic-warped views.
    Each warped prediction is unwarped with the negated displacement field
    (an approximation of the true inverse, accurate for smooth/small warps)
    before averaging back in the original image's coordinate frame."""
    rng = np.random.RandomState(seed)
    logits = model(img_tensor.unsqueeze(0).to(device))
    total = torch.sigmoid(logits).cpu()[0]
    n = 1
    h, w = img_tensor.shape[1:]
    for _ in range(n_samples):
        dx, dy = _elastic_field(h, w, alpha, sigma, rng)
        x_warp = _remap_chw(img_tensor, dx, dy)
        logits = model(x_warp.unsqueeze(0).to(device))
        p = torch.sigmoid(logits).cpu()[0]
        total += _remap_chw(p, -dx, -dy)
        n += 1
    return total / n


@torch.no_grad()
def predict_tta_crop(model, img_tensor, device, crop_frac=0.85):
    """Averages the full-image prediction with 4-corner + center crop
    predictions. Each crop is resized up to the model's input size, predicted,
    then resized back down and pasted into its exact region of the full-size
    canvas — a deterministic placement, not an approximate inverse."""
    C, H, W = img_tensor.shape
    ch, cw = int(H * crop_frac), int(W * crop_frac)
    positions = [(0, 0), (0, W - cw), (H - ch, 0), (H - ch, W - cw), ((H - ch) // 2, (W - cw) // 2)]

    logits = model(img_tensor.unsqueeze(0).to(device))
    total = torch.sigmoid(logits).cpu()[0]
    weight = torch.ones_like(total)

    for y, x in positions:
        crop = img_tensor[:, y:y + ch, x:x + cw]
        crop_up = F.interpolate(crop.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
        logits = model(crop_up.to(device))
        p = torch.sigmoid(logits).cpu()
        p_down = F.interpolate(p, size=(ch, cw), mode="bilinear", align_corners=False)[0]
        total[:, y:y + ch, x:x + cw] += p_down
        weight[:, y:y + ch, x:x + cw] += 1

    return total / weight


@torch.no_grad()
def evaluate_tta(model, dataset, device, predict_fn=predict_tta, threshold=0.5):
    model.eval()
    total_dice = 0.0
    for i in range(len(dataset)):
        img, mask = dataset[i]
        prob = predict_fn(model, img, device)
        pred = (prob > threshold).float()
        inter = (pred * mask).sum()
        union = pred.sum() + mask.sum()
        total_dice += ((2 * inter + 1e-5) / (union + 1e-5)).item()
    return total_dice / len(dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/ultrasound-nerve-segmentation")
    p.add_argument("--checkpoint", default="models/best_unet.pth")
    p.add_argument("--val-split", type=float, default=0.2)
    args = p.parse_args()

    pairs = find_pairs(args.data_dir)
    n_val = int(len(pairs) * args.val_split)
    imgs, masks = zip(*pairs[:n_val])
    val_ds = NerveDataset(list(imgs), list(masks), get_val_transform())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    plain_dice = 0.0
    with torch.no_grad():
        for img, mask in val_ds:
            logits = model(img.unsqueeze(0).to(device))
            plain_dice += dice_score(logits.cpu(), mask.unsqueeze(0))
    plain_dice /= len(val_ds)

    flip_dice = evaluate_tta(model, val_ds, device, predict_tta)
    elastic_dice = evaluate_tta(model, val_ds, device, predict_tta_elastic)
    crop_dice = evaluate_tta(model, val_ds, device, predict_tta_crop)
    print(f"Plain val Dice:         {plain_dice:.4f}")
    print(f"Flip TTA val Dice:      {flip_dice:.4f}")
    print(f"Elastic TTA val Dice:   {elastic_dice:.4f}")
    print(f"Crop TTA val Dice:      {crop_dice:.4f}")


if __name__ == "__main__":
    main()
