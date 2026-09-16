import argparse

import torch

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


@torch.no_grad()
def evaluate_tta(model, dataset, device, threshold=0.5):
    model.eval()
    total_dice = 0.0
    for i in range(len(dataset)):
        img, mask = dataset[i]
        prob = predict_tta(model, img, device)
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

    tta_dice = evaluate_tta(model, val_ds, device)
    print(f"Plain val Dice: {plain_dice:.4f}")
    print(f"TTA val Dice:   {tta_dice:.4f}")


if __name__ == "__main__":
    main()
