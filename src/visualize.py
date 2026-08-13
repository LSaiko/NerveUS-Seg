import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics import dice_score

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def plot_predictions(model, dataset, n=5, device="cpu", out_path="predictions.png"):
    model.eval()
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, axes = plt.subplots(len(idxs), 3, figsize=(9, 3 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]

    for row, idx in enumerate(idxs):
        img, mask = dataset[idx]
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > 0.5).float().cpu()
        d = dice_score(logits.cpu(), mask.unsqueeze(0))

        base = _denormalize(img)
        axes[row, 0].imshow(base)
        axes[row, 0].set_title("image")

        gt_overlay = base.copy()
        gt_overlay[mask[0].numpy() > 0.5] = [0, 1, 0]
        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].set_title("ground truth (green)")

        pred_overlay = base.copy()
        pred_overlay[pred[0, 0].numpy() > 0.5] = [1, 0, 0]
        axes[row, 2].imshow(pred_overlay)
        axes[row, 2].set_title(f"prediction (red) dice={d:.2f}")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return out_path
