import argparse
import glob
import os

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from dataset import NerveDataset, get_train_transform, get_val_transform
from loss import DiceBCELoss
from metrics import dice_score


def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_dice = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_dice += dice_score(logits, masks) * imgs.size(0)
    return total_dice / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed", help="dir with images/ and masks/ subfolders")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--out", default="models/best_unet.pth")
    args = p.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.data_dir, "images", "*.png")))
    masks = sorted(glob.glob(os.path.join(args.data_dir, "masks", "*.png")))
    assert len(imgs) == len(masks) and len(imgs) > 0, "no matching image/mask pairs found"

    n_val = int(len(imgs) * args.val_split)
    train_imgs, val_imgs = imgs[n_val:], imgs[:n_val]
    train_masks, val_masks = masks[n_val:], masks[:n_val]

    train_ds = NerveDataset(train_imgs, train_masks, get_train_transform())
    val_ds = NerveDataset(val_imgs, val_masks, get_val_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = DiceBCELoss()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_dice = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_dice = evaluate(model, val_loader, device)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), args.out)
        print(f"Epoch {epoch}: loss={train_loss:.4f} val_dice={val_dice:.4f}")

    print(f"Best val Dice: {best_dice:.4f} -> saved to {args.out}")


if __name__ == "__main__":
    main()
