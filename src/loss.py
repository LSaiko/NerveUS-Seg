import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        pred = torch.sigmoid(logits)
        inter = (pred * target).sum()
        dice = 1 - (2 * inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return bce + dice


def demo():
    loss_fn = DiceBCELoss()
    logits = torch.full((1, 1, 4, 4), 10.0)
    target = torch.ones(1, 1, 4, 4)
    loss = loss_fn(logits, target).item()
    assert loss < 0.1, f"expected near-zero loss for correct prediction, got {loss}"
    print("loss.py self-check passed")


if __name__ == "__main__":
    demo()
