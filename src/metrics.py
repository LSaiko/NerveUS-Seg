import torch


def dice_score(logits, target, smooth=1e-5, threshold=0.5):
    """logits: raw model output (pre-sigmoid), target: 0/1 mask, same shape."""
    pred = (torch.sigmoid(logits) > threshold).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + smooth) / (union + smooth)).mean().item()


def iou_score(logits, target, smooth=1e-5, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


def demo():
    perfect_pred = torch.ones(1, 1, 4, 4) * 10  # logits -> sigmoid ~1
    target = torch.ones(1, 1, 4, 4)
    assert abs(dice_score(perfect_pred, target) - 1.0) < 1e-4
    assert abs(iou_score(perfect_pred, target) - 1.0) < 1e-4

    empty_pred = torch.ones(1, 1, 4, 4) * -10  # sigmoid ~0
    assert dice_score(empty_pred, target) < 0.01
    print("metrics.py self-check passed")


if __name__ == "__main__":
    demo()
