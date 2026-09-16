import torch
from torch.utils.data import DataLoader

from dataset import NerveDataset, get_train_transform
from loss import DiceBCELoss
from train import build_model, find_pairs, set_seed, train_one_epoch


def demo():
    pairs = find_pairs("data/ultrasound-nerve-segmentation")
    imgs, masks = zip(*pairs[:16])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(seed):
        set_seed(seed)
        ds = NerveDataset(list(imgs), list(masks), get_train_transform(seed=seed))
        dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
        model = build_model().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        return train_one_epoch(model, dl, opt, DiceBCELoss(), device)

    a, b, c = run(42), run(42), run(123)
    assert a == b, f"same seed should reproduce exactly: {a} != {b}"
    assert a != c, "different seeds should not coincidentally match"
    print("test_seed.py self-check passed")


if __name__ == "__main__":
    demo()
