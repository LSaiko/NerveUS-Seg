# Ultrasound Nerve Segmentation

Binary segmentation of nerve structures in ultrasound images using a U-Net
with a ResNet34 encoder.

## Problem

Given an ultrasound image, predict a pixel-level mask of the nerve region.
Accurate automatic segmentation reduces the manual annotation burden in
procedures like nerve-block placement.

## Dataset

[Kaggle Ultrasound Nerve Segmentation](https://www.kaggle.com/c/ultrasound-nerve-segmentation) —
grayscale ultrasound images paired with binary nerve masks.

## Architecture

U-Net (via [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch))
with a ResNet34 encoder pretrained on ImageNet. Input images are resized to
128x128 and expanded to 3 channels to match the pretrained encoder. Skip
connections between encoder and decoder preserve fine boundary detail lost
during downsampling.

## Training

- Loss: combined Dice + BCE (`src/loss.py`) — BCE stabilizes per-pixel
  gradients, Dice directly optimizes mask overlap.
- Optimizer: Adam, lr=1e-3, `StepLR` decay (x0.5 every 4 epochs)
- Epochs: 30
- Augmentation: horizontal flip, `ElasticTransform`, brightness/contrast
  jitter, ImageNet normalization (`src/dataset.py`)
- Checkpointing: best model saved by validation Dice (`src/train.py`)
- Reproducibility: `--seed` (default 42) fixes `random`/`numpy`/`torch` and
  disables cuDNN's non-deterministic algorithm selection; Albumentations 2.x
  uses its own independent RNG so the seed is also passed into
  `get_train_transform(seed=...)` — verified bit-for-bit reproducible in
  `src/test_seed.py`, and confirmed at full scale: two independent 30-epoch
  `--seed 42` runs produced identical loss/Dice at every epoch and
  bit-identical final model weights (checked tensor-by-tensor, since the
  `.pth` file's own zip container metadata can differ even when the weights
  don't)
- Inference: test-time augmentation (`src/predict.py`) — flip, elastic,
  crop, and multiscale variants implemented; see Results for which (if any)
  helps a given checkpoint

## Results

| Metric           | Value      |
|--------------------|------------|
| Val Dice (plain)   | **0.6875** |

Shipped checkpoint (`models/best_unet.pth`) is from epoch 11/30, trained
with `ElasticTransform` + `StepLR`. No TTA variant improves on this
checkpoint's plain score — see the TTA section below — so plain inference is
the recommended path for the current model.

**Training-run variance turned out to be the biggest lever tried.** The
same `ElasticTransform` recipe was trained twice: an earlier run scored
0.6701 plain, this run scored 0.6875 — a 0.017 spread from nothing but
different weight init / batch order, since none of the experiments below
fixed a seed. `train.py` now seeds `random`/`numpy`/`torch`/cuDNN and
Albumentations' own RNG (`--seed`, default 42), so this specific spread is
no longer reproducible by accident — but it's also not eliminated: a
*different* seed still lands somewhere in that same range, it's just fixed
per-run now instead of drifting silently. That's a bigger effect than any
single augmentation or TTA choice tested below. The table is kept for the
TTA findings; keep the variance in mind when comparing rows.

**TTA experiments — nine training-run/TTA pairings tried across two rounds.
Multiscale TTA won on 3 of the 6 checkpoints it was tried on and lost on the
other 3 — genuinely a coin flip, not a reliable technique:**

| Training run                          | Plain      | Flip TTA | Multiscale TTA | Other matching TTA  |
|-----------------------------------------|------------|----------|-----------------|----------------------|
| none (original baseline)                | 0.6752     | 0.6715   | 0.6777          | 0.6582 (elastic) / 0.6571 (crop) |
| + `RandomRotate90`                      | 0.6719     | 0.6703   | 0.6716          | 0.6384 (D4 rotation) |
| + `ElasticTransform` (earlier run)      | 0.6701     | 0.6736   | not tested      | 0.6486 (elastic)     |
| + `RandomResizedCrop`                   | 0.6496     | 0.6433   | not tested      | 0.6694 (5-crop)      |
| + `ElasticTransform` (**shipped run**)  | **0.6875** | 0.6788   | 0.6855          | 0.6666 (elastic) / 0.6619 (crop) |
| + `ElasticTransform` (`--seed 42`)      | 0.6700     | 0.6719   | 0.6783          | 0.6511 (elastic) / 0.6465 (crop) |
| + `ElasticTransform` (`--seed 1`)       | 0.6706     | 0.6646   | 0.6811          | 0.6585 (elastic) / 0.6542 (crop) |
| + `ElasticTransform` (`--seed 7`)       | 0.6856     | 0.6824   | 0.6814          | 0.6787 (elastic) / 0.6725 (crop) |

Adding rotation, elastic, or crop augmentation to training doesn't
consistently help plain accuracy on its own (crop was worst on average — a
small nerve region can get cropped out entirely at 72% scale); the shipped
run's 0.6875 is more a favorable roll of training variance than proof
elastic augmentation reliably helps — the same recipe with three other seeds
landed at 0.6700, 0.6706, and 0.6856 (closest, but still short). The
pretrained ResNet34 encoder isn't equivariant to rotation or non-rigid
elastic warps, so rotation/elastic TTA consistently make things worse than
their own model's plain score on every checkpoint tried. 5-crop TTA is the
one case (on the crop-trained model) where matching TTA clearly helped its
own model (+2 points) without beating the untouched baseline outright.

**Multiscale TTA is checkpoint-dependent, not a free win.** Across the 6
checkpoints it was tried on: helped 3 (baseline +0.0025, seed 42 +0.0083,
seed 1 +0.0105) and hurt 3 (rotation -0.0003, shipped run -0.0020, seed 7
-0.0042) — no pattern by recipe, seed, or plain-Dice level explains which
side a checkpoint lands on. It's not a property of the training
augmentation or a reliably-transferable trick; it's essentially a coin flip
per checkpoint. Always check `python src/predict.py` on the specific
checkpoint you're shipping rather than assuming a TTA result carries over.

Implementation note: crop and multiscale TTA need no approximate inverse —
each is a deterministic resize round-trip (crop: resize up, resize back down
to its exact pixel region; multiscale: resize the whole image down and back
up), unlike elastic's non-rigid warp, which only has an approximate inverse
(round-trip error ~0.07 for crop/multiscale on synthetic noise vs. ~0.23 for
elastic on a real image, verified against a synthetic identity model in
`predict.py`'s dev history).

All four `predict_tta_*` functions (flip, elastic, crop, multiscale) live in
`predict.py` — run `python src/predict.py` to check all of them against
whatever checkpoint you're evaluating, since no single one wins consistently.

![predictions](predictions.png)

## How to run

```bash
pip install -r requirements.txt
python src/train.py --data-dir data/ultrasound-nerve-segmentation --epochs 30 --seed 42
```

Data is expected in the Kaggle layout: `<data-dir>/image/{id}_{n}.tif` paired
with `<data-dir>/mask/{id}_{n}_mask.tif`. `--seed` (any int, default 42) makes
the run reproducible; different seeds land anywhere in roughly the 0.67-0.69
range seen in the seed sweep above, so don't expect one run to reproduce the
shipped 0.6875 exactly unless you match its exact (unseeded) run.

Evaluate with test-time augmentation (compares plain vs. TTA val Dice):

```bash
python src/predict.py --checkpoint models/best_unet.pth
```

Generate side-by-side prediction visualizations:

```python
from src.visualize import plot_predictions
plot_predictions(model, val_dataset, n=5, device="cuda")
```

## Project structure

```
src/
  dataset.py    NerveDataset + Albumentations pipelines
  loss.py       DiceBCELoss
  metrics.py    dice_score, iou_score
  train.py      training loop with best-checkpoint saving
  predict.py    TTA inference/eval (multiscale, flip, elastic, crop variants)
  visualize.py  plot_predictions for image/GT/prediction comparisons
  test_seed.py  self-check that --seed makes training reproducible
models/         saved checkpoints
data/           raw/ and processed/ image-mask pairs
```
