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
- Augmentation: horizontal flip, brightness/contrast jitter, ImageNet
  normalization (`src/dataset.py`)
- Checkpointing: best model saved by validation Dice (`src/train.py`)
- Inference: test-time augmentation (`src/predict.py`) — multiscale TTA is
  the recommended default (see Results); flip, elastic, and 5-crop TTA were
  also tried for comparison

## Results

| Metric                     | Value      |
|-----------------------------|------------|
| Val Dice (plain)            | 0.6752     |
| Val Dice (multiscale TTA)   | **0.6777** |

Best checkpoint from epoch 15/30, with `StepLR` decaying LR x0.5 every 4
epochs (0.6603 without the scheduler).

**TTA experiments — five variants tried; multiscale is the only one that
beats plain inference on the untouched baseline, with no retraining:**

| Training augmentation      | Plain      | Flip TTA | Multiscale TTA | Other matching TTA  |
|-----------------------------|------------|----------|-----------------|----------------------|
| none (baseline)             | 0.6752     | 0.6715   | **0.6777**      | 0.6582 (elastic) / 0.6571 (crop) |
| + `RandomRotate90`          | 0.6738     | —        | —               | 0.6384 (D4 rotation) |
| + `ElasticTransform`        | 0.6701     | 0.6736   | —               | 0.6486 (elastic)     |
| + `RandomResizedCrop`       | 0.6496     | 0.6433   | —               | 0.6694 (5-crop)      |

Adding rotation, elastic, or crop augmentation to training consistently hurts
plain accuracy (crop worst of all — a small nerve region can get cropped out
entirely at 72% scale, which is a much harder training signal than the others).
The pretrained ResNet34 encoder isn't equivariant to any of these transforms;
its ImageNet-trained convolutions expect a fairly fixed input scale and
orientation. Matching TTA to the training augmentation is a mixed bag:
rotation and elastic TTA make things *worse* than their own model's plain
score (out-of-distribution inputs at inference outweigh any invariance
gained), but 5-crop TTA is one case where matching TTA genuinely helps its
own model (0.6496 → 0.6694, +2 points) — still short of the untouched
baseline, though.

**Multiscale TTA is the actual win**, on the baseline model with *no
retraining at all*: average predictions made at 96/112/128/144/160px, each
resized back to 128px (0.6752 → 0.6777). Unlike rotation/elastic, a CNN
encoder tolerates moderate scale changes reasonably well since resizing
doesn't change apparent content the way a 90° rotation or a warp does — but
it's scale-range-sensitive, not free of the same failure mode at the
extremes: a wider range (80/128/176px) drops to 0.6422, likely because 80px
loses too much of the small nerve region's detail. Narrow-to-moderate ranges
centered on the training resolution (128px) are the sweet spot.

Implementation note: both crop and multiscale TTA need no approximate
inverse — each is a deterministic resize round-trip (crop: resize up, resize
back down to its exact pixel region; multiscale: resize the whole image down
and back up), unlike elastic's non-rigid warp, which only has an
approximate inverse (round-trip error ~0.07 for crop/multiscale on synthetic
noise vs. ~0.23 for elastic on a real image, verified against a synthetic
identity model in `predict.py`'s dev history).

**Recommended default: multiscale TTA**, `predict_tta_multiscale` in
`predict.py`. The other `predict_tta_*` functions (flip, elastic, crop) are
kept for reference/comparison but aren't the recommended path.

![predictions](predictions.png)

## How to run

```bash
pip install -r requirements.txt
python src/train.py --data-dir data/ultrasound-nerve-segmentation --epochs 30
```

Data is expected in the Kaggle layout: `<data-dir>/image/{id}_{n}.tif` paired
with `<data-dir>/mask/{id}_{n}_mask.tif`.

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
models/         saved checkpoints
data/           raw/ and processed/ image-mask pairs
```
