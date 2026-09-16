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
different weight init / batch order (no seed is fixed in `train.py`). That's
larger than the effect of any single augmentation or TTA choice tested
below. Below is the full history, kept for the TTA findings, but keep the
variance in mind when comparing rows — the same recipe re-run can land
anywhere in roughly that range.

**TTA experiments — six training-run/TTA pairings tried. Multiscale TTA
beat plain inference exactly once (on the original untouched baseline), and
was neutral-to-negative on every other checkpoint, including the current
shipped one:**

| Training run                          | Plain      | Flip TTA | Multiscale TTA | Other matching TTA  |
|-----------------------------------------|------------|----------|-----------------|----------------------|
| none (original baseline)                | 0.6752     | 0.6715   | 0.6777          | 0.6582 (elastic) / 0.6571 (crop) |
| + `RandomRotate90`                      | 0.6719     | 0.6703   | 0.6716          | 0.6384 (D4 rotation) |
| + `ElasticTransform` (earlier run)      | 0.6701     | 0.6736   | not tested      | 0.6486 (elastic)     |
| + `RandomResizedCrop`                   | 0.6496     | 0.6433   | not tested      | 0.6694 (5-crop)      |
| + `ElasticTransform` (**shipped run**)  | **0.6875** | 0.6788   | 0.6855          | 0.6666 (elastic) / 0.6619 (crop) |

Adding rotation, elastic, or crop augmentation to training doesn't
consistently help plain accuracy on its own (crop was worst on average — a
small nerve region can get cropped out entirely at 72% scale); the shipped
run's 0.6875 is more a favorable roll of training variance than proof
elastic augmentation reliably helps. The pretrained ResNet34 encoder isn't
equivariant to rotation or non-rigid elastic warps, so rotation/elastic TTA
consistently make things worse than their own model's plain score on every
checkpoint tried. 5-crop TTA is the one case (on the crop-trained model)
where matching TTA clearly helped its own model (+2 points) without beating
the untouched baseline outright.

**Multiscale TTA is checkpoint-dependent, not a free win.** It genuinely
improved the original baseline (0.6752 → 0.6777, averaging predictions at
96/112/128/144/160px, each resized back to 128px) but was a wash or slightly
negative on every other checkpoint tested (rotation: 0.6719 → 0.6716;
shipped elastic run: 0.6875 → 0.6855). Whatever gave that first checkpoint
its scale tolerance isn't a property of the training augmentation — it's
closer to an accident of that specific checkpoint's weights. Worth checking
per-checkpoint, never assume it transfers.

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
