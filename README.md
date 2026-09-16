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
- Inference: test-time augmentation (`src/predict.py`) averages predictions
  over the original image plus horizontal/vertical flips (rotation, elastic,
  and 5-crop TTA were also tried — see Results for the full comparison)

## Results

| Metric              | Value  |
|---------------------|--------|
| Val Dice            | 0.6752 |
| Val Dice (with TTA) | 0.6715 |

Best checkpoint from epoch 15/30, with `StepLR` decaying LR x0.5 every 4
epochs (0.6603 without the scheduler).

**TTA experiments — four training/TTA pairings tried, none beat plain
inference on the untouched baseline:**

| Training augmentation      | Plain      | Flip TTA | Matching TTA        |
|-----------------------------|------------|----------|----------------------|
| none (baseline)             | **0.6752** | 0.6715   | 0.6582 (elastic) / 0.6571 (crop) |
| + `RandomRotate90`          | 0.6738     | —        | 0.6384 (D4 rotation) |
| + `ElasticTransform`        | 0.6701     | 0.6736   | 0.6486 (elastic)     |
| + `RandomResizedCrop`       | 0.6496     | 0.6433   | 0.6694 (5-crop)      |

Adding rotation, elastic, or crop augmentation to training consistently hurts
plain accuracy (crop worst of all — a small nerve region can get cropped out
entirely at 72% scale, which is a much harder training signal than the others).
The pretrained ResNet34 encoder isn't equivariant to any of these transforms;
its ImageNet-trained convolutions expect a fairly fixed input scale and
orientation. Matching TTA to the training augmentation is a mixed bag:
rotation and elastic TTA make things *worse* than their own model's plain
score (out-of-distribution inputs at inference outweigh any invariance
gained), but 5-crop TTA is the one case where matching TTA genuinely helps
its own model (0.6496 → 0.6694, +2 points) — likely because crops are still
"natural-looking" sub-images to the encoder, unlike a 90°-rotated or
elastically-warped one. Even so, that combination still falls short of the
untouched baseline's plain score.

Implementation note: crop TTA needs no approximation — each crop is resized
up for the model then resized back down to its exact pixel region, a
deterministic placement (round-trip error ~0.07 on synthetic noise, far
below elastic's ~0.23 on real images, since elastic's "inverse" is only an
approximation of a true inverse for a non-rigid warp with no closed form).

Flip-only TTA remains the best default: a true involution, no training
augmentation needed to justify it, roughly breaks even on the baseline model.
Shipped model stays flip-only training + optional flip TTA; the other
`predict_tta_*` functions in `predict.py` are kept for reference.

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
  predict.py    test-time augmentation (flip averaging) inference/eval
  visualize.py  plot_predictions for image/GT/prediction comparisons
models/         saved checkpoints
data/           raw/ and processed/ image-mask pairs
```
