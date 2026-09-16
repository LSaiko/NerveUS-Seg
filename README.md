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
  over the original image plus horizontal/vertical flips

## Results

| Metric              | Value  |
|---------------------|--------|
| Val Dice            | 0.6752 |
| Val Dice (with TTA) | 0.6715 |

Best checkpoint from epoch 15/30, with `StepLR` decaying LR x0.5 every 4
epochs (0.6603 without the scheduler).

**TTA experiments, both negative:** flip-only TTA (h+v flip) is roughly
neutral (0.6715 vs. 0.6752 plain). Adding `RandomRotate90` to training and
testing full D4 TTA (4 rotations x flip) made things worse on both counts —
plain Dice dropped to 0.6738 and TTA Dice collapsed to 0.6384. Root cause:
the ResNet34 encoder's ImageNet-pretrained convolutions aren't actually
rotation-equivariant, so 90°-rotated inputs are more out-of-distribution for
it than in-distribution ultrasound images ever are — rotating at inference
time hurts more than the training-time augmentation helps. Reverted;
`dataset.py`/`predict.py` stay flip-only.

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
