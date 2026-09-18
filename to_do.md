# To Do

## Methodological concern (highest priority)

We've now compared 13 checkpoints against the *same fixed* 80/20 val split
(picked once, first 20% of sorted pairs, never reshuffled). Selecting the
"best" of 13+ runs on one static validation set is a multiple-comparisons
problem — some of the shipped checkpoint's 0.6875 is likely that split's
noise, not a real quality edge over seed 7's 0.6856. Before trusting 0.6875
as *the* number:
- [ ] Re-evaluate the shipped checkpoint and seed 7 on a different val split
      (e.g. last 20% instead of first 20%, or k-fold) to see if the ranking
      holds.
- [ ] Consider reporting a mean +/- std over the 8 `ElasticTransform` seed
      runs (currently 0.6660-0.6875) instead of a single cherry-picked best,
      if this is ever used for anything beyond a portfolio demo.

## Multiscale TTA follow-up

Found a plain-Dice threshold (~0.68) where multiscale TTA flips from
helping to hurting, within the `ElasticTransform` recipe family (11
checkpoints). Worth strengthening before treating it as more than a
pattern-in-one-sample:
- [ ] Test multiscale TTA on the `RandomRotate90` and `RandomResizedCrop`
      recipes at multiple seeds each — right now the threshold claim rests
      entirely on one augmentation family.
- [ ] Try combining flip + multiscale TTA (average both) — not tried yet,
      might beat either alone on the low-scoring seeds.
- [ ] The `ElasticTransform` "earlier run" and `RandomResizedCrop` rows in
      the results table are marked "not tested" for multiscale — their
      checkpoints were deleted; would need retraining to fill the gap.

## Model quality

- [ ] Per-image error analysis: pull the worst-Dice validation samples and
      look at them (small nerve, occlusion, poor contrast?) to see if the
      failure mode is data-related rather than architecture-related.
- [ ] No true held-out test set exists (only the 80/20 train/val split) —
      the original Kaggle test images have no public ground truth, so this
      may not be fixable, but worth noting as a ceiling on how much the
      current numbers can be trusted as "real-world" performance.

## Not urgent / nice-to-have

- [ ] `visualize.py`'s `plot_predictions` always uses plain inference; could
      add a `predict_fn` param to visualize TTA predictions too.
- [ ] `requirements.txt` has no pinned versions — fine for a single-machine
      project, would matter if this needs to run somewhere else later.
