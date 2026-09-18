# To Do

## Methodological concern (highest priority) — partially resolved

We compared 13 checkpoints against the *same fixed* 80/20 val split (first
20% of sorted pairs, never reshuffled) — a multiple-comparisons setup where
some of the "best" checkpoint's edge could just be that split's noise.

- [x] Re-evaluate two already-ranked seeds (7 and 123) on a different,
      non-overlapping val split (`--val-side end`, added to `train.py` /
      `predict.py`) to see if the ranking holds. **Result:** ranking held
      (seed 7 still beats seed 123) but the gap shrank from 0.0196 to
      0.0081 — partial confirmation the fixed split was inflating
      differences. Bigger finding: absolute Dice for *both* seeds jumped
      ~0.04-0.06 points higher under the new split (subjects are grouped by
      filename prefix, so a slice-based split holds out different subjects,
      not a random sample of frames — some subjects are just easier). Full
      writeup in README's "Validation-split robustness check".
- [ ] The shipped checkpoint (`best_unet.pth`, unseeded, predates
      `--val-side`) hasn't itself been re-evaluated this way since its exact
      training run isn't reproducible — only the two seeded runs were
      checked. Low priority: the pattern from seeds 7/123 likely generalizes,
      but this is technically still unverified for the actual shipped model.
- [ ] The real fix, not yet done: subject-grouped k-fold CV instead of a
      single slice-based split, so every fold's val set is randomly sampled
      across subjects rather than being a specific contiguous block. Would
      give a defensible mean +/- std instead of a single split-dependent
      number. Bigger lift (5x the training compute of one run) — worth
      doing before quoting 0.6875 (or any single number here) as "the"
      model's real-world performance.

## Multiscale TTA follow-up

Found a plain-Dice threshold (~0.68) where multiscale TTA flips from
helping to hurting, within the `ElasticTransform` recipe family (11
checkpoints, all on the `start` val split). The seed 7/123 `end`-split
retrain (above) supports this in *relative* terms — the lower scorer of
the pair still gained from multiscale TTA and the higher scorer didn't,
even though both scored ~0.05 higher in absolute terms — so it's more a
comparative pattern ("weaker of two similar models gets helped more") than
a fixed absolute threshold. Worth strengthening before treating either
framing as more than a pattern-in-one-sample:
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
