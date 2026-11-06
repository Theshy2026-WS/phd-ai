# Continue the learned CNN/ViT fusion and test its contribution

This is a new finite suite. It reads the completed
`runs/vit_followup_tiny/suite_20261105T033555_5ba88b` as immutable evidence and
initialization. It never edits that suite's30-epoch horizon or checkpoints.
There is no shallow64/4096-token global-attention job and no new ViT pretraining.

## Queue and training budgets

| Job | Initialization | New epochs | Purpose |
|---|---|---:|---|
| `dual_continue` | Completed joint30 `last.pt`, whose raw selection score was64.69% |100|Test further training of the learned fusion|
| `cnn_head_control` | Original CNN100 best |30|Match trunk5e-5 and classifier5e-4, including the faster classifier used by the old fusion|
| `cnn_head_continue` | Last checkpoint of the matched CNN30 control |100|Give the CNN the same additional continuation phase|
| `dual_frozen_head` | Original independent CNN100 and ViT100 best |30|Train only the same concatenated-feature classifier; keep both feature extractors fixed|

The main joint model receives **100 additional epochs**, for130 joint epochs in
total, after the separate CNN and ViT pretraining. The queue has260 new
model-epochs, including the matched CNN control. No time cutoff, early stopping,
seed sweep or expensive native64 attention is scheduled. Independent jobs can
continue after one failure; dependent CNN continuation needs its control30 to
finish successfully. Incomplete results are labeled accordingly.

## What continuation preserves

The main continuation loads the original raw model, AdamW moment/step state,
EMA weights, AMP scaler and random-number states from `dual_fusion/last.pt`.
The optimizer groups must retain the encoder/ViT/fusion role ordering and shapes.
The original EMA and raw models are measured separately at the new stage's
epoch0. Original source files are hashed before and after the suite.

The old cosine schedule has finished. This new phase explicitly resets the
scheduler while preserving optimizer moments: peak CNN/ViT LR5e-5, head5e-4,
two warmup epochs from2% of those peaks, then cosine decay to2%. It is a
`30 + additional100` procedure, not the same experiment as a fresh130-epoch
cosine schedule. The new local epochs1..100 correspond to joint epochs31..130
and augmentation epochs131..230. The matched CNN first runs augmentation
epochs101..130 and then the same131..230 continuation phase.

For the frozen-feature control, CNN and ViT are independently pretrained
checkpoints before joint adaptation. Both encoder parameters and their BN/
normalization buffers are fixed; backbone dropout is disabled in evaluation
mode. The new fusion classifier alone learns for30 epochs. Frozen backbone
weights/buffers are also copied exactly into EMA to avoid floating-point drift.
This is an explicit fixed-representation training procedure, not a claim to
change just one isolated mathematical variable of the full joint model.

The old augmentation, Mixup/CutMix, label smoothing, batch64, seed42 and
80K/10K/10K split are retained. Bias/norm/CLS/position parameters receive the
recorded no-decay handling. Audit and official validation are not evaluated.

## Analysis delivered automatically

Training reports retain initial, best, best actually trained, actual last raw
and last EMA results, with both local and inherited epoch counts. Best and last
receive clean80K training and10K selection diagnostics. Costs include inherited
pretraining and the new phase separately. Better scores are not assumed.

The feature-analysis material computes the following before notification:

- Independent CNN/ViT probability mixtures at predeclared ViT weights
  `0, .05, .1, .2, .25, .5, .75, 1`.
- Paired corrections/harms versus original CNN, original joint30 and the
  static `0.8 CNN +0.2 ViT` reference.
- Exact eight-cell CNN/independent-ViT/fusion correctness intersections.
- Fusion with each feature source masked, plus replacing ViT features with
  their mean estimated only from a fixed clean training probe.
- Separate CNN/ViT logit magnitudes and correct-class margin contributions.
- Class summaries, confusion and small corrected/harmed galleries.

Mean-feature replacement tests whether the ViT effect depends on the matching
image beyond a mean shift; it is an inference intervention on co-adapted
weights, not an independently trained model. Feature norms are not percentages
of predictive contribution. Predeclared probability weights are still compared
on the development selection set; a selected maximum is not a final-test result.

The message contains one ordinary report ZIP with `training_analysis/` and
`feature_analysis/`. Aggregate mixture and three-way statistics are retained
even if bulky per-image tables or galleries must be omitted. Checkpoint and
full probability NPZ files remain on the VM. Scripts, nested archives and
credentials are excluded from the email. Source hashes preserve provenance.

## Workbench command

After the earlier experiment has finished:

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
tmux new-session -A -s fusion_continue
```

Then run:

```bash
python -u run_fusion_continuation_overnight.py \
  --source-suite runs/vit_followup_tiny/suite_20261105T033555_5ba88b \
  --data-dir ./data --num-workers 4 \
  --continue-epochs 100 --control-epochs 30 \
  --complete-all --fallback-os-shutdown
```

The wrapper checks the source files, GPU/software, data, SMTP authentication
and stop permissions before training. It sends no start or preflight email.
Add `--check-only` to perform only those checks. Existing ignored `email.txt`
or exported Gmail application credentials are supported; credentials are not
passed to the training child or put in archives.

After all scheduled work and analysis, one completion email is attempted,
followed by managed Workbench stop. The explicit OS fallback is retained.
Packaging/email failure still requests stop and leaves local results intact;
SMTP acceptance does not prove inbox delivery. A hung process is not completed
and has no hard watchdog. Native64 global-attention jobs are never launched.

Resume the new suite, not the old30-epoch suite:

```bash
python -u run_fusion_continuation_overnight.py \
  --resume runs/fusion_continuation_tiny/SUITE_DIRECTORY \
  --fallback-os-shutdown
```

Resume keeps the recorded source, initialization, GPU/software and horizons.
A separate `--smoke` uses one epoch per job and must not be extended or
presented as the formal260-epoch queue.
