# Strongest independent CNN + ViT: one joint30 experiment

This queue runs **one new30-epoch joint feature-fusion experiment**. It uses
the strongest measured independent CNN and independent ViT, rather than an
already fused model. It does not repeat the ViT search or independently train
either backbone first.

| Source | Selected checkpoint | Development Top-1 |
|---|---|---:|
| CNN | `runs/fusion_continuation_tiny/suite_20261106T194319_c2dd7a/training/cnn_head_continue/best.pt` |64.80%|
| ViT | `runs/vit_augreg_tiny/suite_20261107T182435_87170e/training/pretrain_vit_small_p8_augreg/best.pt` |50.18%|

CNN: EMA at local continuation epoch83, cumulative adaptation113, after an
original100-epoch CNN. Thus the selected weights reflect213 training epochs;
the source run itself finished230. The saved `CNNHeadControl.encoder.*` keys
are stripped exactly once and loaded strictly into the same pure CNN8.
ViT: Small/8 AugReg,12 layers,384 channels,6 heads,64 patch tokens, raw epoch273
from a completed300-epoch independent run.

Historical checkpoint SHA256 values are checked against metadata and stored
analysis identities. The source CNN hash begins `9a230237`; the ViT hash
begins `7728fa7c`. Source directories and checkpoints remain read only.

## Model and training

The same image passes through two independent trainable encoders:

`CNN -> 2048D pooled features`

`ViT -> 384D normalized CLS features`

The concatenation is classified by a2432-to-200 linear head. CNN columns and
bias inherit the strongest CNN classifier; ViT columns use small nonzero
initialization. No new cross-attention or nonlinear fusion block is added.
Both encoders and the fusion head update together, starting a fresh optimizer
and EMA for the new joint stage.

The fusion recipe matches the previous30-epoch joint experiment: batch64,
CNN/ViT peak learning rate5e-5, classifier5e-4, one warmup epoch, cosine decay,
the original augmentation, Mixup/CutMix and label smoothing, seed42. The new
stage deliberately uses the same augmentation-view indices101..130 as the
previous joint30, independently of how many epochs the selected CNN already
saw. Actual inherited training counts and augmentation-view indices are
recorded separately. This is a new joint stage, not an extension of an old
completed fusion schedule.

Best raw/EMA, actual last raw/EMA, epoch-zero and best actually trained scores
are kept separately. Resume preserves optimizer, EMA, scaler, scheduler, RNG,
source identities, horizons and training configuration.

## Analysis

The reports evaluate the selected CNN and independent ViT, probability
mixtures, joint best/last, per-class errors, paired corrections/harms and
three-way correctness. The previous65.25% joint checkpoint, when retained in
the source suite, is also restored completely (including its earlier CNN) and
reevaluated on the same images for a direct paired comparison. Branch masking
and training-probe-mean replacement
describe the trained model and do not replace standalone baselines.

The old mismatch control rolled features inside class-ordered batches and
was confounded. This run instead establishes a **global, label-blind random
derangement by image ID**, then substitutes the corresponding ViT features.
The mapping is fixed and has no image paired with itself. Actual same-class
donor frequency is reported after constructing the mapping. Wrong-image
results remain counterfactual diagnostics, not deployable model scores.

All selection results are development evidence. The10K reserved audit and
official validation are not evaluated. Full checkpoints/probability NPZs stay
on the VM. The email contains ordinary reports-only ZIP files, excluding
source scripts, credentials, datasets, weights and nested archives.

## Workbench command: email only, keep instance running

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
tmux new-session -A -s best_pair_joint
```

Inside tmux:

```bash
python -u run_best_pair_joint.py \
  --cnn-suite runs/fusion_continuation_tiny/suite_20261106T194319_c2dd7a \
  --vit-suite runs/vit_augreg_tiny/suite_20261107T182435_87170e \
  --data-dir ./data --num-workers 4 --epochs 30
```

The runner checks the source checkpoints, GPU/software, data and SMTP login,
without sending a start email. After training and analysis, one completion
mail with a report ZIP is attempted. The instance stays running on success,
training failure, packaging failure or email failure. There is **no managed
stop, OS shutdown, sudo or GCP stop-permission check** in this runner.

Add `--check-only` for preflight only. Existing ignored `email.txt`/Gmail
application credentials are used; credentials are not passed to the child or
included in the report archive. SMTP acceptance does not confirm inbox delivery.

Resume a new suite with:

```bash
python -u run_best_pair_joint.py \
  --resume runs/best_pair_joint_tiny/SUITE_DIRECTORY
```

Previous timing context: oldViT+originalCNN joint30 trained1.46h and reached
64.69%; newViT+originalCNN joint30 trained1.37h and reached65.25% (EMA2), ending
at64.02% raw/64.08% EMA. The full AugReg search/fusion suite took30.69h. These
are recorded past times, not a guarantee of this run's duration.
