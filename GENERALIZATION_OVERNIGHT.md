# Tiny ImageNet: investigate the gap, then test bounded remedies

The current parent fits the clean80K training images at99.97% and scores
64.28% on the held-out10K selection set. This is a real generalization gap.
It is not, by itself, evidence that the loaders use different datasets.
Both sets are stratified subsets of the same Tiny ImageNet training folder,
with the same class-index mapping and clean preprocessing. A semantic label
mistake or pipeline defect still needs an explicit audit rather than inference
from the size of the gap.

The last study established that direct full8 inference pooling, train-only BN
recalibration, a residual U decoder, and mild Gaussian noise did not improve
Top1 under those tested settings. Other readout phases were much less reliable
on held-out images. This motivates controlled readout/optimization tests;
it does not establish the entire cause or promise a particular accuracy.

## What runs overnight

The default total active child budget is **12hours**, including audit and
evaluation. A **45minute final-analysis reserve** is excluded from new training.
Budget state is persisted every30seconds and retained on interruption/resume.
The job stops early when all work finishes. It never burns GPU time just to
fill the budget.

### 1. Source, content and augmentation audit

`data_audit/` examines the actual80K training and10K selection image files:

- Verify their shared source folder, image-ID disjointness, path/label/class
  mapping agreement, per-class counts, decodability and native64x64 dimensions.
- Hash compressed files and decoded RGB pixels, report exact cross-set
  duplicates and conflicting labels. Coarse perceptual-hash bucket matches
  are candidates only, not proof of duplicate or semantically identical images.
- Compare brightness, color, dimensions and per-class descriptive statistics.
  Similar aggregate statistics do not prove semantic distribution equivalence.
- Replay32 identical source images through the clean training and selection
  views and require identical tensors, targets and IDs.
- Evaluate a fixed4K clean training probe and10K selection images under clean,
  flip, four reflective2-pixel shifts, crop/flip only, RandAugment/flip only,
  and the inherited full augmentation. These are inference diagnostics;
  selection images are never used to train or fit augmentation parameters.

Concrete decoding, mapping, dimension or same-image preprocessing failures
set `training_allowed=False` and block new training. The workflow does not
delete data or relabel images automatically. Duplicate candidates or color
statistics differences alone do not stop training.

### 2. Actual last checkpoints from the previous screen

`previous_last/` evaluates the three saved epoch30 `last.pt` models on clean
80K/selection10K and compares each with the original parent. The previous
selected `best.pt` was epoch0, so reusing its class/error reports cannot
explain what the trained U models changed.

For each U model, a hook captures original-head and correction logits. The
analysis compares the complete output with removal of only the correction
at inference, reports correction/base logit RMS, changed predictions and
corrections/harms. This is an intervention on its co-adapted encoder, not a
newly trained baseline. Original checkpoint files remain unchanged. If the
old fusion suite is absent, its report explicitly says unavailable and the
new experiments can still run.

### 3. Matched training remedies

Every job retains seed42 and the original80/10/10 split. All encoder weights
are trainable. Parameter count stays equal to CNN8; no bigger decoder is added.

| Queue | Job | Initialization | Horizon | Question |
|---:|---|---|---:|---|
|1|`warm_control`|Original CNN8 best|30|New short-schedule control, AdamW peak1e-4|
|2|`warm_full8`|Same parent|30|Can full8 pooling work when trained that way?|
|3|`warm_phasebalanced`|Same parent|30|Does directly supervising all four grids improve stability?|
|4|`warm_gentle`|Same parent|30|Does less destructive crop/RA and removal of CutMix help?|
|5|`warm_fp32_control`|Same parent|30|Precision-matched control for SAM|
|6|`warm_sam`|Same parent|30|Does a sharpness-aware objective help?|
|7|`fresh_control`|Random seed42|100|Matched fresh-training reference|
|8|`fresh_phasebalanced`|Identical random initialization|100|Does supervision help when introduced before memorization?|
|9|`fresh_sam`|Identical random initialization|100, budget permitting|Exploratory early SAM; FP32 versus AMP limitation applies|

Warm jobs use a fresh optimizer, peak1e-4, one warmup epoch and cosine decay
to1e-6. They do not resume the old optimizer. Fresh jobs use the inherited
5e-4 peak, two warmup epochs and cosine decay to1e-5. Fresh results must be
compared with fresh controls at matching horizons, rather than with a model
that already received100 training epochs.

Full8 normalizes and pools the complete feature map. Phase-balanced training
uses `0.5*CE(full8) + 0.125*sum(CE(four phases))` on the same soft targets.
The head BN is called once on the full map, avoiding four biased running-stat
updates. This is a mechanism hypothesis, not a claimed new method.

Gentle augmentation changes crop scale to.85..1, RandAugment to1operation/M3,
and disables CutMix while retaining Mixup.2/probability.5, smoothing.1 and
weight decay.05. It tests a recipe bundle; it does not isolate all three
augmentation factors. The baseline already had regularization, so this is
not presented as fixing a previously unregularized pipeline.

SAM uses rho.05 and AdamW, two FP32 passes on the identical already-mixed
batch/targets. Perturbed weights are restored exactly even on failure. The
second pass uses batch normalization without updating running statistics or
counters again. `warm_fp32_control` matches precision and learning rate;
the fresh SAM comparison is exploratory because its fresh control uses AMP.
This is a SAM-inspired AdamW experiment, not an exact author-code reproduction.

Every job also maintains an EMA (decay.999) of parameters and floating BN
buffers; integer counters are copied. Raw and EMA validation metrics are
recorded every epoch. Selection Top1 chooses raw/EMA best checkpoints,
including epoch0, with strict improvement and earlier/raw ties retained.
This increases development-set comparisons and does not provide a final test
estimate. Clean4K probes run at epoch0/1/every5/final, never choosing a model.

Best and **actual last raw** checkpoints each receive full clean80K/selection
10K class/confusion/error analysis. If best remains epoch0, trained-model
errors are still retained. The EMA last score is recorded separately. The
criterion is improved held-out performance, not merely a reduced training
accuracy or smaller gap.

### Budget and recovery

The cutoff is cooperative: it is checked between epochs with a small estimate
for the next epoch. One in-flight epoch or complete evaluation can finish
after the cutoff; the reserve covers normal final evaluation and packaging.
This is not a hard kill watchdog for a hung GPU, network call or loader.
Data and previous-model audits each have45minute stage limits. Budget-limited
jobs are checkpointed and labeled incomplete/deferred rather than successful
full runs. Independent model failures do not discard other authorized work.

Resume retains the original source, parent, dataset, software/GPU, horizons,
and total active-time budget. It continues interruptions within that remaining
budget; it does not create another12hours after an already exhausted run.

## Workbench command

Use the existing environment and original spatial checkpoint. First:

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
tmux new-session -A -s generalization12h
```

Skip the last line if already inside tmux. Then:

```bash
python -u run_generalization_overnight.py \
  --parent-run runs/spatial_tiny/suite_20261103T175622_a19c86/cnn8_seed42 \
  --previous-fusion runs/mechanism_tiny/suite_20261104T014344_7caf47/fusion30 \
  --data-dir ./data --num-workers 4 \
  --warm-epochs 30 --fresh-epochs 100 --max-hours 12 \
  --fallback-os-shutdown
```

Preflight validates GPU, data/checkpoint, SMTP login and stop permission without
sending an email. Add `--check-only` to stop after those checks. Existing email
credentials and recipient are reused. Ctrl+B then D detaches tmux.

After all work ends or fails, the wrapper packages **three** materials
(`data_audit_analysis.tar.gz`, `previous_last_analysis.tar.gz`,
`training_analysis.tar.gz`) into one completion email attachment, then requests
managed Workbench stop with optional OS fallback. Budget-limited results are
marked `BUDGET LIMIT / PARTIAL`. Credentials/datasets/checkpoints/NPZ are not
attached. Large per-image files are compacted with an omission manifest;
full outputs remain on disk. Packaging/mail failure still reaches the stop
request after the child has started.

```bash
python -u run_generalization_overnight.py \
  --resume runs/generalization_tiny/SUITE_DIRECTORY \
  --fallback-os-shutdown
```

## Related primary sources and what they do establish

- [Shcherbina, Tiny ImageNet Challenge, Stanford CS231n2016](https://cs231n.stanford.edu/reports/2016/pdfs/401_Report.pdf),
  Table2: a9-layer ConvNet has98.8% training and45.5% validation accuracy.
  This is a course report with an older/different setup, not a peer-reviewed
  matched baseline. It illustrates a large gap without proving a dataset bug.
- [Guo et al., ALPS, IJCAI2021](https://www.ijcai.org/proceedings/2021/0343.pdf),
  Table4: Tiny ImageNet/PreActResNet18 standard-classification accuracies are
  60.95% ERM,62.48% dropout,61.59% smoothing,62.95% Mixup,63.77% ALPS.
  They train100K images with SGD and use official validation; our80K/10K and
  AdamW protocol differs. These are context, not direct performance claims.
- [Foret et al., SAM, ICLR2021](https://arxiv.org/abs/2010.01412)
  motivates optimizing a neighborhood of parameters instead of relying only
  on low training loss. It does not guarantee this Tiny ImageNet fix works.
- [Sarfi et al., SEAL, CVPR2023](https://arxiv.org/abs/2304.04858)
  studies iterative generalization improvements on Tiny ImageNet. Its long
  multi-generation schedule is not being claimed or reproduced overnight.

None of these references imply a64% data ceiling, a guarantee that gap can be
closed, or evidence that all wrong labels can be identified automatically.
