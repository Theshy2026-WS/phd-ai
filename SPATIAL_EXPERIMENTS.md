# Tiny ImageNet: paired CNN downsampling study

This experiment follows the clean baseline diagnosis. It is a development
study, not a new model claim or a final benchmark. It trains two CNNs from
scratch for **100 epochs each**. It neither extends the old checkpoint nor
runs the optional `longer200` control.

## What the existing evidence establishes

The complete clean evaluation of the reference checkpoint found 79,966/80,000
training images correct (99.9575%) and 6,345/10,000 selection images correct
(63.45%). All classes have 400 training and 50 selection images; the image IDs
are disjoint. This shows strong in-sample fit and a substantial generalization
gap. It does not identify spatial compression as its cause.

The archived 100-epoch selection curve for the same reference was:

| Epoch | Selection Top-1 | Selection NLL |
|---|---:|---:|
|50|59.30%|1.79736|
|75|61.60%|1.77552|
|90|63.34%|1.73041|
|94|63.42%|1.70435|
|100|63.20%|1.70764|

Epoch 94 had both the best accuracy and lowest NLL. There was no sustained late
validation collapse in this schedule. The historical logs lack a clean training
probe curve, so they cannot establish when the train-selection gap emerged.
The later checkpoint replay differed by 3 correct selection predictions
(+0.03 percentage points); differences of that scale are not evidence of a new
method's benefit.

## Controlled intervention

Stem and head are retained. Both models use the same 3/4/6/3 preactivation
bottlenecks, channel widths 256/512/1024/2048, native 64x64 inputs, 200 classes,
23,902,472 parameters, and identical weights at initialization for each seed.

| Variant | Stage outputs | Reduction before the unchanged head |
|---|---|---|
| `cnn4` |32 ->16 ->8 ->4| Original stride-two first block in stage 4 |
| `cnn8` |32 ->16 ->8 ->8| Even-index point sampling after all stage 4 blocks, restoring 4x4 |

`cnn4` is bitwise equivalent to the existing `resnet50_preact` model for the
same initialized weights. In `cnn8`, the final stage's first spatial convolution
and projection shortcut both use stride1. A parameter-free
`x[..., ::2, ::2]` then restores the original 4x4 head input. There is no image
upsampling, extra stage, extra attention, learned parameter, or average-pooling
filter. The timing of reduction is the declared intervention.

The intervention also changes receptive fields, training-time BatchNorm
statistics, and compute. Estimated forward arithmetic rises from 2.596G to 3.981G
FLOPs, about 53.3%, under the existing estimator. The pair is parameter-matched,
not compute-matched. Improved accuracy alone cannot establish which of these
mechanisms caused it or demonstrate an attention advantage. Actual timing and
memory are recorded; FLOPs exclude normalization and elementwise operations.

## Training and evaluation

- Inherit the trusted reference checkpoint's archived training configuration,
  not the default 15-epoch screening YAML. Reference weights are never used for
  initialization. The default seed is 42.
- Keep the original 80K/10K/10K split, native resolution, batch 64, AdamW,
  normalization/bias weight-decay exclusions, augmentations, mixed-precision
  policy, and 100-epoch cosine schedule. Checkpoint selection uses selection
  Top-1 only, with strict improvement and the earlier checkpoint retained on ties.
- Disable early stopping. Evaluate the 10K selection set every epoch.
- Evaluate a fixed clean probe of 20 training images per class (4,000 total) at
  epoch 1, every 5 epochs, and the last epoch. Probe observations never select a
  checkpoint or stop training. Extra probe evaluation preserves RNG states.
- At the selected checkpoint, evaluate all 80K clean training images and 10K
  selection images. Record per-class gaps, confusion, per-image predictions,
  and example galleries. Neither audit nor official validation is evaluated.
- Record common-device FP32 batch 1/batch 64 profiles and total training cost.
- Pair predictions on exactly matching image IDs and labels. Report the number
  of CNN errors corrected by `cnn8`, the number of formerly correct predictions
  harmed, the net accuracy difference, and NLL differences. Report broad
  behavior rather than selecting only favorable classes after seeing results.

One seed screens this mechanism. A useful result needs paired seed replication
and a subsequent convolution/attention comparison before a stronger claim.

## Workbench commands

Use the existing CUDA environment. Do not pull into an active older training
job. To keep the process alive when the laptop disconnects, first enter a tmux
session if not already in one:

```bash
tmux new-session -A -s spatial100
```

Then, in that tmux terminal:

```bash
cd ~/wenfei-phd-ai
git pull --ff-only

# No training or shutdown: validate checkpoint/data, GPU, email and stop permission.
python -u run_spatial_overnight.py \
  --base-run runs/literature_tiny/suite_20261013T173206_5ac7cd/resnet50_preact_seed42 \
  --data-dir ./data --num-workers 4 --check-only --fallback-os-shutdown
```

After the preflight succeeds, start the actual run:

```bash
python -u run_spatial_overnight.py \
  --base-run runs/literature_tiny/suite_20261013T173206_5ac7cd/resnet50_preact_seed42 \
  --data-dir ./data --num-workers 4 --fallback-os-shutdown
```

The default is both `cnn4` and `cnn8`, seed 42, 100 epochs each. The wrapper uses
the existing private `email.txt` or environment credentials, with the same
default recipient as the earlier overnight runner. It strips Gmail credentials
from the training child's environment. It never prints credential values.

Ctrl+B then D detaches tmux. Closing the laptop connection does not terminate
the remote tmux process. A VM failure, termination or Workbench idle-shutdown
policy can still interrupt it.

After the child finishes or fails, the wrapper flushes the log, packages
analysis files, attempts email up to three times, and requests managed instance
stop. Optional OS shutdown is a fallback if managed stop fails. Packaging or
notification failure still reaches the stop request. Files remain on the VM;
an offline laptop cannot download them until it reconnects. A hung training
process has not finished and is not covered by an automatic wall-clock timeout.

For a separate one-epoch smoke suite without email or automatic stop:

```bash
python -u spatial_experiment.py \
  --base-run runs/literature_tiny/suite_20261013T173206_5ac7cd/resnet50_preact_seed42 \
  --data-dir ./data --num-workers 4 --smoke
```

A smoke suite is a separate experiment and cannot be extended to 100 epochs by
resume. Formal training starts fresh. For training without notifications or
automatic shutdown, use the same command without `--smoke`.

## Results and resuming

The printed `runs/spatial_tiny/suite_...` directory contains:

- `suite.json`: inherited configuration, split, source identities and fixed jobs.
- `cnn4_seed42/` and `cnn8_seed42/`: `epochs.csv`, `metadata.json`, best/last
  checkpoints, fixed probe IDs, config, summary, inference profiles, and full
  clean diagnosis. The summary identifies the authoritative diagnosis folder
  if an interrupted diagnosis had to be retried.
- `comparison.csv`, `paired_summary.json`, and `paired_selection_seed42.csv`:
  summary comparison and paired correction/harm outcomes.
- `completion.json`: completion marker, explicitly distinguishing smoke runs.
- `overnight.log`: the child's durable combined output when using the wrapper.

Return the emailed `suite_..._analysis.tar.gz`. It includes curves, summaries,
class diagnostics, paired analysis and sample images; checkpoints, datasets,
credentials and full probability NPZs remain on the instance. If the attachment
would exceed 18 MiB, the wrapper builds a smaller archive omitting bulky
per-image CSVs and manifests, and records the omissions. If even that is too
large, the email gives the on-disk archive path. SMTP acceptance is not proof
of inbox delivery; check spam after reconnecting.

To resume an interruption, use the printed suite path with the same source,
software, GPU and precision. Do not pass a new epoch horizon or start a new
suite accidentally:

```bash
python -u run_spatial_overnight.py \
  --resume runs/spatial_tiny/SUITE_DIRECTORY --fallback-os-shutdown
```

Completed model summaries are reused; an incomplete model resumes from the
last completed epoch. The interrupted current epoch repeats. Pair analysis is
regenerated from saved predictions. A source/environment mismatch is rejected.
