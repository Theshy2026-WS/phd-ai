# Tiny ImageNet: mechanism diagnosis and 30-epoch fusion screen

This is the next development experiment after the paired CNN4/CNN8 study.
It uses the **existing completed CNN8 best checkpoint**, seed42, and the same
80,000 training / 10,000 selection / 10,000 audit split. The audit partition
and official validation remain unused. There is no new seed sweep.

## One overnight job, two analysis materials, one completion email

1. Diagnose the original CNN8 and a train-only BN-recalibrated copy.
2. Run three matched groups, each for **30 additional fine-tuning epochs**.
3. Package `mechanism_analysis.tar.gz` and `fusion30_analysis.tar.gz` together
   in one email attachment. Send only after all work finishes or fails.
4. Request managed Workbench stop, with optional OS shutdown fallback.

Preflight authenticates the SMTP connection without sending a message. No
start email or per-model emails are sent. `--check-only` validates credentials,
GPU, source checkpoint, dataset split and stop permission; it neither creates
a suite nor trains, sends mail, or stops the instance.

## Material 1: fixed-checkpoint mechanism diagnosis

The original CNN8 computes an 8x8 final-stage map but samples `[..., ::2, ::2]`
before its BN/ReLU/global-average-pooling/linear head. At each model state,
evaluate all five fixed readouts:

| Readout | Feature positions |
| --- | --- |
| `phase00` | even rows, even columns: the original trained readout |
| `phase01` | even rows, odd columns |
| `phase10` | odd rows, even columns |
| `phase11` | odd rows, odd columns |
| `full8` | all 8x8 positions with the existing adaptive pooling head |

The encoder runs once per batch and all readouts reuse its feature tensor.
In eval mode, the mean of the four phase **logits** equals the full8 logits
up to floating-point rounding; an explicit numerical check verifies this.
Probability averaging is a different operation. All phases are reported;
the best observed selection phase is not promoted to a new method.
The five classifier-head calls temporarily use strict FP32 matmul, restoring
the original policy immediately afterward; the encoder retains its recorded
TF32 policy. This avoids treating TF32 rounding as a failed mathematical
identity. The original phase00 replay and numerical error are recorded.

BN recalibration uses a separate model copy, **all clean training images
only**, no gradients or optimizer, reset running statistics, cumulative BN
averaging, and deterministic shuffled batches. Only BN running buffers may
change. Shuffling avoids class-sorted calibration batches. Default batch64
divides 80,000 exactly. Other modules stay in eval mode. Selection images and
labels are never used to fit calibration statistics.

Both states receive the same complete clean80K/selection10K evaluation of
all five readouts. Reports include Top1/Top5/NLL, class recall and confusion,
generalization gaps, phase disagreement, and paired corrected/harmed images.
Prediction arrays stay on disk; compact email packages can omit per-image
CSVs. Reports distinguish original and recalibrated results explicitly.

This is a fixed-model inference intervention. The original BN/head were
trained with phase00, so a negative full8 or calibration result does not prove
that retraining a different readout could never help. Simultaneous BN
recalibration is a bounded diagnostic, not a guarantee of optimal statistics.

## Material 2: matched short adaptation

| Group | Model | Additional epochs | Extra noise |
| --- | --- | --- | --- |
| `cnn8_control` | Existing CNN8, original phase00 head |30| None |
| `unet32` | Same CNN8 plus a residual 8->16->32 skip decoder |30| None |
| `unet32_noise` | Identical initialization to `unet32` |30| Gaussian, p=.5, RGB sigma=.02 |

Every group loads the **same original, uncalibrated parent checkpoint**. Each
gets a fresh AdamW optimizer and a 30-epoch cosine schedule: learning rate
5e-5, one warmup epoch, minimum1e-6. Existing weight decay, normalization/bias
exclusions, Mixup/CutMix, label smoothing and image augmentation are retained.
All encoder weights are trainable. This is neither a fresh30-epoch baseline
nor resumption of the old optimizer/scheduler. The control separates continued
training from the proposed structure's effect.
Augmentation, sample shuffling and batch-mixing draws use a common data epoch
of100 plus the additional epoch (101..130), rather than replaying the old
first30 epochs. The seed remains42; the optimization schedule and report epoch
remain1..30. `augmentation_epoch` records this distinction in the history.

The decoder projects the deep8x8x2048 features to128 channels, upsamples to16
and fuses the projected stage2 (16x16x512), then upsamples to32 and fuses
projected stage1 (32x32x256). Lightweight convolutions precede pooling and a
200-class correction. The correction is added to the original CNN8 logits.
Its final linear layer starts at zero, so all groups have the original
prediction function at epoch0. There is no64x64 decoder or frozen encoder.
The implemented fusion has24,584,144 parameters versus23,902,472 for CNN8
(681,672 extra, about2.85%). The existing forward estimator reports4.492G
versus3.981G FLOPs (about12.9% extra); this is an arithmetic estimate excluding
normalization/elementwise work, not a measured training-time guarantee.

The noise group changes one factor: independently apply mild Gaussian noise
to half the augmented training images, clamp in RGB[0,1], then restore normal
normalization. Sigma.02 is about5.1/255. Its local per-image/per-epoch RNG does
not consume the model or Mixup RNG. Validation, clean training probes and
final clean training evaluation never receive this extra noise. This tests
whether mild appearance variation helps; it does not assume noise is helpful.

Selection is evaluated at epoch0 and every epoch. A fixed4,000-image clean
training probe is evaluated at epoch0, epoch1, every5 epochs and the last
epoch; it never chooses a checkpoint. Strict selection Top1 improvement
chooses `best.pt`, retaining the earlier checkpoint on ties. Epoch0 can remain
best if adaptation only hurts. Training still completes its30-epoch horizon.
Final best-checkpoint diagnostics cover all clean80K/selection10K images.

`epochs.csv`, per-class outputs, paired corrections/harms and the final report
show whether structure or noise changed accuracy, NLL and generalization gap.
`learning_curves.png` shows the gap, while `selection_top1_zoom.png` and
`selection_nll.png` make small selection changes visible. Both best-checkpoint
and final-epoch selection metrics are reported.
This short screen can identify promising directions; a generic U-shaped
decoder by itself is not a novelty claim. A later capacity/no-skip control
would be needed before attributing gains specifically to skip information.

## Workbench command

Use the existing CUDA environment and saved spatial suite. First update and
enter tmux (skip the tmux line if already inside a tmux session):

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
tmux new-session -A -s mechanism30
```

Then run this inside tmux:

```bash
python -u run_mechanism_overnight.py \
  --parent-run runs/spatial_tiny/suite_20261103T175622_a19c86/cnn8_seed42 \
  --data-dir ./data --num-workers 4 --epochs 30 \
  --fallback-os-shutdown
```

Add `--check-only` to validate without starting anything. The existing private
`email.txt` / Gmail environment credentials are reused, with the earlier
default recipient. No credential values are printed or packaged.

Ctrl+B then D detaches tmux. A laptop disconnect does not stop the remote
job; provider failures or VM policies can still interrupt it. A hung child
has not finished; there is no automatic wall-clock kill deadline.

For a non-notified smoke check, use `mechanism_suite.py` with the same parent
and data arguments plus `--smoke`. This makes a separate one-epoch adaptation
suite; its mechanism diagnosis still covers the full development partitions.
Smoke and formal horizons cannot be switched during resume.

## Results, resuming and notification failures

The printed `runs/mechanism_tiny/suite_...` directory contains `suite.json`,
`status.json`, `overnight.log`, `mechanism/`, and `fusion30/`. The top-level
completion marker is written only when both stages succeed. An independent
stage failure is recorded and the other authorized stage is still attempted;
the final status and process exit remain failed. Interrupts propagate.

To resume an interruption with unchanged source, environment and checkpoint:

```bash
python -u run_mechanism_overnight.py \
  --resume runs/mechanism_tiny/SUITE_DIRECTORY \
  --fallback-os-shutdown
```

Completed work is reused; interrupted training resumes from the last saved
epoch, and an interrupted current epoch repeats. Original runs/checkpoints
are never overwritten. Diagnostic attempts are kept in separate directories.

The wrapper packages analysis, tries the single completion notification up
to three times, and requests managed stop even if training, packaging or mail
fails after the child has started. Full outputs remain on instance disk.
Attachments exclude checkpoints, probability NPZs, datasets and secrets.
If the combined archive exceeds18MiB, a compact version omits bulky per-image
outputs with an omission manifest. If it remains too large, the email lists
disk paths. SMTP acceptance confirms server acceptance, not inbox delivery.
