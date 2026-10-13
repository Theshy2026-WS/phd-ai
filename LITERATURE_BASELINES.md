# Tiny ImageNet literature baseline experiments

## CoAtNet-0 baseline (2026-10-12)

Run **only the added model**; the four completed 100-epoch runs need not be retrained:

```bash
git pull --ff-only
python -m pip install -r requirements-literature.txt
python literature_experiment.py train --variant coatnet0 --config configs/tiny_imagenet_coatnet.yaml --seed 42
```

The new YAML defaults to **100 epochs**, with exactly the same recipe as the
previous literature YAML plus `--epochs 100`. For a smoke test append `--epochs 1`;
start a NEW run without that override for the full experiment (do not extend a
finished one-epoch cosine schedule). A smoke run is not a comparison result.
Default `suite` still runs the original four models, avoiding unexpected extra
work. For all five from scratch:

```bash
python literature_experiment.py suite --variants resnet50 botnet50 preact50 alternet50 coatnet0 --seeds 42 --epochs 100
```

### Architecture and provenance

- Source: `timm==1.0.25`, `coatnet_0_224`, **not** the `_rw` variant. This is
  timm's third-party paper-derived implementation, not an official author port.
  Sources: https://arxiv.org/abs/2106.04803 and
  https://github.com/huggingface/pytorch-image-models/blob/v1.0.25/timm/models/maxxvit.py
- Native 64x64 input, 200 classes, `pretrained=False`; no weights downloaded.
- Preserve the two-convolution stem and C-C-T-T stages: MBConv (with SE channel
  gating), MBConv, relative global self-attention + MLP, relative global
  self-attention + MLP. Depths 2/3/5/2; widths 96/192/384/768.
- Preserve original stride-32 pyramid: stem 32x32, stages **16/8/4/2**.
  Seven attention blocks operate on 4x4 (16 tokens, five blocks) and 2x2
  (4 tokens, two blocks). This is deliberately NOT the ResNet stride-16
  adaptation. It is a whole-model reference, not a position-only ablation.
  A stride-16 CoAtNet would be a separately named future experiment.
- Use explicit timm attention math in FP32 islands for AMP safety; no global
  timm backend mutation. Full FP32 evaluation/inference, TF32 setting unchanged.
  Dropout and stochastic depth are zero. Relative position parameters have no
  weight decay, as do positional parameters in the existing baselines.
- FLOPs include QK and AV in addition to Conv/Linear (including SE/MLP and
  projections). Same partial estimate convention: one MAC = two FLOPs;
  excludes normalization, softmax, bias additions, pooling and elementwise ops.
- No matched pure-CNN CoAtNet control is claimed: comparison `control` is blank.

### Compare against already completed runs

Use the new CoAtNet directory printed by `Run:` in place of `NEW_COATNET_RUN`:

```bash
python literature_experiment.py compare --runs runs/literature_tiny/suite_20261012T162914_11b9ec/resnet50_seed42 runs/literature_tiny/suite_20261012T162914_11b9ec/botnet50_seed42 runs/literature_tiny/suite_20261012T162914_11b9ec/preact50_seed42 runs/literature_tiny/suite_20261012T162914_11b9ec/alternet50_seed42 NEW_COATNET_RUN --output runs/literature_tiny/comparison_with_coatnet.csv
```

Only exact completed `4506e0d` source fingerprints for the four old variants
receive the additive-registration compatibility exception. Actual commit and
source metadata are preserved. Recipe, split, epochs, environment and evaluation
precision must still match. This exception does NOT enable resuming old training
under changed source; unknown edits are rejected. Do not edit JSON to bypass checks.

Research sequence: literature baselines -> choose a backbone -> one controlled,
literature-motivated modification at a time -> multi-seed ablations and final
whole-model comparison. Adding this baseline is not a novelty claim.

Start with existing ResNet, AlterNet and BoTNet designs before proposing another
fusion method. This runner compares **whole-model performance and cost**; it does
not produce rescued/harmed-image tables, interventions or occlusion sweeps.
Existing ResNet-18 and RepViT code and runs are unchanged.

## Current recovery after epoch 6

The second report from `85dd1e7` confirms finite model state and finite full-FP32
logits on the failing batch, but non-finite mixed-precision evaluation. Protecting
only attention was insufficient. All selection, held-out evaluation and inference
benchmarks now use complete FP32 (with the existing TF32 runtime setting).
Training, its AMP/scaler, augmentation, optimizer and LR schedule are unchanged.

Run once for the reported suite:

```bash
git pull --ff-only
python literature_experiment.py migrate-evaluation --from-suite runs/literature_tiny/suite_fp32fix_20261007T163922_803c32
```

This creates `suite_fp32eval_...`, copying existing checkpoints without deleting
or changing the originals. ResNet-50 is NOT retrained: its saved selected model
is re-evaluated and re-benchmarked in FP32. BoTNet continues from the last saved
epoch (6 in the report), with optimizer/scaler/scheduler/RNG restored, so epoch 7
is repeated. Pre-activation CNN and AlterNet then run their planned 15 epochs.
Checkpoint copies need additional disk space. Append `--max-minutes 75` if needed.

After a pause, use `suite --resume` with the NEW printed `suite_fp32eval_...`
directory. Repeating `migrate-evaluation` creates another copy and should not be
used as the normal resume command.

The source migration accepts only the recorded `85dd1e7` training implementation
and the unchanged CNN from `ff14146`; unknown source/recipe/environment/split
changes are rejected. `origin_metadata.json` and `evaluation_migration` preserve
the original provenance. Earlier epoch metrics remain labelled `legacy AMP`.
The saved best candidate is re-scored in FP32 before further checkpoint selection;
discarded historical checkpoints cannot be retrospectively ranked in FP32.
These migrated runs remain exploratory screening, not a pristine uniform-policy
reproduction. Final comparisons should start fresh with the corrected protocol.

The new table reports FP32 evaluation and inference for ALL models, including
the already-trained CNN; old AMP latency numbers are not mixed into the comparison.

## Historical first attention repair

The old `repair-suite` CLI is now retired; use the current recovery above.

The `ff14146` release computed attention dot products under FP16 autocast.
Large finite Q/K values can overflow before the FP32 softmax conversion. This
failure is reproduced by a synthetic FP16 regression test; it is a plausible
explanation for the reported second-epoch BoTNet evaluation failure, not a
claim that the original L4 failing batch has been inspected.

The fix keeps Q/K/V projections, content/relative dot products, softmax and
attention aggregation in FP32 in both hybrid models. Convolutional paths keep
the original AMP setting. This changes numerical precision and potentially
runtime, not the architecture or parameter count. Inference measurements use
the same corrected precision policy; the summary records that policy.

The first recovery used the following command (historical, not for current use):

```bash
git pull --ff-only
python literature_experiment.py repair-suite --from-suite runs/literature_tiny/suite_20261007T160009_2cba87
```

That command created a NEW `suite_fp32fix_...` directory. It reused the completed CNN's
summary unchanged and restarts BoTNet from epoch 1 of the original 15-epoch
schedule, followed by the remaining models. The old suite and checkpoints are
not moved, deleted or overwritten. Old attention checkpoints are not resumed
under a different precision policy. Only the exact original source snapshot
and matching recipe, software, GPU and split are accepted for CNN reuse.

For an office time limit, append `--max-minutes 75`. If paused, continue with
`suite --resume` and the NEW printed `suite_fp32fix_...` directory; do not run
`repair-suite` repeatedly, as each repair invocation starts a new suite.

Original CNN source hashes and commit remain intact. The comparison table
records the original commit and `reused_legacy_cnn`, rather than relabeling old
results as generated by the patched code. The source compatibility exception
is restricted to completed CNN controls from `ff14146`, not arbitrary runs.

## Start on Workbench

```bash
git pull --ff-only
python -m pip install -r requirements-literature.txt
python literature_experiment.py suite --seeds 42 --epochs 15
```

This trains four models sequentially, 15 epochs **each**, not 15 epochs total.
Default order: ResNet-50, BoTNet-50, matched pre-activation CNN, AlterNet-50.
No pretrained weights are downloaded. Tiny ImageNet is reused from `./data`;
the existing dataset helper downloads it on Workbench only if it is missing.

For an office session with a soft 75-minute budget:

```bash
python literature_experiment.py suite --seeds 42 --epochs 15 --max-minutes 75
```

The script prints a suite directory. Continue the same suite later:

```bash
python literature_experiment.py suite --resume runs/literature_tiny/suite_ACTUAL_DIRECTORY
```

Replace `suite_ACTUAL_DIRECTORY` with the exact printed folder name. Completed
models are skipped; the current model resumes from its last completed epoch.
The budget is soft: first-epoch startup, checkpoint writing and final evaluation
can overrun it. It does not stop or shut down your VM. Each epoch prints an
estimated remaining training/selection time for the current model, not the
whole suite. Measure the first L4 epoch before planning the full session.

Run just the AlterNet comparison first:

```bash
python literature_experiment.py suite --variants preact50 alternet50 --seeds 42 --epochs 15
```

Or run one model:

```bash
python literature_experiment.py train --variant resnet50 --seed 42 --epochs 15
python literature_experiment.py train --variant botnet50 --seed 42 --epochs 15
python literature_experiment.py train --variant preact50 --seed 42 --epochs 15
python literature_experiment.py train --variant alternet50 --seed 42 --epochs 15
```

Resume a single model using its run directory, not a checkpoint filename:

```bash
python literature_experiment.py train --resume runs/literature_tiny/ACTUAL_RUN_DIRECTORY
```

Use Ctrl+C if necessary. An incomplete epoch repeats; `last.pt` contains model,
optimizer, scheduler, AMP scaler and RNG states at the last completed epoch.
If interrupted before the first checkpoint, restart the single-model command;
a suite can restart that pending job with its suite `--resume` command.
Do not change source, environment or the planned total epochs during a resume.
For a longer research schedule, start a **new** equally budgeted run rather than
relabeling a completed 15-epoch cosine schedule as a longer experiment.

The process must remain running on the VM. A foreground terminal session is not
a reliable disconnect strategy; if already familiar with tmux, run the command
inside it and detach. Stopping the VM stops training, but saved checkpoints remain
on its persistent disk. This script never changes Workbench idle-shutdown settings.

## Architectures and provenance

| Variant | Architecture | Correct CNN control | Params | Estimated forward GFLOPs |
| --- | --- | --- | ---: | ---: |
| `resnet50` | torchvision ResNet-50 v1.5, depths 3/4/6/3 | — | 23.910M | 2.596 |
| `botnet50` | Replace all three final-stage 3x3 convolutions with global relative MHSA | `resnet50` | 19.199M | 2.533 |
| `preact50` | Author-style pre-activation bottlenecks, matched depths 3/4/6/4 | — | 28.365M | 2.739 |
| `alternet50` | Author dnn_50 layout, local attention counts 0/1/3/2 | `preact50` | 34.267M | 3.161 |

All use native 64x64 input, 200 classes, a 3x3 stride-2 stem without maxpool,
and stage resolutions 32/16/8/4. This common Tiny-specific stem is a deliberate
adaptation: it is neither the original ImageNet stem nor our older stride-1
ResNet-18 stem. Compare within this suite; prior numbers used different models
and training recipes.

### AlterNet

[How Do Vision Transformers Work?](https://arxiv.org/abs/2202.06709), ICLR 2022.
The vendored core is extracted from the
[authors' repository](https://github.com/xxxnell/how-do-vits-work/tree/9571ff1370f31af2f433558fbfa7bc87a97b0bab),
commit `9571ff1370f31af2f433558fbfa7bc87a97b0bab`.
See `third_party/ALTER_NET_NOTICE.md` and the accompanying Apache license.

Preserved: pre-activation convolutional bottlenecks, AttentionBlockB, BatchNorm,
GELU in attention blocks, local Attention2d, shared-head relative positional bias,
head counts 3/6/12/24 and dimension 64 per head. Attention calculations now use
an FP32 island under AMP for numerical stability. No invented FFN, gate or residual
scaling is inserted. The author's dnn_50 has stage depths 3/4/6/4 and six attention
blocks (0/1/3/2); we preserve that public-code layout. The paper also discusses
a four-attention-block CIFAR configuration; that is **not** this preset.

Tiny adaptations: 200-way classifier, 4x4 windows, stride-2 small-image stem,
and the common training recipe below. The CNN control uses the same 17-block
stage layout, not canonical 3/4/6/3 ResNet-50. Model names denote their source
family, not an assertion of identical layer counts or parameter budgets.

The official repository also contains an AlterNet-18 constructor. Thus an
18-based author-derived variant is possible; the 50-family is chosen here to
align with the representative AlterNet and BoTNet comparisons, not because
AlterNet inherently requires ResNet-50.

### BoTNet

[Bottleneck Transformers for Visual Recognition](https://arxiv.org/abs/2101.11605),
CVPR 2021, Table 1 and Figure 4. This is a **paper-based implementation**, not
official author code. It uses torchvision's v1.5 ResNet as the CNN substrate.

The final three bottlenecks replace their 3x3 convolution with four-head global
self-attention. Relative height/width embeddings contribute query-dependent
`q*r` terms, not just a scalar positional bias. The first BoT block attends at
8x8 before 2x2 average pooling; the next two attend at 4x4. Existing bottleneck
normalization, activations, expansion projections and shortcuts are retained.
This is the downsampling BoT variant, **not BoTNet-S1**.

Both architectures are Tiny migrations, not claims to reproduce a published
Tiny ImageNet score. The original AlterNet paper does not supply such a target.

## Shared screening protocol

- From scratch, seed 42 initially; 15 epochs is screening, **not convergence**.
- Preserve existing stratified train/selection/held-out split: 80k/10k/10k
  images from the Tiny training set. The official 10k validation set is untouched.
- Choose `best.pt` only using selection Top-1; evaluate it once on the clean
  held-out split after training. This development split has already informed
  earlier research, so do not present it as a pristine final test benchmark.
- AdamW, LR 0.0005, weight decay 0.05, two warmup epochs and cosine decay;
  biases, normalization and positional parameters excluded from weight decay.
- RandomResizedCrop, flip, RandAugment, Mixup/CutMix, label smoothing; same
  deterministic per-sample/per-epoch augmentation stream across variants.
- Training: batch 64, CUDA AMP float16 with FP32 attention islands, TF32 enabled.
  Evaluation and inference benchmark: complete FP32 for every model.
  Source/split/software/training and evaluation precision are recorded.
  This common recipe is not either paper's original training procedure.
- The two model families differ in normalization, block count and attention
  formulation. Whole-model comparisons are valid descriptive results; pair each
  hybrid with its own CNN control when discussing added attention value.

## Results to send back

Each suite writes `runs/literature_tiny/suite_.../comparison.csv`, refreshed as
models finish. Send this first. Each model directory also has `epochs.csv`,
`summary.json`, `config.yaml`, `metadata.json`, `best.pt`, `last.pt` and the split
manifest. Send the CSV/JSON files, not the model weights.

The comparison includes Top-1/Top-5, parameters, estimated forward FLOPs,
training/experiment minutes, batch-1 latency, batch-64 throughput, and hybrid
accuracy differences in **percentage points versus the appropriate control**.
FLOPs count 1 MAC as 2 FLOPs and include attention QK/AV and relative dot
products, but exclude softmax, normalization and elementwise operations.
Latency is forward-only on synthetic device-resident inputs, not end-to-end
data loading. Timing and GPU memory depend on the hardware/runtime.

Separate completed runs can be combined:

```bash
python literature_experiment.py compare --runs RUN_A RUN_B RUN_C RUN_D
```

The comparison rejects mismatched protocols, recipes, splits, source hashes,
environments, epoch budgets, and duplicate variant/seed entries.

After screening, first choose a common longer training budget based on learning
curves, then repeat shortlisted comparisons with `--seeds 42 43 44`. Novelty
requires a literature gap plus evidence beyond merely migrating these models.

## Local validation

```bash
python -m unittest discover -s tests -v
```

Tests cover actual model forward/backward passes, stage/window layout,
query-dependent relative attention against an explicit formula, attention FLOP
accounting, a synthetic train/pause/resume equivalence test, suite resumption,
and incompatible comparison rejection. No real dataset or L4 training was run
locally. Data, runs, and checkpoints remain git-ignored.
