# Tiny ImageNet literature baseline experiments

Start with existing ResNet, AlterNet and BoTNet designs before proposing another
fusion method. This runner compares **whole-model performance and cost**; it does
not produce rescued/harmed-image tables, interventions or occlusion sweeps.
Existing ResNet-18 and RepViT code and runs are unchanged.

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
head counts 3/6/12/24 and dimension 64 per head. No invented FFN, gate or residual
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
- Batch 64, CUDA AMP float16, TF32 enabled; source/split/software recorded.
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
