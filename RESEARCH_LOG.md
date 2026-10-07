# Research Progress Log

Last updated: 2026-10-07

## 2026-10-07 FP16 attention stability repair

- Workbench report: ResNet-50 completed 15 epochs, held-out Top-1 48.38%,
  Top-5 74.57%, training 974.1 seconds. BoTNet completed epoch 1, then failed
  during epoch 2 evaluation with non-finite loss; no valid epoch 2 checkpoint.
- Reproduced an implementation-level FP16 attention overflow on synthetic
  inputs: QK became infinite before conversion to FP32 for softmax. This
  supports, but does not prove, the cause of the specific uninspected L4 batch.
- Keep attention projections, dot products and aggregation in FP32 for both
  BoTNet and AlterNet; retain AMP for convolutions. Record precision explicitly.
- Added exact-snapshot recovery: preserve completed CNN results and original
  provenance, create a new suite, restart attention from epoch 1, and leave
  old runs intact. No blanket bypass of source/config/split/environment checks.
- CPU FP16 regression, source compatibility and recovery tests added. Actual
  L4 confirmation remains the user's next run; no accuracy gain is inferred.

## 2026-10-07 Literature first ResNet 50 family screening

Current direction supersedes the historical next-step proposals below: compare
whole-model results from established designs before designing another fusion
mechanism. Per-image rescue/harm audits are no longer the main experiment.

- New entry point: `literature_experiment.py`; commands and exact architectural
  adaptations: [LITERATURE_BASELINES.md](LITERATURE_BASELINES.md).
- Two matched comparisons: torchvision ResNet-50 vs paper-based BoTNet, and
  author-style pre-activation CNN (3/4/6/4 blocks) vs author-derived AlterNet.
  AlterNet core is pinned to upstream commit
  `9571ff1370f31af2f433558fbfa7bc87a97b0bab`, with license and attribution.
- Native 64x64 Tiny ImageNet, common stride-2 small-image stem; existing
  80/10/10 development split preserved, official validation still reserved.
- Shared AdamW/augmentation recipe, default 15 epochs for initial screening.
  These are architecture migrations, not original-paper benchmark reproductions
  or evidence that 15 epochs gives a converged baseline.
- Aggregate Top-1/Top-5, parameters, attention-aware FLOP estimate, timing,
  batch-1 latency and batched throughput; no automatic per-image audit.
- Optional soft time budget and epoch-boundary checkpoint resume for office
  sessions; resumable suite skips completed models.
- Local CPU architecture/backprop and synthetic runner tests pass. No real
  Tiny dataset downloaded, no measured L4 runtime or accuracy available yet.
- Next: user runs 15-epoch screening and shares `comparison.csv`; select a
  common longer training budget and multi-seed checks before proposing a
  gap-driven modification. Earlier ResNet-18/RepViT experiments remain intact.

## 2026-10-06: new ResNet-18 generalization check (not yet trained on real data)

The next question is whether a better-controlled, regularized ResNet-18 baseline
benefits from a single late self-attention layer. Per the research decision, this
is **only ResNet-18 vs ResNet-18 + attention**, with no added Conv Control/FFN or
new fusion proposal. Historical RepViT experiments remain unchanged.

- New entry point: `resnet_experiment.py`; protocol and commands:
  [RESNET18_AUDIT.md](RESNET18_AUDIT.md).
- Small-image stem (3x3 stride 1, no initial max-pool); late attention at 8x8,
  64 tokens, 512 channels, four heads, fixed residual scale 0.1.
- Same stratified 80/10/10 split, shared initialization and training recipe.
  SGD, 100 epochs, warmup/cosine, crop/flip/RandAugment, Mixup/CutMix and smoothing.
- A fixed clean training probe tracks accuracy/NLL against selection data; mixed
  labels are not used to report a misleading training accuracy.
- Epoch-boundary resume, parameters/FLOPs/timing/memory/inference metadata and
  two-model paired audit included. No real dataset downloaded locally.
- First run the baseline, assess learning curves, then reuse it for the attention
  comparison under an unchanged recipe. Do not infer that larger models or
  regularization guarantee a gain. Single-seed findings remain preliminary.

## 2026-10-06: first 25-epoch controlled run and frozen-checkpoint diagnostics

Seed 42 completed on L4, with split fingerprint
`cdebb937661395d535e1ed07bf8898d44062f63359eecc50422f4311402cc26d`.
Training commit: `e7a83a7`; recorded dirty working tree contained historical
configuration/results changes. The three variants share a RepViT trunk architecture;
`attention` is a hybrid, not a pure ViT. All best-selection checkpoints are epoch 16.

| Variant | Best selection accuracy | Clean audit accuracy | Clean audit NLL |
| --- | ---: | ---: | ---: |
| CNN | 0.4212 | 0.4125 | 2.68977 |
| CNN + Conv Control | 0.4102 | 0.4058 | 2.71008 |
| CNN + Attention | 0.4107 | 0.4139 | 2.66411 |

Final training accuracies are 0.93866/0.95453/0.93875; selection NLL bottoms
around epochs 11-12 then rises. This is evidence of overfitting under this recipe,
not evidence that extending unchanged training will help. The official validation
set remains reserved. Audit now informs exploratory development, not final testing.

On 10000 clean audit images, attention corrects 941 CNN mistakes but harms 927
correct CNN predictions. Extra conv corrects 916 and harms 983. Label-informed oracle
accuracies are 0.5066 (CNN/attention) and 0.5041 (CNN/conv), so error diversity is
not yet evidence of attention-specific complementarity.

For label-defined corrected-versus-harmed samples only, CNN entropy gives AUC
0.58496 for attention and 0.58195 for conv; negative top-two probability margin gives
0.62321 and 0.63417. These are exploratory ranking associations, not learned-router
performance, significance tests or seed-replicated conclusions. In clean entropy
bins, more uncertain CNN samples show larger relative top-1 gains for both added
modules. Conditioning on reference-model uncertainty can itself expose ordinary
model diversity. NLL utility and accuracy gains must be examined separately.

The next executable step is `utility_diagnostics.py checkpoint`: replay the saved
conv/attention models, then remove just the mixer or the entire addon without
retraining. It checks replay, parameter changes and frozen-model sensitivity. It
does not establish a deployable skip policy or prove unique global-information use.
No new fusion mechanism or training-recipe change has been implemented in this step.

Sources: user-provided renamed `cnn_epochs.csv`, `cpnv_epochs.csv`, `att_epochs.csv`,
the three corresponding summaries, and `audit_seed42_4dc7bb76` paired predictions
and provenance. Data/checkpoints are not committed. See `UTILITY_AUDIT.md` for commands.

## 2026-09-09: controlled utility protocol

See `UTILITY_AUDIT.md` for the executable protocol. The new experiment uses timm
RepViT-M0.9, from scratch, plus either a matched-shell convolution addon or fixed
single-head MHSA after the penultimate stage. It is not the old self-built CNN,
nor an implementation of SHViT SHSA. Training/selection/audit are stratified
80/10/10 partitions, with official validation reserved. Historical numbers below
remain exploratory and are not directly comparable to the new protocol.

The first audit exports paired predictions, loss/margin utility, correction/harm
counts, uncertainty bins, calibration and controlled occlusion. New fusion design,
representation-level claims and conditional execution await repeatable evidence.
At the time of that implementation, no Tiny ImageNet results from this protocol
were available; the first completed run is recorded in the October entry above.

## Research objective

Develop and evaluate an efficient CNN–Transformer hybrid model, with emphasis
on where global attention should be introduced and how local CNN information
should be fused with global self-attention under a limited compute budget.

Current working question:

> Can late, input-adaptive channel-wise attention fusion improve the
> accuracy–efficiency trade-off over CNN, ViT, serial hybrid, and fixed
> residual-attention baselines?

## Implemented infrastructure

- YAML-driven experiments and model construction.
- CIFAR-10, CIFAR-100, and Tiny ImageNet loaders; datasets are not committed.
- CNN, ViT, and serial CNN–ViT hybrid baselines.
- Pluggable BCHW blocks: `conv`, `residual`, `eca`, and
  `self_attention_2d`.
- Attention placement configs: Early, Middle, and Late.
- Fusion modes: Direct, Residual, scalar Gated, two-branch Dynamic, and
  input-dependent Dynamic Channel.
- CUDA AMP/GradScaler, TF32, non-blocking transfers, and fixed seeds.
- Per-epoch CSV, summary CSV, resumable best checkpoints, parameters, FLOPs,
  timing, peak GPU memory, latency, and throughput reporting.

## Baseline results

Results below were reported from the GCP Workbench L4 instance. Results from
different epoch budgets are not directly comparable.

### CIFAR-10, 25 epochs

| Model | Best Top-1 |
|---|---:|
| CNN with residual and lightweight attention | about 90% |
| ViT baseline | 78.86% |
| CNN–ViT serial hybrid | 90.87% |

### CIFAR-100, 25 epochs

| Model | Best Top-1 | Experiment time |
|---|---:|---:|
| CNN | 65.56% | 285.48 s |
| ViT | 55.57% | 462.07 s |
| CNN–ViT serial hybrid | 70.29% | 426.36 s |

### Tiny ImageNet baselines, 25 epochs before runtime upgrade

| Model | Best Top-1 | Notes |
|---|---:|---|
| CNN | 49.95% | Preliminary result |
| ViT | 42.68% | Top-5 at epoch 25: 68.67% |

The old 25-epoch measurements predate the current AMP/profiling setup and must
not be used as final runtime comparisons with current experiments.

## Tiny ImageNet placement screening

All experiments below used 5 epochs, seed 42, batch size 64, AMP, and the
development split. The common backbone and training recipe were held fixed.

| Placement | Top-1 | Top-5 | Train time | Peak memory | Throughput |
|---|---:|---:|---:|---:|---:|
| CNN, no self-attention | 28.91% | 56.62% | 199.51 s | 1056.15 MB | 9368.08 img/s |
| Early residual attention | 28.73% | 56.73% | 333.64 s | 1279.19 MB | 5163.92 img/s |
| Middle residual attention | 33.06% | 61.52% | 251.46 s | 1058.17 MB | 7109.28 img/s |
| Late residual attention | **37.15%** | **65.13%** | 250.42 s | 1058.17 MB | 7101.57 img/s |

Interim finding: Late placement improved Top-1 by 8.24 percentage points over
the 5-epoch CNN baseline. Middle and Late had essentially identical measured
cost, while Late was 4.09 points better. Early attention was the slowest and did
not improve Top-1.

## Tiny ImageNet Late fusion screening

| Fusion | Top-1 | Top-5 | Train time | Throughput |
|---|---:|---:|---:|---:|
| Direct | 35.81% | 63.40% | 245.45 s | 7280.89 img/s |
| Residual | 37.15% | **65.13%** | 250.42 s | 7101.57 img/s |
| Scalar gated | **37.17%** | 64.81% | 247.28 s | 7034.61 img/s |
| Two-branch dynamic | 34.78% | 63.06% | 266.40 s | 6779.40 img/s |
| Dynamic channel, init -2 | 34.61% | 62.92% | 253.05 s | 6990.13 img/s |
| Dynamic channel, init +2 | 35.38% | 63.28% | 254.31 s | 6929.05 img/s |

Interpretation:

- Direct fusion underperformed, supporting the need for a CNN residual path.
- Scalar gated and residual results were effectively tied.
- The first two-branch dynamic design was worse and slower. It changed both the
  local branch and gate simultaneously, and mixed BatchNorm and LayerNorm
  feature distributions, so it is retained as a negative result rather than a
  candidate for longer training.
- Increasing the Dynamic Channel initial attention weight from about 0.119 to
  0.881 improved Top-1 by 0.77 points, but it remained 1.77 points below fixed
  Residual. Gate initialization tuning is therefore stopped.

## Dynamic Channel fusion

The new, controlled design keeps the proven Late residual structure and changes
only the attention coefficient:

```text
A = SelfAttention(x)
w = sigmoid(ChannelGate(GlobalPool(x)))
y = x + w(x) * A
y = y + MLP(LayerNorm(y))
```

`w` has shape `[batch, channels, 1, 1]`, so it varies by image and channel.
There is no additional local CNN branch. Both initial gate biases -2 and +2
were screened. Neither exceeded fixed Residual in the 5-epoch experiment.

## Current experiment: parameter-matched CNN control

The control replaces the Late attention block with a residual ConvNeXt-style
convolutional block:

```text
x -> depthwise 7x7 -> BN -> pointwise expansion -> GELU
  -> pointwise projection -> residual add
```

At 256 channels and expansion ratio 4, its learnable parameter count is close
to the removed Late attention-plus-MLP block. FLOPs cannot be exactly matched
because global attention contains a parameter-free token-interaction term; the
recorded parameter and FLOP values in the summary CSV must therefore be used in
the interpretation.

Run:

```bash
python main.py tiny_imagenet_late_conv_control.yaml
```

Result:

| Model | Top-1 | Top-5 | Train time | Throughput |
|---|---:|---:|---:|---:|
| Late residual attention | 37.15% | 65.13% | 250.42 s | 7101.57 img/s |
| Late convolutional control | **39.53%** | **67.56%** | **242.53 s** | **7595.34 img/s** |

The convolutional control was 2.38 Top-1 points better and faster than Late
attention. Therefore, the earlier Late attention gain cannot be attributed to
global attention alone. The strong local convolutional block becomes the new
5-epoch baseline.

## Current experiment: local-global correction

This experiment preserves the full convolutional control path and adds a
reduced-dimension global attention correction:

```text
L = ConvControl(x)
G = ProjectOut(SelfAttention(ProjectIn(L)))
y = L + alpha * G
```

`alpha` is a learnable scalar initialized to zero, so the initial forward pass
is exactly the convolutional control. Attention uses dimension 128 rather than
256 to limit its parameter, FLOP, and latency overhead. This tests whether
global information can improve the strong local baseline as a learned residual
correction instead of competing with it.

Run:

```bash
python main.py tiny_imagenet_late_local_global_correction.yaml
```

## Experiment policy

- Use 5 epochs only for smoke tests and candidate screening.
- Do not claim final superiority from a single seed or 5-epoch result.
- Change one experimental factor at a time.
- Compare accuracy together with parameters, FLOPs, memory, latency, throughput,
  and training time.
- Promote only promising candidates to longer training and at least three seeds.
- Final comparisons must use identical splits, epochs, optimizer, augmentation,
  seed policy, and runtime settings.

## Next decisions

1. Compare Local-Global Correction with the Late convolutional control.
2. Record exact parameters and FLOPs from both summary CSV files.
3. If the correction is competitive, inspect the learned `correction_scale`
   from its checkpoint and repeat leading candidates with multiple seeds.
4. Train only the shortlisted models for a longer, common epoch budget.
5. Validate the final method on CIFAR-100, Tiny ImageNet, and one real-world
   dataset.

## Important limitations

- Tiny ImageNet currently uses a fixed development split from training data;
  preserve the official validation set for final evaluation.
- Current evidence establishes an attention-placement trend, not yet a new
  state-of-the-art model.
- Novelty must be checked against current literature before naming the final
  contribution.
