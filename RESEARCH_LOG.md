# Research Progress Log

Last updated: 2026-10-20

## 2026-10-20 Daytime execution preference

- Added run_correction_notified.py: only attention correction,25 epochs by
  default, seed42 and the same trained CNN checkpoint. No ViT or shutdown.
- Existing Gmail notifier now optionally attaches the small analysis tar.gz,
  excluding checkpoints and images, after success/failure. App-password preflight
  blocks training if email authentication fails; no real emails sent in local tests.
- Existing overnight commands remain available but are not the daytime command.

## 2026-10-20 Frozen CNN correction candidate and unified compact ViT

- New experiment starts from the actual selection-best checkpoint of a completed
  seed42/100-epoch resnet50_preact run. CNN weights and BN buffers are frozen;
  a zero-output-initialized branch adds class-logit corrections using 8x8 spatial
  self-attention and a base-prediction-conditioned cross-attention query.
- This is a candidate built on side-tuning precedents, NOT a confirmed novelty,
  guaranteed improvement, or claim of independent/complementary information.
- Default:25 extra correction epochs plus100-epoch compact pure ViT (existing
  width224/depth5/patch4 structure), identical parent split and augmentations,
  FP32 recorded explicitly. Optional pooled branch control; equal-budget
  fine-tuning, matched-capacity controls and multiseed validation remain future work.
- Original models are retained. Parent-inclusive and incremental costs are
  separately reported. Selection can retain the unchanged epoch0 classifier.
- No email. New overnight wrapper packages analysis files before managed stop,
  including on failure. Optional private GCS export and a local PowerShell
  downloader support transfer independent of the stopped GPU VM. No claim that
  an offline laptop can receive a direct Workbench download.
- No real-data training or GPU deployment executed locally.
- Validation:30 CPU tests passed, covering exact initial base predictions,
  unchanged frozen parameters/BN, nonzero attention gradients, pure ViT forward/
  backward, checkpoint resume, epoch0 selection fallback, archive exclusion,
  shutdown on archive/export failure, and existing literature baseline regressions.

## 2026-10-07 Full FP32 evaluation and checkpoint preserving recovery

- Second L4 report: BoTNet completed six epochs (selection Top-1 27.24%), then
  failed during epoch 7 evaluation. Diagnostic state: all parameters/buffers
  finite, and the exact failing batch produced finite logits in full FP32.
  Attention-only FP32 protection did not cover the remaining AMP evaluation path.
- All selection, held-out evaluation and inference profiling now run in FP32;
  training and its AMP/scaler/optimizer/schedule are unchanged.
- New `migrate-evaluation` clones the known 85dd1e7 suite, preserves original
  files/provenance, re-evaluates the existing CNN without training it again, and
  resumes BoTNet from epoch 6. Later models retain their 15-epoch plan.
- Saved best candidates are re-scored in FP32; old epoch metrics remain labelled
  AMP and unretained old checkpoints cannot be retrospectively ranked. These are
  pilot runs; final results need fresh uniform-policy runs.
- Added complete-model FP16 overflow regression and exact optimizer-state resume
  tests, including a completed CNN. No local L4 or real-data training performed.

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

## Historical next decisions (superseded by the literature-baseline phase below)

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

## 2026-10-12: literature baselines and CoAtNet extension

- Completed user-reported 100-epoch seed-42 suite: ResNet-50 held-out Top-1
  62.16%, BoTNet-50 62.60%, matched PreAct CNN 63.36%, AlterNet-50 59.50%.
  These are 80k/10k/10k train/selection/held-out results, not the official
  Tiny ImageNet validation benchmark. BoTNet's +0.44 pp is single-seed evidence;
  its best selection accuracy is below ResNet's, so stable superiority is not established.
- Add `coatnet0` from pinned timm's paper-oriented CoAtNet-0 configuration,
  native 64x64 input, 200 classes, no pretraining; default new config 100 epochs.
  Preserve C-C-T-T and stride 32, explicitly reporting 4x4/2x2 attention maps.
  Keep old architecture construction/initialization and training recipes intact.
- Use the same timing, parameter, FLOPs, memory and FP32 inference reporting.
  Completed exact 4506e0d baseline summaries can be compared without retraining.
- Current plan: complete representative literature-method baselines, then
  design controlled improvements based on their limitations and literature gaps.
  Retain strong CNN controls and existing hybrids for the final comparison.
  Use multiple seeds before claiming a small accuracy improvement.

## 2026-10-13: CoAtNet result and two additional fusion-method baselines

- User archive `coatnet_results.tar.gz` confirms 100 complete epochs and exit
  code 0. CoAtNet-0 seed 42: best selection 58.63% at epoch 97; held-out Top-1
  58.22%, Top-5 80.31%; train 6415.32 s, total 6791.07 s; 24.373M parameters,
  0.720802 estimated forward GFLOPs; FP32 batch-1 8.983 ms, batch-64 6131.99 img/s.
  Same recipe hash and identical split manifest as the completed ResNet run.
  Lower FLOPs/high batch throughput did not imply higher accuracy or faster
  single-image inference. No causal conclusion about spatial compression yet.
- User priority: improve Top-1 and validate the improvement on other datasets,
  not efficiency alone. Existing results are single-seed, common-recipe method
  adaptations, not published-score reproductions or proof of a universal optimum.
- Add `acmix50` and `conformer_tiny`; pinned method provenance, method-specific
  dual-head loss, explicit FP32 protection, attention FLOPs and old-result
  comparison compatibility. Defaults in the new shared YAML: 100 epochs.
  Smoke runs are separate. No old experiment outputs or data are committed.
- ACmix: official ResNet structure/equations independently implemented and
  checked with reference-weight output/gradient comparisons for both strides.
  Conformer: Apache-licensed author core, tiny/16 settings and two-way FCUs.
  See LITERATURE_BASELINES.md and the two third-party notices for adaptations.
- Literature findings guiding the next step:
  - TransXNet uses input-dependent depthwise convolution and global attention
    on channel partitions, not adaptive selection of the channel split itself.
    The authors retain a fixed 1:1 split and explicitly flag stage-dependent
    ratios as a future possibility. This is a prior-art lead, NOT a verified
    unoccupied gap as of 2026. Follow-up literature must be checked before claiming
    novelty. Source: https://arxiv.org/html/2310.19380v4 (Tables VII, IX; Section V).
  - AdaViT learns per-image patch/head/block usage. Its central experiment
    reduces T2T-ViT-19 from 8.5 to 3.9 GFLOPs while Top-1 falls from 81.9% to
    81.1%; it starts with pretrained backbone weights. This supports selective
    computation, not a claim that routing improves unrestricted Top-1.
    Source: https://arxiv.org/pdf/2111.15668 (Table 1 and implementation details).
- Plan: finish these two method references, select one accuracy-oriented parent
  method with a recent nearest-method comparator, specify one inherited result
  and one concrete limitation, then implement a narrowly scoped change. Develop
  with selection data, confirm multiple seeds, and lock the method before testing
  ImageNet-100. Do not label more attention or a new gate alone as novelty.

# 2026-10-13: preactivation / two-stage architecture experiments

- User confirmed two **stages**, not two individual blocks; keep original models.
- Added `resnet50_preact` (3/4/6/3), `botnet_twostage` (C3/B3),
  `coatnet_twostage` (MBConv2/Transformer2). Attention runs at 16x16; input 64x64.
- New four-model suite includes unchanged Conformer, seed42, 100 epochs, unchanged
  fusion training recipe. No ACmix included. These are capacity-changing
  architecture experiments, not isolated resolution ablations or original
  paper-model reproductions. See TWOSTAGE_EXPERIMENTS.md for exact layouts.
- Added managed Workbench auto-stop wrapper with read-only permission/CUDA
  preflight, opt-in OS fallback, live logging and suite resume. Cloud stop calls
  tested with mocks only; actual GCP permissions must pass Workbench preflight.
- All seven old model forward outputs verified bitwise unchanged against
  0049d1a at seed42 on CPU. Completed old results retain strict source-whitelist
  comparison compatibility; checkpoint resume source checks remain strict.
- Local CPU suite: 40 tests pass, including new shapes, BF16 gradients,
  checkpoint roundtrip, FLOPs, and managed-stop branches. No training dataset
  downloaded, no full experiment/GPU speed or accuracy claims.
- 2026-10-14: user requested BoT CNN preactivation and ResNet stage4 attention,
  preserving all models, seed42/100epochs. The latter is existing `botnet50`;
  added only `botnet50_preact_cnn` (C-pre3/C-pre4/C-pre6/B-post3) and explicitly
  documented this interpretation. Full widths/depths retained, original BoT
  attention/stem/head unchanged; no Conformer in the requested run pair.
  New parameter count 19,193,480, estimated FLOPs 2,532,802,560. Same recipe.
  Extended overnight wrapper with explicit model selection, preserving defaults.
  Kept strict resume checks and added exact 392a385 completed-result comparison
  compatibility. Added tests for control-weight identity, gradients/optimizer,
  spatial sizes, checkpoint loading, FLOPs and run selection. All 43 local CPU
  tests pass; all ten pre-existing model outputs verified bitwise unchanged
  against 392a385. No full Tiny ImageNet or GPU training performed locally.

## 2026-10-14: four modern mixer structure baselines

- User requested ConvMixer, ConvNeXt V2, ConvFormer and CAFormer after the
  currently running BoT experiment, without removing existing models.
- Added pinned timm1.0.25 adaptations: `convmixer768_tiny64`,
  `convnextv2_tiny64`, `convformer_s18_tiny64`, `caformer_s18_tiny64`.
  Original depths/widths/core blocks retained. ConvMixer patch4 gives a constant
  16x16 map; hierarchical stems use stride2 for 32/16/8/4 maps. All adaptations
  and original settings are documented in MIXER_BASELINES.md.
- Shared seed42/100epoch fusion recipe, no pretraining. ConvNeXt V2 includes
  GRN but NOT FCMAE pretraining. These are structure adaptations, not paper
  score reproductions. CAFormer attention has FP32 islands and QK/AV FLOPs.
- Parameters: 20.419M / 28.017M / 25.135M / 24.702M. Estimated GFLOPs:
  10.299 / 2.902 / 2.575 / 2.582. ConvMixer may be expensive; run a separate
  one-epoch smoke suite on L4 before the new 100-epoch suite.
- Existing eleven models' FP32 outputs and FLOPs verified unchanged against
  0c160b3. Completed old results remain comparable via exact source whitelist;
  resume remains strict. Do not pull until the user's active training finishes.
- Tests cover direct timm equivalence, stage/token shapes, BF16 gradients and
  optimizer updates, GRN/SepConv counts, checkpoint loading, attention FLOPs,
  FP16 overflow protection, and synthetic CAFormer training/evaluation/resume.
  No real dataset, pretrained weights, or cloud training downloaded/run locally.
- Full local CPU regression: 49 tests passed. Used workspace-local temporary
  files for the successful run after Windows sandbox temp cleanup permissions
  blocked the first full-suite attempt. GPU memory/runtime require Workbench smoke.
