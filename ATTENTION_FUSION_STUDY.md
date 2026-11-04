# Attention-first Tiny ImageNet feature-fusion study

The thesis question is whether attention can supplement a strong CNN's
features, not whether every source of overfitting can be eliminated first.
This suite replaces the19-model generalization queue as the next primary
experiment. It does **not** invoke that queue. It uses its generic tested
checkpoint/evaluation helpers without changing their code.

The working reference is the completed CNN8, approximately64.28% selection
Top1. That is a reference under our protocol, not a proved Tiny ImageNet
ceiling. The original80K/10K/10K split, seed42, native64 inputs, parent files,
and phase00 classification head are retained. Neither audit nor official
validation selects models or contributes gradients.

## What changes relative to the previous experiments

| Previous experiment | New study |
|---|---|
|FrozenCNN, singlequeryfromclassprobabilities, stage3 8x8 tokens, logits correction|CNN trainable, spatial semantic queries,8/16/32feature sources, fusion before classification|
|BoT replacements could remove convolution and reduce to4x4|Strong preactivation CNN8 path retained; residual feature updates supplement it|
|U-shaped convolutional interpolation/fusion|Content-dependent attention retrieval compared with explicit convolution controls|
|Uniformly low new-branch LR and zero initial correction|Warm fusion peakLR5e-4 versusCNN5e-5; gamma starts.1 so all branch parameters can learn immediately|
|Best epoch0 made final error tables describe the parent|Best, actual last, and best-post-training checkpoints are distinguished; learned peak weights retained|
|General overfitting tests dominated the queue|Only attention integration/control training and its feature/routing analysis run here|

These differences are hypotheses to test. They do not guarantee a gain or
turn a generic cross-attention module into a novel contribution.

## Common encoder and readouts

The shared CNN has3/4/6/3 preactivation bottlenecks:

| Feature | Channels | Spatial dimensions | Intended information |
|---|---:|---|---|
|C1|256|32x32|Early local detail|
|C2|512|16x16|Middle-level shape/detail|
|C3|1024|8x8|Deeper representation|
|C4|2048|8x8|Final semantic representation|

All models retain the same original classifier (phase00 reduction then
BN/ReLU/global-average-pooling/linear). Attention reads spatial CNN features,
not class probabilities. Width128, four heads, LayerNorm and no attention
dropout are common. Q/K/V/softmax operate inFP32 while the encoder can useAMP.
There is no explicit positional embedding in this first screen; geometry is
provided by the convolutional features. This policy is recorded explicitly.

The insertion is `F_after = F_before + gamma * update`, with trainable scalar
gamma initialized to.1. The gated version multiplies its update by
`2*sigmoid(gate)`, initialized to1 so its initial function matches cross16.
Nonzero gamma means an inserted epoch0 model may differ slightly from the
parent. Its initial score is measured rather than assumed to be64.28%.

## First stage: nine warm-start screens,30 epochs each

| Variant | Operation | Question |
|---|---|---|
|control|UnmodifiedCNN8|What does the same continued-training schedule do?|
|conv_cross|ProjectC2, pool16->8 and dense3x3 fusion withC4|Does an extra static multiscale convolution path suffice?|
|cross16|Q=C4(8x8),K/V=C2(16x16), updateC4|Can deep semantics retrieve useful middle-level evidence?|
|self8|Self-attention onC4(8x8)|Does same-scale semantic interaction suffice?|
|conv16|Local convolution bottleneck afterC2|Control added capacity/processing at16x16|
|self16|Self-attention afterC2, beforeC3/C4|Does earlier context help before further CNN processing?|
|cross32|Q=C4,K/V=C1(32x32), updateC4|Does finer but less semantic detail help?|
|bidir16|C4readsC2 andC2readsC4; fuse both context outputs intoC4|Does exchanging information both ways help?|
|gated_cross16|cross16 with content-dependent residual gate|Does selective use outperform unconditional retrieval?|

Bidir16 computes both directions from the original features and merges their
outputs intoC4. It does not rerun the CNN or claim an iterative Conformer
coupling architecture. Self16 changes the representation before later CNN
stages; the other attention variants supplementC4 before the common head.

All encoder parameters are trained. Warm jobs start at the same original
CNN8 best checkpoint, with new optimizers. Encoder peakLR5e-5 and fusion
peakLR5e-4 share a common warmup/cosine multiplier, preserving their10:1 ratio.
The original weight decay, bias/norm exclusions, augmentation, Mixup/CutMix,
smoothing and gradient clipping are retained. Data augmentation streams use
epochs101..130, without changing seed42. EMA.999 is common to every job.

## Second stage: four matched fresh models,100 epochs each

`fresh_control`, `fresh_conv_cross`, `fresh_self8`, `fresh_cross16` share
identical random encoder initialization, the same100epoch budget and5e-4 peak
LR for encoder and fusion parameters. This tests the core retrieval hypothesis
before memorization, rather than rejecting it only on a30epoch adaptation of
an already fitted CNN. These four are predeclared; they are not selected
post-hoc from the warm results. Other warm candidates can later receive a
fresh test if their evidence warrants it.

The queue has13 runs,670 total model-epochs. Complete-all has no artificial
wall-clock cutoff, so it may continue past one night. It is finite and stops
after its reports are prepared. An optional explicit `--max-hours` preserves
bounded/checkpointed behavior with a final analysis reserve.

## What will be delivered

Training reports distinguish epoch0, highest post-training raw/EMA score,
overall best, actual last raw, and lastEMA. Best and actual last receive full
clean80K/selection10K class/error diagnostics. `best_trained.pt` is retained
even when epoch0 remains overall best. Costs include actual training time,
parameters and a FLOPs estimate that explicitly counts attention QK/AV.

Approximate model costs at width128 (normalization/elementwise costs excluded):

|Variant|Parameters|EstimatedGFLOPs|
|---|---:|---:|
|control|23,902,472|3.9806|
|conv16|24,182,793|4.1232|
|conv_cross|24,647,305|4.1001|
|self8|24,961,673|4.1169|
|self16|24,167,561|4.1483|
|cross16|24,565,385|4.1232|
|cross32|24,499,337|4.2154|
|bidir16|25,441,417|4.2742|
|gated_cross16|24,567,562|4.1234|

These are explicit approximate controls, not perfectly parameter/compute
matched claims. Conv16 is the primary same-position control for self16;
conv_cross is the primary static-multiscale control for cross16. The key
comparisons are attention versus both pureCNN and the appropriate added
convolution path, within the same warm/fresh scope.

The second analysis material evaluates best and last checkpoints on the same
fixed4K clean training probe and10K selection images under:

- Learned attention.
- Uniform attention weights, retaining learned projections and the encoder.
- Removal of the complete feature update, retaining the co-adapted encoder.

Convolution controls receive learned/removal conditions; pureCNN has only its
unchanged condition. When overall best is epoch0, the best actually trained
snapshot is also analyzed. Reports contain paired corrections/harms,
per-class confusion, gamma, effective feature-change RMS, gate values,
normalized attention entropy (divided bylog key-count), peaks and small
retrieval-map galleries. Retrieval maps are averaged over heads/queries and
are descriptive, not class-causal explanations.
Late-stage maps average all64queries, including those not directly read by
the phase00 classifier. Absolute stage2/stage4 feature RMS values also describe
different representations; use within-model intervention comparisons.

Uniform/removal results are inference interventions on a co-adapted model,
not separately trained model scores. They complement the trained convolution
controls. We do not permuteKandV together as an ablation, because a common
permutation can leave attention unchanged.

## Workbench command

This is an independent new suite and does not alter old experiments. Use one
GPU job at a time; do not launch it alongside a previously running19-model
queue. First update and enter tmux (skip tmux if already inside):

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
tmux new-session -A -s attention_fusion
```

Then run:

```bash
python -u run_attention_overnight.py \
  --parent-run runs/spatial_tiny/suite_20261103T175622_a19c86/cnn8_seed42 \
  --data-dir ./data --num-workers 4 \
  --warm-epochs 30 --fresh-epochs 100 --complete-all \
  --fallback-os-shutdown
```

Preflight verifies GPU/checkpoint/split, SMTP authentication and stop
permissions without sending a start email. `--check-only` performs only those
checks. Ctrl+B thenD detaches tmux. One completion email contains
`training_analysis.tar.gz` and `attention_analysis.tar.gz` in a common delivery
attachment; then managed Workbench stop is requested, with OS fallback when
enabled. Failure still produces partial reports/notification/stop attempts.
There is no hard watchdog for a hung process. Full checkpoints and probability
arrays remain on disk; large email bundles are compacted with omissions listed.

```bash
python -u run_attention_overnight.py \
  --resume runs/attention_fusion_tiny/SUITE_DIRECTORY \
  --fallback-os-shutdown
```

Resume requires the recorded source/config/parent/software/GPU and schedule.
It never silently changes the horizon, smoke protocol or completion mode.

## Literature context and contribution limits

- [BoTNet, CVPR2021](https://arxiv.org/abs/2101.11605) studies late spatial
  convolution replacement by attention. Our earlier BoT runs already explored
  that family; this screen retains the strongCNN8 path and compares feature
  supplementation instead of wholesale replacement.
- [CrossViT, ICCV2021](https://arxiv.org/abs/2103.14899) exchanges multi-scale
  token information with cross-attention. Our CNN-feature retrieval is related,
  not a claim that cross-scale attention is a new idea.
- [Conformer, ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Peng_Conformer_Local_Features_Coupling_Global_Representations_for_Visual_Recognition_ICCV_2021_paper.html)
  couples convolutional local features and transformer global representations.
- [ACmix, CVPR2022](https://arxiv.org/abs/2111.14556) provides prior work on
  integrating convolution and attention. Our simple controls are not exact
  paper reproductions.

The first decision is which information source and insertion strategy yields
useful held-out corrections beyond static convolution fusion, at a measured
cost. A paper contribution would require a specific supported failure mode,
targeted mechanism and comparison with relevant prior designs. Merely adding
attention, seeing an attractive heatmap, or improving one development score
does not establish novelty or a final-test result.
