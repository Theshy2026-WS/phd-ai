# AugReg ViT training and matched CNN–ViT fusion

This finite study builds a stronger independent ViT before testing fusion.
It follows the practical principles of [Steiner et al., How to train your ViT?](https://arxiv.org/abs/2106.10270): consider compute, augmentation, regularization and model scale together; augmentation and regularization are separate choices; a larger patch size can make a wider model practical.
The paper trains on ImageNet, not native64 Tiny ImageNet. This study adapts its
principles and does **not** reproduce its results or its exact hyperparameters.

## Independent ViT candidates

All candidates train **from scratch for 300 epochs**, seed42, on the existing
80,000-image training partition. The 10,000-image development selection
partition chooses checkpoints and the candidate. Neither the audit partition
nor official Tiny ImageNet validation is evaluated. No ImageNet-pretrained
checkpoint, distillation teacher, extra dataset, image upscaling or seed sweep
is introduced. Tiny ImageNet derives from ImageNet, so adding external
ImageNet pretraining would also require a separate overlap and protocol study.

| Profile | Width / depth / heads | Patch / patch tokens | Dropout / maximum stochastic depth | Torch RandAugment | Mixup alpha | AdamW weight decay |
|---|---|---|---|---|---|---|
| `vit_tiny_p4_aug` |192 /12 /3|4 /256|0 /0|2 operations, magnitude7|0.2|0.1|
| `vit_small_p8_aug` |384 /12 /6|8 /64|0 /0|2 operations, magnitude7|0.2|0.1|
| `vit_small_p8_augreg` |384 /12 /6|8 /64|0.1 /0.1|2 operations, magnitude9|0.5|0.03|

Each uses full pre-LN attention and 4x GELU MLP blocks, learned CLS/position
embeddings, LayerNorm epsilon1e-6, a linear classifier, and explicit FP32 ViT
arithmetic. SDPA computes ordinary global attention within the ViT's65 or257
tokens; there is no4096-token attention path. Stochastic-depth rates grow
linearly across blocks. The three profiles are a bounded search over complete
recipes; the third changes several factors and is not a one-variable ablation.

The shared independent training schedule uses AdamW beta=(0.9,0.999),
epsilon1e-8, peak learning rate5e-4, 20 warmup epochs from1% of peak, and a
single300-epoch cosine ending at1e-6. Microbatch64 and four gradient
accumulation steps give effective batch256 except a possible partial final
window. Each microbatch loss is weighted by its actual fraction of the window
before backward; gradients are then unscaled, clipped to global norm1 and
stepped once per window. Mixup pairs images within each64-image microbatch,
not across the whole256-image accumulation window. EMA updates
after successful optimizer updates. There is no early stopping.

Native64 augmentation uses bicubic random resized crops with area0.5–1.0,
horizontal flip and torchvision RandAugment, followed by the established
ImageNet normalization. Mixup is applied with probability1, CutMix is disabled
for independent ViT training, and label smoothing is0.1. This is a practical
native64 recipe, not the paper's batch4096 recipe. Torch RandAugment magnitudes
are not numerically equivalent to the paper's TensorFlow implementation.

All scheduled candidates must finish before a formal winner is selected.
The rule is highest best development Top-1, then lower NLL, then declared
profile order. Raw and EMA checkpoints are evaluated explicitly. The winner
record includes the exact checkpoint SHA256 and architecture. This selection
does not guarantee the new ViT beats the old45.97% model; reports show the
actual result and the full search cost.

## Fusion and missing control

Every job below runs30 epochs and starts from the **same original CNN100 best**,
not the CNN that was later adapted to64.80%. New-ViT jobs use the same selected
independent ViT checkpoint. This holds the CNN source fixed when comparing
old and new ViT features.

| Job | Trainable part | Purpose |
|---|---|---|
| `cnn_frozen_head` |Original CNN linear classifier only|Missing frozen-CNN readout control|
| `new_frozen_concat` |Concatenated-feature linear classifier|Compare new ViT with old frozen fusion65.02%|
| `new_joint_concat` |Both encoders and concatenated-feature classifier|Compare with old joint30 result64.69%|
| `new_frozen_normalized` |Linear classifier after parameterless branch LayerNorm|Compare branch normalization with ordinary concat|
| `new_frozen_residual_linear` |Bounded linear correction, fixed original CNN classifier|Anchor and correction control|
| `new_frozen_residual` |Bounded MLP correction, fixed original CNN classifier|Test nonlinear correction on fixed representations|
| `new_frozen_cross_patch` |One-query cross-attention and correction, fixed CNN classifier|Read independent ViT patch evidence using the CNN summary|
| `new_frozen_mlp_patch` |Mean-pooled patch MLP and correction, fixed CNN classifier|Similar-parameter non-attention control|

The standard concat head has inherited original CNN columns and bias, with
small nonzero new-ViT columns. The normalized variant adds parameterless
LayerNorm to each branch before that same type of head; normalization changes
epoch-zero predictions because inherited CNN-head calibration changes.
Epoch-zero scores are therefore retained rather than assumed equal.

Anchored corrections have the form
`z = z_original_CNN + scale * tanh(correction(features))`.
The learned scale is sigmoid bounded, starts at0.1 and is below1.0. These are
experimental constraints, not a guarantee that correct predictions are never
changed. The linear and MLP correction variants share that anchor, branch
normalization and output bound. Comparing them to unconstrained concat changes
the whole design and must not be described as an isolated test of one factor.

Cross-attention projects the pooled CNN representation into a single query.
Keys/values come from the selected independent ViT's64 or256 patch tokens.
The patch MLP control uses mean pooling and approximately matches trainable
parameter count; actual counts and FLOPs are reported, not assumed identical.
This structure has prior art, including [CrossViT](https://arxiv.org/abs/2103.14899),
and this suite makes no claim that cross-attention itself is new.

Fusion jobs retain the old fusion augmentation, Mixup/CutMix, smoothing,
microbatch64, seed42 and30-epoch schedule. There is no fusion gradient
accumulation. Joint encoder learning rates are5e-5 and the classifier rate is
5e-4. Frozen jobs fix both encoder parameters and BN buffers, with encoder
dropout disabled. Exact frozen state is also copied into EMA to avoid drift.

## Analysis and interpretation

The study delivers candidate histories, best/last/raw/EMA metrics, full clean
training and development diagnostics, actual compute/memory/time, original
checkpoint identities, and independent old/new ViT comparisons. It evaluates
probability mixtures and feature-fusion results separately, with correction
and harm counts and three-way correctness intersections. Full prediction NPZs
remain on the VM for later analysis.

Fusion interventions reuse the same extracted features: remove the ViT,
replace it by means estimated only from a fixed clean training probe, and
substitute another image's ViT representation. Cross-attention additionally
uses uniform weights as a control. Permuting the order of complete patch
tokens after ViT is *not* used as evidence: attention pooling is invariant to
joint key/value permutations. These interventions describe a trained model;
they do not replace independently trained controls or establish a causal
percentage contribution from feature/logit norms.

For bounded corrections, analysis records the learned scale, maximum absolute
logit change and the number of anchor predictions whose first-versus-second
logit gap exceeds twice that scale. It verifies that predictions inside this
safe region are unchanged. This follows directly from the per-logit bound;
it is not a guarantee that the anchor is correct or a claim of a new theorem.

Checkpoint selection uses development labels. Scores and exploratory paired
statistics are not final held-out claims. Old joint130 score65.14% is a
longer-training reference; new joint30 should first be compared to old joint30.
Improvements in an independent ViT do not guarantee better fusion.

## Workbench execution

The default queue is three300-epoch ViTs plus eight30-epoch controls/fusions:
**1140 new model-epochs**. It runs the complete queue without a time cutoff.
Total time depends on GPU throughput; this is longer than the previous suite,
but contains no shallow64 global-attention run. Original results are read only.

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
tmux new-session -A -s vit_augreg
```

Inside tmux:

```bash
python -u run_vit_augreg_overnight.py \
  --source-suite runs/vit_followup_tiny/suite_20261105T033555_5ba88b \
  --data-dir ./data --num-workers 4 \
  --vit-epochs 300 --fusion-epochs 30 \
  --complete-all --fallback-os-shutdown
```

The wrapper checks GPU/software, source/data identity, SMTP authentication and
Workbench stop access before launching. It sends no start email. After all
training and analysis, it sends one reports-only ZIP and then requests managed
instance stop, with the explicit OS fallback. Mail/package failure also
requests stop while leaving results on disk. SMTP acceptance does not prove
inbox delivery. A hung training process has no hard watchdog.

Credentials use the existing ignored `email.txt` or Gmail environment settings,
are removed from the child environment, and are excluded from archives.
Source scripts, checkpoints, NPZ arrays, datasets and nested archives are not
mailed. Aggregated class/mixture/intervention statistics remain in compact ZIPs.
Add `--check-only` for preflight without training, notification or shutdown.

Resume the **new** suite:

```bash
python -u run_vit_augreg_overnight.py \
  --resume runs/vit_augreg_tiny/SUITE_DIRECTORY \
  --fallback-os-shutdown
```

Resume preserves the recorded profiles, full schedules, sources, software and
checkpoints. `--smoke` creates a separate one-epoch-per-job fixture; it cannot
be extended or presented as the formal study. CPU tests mock mail and VM control.
