# Independently trained ViT evidence for a mature CNN: HCSI comparison

The research question is whether a CNN that has reached an observed development
plateau under a fixed recipe benefits from an independently task-trained ViT,
and which fusion mechanism produces useful corrections with fewer harmful
changes. A plateau under one recipe is not a theoretical CNN accuracy ceiling.
The experiment can support or reject this question; it does not assume the
desired answer or establish architectural novelty by itself.

## Evidence motivating this comparison

The completed strongest-pair suite is
`runs/best_pair_joint_tiny/suite_20261109T014414_c55052`.

| Result | Development Top-1 | Details |
|---|---:|---|
| Strong independent CNN |64.80%|Selected source checkpoint reflects213 original training epochs|
| Strong independent ViT |50.18%|Small/8 AugReg, selected raw epoch273 from300|
| Previous joint concatenation |65.25%|Earlier CNN source; EMA@2|
| Strongest-pair joint concatenation |65.45%|EMA@7; both full encoders update|
| Actual joint last raw / EMA |65.17% /65.29%|Epoch30|

The new joint stage took83.06 minutes of training and97.16 minutes overall.
Its best checkpoint corrects455 source-CNN mistakes and harms390 correct CNN
predictions, a net65/10,000 or0.65 percentage points. It also corrects199 images
that both independent Top-1 predictions missed. The best joint NLL1.79313 is
almost identical to the source CNN1.79458 and worse than the previous joint
model1.70535. Thus accuracy improved, while probability quality did not improve
across the board.

Removing the co-adapted ViT contribution reduces65.45% to63.96%; replacing it
with a training-only feature mean gives63.97%; global wrong-image substitution
gives62.78%. These interventions support use of image-specific ViT information.
The masked CNN is not an independently trained CNN baseline. No reserved audit
or official validation results were used.

The missing necessary control is a new CNN-only continuation from the **same
64.80% checkpoint** with the same30-epoch adaptation recipe and training views.
The inherited earlier CNN continuation is not this matched new phase.

## Relationship to prior work

The closest work for this comparison is
[HCSI-Net, IWBF2026](https://lmi.fe.uni-lj.si/wp-content/uploads/2026/03/IWBF_2026_paper.pdf).
It separately trains CNN/Transformer backbones, then loads them for end-to-end
training. Its main mechanism is multistage bidirectional spatial cross-gating;
its no-interaction ablation retains only final pooled feature fusion. This is
a relevant methodological comparison, not evidence that all our implementation
details were already identical.

[Wrist2025](https://www.mdpi.com/2076-3417/15/22/11865), Tables5-6, also builds
hybrids from individually task-adapted weights and updates parts of both
branches. It uses partial unfreezing, SE refinement and a different head.
It remains relevant related work even though our main route updates all
participating encoder parameters. Earlier experiments freezing both encoders
cannot reject every possible partial-freezing policy.

A focused review should compare serial CNN-token Transformers, parallel
intermediate coupling (Conformer/HCSI), final feature fusion, and independent
pretraining followed by frozen or joint adaptation. For each source, record the
actual formula, feature readout, initialization, trainability, supervision and
matched controls. Neither a paper title nor a different dataset is enough to
decide whether the method is the same. The broad independent-training/fusion/
joint-fine-tuning route already has precedents; a contribution must specify the
new technical rule, mechanism or generalizable experimental finding.

## What is reproduced, and what is adapted

This study reproduces the **HCSI interaction mechanism on Tiny ImageNet** with
the same independently trained CNN and ViT used by our existing experiment.
It is not an author-faithful rerun of the SBI/deepfake benchmark. The published
EfficientNet-B4/Swin-B backbones,224px inputs, binary task, data construction and
cross-dataset AUC are not used; their reported scores are not numerically ranked
against our200-class Top-1.

The reference implementation was read at fixed commit
`85944b39ee2e509f5b8d10b03eb95a82e9c17752`,
[models/architecture/HCSINet.py](https://github.com/markobrodaric/HCSI-Net/blob/85944b39ee2e509f5b8d10b03eb95a82e9c17752/models/architecture/HCSINet.py).
It uses descriptor width64, position coefficient0.2, temperature1.0, bias-free
feature/position projections, normalization `x/(norm(x)+1e-6)`, and one shared
nonaffine LayerNorm for retrieved context. The model constructor does not freeze
backbone parameters; the official training script optimizes `model.parameters()`.
The upstream MIT notice is retained in `third_party/HCSI-Net-LICENSE.txt`.

At each stage, projected and normalized CNN nodes and ViT patch tokens receive
normalized2D position descriptors. Their cross-similarity matrix gives separate
row-softmax weights in each direction. A descriptor's dot product with normalized
retrieved context produces a sigmoid spatial gate, which multiplies the original
stream features. Gates remain in[0,1]; this implementation does not silently
replace them with residual addition, identity mixing or twice-sigmoid gates.

The native64 mapping uses four CNN stages at32,16,8,8 grids and ViT block groups
1-3,4-6,7-9,10-12 with64 spatial tokens throughout. Rectangular cross-attention
matrices allow unequal CNN and ViT node counts without upsampling. CLS remains
in the Transformer computation but is excluded from spatial gates and final
patch-mean readout. The plain ViT's final LayerNorm is applied **before** the last
gate: applying per-token LayerNorm after a scalar gate would largely cancel its
effect. This ordering is an explicit adaptation to the existing plain ViT.

Both fusion arms use the same CNN readout and final normalized patch-mean ViT
readout. Their identical3-layer classifier is2432->1024->512->200, with BatchNorm,
ReLU and Dropout0.5 after hidden layers, using the author's default Linear
initialization. The head is initialized independently of gate construction so
the paired no-interaction and HCSI heads start identically. Epoch-zero scores
must be reported: a new nonlinear head and multiplicative gates may initially
degrade a strong pretrained representation.

## Five new arms, each at most30 epochs

| Job | Purpose |
|---|---|
| `cnn_continue` | Same64.80% CNN, inherited linear head; controls additional adaptation training |
| `late_mlp` | Independently trained CNN/ViT, patch-mean pooling and HCSI-sized MLP; no intermediate interaction |
| `hcsi` | Same weights, pooling and classifier as `late_mlp`, plus four HCSI stages |
| `cnn_mlp` | Same CNN with the same hidden classifier widths; controls a deeper classifier without ViT |
| `late_mlp_random_vit` | Same fusion architecture and CNN, but ViT begins randomly; tests independent ViT training rather than merely adding a branch |

The completed65.45% CLS-plus-linear experiment is reused as a reference; it is
not rerun. Changes between it and `late_mlp` include both readout and classifier,
so that comparison does not isolate either change. `hcsi` versus `late_mlp`
isolates the HCSI interaction in this adaptation. CNN-MLP has a smaller input
dimension than the fusion MLP; report actual parameters rather than calling
their full models parameter matched.

All participating encoders update; there is no partial freezing policy. Both
pretrained and random-ViT arms use one declared seed, not a seed sweep. The random
ViT has the same architecture as the pretrained one; its extra independent
pretraining cost is deliberately absent and the inherited training cost of the
pretrained branch is reported separately.

The new stages use the existing fusion recipe: batch64, AdamW encoder LR5e-5,
classifier/HCSI-module LR5e-4, weight decay0.05 with norm/bias exclusions, gradient
clipping, Mixup/CutMix, label smoothing, one warmup epoch then cosine, and EMA0.999.
All use the same augmentation-view indices101..130. The source CNN's actual
training history is recorded separately. No new independent pretraining/search
or4096-token global attention is scheduled.

## Interpretation and completion policy

Check whether ordinary joint fusion exceeds the new same-source CNN-only
continuation; whether trained-ViT MLP fusion exceeds both the CNN-MLP and random
ViT controls; and whether HCSI exceeds its no-interaction MLP control. These are
distinct claims. If HCSI loses, it does not erase the observed benefit of ordinary
late fusion. If random ViT matches trained ViT, independent ViT pretraining has
not demonstrated an advantage under this adaptation budget.

Report epoch zero, best raw/EMA, best actually trained, actual last, NLL, training
probe gap, trainable parameters, correction/harm counts, gate/routing statistics
and measured time. Comparisons are matched by data and update schedule, not
identical FLOPs or wall time; report the additional computation rather than
claiming a free gain. Independent confirmation and a second suitable dataset
are later steps after selecting a justified candidate. The reserved10K audit and
official validation remain unused during screening.

The cumulative queue budget is10 hours, including preflight and delivery, with
15 minutes reserved by default for reports/email. Incomplete jobs remain marked
partial; no extra hours are granted on resume. Training, source validation,
timeout and email failures never request a Workbench stop or OS shutdown.
Only experiment child processes are terminated on timeout; **the instance stays
running**. New source files leave the existing best-pair/pairwise run protocols
unchanged.

## Workbench command and local validation

Run this after another GPU queue, including pairwise, has finished. Both runners
keep the instance running. They have separate suite directories and do not
modify an already created queue's source files or checkpoints.

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
tmux new-session -A -s hcsi_tiny
```

Inside tmux:

```bash
python -u run_hcsi_tiny.py \
  --data-dir ./data \
  --num-workers 4 \
  --epochs 30 \
  --max-hours 10
```

The three source suites, including the completed65.45% joint reference, must
remain on the same VM disk. Source identities, data split and execution
environment are checked before new training. Add `--check-only` for preflight
without suite creation, training or a transmitted email. Existing ignored
`email.txt` credentials are used and excluded from child environments and ZIPs.

Resume with remaining cumulative budget:

```bash
python -u run_hcsi_tiny.py \
  --resume runs/hcsi_tiny/ACTUAL_SUITE_DIRECTORY
```

The runner attempts one completion or partial-result email, retrying failed SMTP
transmissions within its reserved budget. The reports-only ZIP contains
`training_analysis` and `feature_analysis`. Full weights, optimizer state and
prediction NPZs remain on disk. SMTP acceptance is recorded separately from
unverified inbox delivery. Missing or interrupted jobs are explicitly partial.
An ordinary failure in one arm is recorded while the remaining independent arms
continue within the budget; a time limit stops further training.

All52 new CPU tests pass. They include an independent NumPy reference for HCSI
gating, full-gradient and final-gate-effect checks, matched no-interaction output,
random/pretrained ViT isolation with identical CNN and classifier initial values,
optimizer roles, checkpoint transaction recovery, exact interrupted-versus-
continuous training state, failure isolation, read-only diagnostics, full-split
paired comparisons, cumulative budgets and mocked email/no-VM-stop paths.

On synthetic native64 input, the actual CNN8 plus12-block Small/8 ViT
`no_interaction` HCSI path exactly matches its corresponding late-MLP path
(maximum FP32 logit difference0.0). Late-MLP has48,008,328 trainable parameters;
HCSI has48,353,416, with345,088 additional gate parameters. The new modules compile
and the runner CLI parses. These checks are not new Tiny ImageNet results.
Actual GPU performance, elapsed training duration and mail delivery remain
unverified until the user runs the queue.
