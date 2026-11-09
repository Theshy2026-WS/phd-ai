# Candidate-pair evidence retrieval: first controlled experiment

This is a new, falsifiable module experiment. It does not establish architectural
novelty or promise an accuracy improvement. The existing strongest-independent-
CNN plus strongest-independent-ViT experiment still trains both encoders and the
concatenation classifier jointly. This new study fixes a **complete trained joint
model** so that the contribution of an additional evidence-retrieval module can
be isolated before considering backbone fine-tuning.

## What is different from the cited work?

| Method | Feature interaction | Training relevant to this comparison |
|---|---|---|
| Our existing joint concatenation | CNN pooled vector and ViT CLS are concatenated, then classified by one linear layer | Independently trained Tiny ImageNet encoders initialize a new stage; both encoders and the classifier update together |
| Conformer | Repeated bidirectional Feature Coupling Units connect intermediate CNN maps and Transformer tokens; two output classifiers | The published ImageNet recipe jointly trains the coupled architecture for 300 epochs; its standard command does not first independently train two encoders and then concatenate their final vectors |
| DBFF-Net | Final features from ShuffleNetV2 and DeiT-Tiny are projected and combined; five fusion alternatives include concatenation and gating | The selected backbones are frozen during fusion training |

Sources: [Conformer paper, sections 3.1 and 4.2](https://arxiv.org/html/2105.03889v1),
[official training code](https://github.com/pengzhiliang/Conformer), and
[DBFF-Net, sections 3.4 and 4.3](https://www.mdpi.com/2079-9292/15/9/1960).
Neither paper is identical to our complete training pipeline. However, final
feature concatenation and joint fine-tuning are established ingredients; a
different initialization, dataset or freezing policy alone is not sufficient
evidence of a new fusion mechanism. An exhaustive novelty claim remains unverified.

Reproducing an author's original result and adapting its mechanism to our setup
are different experiments. A useful local comparison must use our same split,
backbones, initialization, augmentation and computation budget, and disclose the
adaptations. The historical Conformer transfer score is not an author-faithful
reproduction or a matched comparison with our current joint model. We cannot
predict which method will win before running a controlled comparison. This queue
does not label its controls as reproductions of Conformer or DBFF-Net.

## Hypothesis and model

A pooled joint representation may identify plausible classes without using all
the spatial evidence needed to distinguish them. The hypothesis is that asking
**which of two plausible classes is better supported by the image's intermediate
ViT tokens** can improve that trained model more than a static query or a similarly
sized MLP. This mechanism and its advantage are hypotheses, not measured facts.

Let `g = [CNN_GAP; ViT_CLS]`, and let the frozen joint classifier give
`z0 = W g + b`. Obtain the five leading classes from `z0`, without labels.
For each unordered candidate pair `(a,b)`, form

```
h_ab = g * (W_a - W_b)
q_ab = normalize(P h_ab)
K = H W_K,  V = H W_V
r_ab = [softmax(q_ab K^T / sqrt(d_head))
        - softmax(-q_ab K^T / sqrt(d_head))] V
s_ab = u^T r_ab
```

`H` contains the 64 patch tokens after block 6 of the existing 12-block ViT.
The final normalized CLS is still computed by the entire original ViT, so the
anchor's predictions are preserved. `P`, `W_K`, `W_V` and `u` have no biases.
The retrieval width is 128 with four heads. Aggregate pair scores into candidate
logit changes using `eta/(5-1) * tanh(s_ab)`, adding to one candidate and
subtracting from the other. `eta` is a learned positive scalar initialized to
0.1. Scores outside the candidate set are unchanged. There is no new auxiliary
loss, label-dependent candidate choice, or external pretraining.

Here `normalize` means per-head unit RMS (L2 norm `sqrt(d_head)`), followed
by the displayed attention scaling. The bias-free scalar readout starts with
normal weight standard deviation 0.02. A synthetic gradient check before any
image experiment found that unit-L2 queries combined with the usual additional
square-root scaling and a 0.001 readout could start with gradients around the
AdamW epsilon. The fixed RMS convention and 0.02 readout avoid that numerical
starting condition in the check. This is initialization calibration, not measured
image accuracy or a selected training hyperparameter search.

The construction has checkable properties:

- The existing margin is exactly `z0_a-z0_b = sum(h_ab) + b_a-b_b`.
- Reversing the class pair reverses its correction. All candidate corrections
  sum to zero, and each absolute correction is at most `eta`.
- A base Top-1 margin greater than `2*eta` is protected. This protects some
  confident mistakes as well as confident correct predictions.
- A token-independent constant added to the projected values cancels in the
  signed retrieval. This does not prove that background or CNN-redundant
  information has been removed.
- The maximum unchanged outside score is no greater than the original minimum
  candidate score, which is no greater than the preserved candidate mean. Thus
  an outside class cannot strictly exceed every corrected candidate. With
  consistent tie-breaking, accuracy is bounded by the anchor's Top-5 coverage.
  The report also checks actual outside winners and numerical tolerance.

Class-directed queries, classifier-weight contributions, differential attention,
and residual correction all have prior art. The novelty of this particular
combination is **unverified**. These identities explain its behavior; they are
not a claim of a new mathematical theorem.

## Four arms, each at most 30 epochs

| Arm | What changes | Question |
|---|---|---|
| Pairwise retrieval | Image-conditioned class-pair query reads signed patch evidence | Does the proposed complete mechanism improve the frozen joint anchor? |
| Static query | Remove `g` from the query; retain the class-weight difference | Is adapting the question to the current image useful? |
| Antisymmetric MLP | Replace token retrieval with a similarly sized MLP using `h_ab` and mean image tokens | Is attention useful beyond extra parameters and nonlinear computation? |
| Training-mean tokens | Replace image-specific tokens during both training and inference with per-position training means | Does the added module use additional image-specific patch evidence? |

All arms use the same anchor, seed, training views, horizon and candidate update
rule. The mean-token control retains the image-specific anchor `g`, including its
ViT CLS; it is not a control without ViT information. Its positional token means
are estimated only from the fixed clean training probe, never from selection,
reserved audit, or official validation images.

The checkpoint selector records the completed eligible joint runs it compares.
It can reuse the known AugReg joint checkpoint (65.25% development Top-1) or a
completed newer strongest-pair joint suite. Incomplete or smoke suites are not
eligible. All four arms share the one selected complete checkpoint, including
both encoders and its trained classifier. A newer strongest-pair score remains
unverified until its completed metadata and checkpoint are available.

Train only the new module. Keep the entire anchor in evaluation mode, including
BatchNorm and dropout. A future joint fine-tuning stage is conditional on useful
evidence from these controls and is not included in this queue.

## Decision and runtime policy

Report best and last raw/EMA scores, epoch zero, NLL, correction and harm counts,
candidate coverage, learned correction scale, margin protection, parameters,
actual training time and elapsed suite time. Main metrics use the full 10K
development selection set. The reserved 10K audit and official validation are
not evaluated during this selection experiment.

If complete retrieval cannot improve the anchor and the two learned static/MLP
controls, the proposed benefit is unsupported in this setting. Do not extend
epochs or launch a seed search solely to rescue the hypothesis. A selected
successful candidate still requires independent confirmation and relevant
published-method comparisons before making a thesis contribution claim.

There is no new backbone pretraining, 4096-token attention, or experiment whose
planned runtime exceeds 10 hours. The runner enforces a bounded execution budget,
reserves time for reports/email, and records partial results if the budget is
exhausted. It sends one completion or partial-result email. **The instance stays
running on success, failure and timeout.** Checkpoints and full feature caches
remain on the VM; email attachments contain ordinary reports only.

The default cumulative suite budget is 10 hours. Training and analysis stop at
9 hours 45 minutes from the initial invocation, reserving 15 minutes for delivery.
The independent watchdog also handles a child with stalled output. On Linux it
owns a new process session and terminates that experiment's process group,
including DataLoader descendants; it never controls the VM. Interrupted epochs
replay from the last atomically committed complete epoch.

Resume deducts already spent active wall time, including preflight and delivery.
It cannot enlarge or reset the saved budget. Idle time between cleanly finished
invocations is excluded. If the wrapper itself was killed without finalizing its
budget record, time since that invocation began is conservatively charged until
resume. An exhausted budget does not authorize another 10-hour run.

## Workbench command

If the strongest-independent-pair joint run is still running, let it finish before
starting this queue. This avoids concurrent GPU jobs and makes its completed
checkpoint eligible for selection. The source choice and all compared eligible
scores are written to `suite.json`; inspect the printed suite path and source.

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
tmux new-session -A -s pairwise_resolver
```

Inside tmux:

```bash
python -u run_pairwise_resolver.py \
  --data-dir ./data \
  --num-workers 4 \
  --epochs 30 \
  --max-hours 10
```

The default source paths are the previously completed continuation CNN suite
`runs/fusion_continuation_tiny/suite_20261106T194319_c2dd7a` and AugReg suite
`runs/vit_augreg_tiny/suite_20261107T182435_87170e`. The selector also checks
completed suites in `runs/best_pair_joint_tiny/`. To pin a newer joint source,
add `--base-suite runs/best_pair_joint_tiny/ACTUAL_COMPLETED_SUITE`.

`--check-only` checks sources, GPU/data and SMTP login without starting training
or sending an email. Credentials use the existing ignored `email.txt` setup.
Resume an interrupted queue with remaining budget using:

```bash
python -u run_pairwise_resolver.py \
  --resume runs/pairwise_resolver_tiny/ACTUAL_SUITE_DIRECTORY
```

Completed epochs are preserved; new optimizer/scheduler horizons are not
invented on resume. Best and best-trained indexes can be reconstructed from the
atomic last checkpoint if interruption occurs between checkpoint writes.

The email ZIP contains `training_analysis` and `feature_analysis`. Reports
include globally paired, label-blind wrong-image patch interventions and exact
checkpoint-stored training-mean interventions. Raw image galleries are deferred
and explicitly marked unperformed. A timeout sends partial evidence and does not
label an incomplete comparison as a completed result.

## Local validation

The 53 new tests pass, including gradient and algebraic properties, frozen
BatchNorm/dropout, source selection, complete-epoch recovery, raw/EMA state,
candidate bounds, global mismatch pairing, cumulative time accounting, owned
process-group termination and mocked completion/partial email delivery. The
related pre-existing regression suite has 96 passes and one platform skip.
Both new CLIs parse successfully, and the four new modules compile.

A CPU check of the actual full CNN8 plus 12-block Small/8 ViT architecture found
an FP32 logit difference of exactly zero between the original forward path and
the new frozen extraction path. The anchor has 45,373,064 fixed parameters;
the attention resolver has 409,729 trainable parameters and the MLP has 408,611
(0.27% difference). These checks use synthetic input, not Tiny ImageNet results.
No new GPU accuracy, actual queue duration, SMTP transmission or VM action has
been verified locally.
