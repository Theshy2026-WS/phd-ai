# Tiny ImageNet: investigate the gap, then test bounded remedies

The current primary research queue is [attention feature fusion](ATTENTION_FUSION_STUDY.md).
The19-job generalization study below is supporting work, not a prerequisite
for testing attention and not the command for the new attention-first run.

The current parent fits the clean80K training images at99.97% and scores
64.28% on the held-out10K selection set. This is a real generalization gap.
It is not, by itself, evidence that the loaders use different datasets.
Both sets are stratified subsets of the same Tiny ImageNet training folder,
with the same class-index mapping and clean preprocessing. A semantic label
mistake or pipeline defect still needs an explicit audit rather than inference
from the size of the gap.

The last study established that direct full8 inference pooling, train-only BN
recalibration, a residual U decoder, and mild Gaussian noise did not improve
Top1 under those tested settings. Other readout phases were much less reliable
on held-out images. This motivates controlled readout/optimization tests;
it does not establish the entire cause or promise a particular accuracy.

## What runs overnight

The default is **complete-all**: finish the finite, explicitly listed analyses
and19 training jobs without a wall-clock cutoff. `--complete-all` states this
choice explicitly. Runtime is still recorded every30seconds. It can take more
than one night; completion notification and stop happen after the queue ends.
The number of epochs and jobs remain finite: there is no automatic expansion
into unlimited model or hyperparameter search.

An explicit `--max-hours 12` retains the earlier bounded behavior, including
a45minute final-analysis reserve and45minute audit-stage caps. The two modes
are mutually exclusive. A running older12hour job does not change mode when
new code is pushed: do not pull changed sources into an active job. Resume
retains the recorded code and mode; use a new suite for the new protocol.

### 1. Source, content and augmentation audit

`data_audit/` examines the actual80K training and10K selection image files:

- Verify their shared source folder, image-ID disjointness, path/label/class
  mapping agreement, per-class counts, decodability and native64x64 dimensions.
- Hash compressed files and decoded RGB pixels, report exact cross-set
  duplicates and conflicting labels. Coarse perceptual-hash bucket matches
  are candidates only, not proof of duplicate or semantically identical images.
- Compare brightness, color, dimensions and per-class descriptive statistics.
  Similar aggregate statistics do not prove semantic distribution equivalence.
- Replay32 identical source images through the clean training and selection
  views and require identical tensors, targets and IDs.
- Evaluate a fixed4K clean training probe and10K selection images under clean,
  flip, four reflective2-pixel shifts, crop/flip only, RandAugment/flip only,
  and the inherited full augmentation. These are inference diagnostics;
  selection images are never used to train or fit augmentation parameters.

Concrete decoding, mapping, dimension or same-image preprocessing failures
set `training_allowed=False` and block new training. The workflow does not
delete data or relabel images automatically. Duplicate candidates or color
statistics differences alone do not stop training.

### 2. Actual last checkpoints from the previous screen

`previous_last/` evaluates the three saved epoch30 `last.pt` models on clean
80K/selection10K and compares each with the original parent. The previous
selected `best.pt` was epoch0, so reusing its class/error reports cannot
explain what the trained U models changed.

For each U model, a hook captures original-head and correction logits. The
analysis compares the complete output with removal of only the correction
at inference, reports correction/base logit RMS, changed predictions and
corrections/harms. This is an intervention on its co-adapted encoder, not a
newly trained baseline. Original checkpoint files remain unchanged. If the
old fusion suite is absent, its report explicitly says unavailable and the
new experiments can still run.

### 3. Matched training remedies

Every job retains seed42 and the original80/10/10 split. All encoder weights
are trainable. Most jobs retain CNN8's23,902,472 parameters. The explicitly
declared shallow26 control has14,340,616 parameters; it tests a different depth,
capacity and receptive field. No bigger decoder or attention is added.

| Queue | Job | Initialization | Horizon | Question |
|---:|---|---|---:|---|
|1|`warm_control`|Original CNN8 best|30|New short-schedule control, AdamW peak1e-4|
|2|`warm_full8`|Same parent|30|Can full8 pooling work when trained that way?|
|3|`warm_phasebalanced`|Same parent|30|Does directly supervising all four grids improve stability?|
|4|`warm_gentle`|Same parent|30|Does less destructive crop/RA and removal of CutMix help?|
|5|`warm_fp32_control`|Same parent|30|Precision-matched control for SAM|
|6|`warm_sam`|Same parent|30|Does a sharpness-aware objective help?|
|7|`fresh_control`|Random seed42|100|Matched fresh-training reference|
|8|`fresh_full8`|Identical random initialization|100|Full8 readout trained from the start|
|9|`fresh_phasebalanced`|Identical random initialization|100|Additional phase supervision versus fresh_full8|
|10|`fresh_fp32_control`|Identical random initialization|100|Precision-matched fresh SAM control|
|11|`fresh_sam`|Identical random initialization|100|SAM before memorization, compared with fresh_fp32_control|
|12|`fresh_gentle`|Identical random initialization|100|Gentler augmentation from the start|
|13|`fresh_no_cutmix`|Identical random initialization|100|Remove only CutMix|
|14|`fresh_no_randaugment`|Identical random initialization|100|Remove only RandAugment|
|15|`fresh_sgd`|Identical random initialization|100|SGD-momentum recipe versus AdamW recipe|
|16|`fresh_shallow26`|Shared surviving initialized tensors|100|Two bottlenecks/stage at unchanged widths and strides|
|17|`fresh_groupnorm`|Identical shared parameters; new normalization|100|GroupNorm throughout the backbone/head instead of BN|
|18|`fresh_wd_low`|Identical random initialization|100|Weight decay.01 versus default.05|
|19|`fresh_wd_high`|Identical random initialization|100|Weight decay.1 versus default.05|

Warm jobs use a fresh optimizer, peak1e-4, one warmup epoch and cosine decay
to1e-6. They do not resume the old optimizer. Fresh jobs use the inherited
5e-4 peak, two warmup epochs and cosine decay to1e-5. Fresh results must be
compared with fresh controls at matching horizons, rather than with a model
that already received100 training epochs.

Full8 normalizes and pools the complete feature map. Phase-balanced training
uses `0.5*CE(full8) + 0.125*sum(CE(four phases))` on the same soft targets.
The head BN is called once on the full map, avoiding four biased running-stat
updates. This is a mechanism hypothesis, not a claimed new method.

Gentle augmentation changes crop scale to.85..1, RandAugment to1operation/M3,
and disables CutMix while retaining Mixup.2/probability.5, smoothing.1 and
weight decay.05. It tests a recipe bundle; it does not isolate all three
augmentation factors. Separate fresh no-CutMix/no-RandAugment jobs isolate
those two factors; crop remains part of the gentle bundle. The baseline already had regularization, so this is
not presented as fixing a previously unregularized pipeline.

SAM uses rho.05 and AdamW, two FP32 passes on the identical already-mixed
batch/targets. Perturbed weights are restored exactly even on failure. The
second pass uses batch normalization without updating running statistics or
counters again. Both warm and fresh FP32 controls match their SAM job's
precision and learning rate. Compare those matched pairs for SAM claims.
This is a SAM-inspired AdamW experiment, not an exact author-code reproduction.

The SGD job uses LR.1, momentum.9, Nesterov, WD5e-4, five warmup epochs and
cosine minimum1e-4. This is an optimizer-family recipe comparison, not an
optimizer-only ablation. The GN job replaces all49BN modules with GN (up to32
groups), retaining shared parameter initialization. Its benefit is not assumed
at ourbatch64. The shallow26 model retains the first two initialized blocks
in each stage; its improvement would not isolate capacity from depth/RF.

Every job also maintains an EMA (decay.999) of parameters and floating BN
buffers; integer counters are copied. Raw and EMA validation metrics are
recorded every epoch. Selection Top1 chooses raw/EMA best checkpoints,
including epoch0, with strict improvement and earlier/raw ties retained.
This increases development-set comparisons and does not provide a final test
estimate. Clean4K probes run at epoch0/1/every5/final, never choosing a model.

Best and **actual last raw** checkpoints each receive full clean80K/selection
10K class/confusion/error analysis. If best remains epoch0, trained-model
errors are still retained. The EMA last score is recorded separately. The
criterion is improved held-out performance, not merely a reduced training
accuracy or smaller gap.

### Budget and recovery

In explicit bounded mode, the cutoff is cooperative: it is checked between epochs with a small estimate
for the next epoch. One in-flight epoch or complete evaluation can finish
after the cutoff; the reserve covers normal final evaluation and packaging.
This is not a hard kill watchdog for a hung GPU, network call or loader.
Data and previous-model audits each have45minute stage limits only in bounded mode. Budget-limited
jobs are checkpointed and labeled incomplete/deferred rather than successful
full runs. Independent model failures do not discard other authorized work.

Complete-all passes no deadline to either audit or the training study. It
still handles exceptions, checkpoints, sends a single completion notification
and requests Workbench stop. A genuinely hung process does not finish and has
no watchdog in either mode.

Resume retains the original source, parent, dataset, software/GPU, horizons,
and completion/budget mode. In bounded mode it continues interruptions within
the remaining budget, rather than creating another12hours after exhaustion.

## Workbench command

Use the existing environment and original spatial checkpoint. First:

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
tmux new-session -A -s generalization_all
```

Skip the last line if already inside tmux. Then:

```bash
python -u run_generalization_overnight.py \
  --parent-run runs/spatial_tiny/suite_20261103T175622_a19c86/cnn8_seed42 \
  --previous-fusion runs/mechanism_tiny/suite_20261104T014344_7caf47/fusion30 \
  --data-dir ./data --num-workers 4 \
  --warm-epochs 30 --fresh-epochs 100 --complete-all \
  --fallback-os-shutdown
```

Preflight validates GPU, data/checkpoint, SMTP login and stop permission without
sending an email. Add `--check-only` to stop after those checks. Existing email
credentials and recipient are reused. Ctrl+B then D detaches tmux.

After all work ends or fails, the wrapper packages **three** materials
(`data_audit_analysis.tar.gz`, `previous_last_analysis.tar.gz`,
`training_analysis.tar.gz`) into one completion email attachment, then requests
managed Workbench stop with optional OS fallback. Budget-limited results are
marked `BUDGET LIMIT / PARTIAL`. Credentials/datasets/checkpoints/NPZ are not
attached. Large per-image files are compacted with an omission manifest;
full outputs remain on disk. Packaging/mail failure still reaches the stop
request after the child has started.

```bash
python -u run_generalization_overnight.py \
  --resume runs/generalization_tiny/SUITE_DIRECTORY \
  --fallback-os-shutdown
```

## Baseline and attention

The baseline is PreActResNet50 with3/4/6/3 preactivation CNN bottlenecks. CNN8
keeps the last stage at8x8 before the declared readout. There are **zero
attention blocks**. Full8, multi-phase supervision, BN/GN, EMA and the earlier
U-shaped convolutional decoder do not introduce attention. Historical BoTNet,
CoAtNet, AlterNet and Conformer experiments belong to the wider comparison
history; they are not the current pure-CNN control. Absence of attention is
not established as the cause of the generalization gap.

## Evidence and coverage: what is implemented versus still missing

Implementation is not a positive experimental result or elimination of a
cause. The file audit checks labels against file paths, not human semantic
correctness. Color statistics do not test shape/background shortcuts. One
BN recalibration does not exclude every normalization issue. The matrix below
records these limits rather than calling19jobs an exhaustive explanation.

### Direct Tiny ImageNet evidence

| Primary source | Verified evidence and setting | Coverage in this code |
|---|---|---|
|[SEAL, CVPR2023](https://arxiv.org/html/2304.04858v1)|The paper explicitly states all compared Tiny ImageNet models reach100% training accuracy. Table2: normal160epochs54.37%, normal1600epochs49.27%, LLF56.92%, SEAL59.22%. Their100K/SGD protocol differs from ours.|**Not implemented**: early-layer gradient-ascent/descent over generations is distinct from SAM and fresh training. This is the strongest directly similar gap evidence.|
|[Fortuitous Forgetting/LLF, ICLR2022](https://arxiv.org/pdf/2202.00155)|Tiny/ResNet50 later-layer resets across generations improve generalization; selective reset differs from restarting every layer.|**Not implemented**: current fresh controls test when constraints are introduced, not selective forgetting.|
|[PuzzleMix, ICML2020](https://proceedings.mlr.press/v119/kim20b/kim20b.pdf)|Table4 ordinary Top1: vanilla57.23%, inputMixup56.59%, CutMix56.67%, PuzzleMix63.48% (converted from errors). Saliency/local-statistic mixing differs from random mixing.|**Partially related tests**: no-CutMix/gentle controls; PuzzleMix saliency/transport is not implemented.|
|[Co-Mixup, ICLR2021](https://janghyun1230.github.io/paper/comix21.pdf)|Tiny/PreAct18 Table1: vanilla56.60%, Co-Mixup64.15%; long1200epoch protocol and specialized graph-cut dependency.|**Not implemented**: ordinary Mixup is not joint saliency/diversity mixing.|
|[When Does Re-initialization Work?, PMLR187/2023](https://proceedings.mlr.press/v187/zaidi23a/zaidi23a.pdf)|Table2: tuned regularized Tiny baseline59.12%, Shrink&Perturb58.95%, with distillation61.03%; equal update budget. Gains depend on existing regularization and label noise.|**Not implemented**: S&P/distillation; warns that method gains cannot simply be added.|
|[ALPS, IJCAI2021](https://www.ijcai.org/proceedings/2021/0343.pdf)|Table4 clean Tiny/PreAct18: ERM60.95%, dropout62.48%, smoothing61.59%, Mixup62.95%, ALPS63.77%;100K/SGD/official-validation protocol.|Mixup/smoothing exist; **ALPS itself not implemented**.|
|[Stanford2016 course report](https://cs231n.stanford.edu/reports/2016/pdfs/401_Report.pdf)|Table2 has98.8% training versus45.5% validation for an older9-layerConvNet. This is a course report, not a peer-reviewed matched baseline.|Supporting illustration, not a method implementation or comparable benchmark.|
|[Stanford2017 course report](https://cs231n.stanford.edu/reports/2017/pdfs/937.pdf)|Discusses smallerResNet18, cyclicLR and snapshot ensembles; dropout hurts its ResNets. Its99% line says'testing' and is ambiguous, so is not treated as a verified train score.|Shallow control added; **cyclicLR/snapshot ensembling not implemented**.|
|[Vision Transformers2022 Tiny update](https://arxiv.org/html/2205.10660v1)|The91.35% Swin result uses ImageNet pretraining and384x384 interpolation, not native64 scratch training.|Historical attention comparisons exist; **that pretraining/resize protocol is not included** and cannot establish an attention-only benefit.|

These papers improve generalization partially; they do not demonstrate that
train/test gaps must disappear. None makes our64% an intrinsic data ceiling.

### Mechanism-oriented primary literature

| Cause/method family and primary source | What it supports | Current coverage and remaining gap |
|---|---|---|
|[Confident Learning, JAIR2021](https://arxiv.org/abs/1911.00068)|Estimate/rank likely label issues using appropriate out-of-sample probabilities.|**Missing semantic label audit**. In-sample99% predictions cannot validly perform this audit. Requires cross-fitted predictions and human verification; no automatic relabeling.|
|[Pervasive Label Errors, NeurIPS D&B2021](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/f2217062e9a397a1dca429e7d70bc6ca-Abstract-round1.html)|Evaluation-label mistakes can affect model rankings.|Source checks plus error galleries exist; **human semantic verification is not complete**. Do not call every model error a wrong label.|
|[Class-dependent Regularization/DA, NeurIPS2022](https://papers.neurips.cc/paper_files/paper/2022/hash/f73c04538a5e1cad40ba5586b4b517d3-Abstract-Conference.html)|Average-optimal crops/WD can harm individual classes.|Per-class errors, gentle bundle, no-CutMix/no-RA and WD.01/.05/.1 added. Interaction effects remain unexhausted.|
|[FixRes, NeurIPS2019](https://arxiv.org/abs/1906.06423)|Train/test object-size distributions can differ despite similar image dimensions.|Crop-view and gentler-crop tests; **no FixRes reproduction or object-scale stratification**.|
|[Shape/texture bias, ICLR2019](https://arxiv.org/abs/1811.12231)|CNNs can rely on texture; modified training can change shape bias.|**Not directly tested**: no controlled background replacement or stylized-image experiments. Color statistics are insufficient.|
|[BlurPool, ICML2019](https://proceedings.mlr.press/v97/zhang19a.html)|Anti-aliased downsampling can improve shift stability and accuracy.|Input-shift and phase diagnostics exist; **no actual BlurPool insertion**.|
|[APS, CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chaman_Truly_Shift-Invariant_Convolutional_Neural_Networks_CVPR_2021_paper.html)|Adaptive polyphase selection addresses grid dependence, with boundary assumptions.|Multi-phase CE is a different hypothesis; **APS is not implemented** and exact shift invariance is not claimed.|
|[Rethinking Batch in BatchNorm, 2021](https://arxiv.org/abs/2105.07576)|Batch composition and running-statistic behavior deserve explicit scrutiny.|Train-only recalibration was tested; source/clean-view checks and freshGN extend it. Batch-size effects are still not fully separated.|
|[GroupNorm, ECCV2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html)|Normalization without batch-axis statistics, particularly helpful for tiny batches.|**Fresh GN backbone/head implemented**; ourbatch64 is not the paper's strongest tiny-batch case.|
|[SAM, ICLR2021](https://arxiv.org/abs/2010.01412)|Optimize parameter-neighborhood loss instead of relying on low pointwise training loss.|Warm/fresh SAM with FP32 controls implemented; AdamW adaptation, not an exact author reproduction.|
|[AdamW, ICLR2019](https://arxiv.org/abs/1711.05101)|Decoupled WD differs from an L2 penalty in adaptive optimizers.|AdamW baseline, SGD-family recipe and WD controls implemented; SGD comparison also changes LR/WD/warmup.|
|[SWA, UAI2018](https://auai.org/uai2018/proceedings/papers/313.pdf)|Equal-weight trajectory averaging with an appropriate LR path and BN treatment.|**Not implemented**. EMA is not SWA or snapshot ensembling.|
|[EMA dynamics/benefits, TMLR2024](https://openreview.net/pdf?id=2M9CUnYnBA)|Studies trajectory smoothing, generalization, calibration and transfer.|Per-stepEMA.999 with raw/EMA evaluation implemented; buffers averaged, integer counters copied.|
|[mixup, ICLR2018](https://arxiv.org/abs/1710.09412)|Interpolated samples/targets constrain behavior between examples.|Baseline already includes it. **No isolated removal of Mixup yet**; no-CutMix retains Mixup.|
|[CutMix, ICCV2019](https://arxiv.org/abs/1905.04899)|Regional sample mixing with area-based target weights.|Baseline includes it; isolated fresh no-CutMix and gentle tests added. No saliency-aware target correction.|
|[AugMix, ICLR2020](https://arxiv.org/abs/1912.02781)|Augmentation mixing/consistency mainly studied for corruption robustness.|**Not implemented**. Clean Top1 and corruption robustness are distinct outcomes.|
|[Rethinking Generalization, ICLR2017](https://research.google/pubs/understanding-deep-learning-requires-rethinking-generalization/)|Networks can memorize even random labels; training accuracy is insufficient evidence of generalization.|Interpretive foundation, not a reason to spend GPU reproducing random-label fitting.|
|[Deep Double Descent, ICLR2020](https://arxiv.org/abs/1912.02292)|Capacity, sample count and training duration do not have a simple monotone relation with test error.|Fresh shallow26 tests one depth/capacity change. **No full data-size/compute-matched learning curve**.|

### How the next decision will be made

1. Fix a concrete provenance/preprocessing error if the audit finds one.
2. Compare matched fresh/warm controls by held-out accuracy/NLL and class
   corrections/harms, rather than the size of the train/selection gap alone.
3. If the controllable recipe/normalization/depth tests fail, prioritize a
   faithful selective-forgetting or saliency-preserving-mixing reproduction,
   and an out-of-fold semantic-label review, guided by the observed errors.
4. Keep audit/final-validation labels untouched until a final method is fixed.

No finite matrix covers every interaction, ambiguity, optimization path or
possible model. The goal is a falsifiable account of the major causes, with
missing tests recorded, rather than an assertion of exhaustive coverage.

### Earlier concise references

- [Shcherbina, Tiny ImageNet Challenge, Stanford CS231n2016](https://cs231n.stanford.edu/reports/2016/pdfs/401_Report.pdf),
  Table2: a9-layer ConvNet has98.8% training and45.5% validation accuracy.
  This is a course report with an older/different setup, not a peer-reviewed
  matched baseline. It illustrates a large gap without proving a dataset bug.
- [Guo et al., ALPS, IJCAI2021](https://www.ijcai.org/proceedings/2021/0343.pdf),
  Table4: Tiny ImageNet/PreActResNet18 standard-classification accuracies are
  60.95% ERM,62.48% dropout,61.59% smoothing,62.95% Mixup,63.77% ALPS.
  They train100K images with SGD and use official validation; our80K/10K and
  AdamW protocol differs. These are context, not direct performance claims.
- [Foret et al., SAM, ICLR2021](https://arxiv.org/abs/2010.01412)
  motivates optimizing a neighborhood of parameters instead of relying only
  on low training loss. It does not guarantee this Tiny ImageNet fix works.
- [Sarfi et al., SEAL, CVPR2023](https://arxiv.org/abs/2304.04858)
  studies iterative generalization improvements on Tiny ImageNet. Its long
  multi-generation schedule is not being claimed or reproduced overnight.

None of these references imply a64% data ceiling, a guarantee that gap can be
closed, or evidence that all wrong labels can be identified automatically.
