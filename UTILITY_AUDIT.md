# Tiny ImageNet: controlled baseline and attention utility audit

This protocol supersedes the exploratory 5-epoch gate sweep for new comparisons.
It does not alter the old `main.py` experiments. No proposed fusion is introduced.

## Model family

The shared backbone is **timm 1.0.25 RepViT-M0.9**, initialized from scratch,
without pretrained weights or distillation. The classification head has 200 outputs.
The original stem/downsampling is retained. At native 64x64 resolution, stage 2
(zero-indexed, the penultimate stage) has 192 channels and a 4x4 map. The optional
module is inserted there, before the final downsampling to 2x2.

- `cnn`: unchanged RepViT trunk and single classification head (B0).
- `conv`: B0 plus a convolutional residual mixer and FFN (B1).
- `attention`: B0 plus fixed residual single-head MHSA and the same FFN (B2).

B1/B2 share positional depthwise convolution, pre-LayerNorm, residual shell,
and FFN. B1 substitutes depthwise 3x3 + pointwise expansion/projection for MHSA.
There is no gate and no additional dropout in either addon. Added parameter counts
are close, not identical; actual counts/FLOPs are saved. This is standard full-channel
single-head MHSA, **not SHViT's partial-channel SHSA**. RepViT retains its existing
SE channel attention: B0 means no added spatial self-attention.

This is a controlled Tiny ImageNet adaptation, not reproduction of published
ImageNet accuracy or deployment latency. Native 64x64 leaves only 16 tokens at the
insertion point; any conclusion is bounded to this choice. Do not silently resize
only one variant or change the stem mid-comparison.

Sources: [timm implementation](https://github.com/huggingface/pytorch-image-models/blob/v1.0.25/timm/models/repvit.py),
[RepViT authors](https://github.com/THU-MIG/RepViT).

## Data and randomization

The 100000 Tiny ImageNet training images are stratified per class into:
80000 training / 10000 selection / 10000 audit. Official validation is untouched.
The split seed remains 42 across training seeds. Each run saves image IDs and a
split fingerprint; pairing rejects mismatches. The fingerprint covers names, class
mapping and split, not file contents. Keep the dataset immutable.

The selection split alone selects `best.pt`. Audit labels never select checkpoints.
Independent generators and sample/epoch-keyed augmentation isolate data order and
crop/flip from model initialization. B0/B1/B2 use identical initial backbone/head
weights within a seed. End-to-end training subsequently changes them independently.
GPU nondeterminism is still possible (`deterministic: false`).

Audit is exploratory held-out analysis, not the final test set. Once it informs
method design, do not present it as untouched final evaluation. A learned utility
predictor will require another fit/validation split or cross-fitting, not fitting
and scoring on these same labels.

## Workbench commands

```bash
git pull origin main
pip install -r requirements-audit.txt

# First verify all variants and audit end-to-end. One epoch is only a smoke test.
python utility_experiment.py suite --seeds 42 --epochs 1

# Common 25-epoch protocol, three seeds, sequential on one GPU; audits automatically.
python utility_experiment.py suite --seeds 42 43 44
```

The configuration is `configs/tiny_imagenet_utility.yaml`. Twenty-five epochs is
a first common budget, not a claim of convergence; inspect `epochs.csv` before
interpreting a negative result. Old 5-epoch numbers use another backbone and split.

For individual runs:

```bash
python utility_experiment.py train --variant cnn --seed 42
python utility_experiment.py train --variant conv --seed 42
python utility_experiment.py train --variant attention --seed 42

python utility_experiment.py audit --cnn runs/tiny_utility/CNN_RUN --conv runs/tiny_utility/CONV_RUN --attention runs/tiny_utility/ATTENTION_RUN
```

Replace the three run paths with the actual directories printed by training.
Use trusted local checkpoints only; audit loads their stored configuration.
The runner saves `best.pt` and `last.pt` model checkpoints; automatic training
resume is not implemented here. A failed suite preserves completed runs, which
can be combined with individual runs using the audit command.

## Outputs

Each run has a unique directory in ignored `runs/tiny_utility/`:

- Effective YAML, split manifest, code/version metadata, parameter/FLOP estimates.
- `epochs.csv`: train loss/NLL/accuracy, selection NLL/accuracy, LR and time.
- `best.pt` and `last.pt`: model weights, epoch, config, selection metrics.
- `summary.json`: training/selection peak memory and separate inference profiles.

Inference runs with optimizer removed and gradients cleared; batch=1 batch latency
is single-request forward latency. Batch=64 per-image time is amortized throughput
time. Synthetic inputs are already on-device; no transfer/decode time is included.
The timm network is not deployment-fused. FLOPs omit elementwise/normalization
operations. CPU measurements report no GPU memory. Benchmarks use final weights;
audit predictions use best-selection weights.

Audit exports:

- Full logits NPZ and per-sample CSV for each model, keyed by image ID and condition.
- NLL, entropy, confidence, top-two probability margin, true-class logit margin,
  correctness and top-5 correctness.
- `paired_utility.csv`: NLL(CNN)-NLL(attention) and NLL(conv)-NLL(attention),
  plus true-class margin gains. Positive utility favors attention.
- `groups.csv`: corrections, harms, both right/wrong, mean utility, oracle accuracy.
- `uncertainty_groups.csv`: utility in fixed CNN entropy bins.
- `calibration.csv`: raw NLL, accuracy, top-5 and 15-bin ECE.
- `provenance.json`: exact run paths/checkpoint hashes and protocol.
- Suite tables: per-seed results and seed-level mean/sample-standard-deviation.
  A single seed has undefined standard deviation, not zero variance.

Occlusion uses a centered square covering approximately 0%, 10%, 25% of image
area, filled with normalized zero (ImageNet mean color), identical across variants.
Rounded square dimensions mean actual pixel area can differ slightly. This is a
single controlled perturbation, not a general corruption benchmark.

## Interpretation and next gate

These are **between-model differences**, not causal contributions of the attention
branch. Different checkpoints can improve on different images by chance; repeat
seeds before claiming predictable utility. Oracle selection uses true labels and
is an upper bound, not an executable policy. NLL comparisons can reflect calibration,
so inspect correctness and margin as well. No feature similarity is equated with
useful information, and no router or complementarity estimator is trained yet.

If stable utility is observed, next steps are shared/frozen-trunk diagnostics,
representation analysis, and held-out utility prediction. CNN-only signals can
support pre-attention execution decisions; signals using attention output can
support fusion but already pay the attention cost. This first audit does not
automatically choose or prove any new fusion mechanism.

## Local validation (no dataset download)

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic images to verify common initialization, gradients, addon shape,
augmentation isolation, sample matching, utility signs and full train/audit outputs.

## Saved-run diagnostics (no retraining)

Names in the runner refer to **CNN-only**, **CNN + extra convolution**, and
**CNN + late self-attention**. There is no pure ViT in this three-way audit.

Analyze an existing paired file on CPU:

```bash
python utility_diagnostics.py paired --paired runs/tiny_utility/AUDIT_RUN/paired_utility.csv
```

This writes pairwise comparisons including CNN-versus-conv, fixed entropy-bin
accuracy/correction/harm counts, and uncertainty-signal AUCs. The AUC population
includes only samples on which exactly one model is correct. This uses ground-truth
labels for analysis and is NOT end-to-end routing accuracy or a validated predictor.
No thresholds are fitted. Compare to the conv control: a signal may predict ordinary
reference-model mistakes rather than attention-specific utility. These are single-run
descriptive associations, with no claim of seed-level reproducibility or causality.

Check trained modules on Workbench, where the original run directories and dataset
still exist:

```bash
python utility_diagnostics.py checkpoint --audit-dir runs/tiny_utility/AUDIT_RUN
```

Replace `AUDIT_RUN` with the directory printed by the original audit. This command
uses its provenance to locate the **same** conv and attention `best.pt` files and
checks checkpoint hashes, timm version, dataset fingerprint and sample pairing.
It does not download data, train, or write checkpoint weights. Other Torch/runtime
versions are recorded; the full-model replay is checked against original predictions.
Use trusted locally produced checkpoints, as in the training/audit runner.

For both models, it evaluates all original occlusion conditions with:

- `full`: unchanged checkpoint.
- `mixer_off`: zero the spatial mixer's output, keeping positional convolution,
  residual connection, normalization and FFN.
- `addon_off`: bypass the complete added module.

Outputs go into a new `runs/tiny_utility/checkpoint_diagnostic_seed...` directory:

- `replay_check.csv`: normal-forward prediction mismatch counts and NLL differences
  from the original audit. Inspect this first; nontrivial mismatches require diagnosis.
- `intervention_summary.csv`: paired accuracy/NLL changes, corrected/harmed counts,
  prediction flips and probability changes under each intervention.
- `parameter_updates.csv`: initial/trained/change L2 norms per addon parameter,
  comparing to the seeded initialization. Changed weights alone do not establish benefit.
- Per-sample intervention CSVs and provenance for follow-up analysis.

Removing a trained component may shift downstream feature distributions. A loss of
accuracy establishes frozen-model sensitivity, not unique global reasoning, causal
mechanistic identification, or superiority over a separately retrained CNN. Moreover,
these hooks compute and discard outputs: intervention timing is **not** a conditional
execution speed benchmark. No new architecture is selected by this diagnostic alone.
