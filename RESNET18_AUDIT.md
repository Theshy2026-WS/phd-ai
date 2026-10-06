# ResNet-18: does adding attention improve generalization?

This experiment replaces neither the old files nor their recorded results. It
starts a new, two-model comparison, not a CoAtNet reproduction or a proposed new
architecture. No extra convolutional-control branch or FFN is added.

## Models

Both models use torchvision ResNet-18, **trained from scratch**. For native 64x64
images the stem is changed to 3x3 stride 1 and the initial max-pool is removed.
The four residual stages produce 64x64, 32x32, 16x16 and 8x8 feature maps.

- `cnn`: adapted ResNet-18 -> global average pool -> 200-class classifier.
- `attention`: identical ResNet-18 -> one pre-LayerNorm, 4-head spatial MHSA
  residual -> the same pooling/classifier. It processes 64 tokens of width 512;
  the residual is `x + 0.1 * MHSA(LayerNorm(x))`. The scale is fixed. There is no
  FFN, added positional embedding, gate, dropout or extra convolution.

This is a limited test of **late self-attention**, not all attention placements.
Shared backbone/classifier tensors have identical initialization for a given
seed. Per-image augmentation and per-batch mixing are model-independent.

| Model | Parameters | Estimated forward FLOPs / image |
|---|---:|---:|
| ResNet-18 | 11,271,432 | 4,443,545,600 |
| ResNet-18 + attention | 12,323,080 | 4,586,151,936 |

FLOPs count convolution, linear and MHSA matmuls, with 1 MAC = 2 FLOPs; they omit
normalization, activation and other elementwise work. These are not measured
latencies. The small-image stem is significantly more expensive than the old
downsampling RepViT configuration. Do not reuse its training-time estimate.

## Training and overfitting checks

Default: 100 epochs, batch 64, SGD (LR .05, momentum .9, Nesterov), five warmup
epochs then cosine decay to 1e-5. Weight decay .0005 excludes norm/bias parameters.
Both models receive identical random resized crops (scale .6–1), horizontal flip,
RandAugment (2 operations, magnitude 5), Mixup alpha .2 / CutMix alpha 1 (combined
application probability .5), label smoothing .1, clipping norm 5 and CUDA AMP.
These are initial shared hyperparameters, not a guarantee of higher accuracy or
an exact published training recipe. References: [RandAugment API](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandAugment.html),
[CutMix](https://arxiv.org/abs/1905.04899).

The fixed stratified split remains 80k training / 10k selection / 10k exploratory
audit, all drawn from the original training set. The official validation set is
untouched. Best checkpoints are chosen **only by selection accuracy**.

Every 5 epochs (also first/final), 2,000 fixed clean training images are evaluated
in eval mode without augmentation/mixing. `train_probe_accuracy` and
`train_probe_nll` can be compared with `selection_accuracy` and `selection_nll`.
The probe is an estimate, not full-training-set accuracy. Probe identities are
saved. Mixed-label training logs only `train_soft_target_loss`, not misleading
hard-label training accuracy; this loss is not directly comparable to clean NLL.

A remaining train/selection gap is not automatically a broken baseline. Check
selection learning curves and clean NLL before tuning. Fix the common recipe
using selection data before looking repeatedly at audit outcomes. Audit-driven
choices are exploratory; reserve independent final evaluation. A single seed
cannot establish a reliable gain, nor can this pair alone separate extra capacity
from the attention-specific effect.

## Workbench commands

Use the existing PyTorch/torchvision/timm environment. No data or weights are
committed. Missing Tiny ImageNet data is downloaded only when you run this code
and `dataset.download: true`; an existing dataset is reused.

```bash
git pull --ff-only
# Smoke test only; not an accuracy comparison:
python resnet_experiment.py suite --seed 42 --epochs 1

# Recommended first real run: establish the ResNet baseline (100 epochs):
python resnet_experiment.py train --variant cnn --seed 42
```

Send its `epochs.csv` and `summary.json` to inspect the generalization gap first.
If the shared recipe is acceptable, reuse that run, train attention, and audit:

```bash
python resnet_experiment.py suite --seed 42 --cnn-run runs/resnet18_utility/cnn_seed42_REPLACE_WITH_ACTUAL_RUN
```

Alternatively run both from scratch sequentially, then audit automatically:

```bash
python resnet_experiment.py suite --seed 42
```

Runs are uniquely named under `runs/resnet18_utility/`. An epoch-boundary
checkpoint includes optimizer/scheduler/scaler and RNG states. Resume a stopped
individual model with its exact directory (the original total epochs is retained):

```bash
python resnet_experiment.py train --resume runs/resnet18_utility/attention_seed42_REPLACE_WITH_ACTUAL_RUN
python resnet_experiment.py audit --cnn CNN_RUN_DIRECTORY --attention ATTENTION_RUN_DIRECTORY
```

Do not pass a new epoch budget when resuming or update source code mid-experiment.
Resume checks source, split, software, device and precision. It repeats an
unfinished epoch rather than resuming a partly completed batch. Default CUDA
kernels are not guaranteed bitwise deterministic. Load only your own trusted
checkpoint files. `--epochs 1` is a separate smoke-test schedule, not the start
of a resumable 100-epoch experiment.

For disconnect-safe execution, use a persistent terminal session such as `tmux`;
keep the Workbench VM running. Neither this script nor resume prevents an idle
shutdown policy or VM stop from interrupting training.

## Output and next decision

Each run saves config, metadata (source hashes/versions/split/params/FLOPs), split
manifest, clean probe IDs, `epochs.csv`, `best.pt`, `last.pt`, and `summary.json`.
The summary separates training, selection and probe time, peak allocated GPU
memory, and synthetic forward-only inference performance. It excludes download,
checkpoint I/O and audit time; throughput excludes data loading/transfer.

The audit saves per-image predictions, `metrics.csv`, `paired_utility.csv`,
`paired_summary.csv` and provenance. Positive `utility_nll` means attention has
lower NLL. `rescued` counts CNN-wrong/attention-right; `harmed` is the reverse.
Centered masking of 0%, 10% and 25% area is a narrow stress test, not a general
robustness benchmark. Use clean accuracy, NLL, rescues/harm, and compute together.

First decide whether the baseline trains and generalizes reasonably. Then test
whether attention helps under the **same** recipe. If promising, repeat seeds
43 and 44 before changing placement/fusion. An improvement is not predetermined;
a null result under this protocol is also evidence about attention's conditions.
