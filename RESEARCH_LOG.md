# Research Progress Log

Last updated: 2026-08-12

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

Interpretation:

- Direct fusion underperformed, supporting the need for a CNN residual path.
- Scalar gated and residual results were effectively tied.
- The first two-branch dynamic design was worse and slower. It changed both the
  local branch and gate simultaneously, and mixed BatchNorm and LayerNorm
  feature distributions, so it is retained as a negative result rather than a
  candidate for longer training.

## Current experiment: Dynamic Channel fusion

The new, controlled design keeps the proven Late residual structure and changes
only the attention coefficient:

```text
A = SelfAttention(x)
w = sigmoid(ChannelGate(GlobalPool(x)))
y = x + w(x) * A
y = y + MLP(LayerNorm(y))
```

`w` has shape `[batch, channels, 1, 1]`, so it varies by image and channel.
There is no additional local CNN branch. The final gate layer starts with zero
weights and bias -2, giving an initial attention weight near 0.119, consistent
with the stable scalar-gated experiment.

Run:

```bash
python main.py tiny_imagenet_attention_late_dynamic_channel.yaml
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

1. Compare Dynamic Channel against Late Residual and scalar Gated at 5 epochs.
2. If it is competitive, repeat the leading candidates with multiple seeds.
3. Add a parameter/FLOP-matched CNN control before attributing gains solely to
   attention or fusion.
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
