# Fixed-model fusion confirmation

This step performs inference only. It does not train another model, select another checkpoint, tune a probability weight or choose a new winner from the confirmation scores. The candidate remains the complete CLS + Linear joint model selected on the development partition at 65.45%.

## Evaluation partition and correction to the earlier description

The target is the **official Tiny ImageNet validation set: 10,000 images, 50 per class**, using its annotation file and the original training class mapping. It is distinct from the 80,000 training images and the 10,000 development images used for the present fusion studies.

The internal 10,000-image partition named `audit` cannot be described as unused: the earlier `literature_experiment.py` evaluated that partition for completed baselines. Consequently this confirmation uses the official validation set instead. Recent supplied fusion records say that official validation was not evaluated, but a complete global history of official-validation use is **unverified**. The registered plan records this distinction; it must not claim that a partition was pristine solely because a recent report contains an `evaluated=false` flag.

## Locked models and comparisons

| Role | Model or prediction rule | Checkpoint rule fixed before confirmation |
| --- | --- | --- |
| Primary reference | Independent CNN | Same strongest source CNN checkpoint, development 64.80% |
| Context and ensemble input | Independent ViT | Same independently trained ViT checkpoint, development 50.18% |
| Primary candidate | Joint CLS + Linear fusion | Complete historical joint checkpoint, development 65.45%, EMA epoch 7 |
| Key secondary reference | Probability ensemble | Exactly `0.9 * CNN probabilities + 0.1 * ViT probabilities`; no grid or fitting |
| Secondary control | Random ViT + CLS + Linear | `best_trained.pt`, not the epoch-zero best |
| Secondary control | Reset CNN classifier columns and bias | Development-selected `best.pt` |
| Secondary control | Patch-mean + Linear | Development-selected `best.pt` |

The three controls come from `suite_20261109T165758_03cb08`. The original source checkpoints, source code, completed histories, original split, class mapping and inference environment are validated before evaluation. Source weights and normalization buffers remain unchanged.

The primary endpoint is the paired Top-1 accuracy difference between the fixed joint candidate and the independent CNN. Its exact two-sided McNemar test uses alpha 0.05. Four secondary comparisons—joint versus the fixed ensemble, random-ViT control, reset-head control and patch-mean control—use Holm-adjusted p-values as one fixed family. Secondary positive-improvement conclusions are gated by a positive primary result.

Each accuracy difference also receives a pointwise 95% percentile interval from 10,000 **class-stratified paired image bootstrap** replicates. Within each true class, the two models' correctness outcomes are resampled together and the class sample count remains fixed. The statistical RNG seed is fixed at 20261109 plus the predeclared comparison index; this does not retrain a model or change a training seed. A positive effect is marked established only when the permitted p-value is below alpha, the estimated effect is positive and the interval excludes zero. An unestablished effect is not evidence of equivalence.

Top-5, NLL and paired NLL improvements are descriptive. Lower NLL alone does not establish better calibration. Image-based uncertainty is conditional on these fixed trained models and sampled classes; it does not estimate variability over training runs, future datasets or alternative training recipes. The report retains the fixed model order and never chooses the highest confirmation score as a replacement winner.

Statistical implementation references: [exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html), [Holm adjustment](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html). The implementation does not add a SciPy dependency.

## Run on Workbench

Run in a persistent `tmux` session from the repository directory:

```bash
git pull --ff-only
python -u run_fusion_audit.py \
  --data-dir ./data \
  --num-workers 4
```

There is no epoch argument because this command performs no training. The default registered output directory is `runs/fusion_confirmatory_audit`. Registration locks model identities, prediction rules, partition, source dependencies and statistical decisions before inference. Exact completed prediction caches are reused on resume; changing the locked experiment is rejected. This prevents an accidental second confirmation from becoming another round of model selection.

Preflight only:

```bash
python -u run_fusion_audit.py \
  --data-dir ./data --num-workers 4 --check-only
```

Preflight validates the official validation files, source inputs, email authentication and managed Workbench stop permission. It does not register an evaluation, run inference, send email or stop the instance.

Resume the same registered evaluation:

```bash
python -u run_fusion_audit.py \
  --resume runs/fusion_confirmatory_audit
```

The wrapper sends one final ordinary reports-only ZIP after all inference and statistics, then requests managed Workbench STOP. Failed or interrupted runs still retain available evidence and attempt the same final delivery/stop policy after execution has started. Startup preflight failure does not arm shutdown. Email acceptance is recorded separately from confirmed inbox delivery; the local stop record cannot by itself prove the cloud instance has finished stopping.

Keep the confirmation report as a fixed result. Subsequent method changes belong to a new development study and require another independently justified evaluation design.
