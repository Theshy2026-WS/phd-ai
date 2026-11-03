# Tiny ImageNet: clean baseline diagnosis

The first command evaluates the existing `resnet50_preact` best checkpoint. It
does not train, change the checkpoint, evaluate official validation, or evaluate
the 10K audit partition. Existing training and model source files are unchanged.

The intended reference is the native64 four-stage preactivation CNN with
3/4/6/3 blocks, trained for 100 epochs at seed42. Its selected checkpoint is epoch
94 in the existing reference run. The slightly larger earlier `preact50`
3/4/6/4 model is a different model and is not selected automatically.

## Workbench: evaluate the existing checkpoint

Use the existing environment with working CUDA/PyTorch/timm. Do not pull while
an older training job is active. These commands run in the Workbench Linux
terminal, inside an existing tmux session if desired:

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
mkdir -p logs
set -o pipefail
python -u baseline_diagnostics.py evaluate \
  --base-run runs/literature_tiny/suite_20261013T173206_5ac7cd/resnet50_preact_seed42 \
  --data-dir ./data \
  --num-workers 4 \
  2>&1 | tee logs/baseline_diagnostics.log
```

If the original run is elsewhere, use its actual directory containing
`best.pt` and `summary.json`. Omitting `--base-run` permits automatic detection
only when exactly one eligible 100-epoch seed42 parent exists. An analysis
archive containing only JSON/CSV is insufficient: the weights must remain on
the VM. Dataset download is disabled by default; `--allow-download` explicitly
permits it if the data are missing.

The default diagnosis evaluates all 80,000 training images without augmentation
and all 10,000 selection images. Add `--train-per-class 20` for a fixed 4,000-image
stratified training probe while retaining the full selection set. Set
`--examples 0` to skip galleries. CPU evaluation requires explicit `--device cpu`.

The checkpoint and summary metadata, selected epoch, core model source hashes,
and original split/class-mapping fingerprint are checked before evaluation.
The source checkpoint and all model parameters/BatchNorm buffers are checked
again after evaluation. Each run creates a new timestamped directory under
`runs/baseline_diagnostics/`, never inside the original training run.

## Outputs and interpretation

- `summary.json`: clean train/selection Top-1, Top-5, NLL, macro accuracy, the
  train-selection gap, checkpoint provenance, runtime settings, and selection
  replay difference from the historical score.
- `report.md`: compact human-readable overview.
- `evaluation_manifest.json`, `class_mapping.json`: evaluated image IDs and
  original synset-to-class mapping. Human-readable labels are added from the
  dataset's `words.txt` when available, without changing that mapping.
- `class_comparison.csv`: per-class clean train versus selection recall/NLL,
  counts and recall gaps, ordered by selection recall to aid inspection.
- Each split directory contains `predictions.csv`, `predictions.npz`,
  `per_class.csv`, `confusion_matrix.csv`, `confusion_pairs.csv`, and galleries
  of representative errors and correct controls. The NPZ stores complete class
  probability vectors, image IDs, and labels for further paired analysis.
- `diagnostics_report.tar.gz`: the analysis report, CSVs, manifests and galleries.
  Full NPZ probability arrays are excluded to keep the shareable archive smaller;
  they remain on the VM. No archive is sent automatically.

Return the printed `diagnostics_report.tar.gz` for review. Inspect the gap,
low-recall classes (each selection class has only 50 images), confused class
pairs, and cases with the true class inside versus outside Top-5. Gallery
examples are illustrative, not a representative error-rate estimate. A small
class score or a large gap does not alone establish a causal explanation.

Evaluation is native64, deterministic image preprocessing, unsmoothed
cross-entropy, model eval mode, and FP32 with autocast disabled. The archived
TF32 setting is retained and reported. A selection replay difference is
reported explicitly rather than silently treated as exact reproduction; kernel,
software and batch-size differences may warrant investigation. The training
loss under mixed labels is not compared directly with this clean NLL.

## Optional training controls: choose after the diagnosis

These are separate, explicitly requested commands. They train the same
`resnet50_preact` architecture from scratch using the archived reference config,
seed, split and training precision. They do not reuse its weights or resume its
optimizer. Early stopping is disabled in both controlled schedules.

```bash
# Longer schedule: 200 epochs with a freshly defined cosine horizon.
python -u baseline_diagnostics.py controls \
  --base-run runs/literature_tiny/suite_20261013T173206_5ac7cd/resnet50_preact_seed42 \
  --data-dir ./data --num-workers 4 --variants longer200 \
  2>&1 | tee logs/baseline_longer200.log

# Mixing-policy control: 100 epochs, mix_probability=0.
python -u baseline_diagnostics.py controls \
  --base-run runs/literature_tiny/suite_20261013T173206_5ac7cd/resnet50_preact_seed42 \
  --data-dir ./data --num-workers 4 --variants no_mix100 \
  2>&1 | tee logs/baseline_no_mix100.log
```

`longer200` changes the total schedule to 200 epochs; it does not append 100
epochs at the terminal learning rate. `no_mix100` disables Mixup/CutMix batch
mixing while retaining label smoothing, random crops, RandAugment, optimizer
and the 100-epoch schedule. A different batch size is rejected for these
comparisons. `plan.json` records the intended differences and provenance.

The existing literature training loop is reused unchanged, including its
selection-only checkpoint choice and legacy development-audit report at the
end of training. That audit is already development data, not a final test.
After each completed control, the new clean diagnosis runs on its best
checkpoint. Official validation is never opened by either command.

Control runs live under `runs/baseline_controls/`. If training is interrupted,
the run's ordinary `last.pt` remains available; restore the recorded source
and resume through the existing literature runner with `python
literature_experiment.py train --resume PATH_TO_CONTROL`. Then evaluate its completed directory with the
`evaluate` command above. Re-running `controls` always starts a new suite.

Neither command sends email nor stops the VM. Stop the Workbench instance
normally after the work completes if it should no longer incur compute cost.
