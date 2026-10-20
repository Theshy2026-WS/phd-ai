# Frozen CNN + prediction-conditioned attention correction

Experimental candidate, not a verified novelty or an accuracy guarantee.
Precedents: Side-Tuning (ECCV 2020), Ladder Side-Tuning (NeurIPS 2022),
CALA (arXiv:2412.12654, a different incremental-classification setting).

## Parent and architecture

Only a completed **100-epoch, seed42 resnet50_preact** literature run is accepted.
Use its `best.pt` (selection-chosen epoch), not random weights or automatically
the last epoch. The runner validates metadata, source model hashes, split hash,
state dict, finite weights, and records checkpoint SHA256.

CNN stem and stages 1--3 produce 1024 x 8 x 8 features; stage4 and original head
still produce the unchanged 200 logits. Everything in this CNN is frozen,
including BatchNorm buffers. There is no backward graph through the CNN.

New branch:

1. Project 8x8 features to 64 tokens of width192; learned position embeddings.
2. One pre-norm four-head self-attention block, MLP ratio2.
3. Form one query from the frozen CNN's 200-class probabilities plus a learned
   query embedding. Four-head cross-attention retrieves the spatial evidence.
4. A small MLP outputs 200 additive logit corrections.
5. Final output projection starts at zero, so initial predictions equal the CNN.

All training labels are used only in the training loss; no true label, selection
label, or held-out label enters the query. A correct base prediction can still
be harmed by correction; freezing is not a prediction-preservation guarantee.
The baseline branch is run in FP32 to preserve its recorded evaluation path.
New attention and ViT are FP32 too; timings must not be mislabelled AMP.

`pooled_correction` is an optional simpler control using the same projection,
prediction query and output head, but pooling instead of either attention block.
It is NOT parameter-matched. A continued-training control and additional seeds
remain necessary before attributing a gain specifically to attention.

## Pure ViT

Reuses `models/vit.py`: 64x64 RGB, patch4, 256 patches + CLS, width224,
5 encoder blocks, 4 heads, MLP ratio4, dropout0.1, 200 classes. No CNN feature
extractor (the non-overlapping patch projection is implemented with Conv2d).
From scratch, 100 epochs by default; this is a compact baseline, not ViT-B.
Same 80/10/10 split and inherited augmentation recipe as the trained parent.
Historical `main.py` ViT splits differ and are not directly comparable.

## Workbench commands

Run from the repository. No dataset/checkpoint is committed or downloaded locally.

```bash
git pull --ff-only
# If not already inside tmux:
tmux new -s correction-night
python -u overnight_correction.py --check-only --fallback-os-shutdown
python -u overnight_correction.py --fallback-os-shutdown
```

Defaults: attention correction25 additional epochs, pure ViT100 epochs, seed42.
The parent is autodetected ONLY if exactly one qualifying run exists.
If missing/ambiguous, pass its directory with `--base-run runs/.../resnet50_preact_seed42`.
Two base variants exist: `preact50` and `resnet50_preact`; only the latter is supported.
The trained checkpoint must actually be present in Workbench, not just a downloaded CSV.

Optional smoke test (still shuts down at completion):

```bash
python -u overnight_correction.py --correction-epochs 1 --vit-epochs 1 --fallback-os-shutdown
```

Optional pooled control:

```bash
python -u overnight_correction.py --variants attention_correction pooled_correction vit_tiny64 --fallback-os-shutdown
```

Resume keeps the recorded schedule/source/checkpoint:

```bash
python -u overnight_correction.py --resume runs/attention_correction/suite_TIMESTAMP_ID --fallback-os-shutdown
```

Progress is printed and saved at `SUITE/overnight.log`.
`correction_experiment.py` accepts the training options but never stops the VM.

## Outputs and costs

Each run saves epochs.csv, summary.json, metadata.json, config.yaml, best.pt,
last.pt. The suite also saves the exact parent provenance, frozen-base
re-evaluation, and comparison.csv (Top1/5, trainable/total params, estimated
FLOPs, batch1 latency, throughput, incremental AND base-inclusive training time).
This is 100+25 versus100 epochs, NOT equal training budget. FP32 vs historical
AMP training time is also not an architecture-only comparison.

Before shutdown, small analysis files, source snapshots and logs are archived:
`runs/attention_correction/SUITE_analysis.tar.gz`. No checkpoints or images.
It is created even after training failure; inspect log/exit code for completion.
Checkpoints remain in persistent storage. Workbench managed stop is attempted;
opt-in OS fallback may leave the UI temporarily showing Active.
Stopping GPU compute does not remove persistent disk charges. A hung job does
not count as finished. Cloud/API failures can still require manual intervention.

**Automatic packaging is not automatic downloading to your laptop.**
An offline laptop/browser cannot receive a file initiated by the VM.
Download the single archive after restarting Workbench, or optionally export
before stopping to an existing, private bucket:

```bash
python -u overnight_correction.py --export-to gs://YOUR_BUCKET/correction --fallback-os-shutdown
```

This requires gcloud and the VM's existing write access; no bucket creation,
public ACL, IAM change or credentials are automated. Upload has a 180s timeout;
failure still triggers shutdown, and the local archive remains on disk.
You can download from Cloud Storage without restarting the GPU VM:

```bash
gcloud storage cp gs://YOUR_BUCKET/correction/SUITE_analysis.tar.gz .
```

For an automatic local download, keep a laptop PowerShell session running:

```powershell
.\download_correction_results.ps1 -ArchiveUri gs://YOUR_BUCKET/correction/SUITE_analysis.tar.gz
```

Use the actual suite name printed at launch. This waits up to72 hours for that
exact object, then downloads to Downloads without overwriting an existing file.
Requires installed/authenticated gcloud and private bucket read access locally.
The laptop must be awake and online to perform the download; training and VM
shutdown do not depend on it. GCS storage/request charges may still apply.

## Interpretation

Higher Top1 is a hypothesis, not an implementation promise. First compare
against the exact frozen parent, then simpler corrections and equal incremental
training budget, then multiple seeds/datasets. The repeatedly inspected held-out
split is development evidence; reserve unseen data for final claims.
