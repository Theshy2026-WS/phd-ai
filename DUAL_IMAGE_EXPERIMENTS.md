# Independent-image fusion on Tiny ImageNet

All jobs use seed42 and the existing 80k/10k/10k split. No shutdown,
email, external weights or dataset download is performed by local tests.
Workbench uses the existing dataset loader (download permitted if missing).
Old models and result directories are preserved.

| Job | CNN | ViT | Combination | Full epochs |
|---|---|---|---|---|
| prediction100 | frozen 100-epoch preactivation ResNet50 checkpoint | trainable scratch | image-dependent mixture of class probabilities | 100 |
| feature100 | same frozen checkpoint | trainable scratch | projected 192D feature vectors, softmax gate, new classifier | 100 |
| prediction25 | identical to prediction100 | identical | same method, shorter cosine schedule | 25 |
| aacnn100 | trainable 3-block CNN 32/64/128 | trainable scratch | 256D projections, 512→64→2 gate, weighted sum, dropout/classifier | 100 |

The ViT receives the same augmented **original image**, not CNN activations:
64x64, patch8 (64 patches plus CLS), width192, depth4, heads4, MLP ratio2.
This intentionally small branch limits overhead; measured FLOPs and latency,
not parameter count alone, determine whether the cost is acceptable.
Both branches execute for every image; this is not conditional execution.
CPU structural estimates (not L4 timing): parent 2.596G FLOPs;
prediction 2.768G / 25.205M parameters; feature 2.768G / 25.637M parameters;
AACNN adaptation 0.254G / 1.564M parameters. The first two add approximately
6.6% forward FLOPs. AACNN is a much smaller scratch model, not a matched-backbone
fusion ablation; lower cost or different accuracy cannot be attributed to its
gate alone.

Prediction mixing is computed stably in log-probability space:
logsumexp(log p_C + log w_C, log p_V + log w_V). Gate starts at
softmax([2,0]); no forced gate warmup. Feature mixing starts with equal weights.
Both frozen-CNN variants optimize fusion CE + 0.5 * standalone ViT CE to
avoid starving the weaker new branch. The CNN and all BN buffers remain frozen.
Feature fusion has a newly initialized classifier: it is not initially identical
to the base CNN. Initial selection is an epoch0 candidate, not a claimed baseline.

## AACNN scope

Source: Rahman & Rahman, *AACNN-ViT: Adaptive Attention-Augmented Convolutional
and Vision Transformer Fusion for Lung Cancer Detection* (2026),
https://www.mdpi.com/2313-433X/12/2/62 .

We transfer the **vector gate structure in section 4.4.3 / Algorithm 1**,
not reproduce the medical-data result or claim its training protocol.
The paper's frozen ImageNet21k-pretrained ViT-B/16 is replaced, at the user's
request, by the same small trainable scratch ViT. CNN inputs are native64 rather
than512; 200 classes rather than3; common Tiny augmentation, AdamW/cosine,
soft-target CE rather than paper focal/CE and early stopping. Gate hidden
activation is ReLU. Thus compare this as `AACNN structure adaptation`, not
an exact paper implementation or a novelty claim. No external pretraining.

All four use FP32, lr0.0003, inherited augmentation/Mixup/CutMix and selection-only
checkpoint choice. The frozen CNN's 100-epoch cost is recorded separately;
AACNN has no parent cost. Held-out labels never drive selection.

## Commands (Workbench terminal, inside tmux)

```bash
git pull --ff-only
python -u dual_image_experiment.py --smoke 2>&1 | tee dual_image_smoke.log
```

If autodetection finds multiple CNN parents, append:
`--base-run runs/literature_tiny/suite_20261013T173206_5ac7cd/resnet50_preact_seed42`

Each of the four jobs runs one epoch; prediction100/prediction25 smoke jobs are
intentionally duplicate checks. Smoke results are not model-quality conclusions.
Full schedules start fresh, not from smoke checkpoints:

```bash
python -u dual_image_experiment.py 2>&1 | tee dual_image_full.log
```

Resume interruption with `python -u dual_image_experiment.py --resume PATH_TO_SUITE`.
Do not change source or schedules when resuming. Both commands leave the VM on.
Send `comparison.csv`, `suite.json` and each job's `summary.json` / `epochs.csv`.
The rough full-training projection excludes setup/profiling and is unreliable
from a cold first epoch. FLOPs include both branches, inference is measured.

## Email then stop (explicit overnight alternative)

`python -u run_dual_overnight.py --fallback-os-shutdown` starts fresh full
100/100/25/100 schedules at seed42. Add `--base-run PATH` if needed.
`--check-only` checks the parent, CUDA, managed stop permission and sends a test
email without training or shutdown. Run inside tmux with existing exported
`GMAIL_USER` and `GMAIL_APP_PASSWORD`; never use a Google login password.
No credentials are stored or passed to the training child.

After success or failure, small analysis files and sources are packaged, email
is attempted up to three times, then managed Workbench stop is requested.
Datasets/checkpoints are excluded from email and remain on disk. If email fails,
the VM is still stopped to limit cost; retrieve the archive after restarting.
OS fallback can leave a temporarily stale Active status in the Workbench UI.
Preflight errors do not stop the VM. This does not impose a timeout on hung
training. Environment variables do not automatically survive a VM restart.
