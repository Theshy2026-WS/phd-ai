# Independent ViT, late feature fusion and full Transformer follow-up

This new suite implements the three requested research routes. It preserves
the original CNN8 checkpoint, old experiments and their source-locked resumes.
The main dataset remains native64 Tiny ImageNet, seed42, with the original
80K training /10K selection /10K reserved split. Selection chooses checkpoints;
the reserved audit and official validation sets are not evaluated here.

## Finite queue

| Job | Initialization and operation | New training |
|---|---|---:|
| `vit_pretrain` | Independent image ViT, trained from scratch |100 epochs|
| `dual_fusion` | Original CNN8 plus the selected ViT from the first job; concatenate final features and classify jointly |30 epochs|
| `shallow64_attention` | Original CNN8 plus global self-attention whose Q/K/V all come from a native64 shallow CNN feature map |30 epochs|
| `cross16_mlp` | Original CNN8 plus two complete latent cross-attention/MLP blocks; deep8 queries middle16 |30 epochs|

The queue has190 new model-epochs. The previous CNN pretraining also contributes
to the total cost of the three warm models. Joint fusion is ViT100 plus joint30,
not a fresh100 model. This suite does not rerun the earlier13-model attention
queue or the19-model generalization queue. No early stopping or artificial
time cutoff is used. A resource or numerical failure is reported, not silently
converted into completion; independent later jobs can still produce results.

Structural estimates at native64 (one multiply-add counts as two FLOPs):

| Model | Parameters | Forward GFLOPs |
|---|---:|---:|
| Independent ViT |3,139,336|1.8489|
| Joint CNN/ViT feature fusion |27,041,608|5.8295|
| Shallow64 global attention |24,208,265|12.9537|
| Cross16 with two attention/MLP blocks |24,896,777|4.1735|

Normalization, softmax and elementwise work are excluded. Shallow64 training
also recomputes attention chunks; these estimates are not runtime predictions.

## Independent ViT and joint late fusion

The ViT uses patch4 on64x64 images (256 patches plus CLS), width224, five encoder
blocks, four heads, MLP ratio4 and dropout0.1. Each block has pre-LayerNorm,
multi-head self-attention, residual addition, and a separate nonlinear MLP
residual. It has learned CLS and positional embeddings. It uses no external
pretrained weights. Its attention is computed inFP32.

The joint model receives one augmented image tensor in both branches. It loads
the original CNN8 best and the independently trained ViT best, then updates both
branches and the new classifier from the fused classification loss. The CNN and
ViT do not share parameters. The CNN contributes its2048D pooled feature after
the original phase00 sampling and BN/ReLU. The ViT contributes its224D normalized
CLS feature. Their2272D concatenation feeds one200-class linear classifier.

The CNN columns of this classifier start from the original CNN classifier; ViT
columns start with small nonzero weights, allowing gradients to reach the ViT
from the first update. The old separate dense classifiers are removed from the
joint model. No branch is frozen and no auxiliary branch loss is silently added.
The initial fused model is measured; it is not assumed to exactly reproduce the
CNN. Separate branch removal is an inference intervention on a jointly trained
model, not the score of an independently trained CNN or ViT.

## Genuine64x64 shallow attention

The existing CNN stem normally has stride2 and outputs32x32. The new branch
applies the **same stem convolution weights with stride1** to produce64x64
features. The original stride2 CNN path is retained. All Q/K/V projections
originate from the shallow64 map; the4096 queries attend globally to all4096
keys. Width128 and four heads are used. The context is spatially pooled to8x8,
projected to2048 channels and added residually to the deep CNN feature before
the unchanged classification readout.

This is not window attention or attention on a downsampled source. Queries are
processed in chunks of64, with gradient checkpointing during training, to avoid
keeping a full batch64 x heads4 x4096x4096 attention tensor. Chunking preserves
the global attention computation; it trades recomputation for memory. It remains
quadratic in token count and can take substantially longer than deep8 attention.
Diagnostic entropy and retrieval summaries are accumulated chunk by chunk.
FLOPs include the extra stride1 stem computation and all QK/AV arithmetic.

## Two complete cross-attention/MLP blocks

Deep2048-channel8x8 and middle512-channel16x16 features are projected to128D.
Each of two blocks updates deep query tokens using pre-normalized cross-attention
to middle tokens, an attention residual, then a pre-normalized128->512->128 GELU
MLP and its residual. The final change relative to the starting query tokens is
projected back to2048D and added to the original deep feature. There is no CLS
token here: the original CNN pooling/classifier still performs the readout.

This implements complete attention-plus-MLP processing in CNN feature space.
It is not the independent image ViT of the first job. Relative to the previous
single cross16 insertion, it also changes depth and latent processing, so it is
an architecture screen rather than an MLP-only ablation.

## Training and analysis

Pure ViT uses the inherited AdamW recipe,100 epochs,5e-4 peak learning rate,
two warmup epochs and cosine decay. Warm models use5e-5 for loaded backbone
parameters and5e-4 for new fusion parameters, one warmup epoch, and a common
cosine multiplier that preserves the learning-rate ratio. The pretrained ViT
in the dual model also uses5e-5. Bias, normalization and positional parameters
receive the recorded no-decay handling. Original augmentation, Mixup/CutMix,
smoothing, batch64 and gradient clipping are retained. EMA is recorded for all
jobs. Formal schedules never resume from a finished smoke schedule.

Reports distinguish initial epoch0, best, best actually trained, last raw and
last EMA. Best and actual last get clean training/selection evaluation, class
diagnostics, confusion and paired correction/harm analysis. A separate feature
analysis stage uses the same fixed clean training probe and selection images to
compare branch interventions. Successful checkpoints are analyzed even when a
different job fails. Missing or failed jobs are explicitly labeled.

The completion email contains **one ordinary ZIP with report directories**, not
source files or nested archives. It keeps JSON/CSV/configuration, reports,
curves, class statistics, paired aggregates and available galleries. If needed,
email compaction first removes bulky per-image CSVs/manifests and then PNG
galleries; every omission is listed. Full outputs/checkpoints stay on the VM.
Source hashes and the Git commit preserve provenance without emailing scripts.
SMTP acceptance alone does not prove inbox delivery. Managed stop is requested
after packaging/notification, including after a child failure. A hung process
has not finished and does not trigger that completion path.

## Workbench

Wait for any running experiment to finish before updating its checkout. In a
Workbench terminal:

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
tmux new-session -A -s vit_followup
```

Optional preflight (no training, email transmission or shutdown):

```bash
python -u run_vit_followup_overnight.py \
  --parent-run runs/spatial_tiny/suite_20261103T175622_a19c86/cnn8_seed42 \
  --data-dir ./data --num-workers 4 \
  --check-only --fallback-os-shutdown
```

Run the requested complete queue:

```bash
python -u run_vit_followup_overnight.py \
  --parent-run runs/spatial_tiny/suite_20261103T175622_a19c86/cnn8_seed42 \
  --data-dir ./data --num-workers 4 \
  --vit-epochs 100 --fusion-epochs 30 \
  --complete-all --fallback-os-shutdown
```

Use existing `GMAIL_USER` / `GMAIL_APP_PASSWORD` or the already supported ignored
Workbench `email.txt`. No credentials are written into reports or passed to the
training child. Ctrl+B thenD detaches tmux. When the finite queue and reports
finish, one email is attempted and managed instance stop is requested; the
explicit OS fallback is available if managed stop fails.

Resume an interrupted suite with its actual printed directory:

```bash
python -u run_vit_followup_overnight.py \
  --resume runs/vit_followup_tiny/SUITE_DIRECTORY \
  --fallback-os-shutdown
```

Resume preserves the source, data, software/GPU environment, initialization and
epoch schedules. A separate `--smoke` suite runs one epoch per job for resource
checking and must not be presented or extended as the formal experiment.
