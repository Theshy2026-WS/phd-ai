# Tiny ImageNet: preactivation and two-stage experiments

These are **new architecture experiments**, not reproductions of the original
BoTNet-50 or CoAtNet-0. All seven existing variants remain unchanged. No ACmix
training is included in this suite.

| Variant | Stages after stem | Stage outputs | Attention | Parameters | Estimated forward GFLOPs |
|---|---|---|---|---:|---:|
| `resnet50_preact` | PreAct bottlenecks 3/4/6/3 | 32/16/8/4, channels 256/512/1024/2048 | none | 23,902,472 | 2.5964 |
| `botnet_twostage` | C3 -> B3 | 32/16, channels 256/512 | 3 global BoT blocks, 4 heads, 256 tokens each | 970,824 | 0.8800 |
| `coatnet_twostage` | MBConv2 -> Transformer2 | 32/16, channels 96/192 | 2 global relative-bias Transformer blocks, 6 heads, 256 tokens each | 1,221,860 | 1.0263 |
| `conformer_tiny` | unchanged author-derived dual branch | unchanged | unchanged | 23,011,104 | 0.8393 |

All inputs remain native 64x64. A stage contains multiple blocks; the stem and
classification head are outside the two stages. BoT pools 32->16 **before** B3;
all B blocks operate at 16. CoAt uses a 96-channel, stride-2 two-convolution
stem, a stride-1 MBConv stage, then a stride-2 Transformer stage (pool before
attention). The existing timm block definitions/relative bias/MLP are reused.
PreAct uses full BN-ReLU-convolution preactivation with projection from the
preactivated input, no activation after residual addition, and final BN-ReLU
before pooling. Unlike old `preact50`, it has 3/4/6/3, not 3/4/6/4 blocks.

The shared `configs/tiny_imagenet_fusion.yaml` recipe is unchanged: seed 42,
100 epochs, same 80k/10k/10k split, augmentation, optimizer, batch 64, FP32
evaluation and attention islands. Architecture metadata records the fixed
depths/channels/spatial sizes per run. No pretraining or input enlargement.

**Interpretation:** removing stages also changes capacity, width, downsampling
and computation. The two small models are not parameter-matched controls. A
gain/loss cannot by itself prove that higher spatial attention is the cause.
Compare whole-model Top-1, parameters, FLOPs, time, latency and throughput first.
GFLOPs exclude softmax, normalization, pooling and elementwise work; not a
wall-time prediction. L4 timing/100-epoch convergence are not tested locally.

## Run on Workbench

Do not pull while an older run is actively training. Completed results from
commit 0049d1a remain comparable; unfinished runs must resume on their recorded
source version, not this revision. This command starts a **new** four-model
suite and does not reuse an earlier Conformer run.

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
python -u overnight_twostage.py --check-only --fallback-os-shutdown
```

Preflight checks CUDA, current VM identity, Workbench resource access, and the
attached service account's project-level `notebooks.instances.stop` permission.
It makes no IAM changes and does **not** stop or train with `--check-only`.
If this fails, send the error before leaving the machine unattended. The
Cloud Resource Manager API must be available for its permission check. An
instance-only IAM grant is conservatively rejected by the project-level check.
No downloaded credentials or authentication tokens are printed/saved.

Then create/attach a tmux session (safe even when already inside tmux):

```bash
if [ -n "$TMUX" ]; then
  echo "Already inside tmux; use this terminal."
else
  tmux new-session -A -s twostage-night
fi
```

Inside that tmux terminal:

```bash
python -u overnight_twostage.py --epochs 100 --fallback-os-shutdown
```

Live progress appears in the terminal and is appended to
`twostage_100ep_overnight.log`. Press Ctrl+B, then D to detach; closing your
laptop or losing **your laptop's** network connection does not terminate tmux
training on the cloud VM. Keep the VM running and avoid an idle-shutdown policy
that could interrupt terminal jobs. This does not protect against VM failure,
cloud maintenance or instance stop.

After the training process exits (success **or error**), the wrapper flushes
results and requests managed Workbench stop, which should update Console to
Stopped after the asynchronous operation completes. If the API repeatedly
fails, the opt-in OS shutdown fallback limits cost but may leave the Workbench
UI temporarily stale, as happened previously. A hung training process does not
exit and will not trigger shutdown. No script can guarantee a remote stop if
both API and OS shutdown fail. Do not use this wrapper while unrelated jobs
must remain running on the same VM.

Follow logs from another terminal:

```bash
tail -f ~/wenfei-phd-ai/twostage_100ep_overnight.log
```

If training failed, restart the instance, fix the issue, and resume the printed
suite directory on the same source version:

```bash
python -u overnight_twostage.py --resume runs/literature_tiny/SUITE_DIRECTORY --fallback-os-shutdown
```

Completed models are skipped; unfinished models resume the last completed epoch.
Do not change the original total epoch schedule on resume.

Training only, without automatic stop:

```bash
python -u literature_experiment.py suite --variants resnet50_preact botnet_twostage coatnet_twostage conformer_tiny --config configs/tiny_imagenet_fusion.yaml --seeds 42 --epochs 100
```

Official managed stop API:
https://docs.cloud.google.com/vertex-ai/docs/workbench/reference/rest/v2/projects.locations.instances/stop
CLI equivalent:
https://docs.cloud.google.com/sdk/gcloud/reference/workbench/instances/stop
