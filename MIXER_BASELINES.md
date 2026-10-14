# Tiny ImageNet mixer structure baselines

All four are supervised **from scratch**, seed42, 100 epochs under the existing
`configs/tiny_imagenet_fusion.yaml` recipe: same data split, augmentation,
AdamW, batch64, precision and evaluation protocol. No model weights or dataset
are downloaded during model construction. Dataset loading on Workbench retains
the existing download behavior. No FCMAE self-supervised pretraining is included.

These reproduce published **structures via pinned timm 1.0.25**, with explicit
Tiny-input adaptations. They do not reproduce the original training recipes,
pretraining, or published ImageNet scores. Do not interpret the shared recipe
as a per-model optimized upper bound. Existing baselines remain unchanged.

| CLI variant | Base architecture | Depths / channels | Spatial sizes | Parameters | Forward GFLOPs |
|---|---|---|---|---:|---:|
| `convmixer768_tiny64` | ConvMixer-768/32 (ReLU) | 32 / 768 | 16 throughout | 20,419,016 | 10.2994 |
| `convnextv2_tiny64` | ConvNeXt V2-Tiny, GRN retained | 3/3/9/3 / 96/192/384/768 | 32/16/8/4 | 28,016,840 | 2.9025 |
| `convformer_s18_tiny64` | ConvFormer-S18, SepConv mixers | 3/3/9/3 / 64/128/320/512 | 32/16/8/4 | 25,135,248 | 2.5746 |
| `caformer_s18_tiny64` | CAFormer-S18, C/C/A/A mixers | 3/3/9/3 / 64/128/320/512 | 32/16/8/4 | 24,702,456 | 2.5824 |

GFLOPs use our existing 1 MAC = 2 FLOPs estimator, including Conv/Linear and
attention QK/AV; excluding normalization (including GRN), softmax, pooling,
activations and elementwise operations. Do not directly compare with paper
MAC counts or infer wall time from this table. ConvMixer has substantially
more arithmetic; measure one epoch on L4 before committing to overnight runs.

## Exact adaptations

- **ConvMixer:** directly use timm's ConvMixer class with dim768, depth32,
  kernel7, ReLU, BN, depthwise-residual/pointwise blocks. The authors' released
  768/32 checkpoint uses ReLU (their generic small code sample uses GELU).
  Patch7 becomes patch4 to tile native 64x64 input without dropped border
  pixels; the feature map is 16x16 throughout. Classifier becomes 200 classes.
- **ConvNeXt V2:** `timm.create_model('convnextv2_tiny', pretrained=False)`.
  Preserve all stages, depthwise7x7, channel MLP and GRN layers. Replace only
  the stem 4x4 stride4 convolution with 2x2 stride2, initialized with the same
  timm truncated-normal scheme. Stage0 keeps stride1; later stages downsample.
  This is structure-only, not the ConvNeXt V2 FCMAE learning method.
- **ConvFormer / CAFormer:** use timm S18 definitions. Both retain the original
  blocks, StarReLU, normalization, residual scaling and MLP heads. Change
  stem7x7 stride4 padding2 to stride2 padding3. Classifier becomes 200 classes.
  ConvFormer uses SepConv in all 18 blocks; CAFormer uses six SepConv blocks
  then twelve self-attention blocks (nine with 64 tokens and three with 16).
  No CLS token is added. CAFormer QKV, QK, softmax, AV and output projection
  run in FP32 islands, with explicit matmul, matching our stability policy.
- `drop_rate=0` and `drop_path_rate=0` are kept for the common recipe. This is
  not the original papers' regularization/optimizer selection. In particular,
  the MetaFormer paper's ConvFormer and CAFormer training recipes differ;
  here both deliberately use our existing AdamW recipe for comparison.
- Feature-info reductions and saved architecture metadata reflect the actual
  Tiny geometries, not the original stride32 layouts.

ConvFormer/CAFormer form the closest new pair, with matching stage depths,
widths and resolution. They are not exactly parameter-matched, and differences
must not automatically be attributed to attention alone. No novelty claim.

## Run after the current training has completed

**Do not pull into an active training checkout.** Source hashes are strict for
resume. Completed results from 0c160b3 remain comparable via an exact whitelist;
unfinished old runs must resume with their recorded source, not this revision.

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
python -m pip install -r requirements-literature.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name())"
```

One-epoch smoke run, no shutdown:

```bash
python -u literature_experiment.py suite --variants convmixer768_tiny64 convnextv2_tiny64 convformer_s18_tiny64 caformer_s18_tiny64 --config configs/tiny_imagenet_fusion.yaml --seeds 42 --epochs 1
```

The smoke suite is separate; it is not the first epoch of the 100-epoch recipe
and cannot be extended by resume. If it passes, start a fresh suite inside
tmux. Already inside tmux? Reuse that session rather than nesting another.

```bash
tmux new-session -A -s mixers-night
```

Inside tmux, check stop permissions, then start the four-model run:

```bash
python -u overnight_twostage.py --check-only --fallback-os-shutdown
python -u overnight_twostage.py --variants convmixer768_tiny64 convnextv2_tiny64 convformer_s18_tiny64 caformer_s18_tiny64 --epochs 100 --log mixers_100ep_overnight.log --fallback-os-shutdown
```

Despite the legacy wrapper name, explicit `--variants` selects **only these
four models**, not the earlier two-stage models or Conformer. The wrapper's
default model list has not changed. It still stops the current Workbench on
training-process exit (success or failure); OS fallback may leave a stale UI.
Do not run unrelated jobs on the VM when using auto-stop. Laptop disconnects
do not stop tmux; VM shutdown, failures or idle shutdown still interrupt it.

Watch from another terminal:

```bash
tail -f ~/wenfei-phd-ai/mixers_100ep_overnight.log
```

If needed, resume on the same source revision using the printed suite path:

```bash
python -u overnight_twostage.py --resume runs/literature_tiny/SUITE_DIRECTORY --log mixers_100ep_overnight.log --fallback-os-shutdown
```

Do not pass `--variants` on resume. Completed models are skipped. Results
include comparison.csv, per-model epochs.csv, summary.json, architecture,
source hashes, parameters, FLOPs estimates, timings, memory and inference speed.

## Primary sources and implementation provenance

- ConvMixer: https://arxiv.org/abs/2201.09792
  Author code: https://github.com/locuslab/convmixer (README documents ReLU variant).
- ConvNeXt V2: https://arxiv.org/abs/2301.00808
  Author code: https://github.com/facebookresearch/ConvNeXt-V2
- ConvFormer / CAFormer: https://arxiv.org/abs/2210.13452
  Author code: https://github.com/sail-sg/metaformer
- Runtime implementation is pinned by requirements-audit.txt to timm 1.0.25:
  `timm/models/convmixer.py`, `timm/models/convnext.py`, `timm/models/metaformer.py`.
  These are external dependency modules, not newly vendored source. Our adapter
  changes are in models/literature_mixers.py and recorded in source hashes.
