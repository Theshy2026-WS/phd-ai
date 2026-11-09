# Targeted late-fusion mechanism controls

This suite tests two missing explanations for the current best development result and one readout difference. It reuses the existing 65.45% joint model as a reference and does not train that reference again. The source CNN is 64.80%; source ViT is 50.18%. These are development selection scores, not independent audit results or universal model ceilings.

| Arm | Encoders at initialization | Fusion head | Readout | Main question |
| --- | --- | --- | --- | --- |
| Existing reference, reused | Independently trained CNN and ViT | Inherit CNN columns/bias; small new ViT columns | CNN pool + ViT CLS, Linear | Existing best reference |
| `random_vit_cls_linear` | Same trained CNN; random ViT | Same inherited CNN head policy | Same CLS + Linear | Does independent ViT training help beyond adding a trainable branch? |
| `reset_cnn_head_cls_linear` | Same trained CNN and ViT | Reset CNN columns/bias; preserve the new ViT columns' initialization | Same CLS + Linear | Does retaining the CNN decision boundary help within the same 30-epoch fusion budget? |
| `patch_mean_linear` | Same trained CNN and ViT | Same inherited CNN head policy | Mean of normalized patch tokens instead of CLS + Linear | Does the ViT readout help explain the previous patch-mean/MLP family's lower score? |

Both encoders and the classifier update in every new arm. No frozen-backbone model, additional HCSI module, pairwise resolver or global 64x64 attention is added. Each arm uses at most 30 epochs; the same source checkpoints, split, data-view schedule, optimizer and learning-rate policy are retained. Changed classifier initialization is intentionally part of the second control.

The reference classifier is

`z = W_C f_C + W_V f_V + b`.

It starts from the trained CNN classifier for `W_C,b` and small new ViT columns. Preserving that initial prediction function may help joint adaptation, but this mechanism is a hypothesis until the controls support it. The previous random-ViT comparison used patch-mean + MLP, so it does not replace the first control in the winning CLS + Linear family.

The diagnostics retain best, best-trained and final checkpoints, paired corrected/harmed counts, NLL, branch contributions and training/development gaps. Reports must distinguish a source checkpoint selected at epoch 0 from improvement caused by new training. Masking a branch of a co-adapted joint model does not measure an independently trained single branch.

Interpretation is restricted to the recorded recipe and 30-epoch budget. If random ViT matches the trained-ViT reference, independent ViT training is not established as the gain's source under this recipe. If resetting the CNN head does not hurt, decision-boundary preservation is not established as the key mechanism. Patch-mean + Linear isolates readout; it is not a new attention method. None of these outcomes by itself establishes novelty, and development-set selection is exploratory.

After these controls, fix a candidate and comparison rules before one confirmatory audit evaluation. This suite never evaluates the reserved audit partition or official validation split.

## Run on Workbench

```bash
git pull --ff-only
python -u run_fusion_mechanism.py \
  --data-dir ./data \
  --num-workers 4 \
  --epochs 30
```

Run inside `tmux` as with previous long experiments. The runner prints its suite path. It has **no default total wall-time cutoff**: all three arms and diagnostics finish, then one ordinary reports-only ZIP is emailed, followed by managed Workbench stop. Existing checkpoints and full prediction arrays stay on disk. There are no intermediate emails or stops.

Preflight authenticates email without sending a message and verifies managed stop permission before creating a suite or starting training. The final email/stop path is armed only after a child starts. Training, analysis, package or email failure still attempts delivery of available evidence followed by stop. SMTP acceptance does not prove inbox delivery. A rejected managed stop is reported as a failure, not a confirmed shutdown.

Preflight only:

```bash
python -u run_fusion_mechanism.py \
  --data-dir ./data --num-workers 4 --epochs 30 --check-only
```

This creates no suite, trains nothing, sends no message and stops nothing.

Resume the printed path without changing source files or experiment identity:

```bash
python -u run_fusion_mechanism.py \
  --resume runs/fusion_mechanism_tiny/suite_PRINTED_NAME
```

Resume retains the original horizon, source identities, split, recipe and delivery policy. Do not replace a saved suite with a different run or mix source versions. A completed arm is reused; existing training identity checks remain active.
