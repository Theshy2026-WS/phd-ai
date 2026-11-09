# Pairwise and HCSI: complete queue, one email, then stop Workbench

This entry point implements the user's latest policy: finish all planned
experiments, send their analysis files together, then stop the instance.
There is **no default total runtime limit**. In particular, the previous
10-hour cutoff is not applied to this combined queue. Each arm still has its
declared30-epoch schedule; no new independent backbone pretraining is added.

## What runs

The queue runs the four pairwise arms first, followed by the five HCSI/control
arms:9 arms and270 new model epochs in total. Each child completes its training
and analysis before the next study begins. The GPU studies run sequentially.

| Study | Arms | Initialization and trainability |
|---|---|---|
| Pairwise | Pairwise retrieval, static query, matched MLP, training-mean tokens | Freeze the completed65.45% full joint model; train only each new resolver |
| HCSI and controls | CNN continuation, trained-ViT late MLP, HCSI, CNN MLP, random-ViT late MLP | Start from the same independent CNN64.80% and ViT50.18% sources, except the explicitly random-ViT control; update all participating encoder parameters |

Pairwise is retained. Its result is **not** used to initialize HCSI: these are
different controlled studies. They share the same completed65.45% joint
checkpoint as anchor/reference, while preserving their own specified training
initializations and controls. The already completed ordinary joint30 experiment
is not repeated.

The underlying child scripts do not send mail or stop the VM. Their saved
`shutdown_after_completion=False` fields remain correct; the new parent runner
alone owns the final notification and shutdown policy. The old standalone
`run_pairwise_resolver.py` and `run_hcsi_tiny.py` retain their documented
email-only/keep-running behavior. Use the combined entry point below for the
requested automatic stop.

## Workbench command

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
tmux new-session -A -s fusion_overnight
```

Inside tmux:

```bash
python -u run_fusion_overnight.py \
  --data-dir ./data \
  --num-workers 4 \
  --epochs 30
```

Do not add `--max-hours 10` to this command: the latest request is to let all
planned arms finish. No separate pairwise or HCSI command is needed.

Preflight checks the saved weights, dataset/split, GPU/software, SMTP login and
managed Workbench stop access before starting training. Existing ignored
`email.txt` credentials are used. `--check-only` performs preflight without
training, a transmitted email or a stop request.

## Delivery and failure behavior

After both studies end, one combined reports-only ZIP is attached to one
completion email. Within that ZIP, the `pairwise/` and `hcsi/` directories each
contain training and feature-analysis reports, including their individual arms.
They are two study directories, not two experimental groups. The archive has no
nested ZIPs, executable source, weights, optimizer state, datasets or credentials.
Compaction removes bulk per-image outputs before aggregate evidence.

The final order is: **finish analysis -> package reports -> attempt the email ->
flush completion records -> request managed Workbench stop**. Sending the files
must happen before the instance is powered off. SMTP acceptance and the stop
request are recorded separately; they do not prove inbox delivery or a final
stopped state observed by a process that may disappear during shutdown.

After an actual training child has started, failures and interruptions also
retain available results, attempt the final email and request the final stop.
A failure in one study does not prevent attempting the other when possible.
Mail failures are retried, and original files remain on the persistent VM disk.
Preflight failure and check-only do not arm shutdown. Optional OS-shutdown
fallback is explicit and its permission is checked before training.

## Resume

Resume the parent queue using its printed directory, rather than launching new
standalone wrappers. Completed child work and its schedules are preserved;
incomplete epochs replay according to each study's atomic checkpoint protocol.
Parent completion and runtime records are flushed before the managed stop can
terminate the process. Powered-off idle time is not treated as active training.

```bash
python -u run_fusion_overnight.py \
  --resume runs/fusion_overnight/ACTUAL_QUEUE_DIRECTORY
```

A persisted execution marker distinguishes a queue that actually started from
one that only passed preflight. Explicitly resuming a completed queue can retry
combined delivery and final stop without rerunning its completed studies.

The new workflow tests have28 passes and one Windows symlink-permission skip
(29 cases). They cover uncapped default execution, pairwise-before-HCSI order,
independent initializations/reference pinning, absence of intermediate mail or
stop calls, resume identities, accounting before stop, ordinary ZIP content and
compaction, one combined delivery, and final managed-stop attempts on failure.
The new modules compile and the CLI parses. All actual SMTP, cloud-stop and GPU
operations were mocked in tests; no real instance operation was performed
during implementation.
