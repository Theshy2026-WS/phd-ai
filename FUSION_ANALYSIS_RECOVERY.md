# Recover the completed queue's HCSI analysis without retraining

Queue `queue_20261109T035729_2d004a` finished all nine30-epoch training jobs.
Pairwise training and analysis completed. HCSI training also completed, but its
post-training reference-model loader raised a source-identity ValueError.

The HCSI dependency map contains42 files: the original36 independent-source
files plus six files belonging to the completed joint reference. The reference
model's own validator correctly expects its historical36-file dependency map.
Passing the enlarged HCSI map into that validator was an analysis-code error.
The archived36 original hashes are unchanged. It is not evidence of damaged
weights or failed training.

The recovery adapter loads and validates the reference through its canonical
loader using its own original dependency scope. The full42-file HCSI dependency
map is independently validated. Real changed source/checkpoint hashes still
cause rejection. Original research source files are left unchanged so historical
source-hash and resume checks remain valid.

The recovery runner verifies every job's completed summary, atomic last history
and selected checkpoint, then hashes completed training and pairwise artifacts.
It runs only the existing HCSI inference/diagnostic routine with the corrected
reference adapter. It does not call a training function or schedule a training
subprocess. The existing pairwise analysis is reused. Historical failure/status
markers are backed up with hashes before publishing recovered completion status.

Start the instance if necessary, then run inside the original checkout:

```bash
cd ~/wenfei-phd-ai
git pull --ff-only
tmux new-session -A -s analysis_recovery
```

Inside tmux:

```bash
python -u recover_fusion_analysis.py \
  --queue runs/fusion_overnight/queue_20261109T035729_2d004a
```

The original run directories and saved weights must remain on the VM. The email
ZIP alone does not contain model weights and cannot reproduce inference.
The existing recorded data configuration and GPU/software environment are used.
`--check-only` validates prerequisites without analysis, a transmitted email or
a stop request.

After analysis, the runner sends one combined reports email and requests managed
Workbench stop, following the latest user instruction. Recovery failure also
retains available evidence, attempts the notification and requests stop after
analysis has actually begun. Original training weights and pairwise results are
checked for byte identity before publication. Full prediction arrays remain on
disk. SMTP acceptance and a stop request do not prove inbox delivery or a stopped
state observed by the process that may disappear during shutdown.

The bug regression exercises the actual source-identity validator: the original
map passes, adding reference-only dependencies reproduces the error, canonical
loading succeeds, and genuine source/checkpoint modifications remain rejected.
Other tests cover restoration of the scoped adapter, no training calls,
completed-job validation, artifact preservation and email-before-stop behavior.
All17 new recovery tests pass. Both recovery modules compile and the CLI parses.
Actual GPU inference and SMTP/Workbench operations were not executed locally;
the user runs the recovery on the VM holding the original checkpoints.
