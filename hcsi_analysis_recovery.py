"""Recover HCSI diagnostics without changing historical training source hashes.

The HCSI dependency manifest includes the completed joint reference, whereas
that reference's own manifest describes only its original independent sources.
Reconstruct it through its canonical loader, never through the enlarged HCSI
manifest. All original source/checkpoint/recipe checks remain active.
"""
from contextlib import contextmanager
from pathlib import Path
import threading

from correction_experiment import file_hash
from literature_experiment import save_json


ROOT = Path(__file__).resolve().parent
PROTOCOL = 'tiny-hcsi-analysis-source-scope-recovery-v1'
_REFERENCE_LOCK = threading.RLock()


def load_reference_model(plan, sources=None):
    """Use the reference's own validated source scope, without relaxing checks."""
    import best_pair_joint
    reference = plan['reference']
    suite = Path(reference['suite']).resolve()
    checkpoint = suite/'training/new_joint_concat/best.pt'
    if Path(reference['checkpoint']).resolve() != checkpoint:
        raise ValueError('Recovery reference path differs from its recorded joint checkpoint')
    if sources is not None and sources['reference']['record'] != reference:
        raise ValueError('HCSI source reference differs from the recovery plan')
    if checkpoint.is_symlink() or file_hash(checkpoint) != reference['checkpoint_sha256']:
        raise ValueError('Recovery reference checkpoint bytes changed')
    canonical = best_pair_joint.validate_resume(suite)
    # Deliberately omit sources: the canonical loader restores and verifies its
    # original dependencies instead of comparing the larger HCSI manifest.
    model, state = best_pair_joint.load_best_pair_model(canonical, suite/'training')
    if (state['epoch'] != reference['epoch'] or state['source'] != reference['source']
            or state['selection'] != reference['selection']):
        raise ValueError('Recovered reference score or selected checkpoint identity changed')
    return model, state


@contextmanager
def reference_loader_scope():
    """Temporarily supply the corrected adapter; always restore the old module."""
    import hcsi_tiny_experiment
    with _REFERENCE_LOCK:
        original = hcsi_tiny_experiment.load_reference_model
        hcsi_tiny_experiment.load_reference_model = load_reference_model
        try:
            yield
        finally:
            hcsi_tiny_experiment.load_reference_model = original


def run_hcsi_recovery_diagnostics(plan, training_dir, output_dir, examples=8,
                                  deadline_unix=None):
    """Re-evaluate saved weights only; no optimizer or training loop is called."""
    import hcsi_tiny_experiment
    from hcsi_tiny_diagnostics import run_hcsi_diagnostics
    training = Path(training_dir).resolve()
    output = Path(output_dir).resolve()
    if output == training or training in output.parents or output in training.parents:
        raise ValueError('Recovered analysis must be separate from immutable training artifacts')
    # Validate the full original HCSI dependency manifest independently. The
    # reference loader then validates the reference's narrower dependency scope.
    sources = hcsi_tiny_experiment.load_sources(plan)
    if sources['reference']['record'] != plan['reference']:
        raise ValueError('Recovery reference is not the pinned HCSI reference')
    output.mkdir(parents=True, exist_ok=True)
    names = ('hcsi_tiny_experiment.py', 'hcsi_tiny_diagnostics.py', 'best_pair_joint.py')
    save_json(output/'analysis_recovery_implementation.json', dict(
        protocol=PROTOCOL, training_repeated=False, original_source_files_modified=False,
        correction='Restore joint reference using its original canonical dependency scope',
        reference_checkpoint_sha256=plan['reference']['checkpoint_sha256'],
        recovery_source_sha256=file_hash(Path(__file__)),
        unchanged_original_source_hashes={name:file_hash(ROOT/name) for name in names}))
    with reference_loader_scope():
        return run_hcsi_diagnostics(plan, training, output, examples=examples,
                                    deadline_unix=deadline_unix)
