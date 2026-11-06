"""Validate immutable CNN/ViT checkpoints used by the separate continuation suite.

The old 30-epoch suite is evidence and initialization, never a writable resume
target. A continuation has its own schedule, metadata and output directory.
"""
import copy
import json
from pathlib import Path

import torch

from correction_experiment import file_hash
from literature_experiment import ROOT, recipe_hash
from mechanism_diagnostics import load_spatial_parent, _relocation_only


SOURCE_PROTOCOL = 'tiny-vit-followup-suite-v1'
TRAINING_PROTOCOL = 'tiny-vit-followup-training-v1'
DEFAULT_SOURCE = ROOT/'runs/vit_followup_tiny/suite_20261105T033555_5ba88b'
MODEL_SOURCES = ('models/vit_followup.py', 'models/vit.py', 'models/spatial_preact.py',
                 'models/literature_twostage.py', 'models/literature_alternet.py')


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Need a regular source metadata file: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _checkpoint(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Need a regular source checkpoint: {path}')
    # These are the user's locally generated, source-validated Workbench runs.
    # Optimizer/RNG continuation requires their complete checkpoint dictionaries.
    return torch.load(path, map_location='cpu', weights_only=False)


def _metadata_matches(checkpoint, summary, job, source_plan):
    metadata = checkpoint.get('metadata', {})
    expected = dict(protocol=TRAINING_PROTOCOL, job=job,
        variant='vit' if job == 'vit_pretrain' else 'dual_fusion',
        seed=source_plan['seed'], split_hash=source_plan['split_hash'],
        parent_checkpoint_sha256=source_plan['parent_checkpoint_sha256'])
    for key, value in expected.items():
        if metadata.get(key) != value or summary.get(key) != value:
            raise ValueError(f'Source checkpoint/summary differs: {job}/{key}')
    for key in ('source_hashes', 'recipe_hash', 'environment', 'initial_state_sha256',
                'vit_checkpoint_sha256'):
        if metadata.get(key) != summary.get(key):
            raise ValueError(f'Source checkpoint metadata differs: {job}/{key}')
    if metadata.get('source_hashes') != source_plan.get('source_hashes'):
        raise ValueError(f'Source checkpoint differs from its training source: {job}')
    if metadata.get('recipe_hash') != recipe_hash(checkpoint['config']):
        raise ValueError(f'Source checkpoint recipe differs: {job}')
    for name in MODEL_SOURCES:
        if metadata.get('source_hashes', {}).get(name) != file_hash(ROOT/name):
            raise ValueError(f'Source model implementation changed: {name}')
    model = checkpoint.get('model', {})
    if not model or any(t.is_floating_point() and not torch.isfinite(t).all() for t in model.values()):
        raise ValueError(f'Invalid source model weights: {job}')


def load_sources(source_suite, config=None):
    source = Path(source_suite).expanduser().resolve()
    if not source.is_dir() or source.is_symlink():
        raise ValueError('Need the original completed ViT follow-up suite directory')
    plan = _read(source/'suite.json')
    completed = _read(source/'completion.json')
    if (plan.get('protocol') != SOURCE_PROTOCOL or completed.get('protocol') != SOURCE_PROTOCOL
            or completed.get('status') != 'complete' or plan.get('smoke')
            or plan.get('vit_epochs') != 100 or plan.get('fusion_epochs') != 30):
        raise ValueError('Need a completed, non-smoke ViT100/joint30 source suite')
    training = source/'training'
    training_plan = _read(training/'suite.json')
    training_done = _read(training/'completion.json')
    if (training_plan.get('protocol') != TRAINING_PROTOCOL
            or training_done.get('status') != 'complete' or training_done.get('failures')
            or training_done.get('skipped_jobs')):
        raise ValueError('Source training did not complete successfully')
    for key in ('seed', 'split_hash', 'parent_checkpoint_sha256'):
        if plan.get(key) != training_plan.get(key):
            raise ValueError(f'Source suite and training identity differ: {key}')
    parent, parent_summary, parent_state, parent_hash = load_spatial_parent(plan['parent_run'])
    if (parent_hash != plan['parent_checkpoint_sha256']
            or parent_summary['split_hash'] != plan['split_hash']
            or parent_summary['seed'] != plan['seed']):
        raise ValueError('Source CNN parent, seed or split changed')
    requested = copy.deepcopy(parent_state['config'] if config is None else config)
    _relocation_only(parent_state['config'], requested)
    vit_run, dual_run = training/'vit_pretrain', training/'dual_fusion'
    vit_summary, dual_summary = _read(vit_run/'summary.json'), _read(dual_run/'summary.json')
    if (vit_summary.get('status') != 'complete' or vit_summary.get('completed_epochs') != 100
            or dual_summary.get('status') != 'complete' or dual_summary.get('completed_epochs') != 30):
        raise ValueError('Source independent ViT100 and joint30 must both be complete')
    vit = _checkpoint(vit_run/'best.pt')
    best, last = _checkpoint(dual_run/'best.pt'), _checkpoint(dual_run/'last.pt')
    _metadata_matches(vit, vit_summary, 'vit_pretrain', training_plan)
    _metadata_matches(best, dual_summary, 'dual_fusion', training_plan)
    _metadata_matches(last, dual_summary, 'dual_fusion', training_plan)
    if (vit['epoch'] != vit_summary['best_epoch'] or vit['selection'] != vit_summary['best_selection']
            or best['epoch'] != dual_summary['best_epoch'] or best['selection'] != dual_summary['best_selection']):
        raise ValueError('Source selected checkpoint chronology/scores differ')
    vit_hash = file_hash(vit_run/'best.pt')
    if (best['metadata'].get('vit_checkpoint_sha256') != vit_hash
            or last['metadata'].get('vit_checkpoint_sha256') != vit_hash):
        raise ValueError('Source joint model used a different independent ViT checkpoint')
    required = ('model', 'ema_model', 'optimizer', 'scheduler', 'scaler', 'rng', 'history', 'epoch')
    if any(name not in last for name in required):
        raise ValueError('Joint last.pt lacks optimizer/EMA/scaler/RNG continuation state')
    if last['epoch'] != 30 or [row['epoch'] for row in last['history']] != list(range(31)):
        raise ValueError('Joint continuation must start after the completed30-epoch history')
    if last['history'][-1]['augmentation_epoch'] != 130:
        raise ValueError('Expected original CNN100 plus joint augmentation epochs101..130')
    if recipe_hash(last['config']) != recipe_hash(best['config']):
        raise ValueError('Joint best/last training recipes differ')
    optimizer = last['optimizer']
    if not optimizer.get('state') or not optimizer.get('param_groups'):
        raise ValueError('Joint continuation requires the original nonempty AdamW state')
    roles = {group.get('role') for group in optimizer['param_groups']}
    if roles != {'encoder', 'vit', 'fusion'}:
        raise ValueError('Source AdamW groups do not match independent CNN/ViT/fusion roles')
    for parameter_state in optimizer['state'].values():
        if any(isinstance(t, torch.Tensor) and t.is_floating_point() and not torch.isfinite(t).all()
               for t in parameter_state.values()):
            raise ValueError('Non-finite source AdamW state')
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in last['ema_model'].values()):
        raise ValueError('Non-finite source EMA weights')
    paths = [source/'suite.json', source/'completion.json', training/'suite.json',
        training/'completion.json', parent/'summary.json', parent/'best.pt',
        vit_run/'summary.json', vit_run/'best.pt', dual_run/'summary.json',
        dual_run/'best.pt', dual_run/'last.pt']
    # When available, the already-delivered diagnostic identity independently
    # verifies the source checkpoint bytes before the new stage is initialized.
    identity_path = source/'feature_analysis/summary.json'
    if identity_path.is_file():
        diagnostic = _read(identity_path)
        identity = diagnostic.get('identity', {})
        if diagnostic.get('status') != 'complete' or identity.get('parent_checkpoint_sha256') != parent_hash:
            raise ValueError('Source feature-analysis provenance is inconsistent')
        recorded = identity.get('checkpoints', {})
        for job, name in (('vit_pretrain', 'best.pt'), ('dual_fusion', 'best.pt'), ('dual_fusion', 'last.pt')):
            if recorded.get(job, {}).get(name) != file_hash(training/job/name):
                raise ValueError(f'Source checkpoint changed since feature analysis: {job}/{name}')
        paths.append(identity_path)
    files = {str(path.resolve()): file_hash(path) for path in paths}
    provenance = dict(source_protocol=SOURCE_PROTOCOL, source_suite=str(source),
        source_training_dir=str(training), parent_run=str(parent), source_files=files,
        parent_checkpoint_sha256=parent_hash, vit_checkpoint_sha256=vit_hash,
        dual_last_checkpoint_sha256=file_hash(dual_run/'last.pt'),
        dual_best_checkpoint_sha256=file_hash(dual_run/'best.pt'),
        source_joint_epochs=30, source_independent_vit_epochs=100,
        source_cnn_epochs=parent_summary['completed_epochs'],
        source_joint_best_epoch=dual_summary['best_epoch'],
        source_joint_best_accuracy=dual_summary['best_selection']['accuracy'],
        source_joint_last_accuracy=last['history'][-1]['selection_accuracy'],
        source_execution_environment=plan['execution_environment'])
    return dict(source_suite=str(source), source_training_dir=str(training),
        parent_run=str(parent), parent_model=parent_state['model'], vit_model=vit['model'],
        dual_best=best, dual_last=last, config=requested, seed=plan['seed'],
        split_hash=plan['split_hash'], parent_completed_epochs=parent_summary['completed_epochs'],
        dual_completed_epochs=30, source_files=files, provenance=provenance)
