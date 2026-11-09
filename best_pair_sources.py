"""Validate the completed strongest independent CNN and ViT as immutable inputs.

Only model weights initialize the new joint experiment. No completed source
schedule, optimizer, report, or checkpoint is changed by this module.
"""
import copy
import hashlib
import json
from pathlib import Path

import torch

from correction_experiment import file_hash
from literature_experiment import ROOT, recipe_hash
from mechanism_diagnostics import _relocation_only


DEFAULT_CNN_SOURCE = ROOT/'runs/fusion_continuation_tiny/suite_20261106T194319_c2dd7a'
DEFAULT_VIT_SOURCE = ROOT/'runs/vit_augreg_tiny/suite_20261107T182435_87170e'
CNN_JOB = 'cnn_head_continue'
CNN_CHECKPOINT_SHA256 = '9a230237ad61f204c76a7df009da04607fc3ebbf3cc33d3258c76d8f8d8592eb'
VIT_CHECKPOINT_SHA256 = '7728fa7c2b13887034e6c4133534b180fd4b5e2457ab4841b0efee828c8346ed'
CNN_SUITE_PROTOCOL = 'tiny-fusion-continuation-suite-v1'
CNN_TRAINING_PROTOCOL = 'tiny-fusion-continuation-training-v1'
VIT_SUITE_PROTOCOL = 'tiny-vit-augreg-suite-v1'
VIT_TRAINING_PROTOCOL = 'tiny-vit-augreg-training-v1'


def _regular(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Need a regular immutable source file: {path}')
    return path


def _read(path):
    return json.loads(_regular(path).read_text(encoding='utf-8'))


def _checkpoint(path):
    # Own locally generated Workbench checkpoints include metadata/config state.
    return torch.load(_regular(path), map_location='cpu', weights_only=False)


def _hash_code(recorded):
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError('Source lacks implementation hashes')
    for name, digest in recorded.items():
        candidate = (ROOT/name).resolve()
        if not candidate.is_relative_to(ROOT.resolve()):
            raise ValueError('Source implementation path escapes repository')
        data = _regular(candidate).read_bytes()
        if digest not in (hashlib.sha256(data).hexdigest(),
                          hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()):
            raise ValueError(f'Source implementation changed: {name}')


def _suite(source, suite_protocol, training_protocol, completion_protocol=None):
    source = Path(source).expanduser().resolve()
    if not source.is_dir() or source.is_symlink():
        raise ValueError('Need a completed real source suite directory')
    plan, completed = _read(source/'suite.json'), _read(source/'completion.json')
    training = source/'training'
    recorded, done = _read(training/'suite.json'), _read(training/'completion.json')
    if (plan.get('protocol') != suite_protocol or completed.get('protocol') != suite_protocol
            or completed.get('status') != 'complete' or plan.get('smoke')
            or recorded.get('protocol') != training_protocol
            or done.get('protocol') != (completion_protocol or training_protocol) or done.get('status') != 'complete'
            or done.get('failures') or done.get('skipped_jobs')):
        raise ValueError('Need complete non-smoke source suite and training')
    for key in ('seed', 'split_hash', 'config', 'execution_environment'):
        if plan.get(key) != recorded.get(key):
            raise ValueError(f'Suite/training source identity differs: {key}')
    _hash_code(plan.get('source_hashes'))
    _hash_code(recorded.get('source_hashes'))
    return source, training, plan, recorded, [source/'suite.json', source/'completion.json',
                                             training/'suite.json', training/'completion.json']


def _validate_state(run, plan, protocol, job, expected_variant):
    metadata = _read(run/'metadata.json')
    summary = _read(run/'summary.json')
    checkpoint = _checkpoint(run/'best.pt')
    if checkpoint.get('metadata') != metadata:
        raise ValueError('Selected checkpoint metadata differs from saved source metadata')
    expected = dict(protocol=protocol, job=job, variant=expected_variant,
                    seed=plan['seed'], split_hash=plan['split_hash'])
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Selected source identity differs: {job}/{key}')
    if (metadata.get('source_hashes') != plan.get('source_hashes')
            or metadata.get('environment') != plan.get('execution_environment')
            or metadata.get('recipe_hash') != recipe_hash(checkpoint['config'])):
        raise ValueError('Selected source implementation, environment or recipe differs')
    if any(summary.get(key) != value for key, value in metadata.items()):
        raise ValueError('Source summary metadata differs from selected checkpoint')
    if (summary.get('status') != 'complete'
            or summary.get('completed_epochs') != checkpoint['config']['training']['epochs']
            or checkpoint.get('epoch') != summary.get('best_epoch')
            or checkpoint.get('source') != summary.get('best_source')
            or checkpoint.get('selection') != summary.get('best_selection')):
        raise ValueError('Source best checkpoint chronology or score differs from completed summary')
    weights = checkpoint.get('model')
    if (not isinstance(weights, dict) or not weights or any(
            not isinstance(t, torch.Tensor) or (t.is_floating_point() and not torch.isfinite(t).all())
            for t in weights.values())):
        raise ValueError('Source contains invalid or non-finite model weights')
    return checkpoint, summary, metadata


def _validate_models(cnn, vit, profile, seed):
    from models.spatial_preact import build_spatial_model
    from models.vit_augreg import build_vit
    cnn_model = build_spatial_model('cnn8', seed)
    cnn_model.load_state_dict(cnn, strict=True)
    del cnn_model
    vit_model = build_vit(profile, seed)
    vit_model.load_state_dict(vit, strict=True)


def _historical_files(plans):
    files = {}
    for plan in plans:
        for name, digest in plan.get('source_files', {}).items():
            path = _regular(name).resolve()
            if file_hash(path) != digest:
                raise ValueError(f'Historical source checkpoint or metadata changed: {path}')
            key = str(path)
            if key in files and files[key] != digest:
                raise ValueError('Conflicting historical source identities')
            files[key] = digest
    return files


def load_best_sources(cnn_suite=DEFAULT_CNN_SOURCE, vit_suite=DEFAULT_VIT_SOURCE, config=None):
    cnn_source, cnn_training, cnn_plan, cnn_recorded, paths = _suite(
        cnn_suite, CNN_SUITE_PROTOCOL, CNN_TRAINING_PROTOCOL)
    vit_source, vit_training, vit_plan, vit_recorded, vit_paths = _suite(
        vit_suite, VIT_SUITE_PROTOCOL, VIT_SUITE_PROTOCOL, VIT_TRAINING_PROTOCOL)
    paths += vit_paths
    for key in ('seed', 'split_hash', 'execution_environment'):
        if cnn_plan.get(key) != vit_plan.get(key):
            raise ValueError(f'CNN and ViT sources use different {key}')
    if cnn_plan['seed'] != 42:
        raise ValueError('This best-pair experiment requires the existing seed42 sources')
    _relocation_only(cnn_plan['config'], vit_plan['config'])
    base = copy.deepcopy(vit_plan['config'])
    requested = copy.deepcopy(base if config is None else config)
    _relocation_only(base, requested)

    cnn_run = cnn_training/CNN_JOB
    cnn, cnn_summary, cnn_metadata = _validate_state(cnn_run, cnn_recorded,
        CNN_TRAINING_PROTOCOL, CNN_JOB, 'cnn_head_control')
    if (cnn_summary['completed_epochs'] != 100 or cnn_metadata.get('augmentation_epoch_offset') != 130
            or cnn_metadata.get('inherited_adaptation_epochs') != 30
            or cnn_summary.get('best_global_adaptation_epoch') != 30+cnn['epoch']):
        raise ValueError('Expected the completed CNN30+100 continuation and its selected best epoch')
    if any(not name.startswith('encoder.') for name in cnn['model']):
        raise ValueError('Expected independent CNNHeadControl weights, not a fused model')
    cnn_weights = {name.removeprefix('encoder.'): value for name, value in cnn['model'].items()}
    cnn_hash = file_hash(cnn_run/'best.pt')
    if cnn_source.name == DEFAULT_CNN_SOURCE.name and cnn_hash != CNN_CHECKPOINT_SHA256:
        raise ValueError('Known strongest CNN checkpoint bytes changed')
    if cnn_source.name == DEFAULT_CNN_SOURCE.name and cnn_summary['best_selection']['accuracy'] != .648:
        raise ValueError('Known strongest CNN score changed')

    selected_path = vit_training/'selected_vit.json'
    selected_record = _read(selected_path)
    if not selected_record.get('complete') or selected_record.get('protocol') != VIT_TRAINING_PROTOCOL:
        raise ValueError('Need the fully completed independent ViT selection')
    profile = selected_record.get('profile')
    if profile not in vit_plan['profiles'] or selected_record.get('job') != 'pretrain_'+profile:
        raise ValueError('Selected ViT profile/job is not a declared independent candidate')
    vit_run = vit_training/('pretrain_'+profile)
    if Path(selected_record.get('checkpoint', '')).resolve() != (vit_run/'best.pt').resolve():
        raise ValueError('Selected ViT checkpoint path differs from its independent candidate')
    vit, vit_summary, vit_metadata = _validate_state(vit_run, vit_recorded,
        VIT_TRAINING_PROTOCOL, 'pretrain_'+profile, profile)
    if (vit_summary['completed_epochs'] != vit_plan['vit_epochs']
            or vit_metadata.get('scope') != 'pretrain' or vit_metadata.get('external_pretraining') is not False
            or vit_metadata.get('profile') != profile):
        raise ValueError('ViT must be the completed independently trained from-scratch candidate')
    vit_hash = file_hash(vit_run/'best.pt')
    for key, value in dict(checkpoint_sha256=vit_hash, best_epoch=vit['epoch'],
                           best_source=vit['source'], best_selection=vit['selection']).items():
        if selected_record.get(key) != value:
            raise ValueError(f'Selected ViT record differs from source checkpoint: {key}')
    candidates = selected_record.get('candidates', [])
    if (len(candidates) != len(vit_plan['profiles'])
            or {item.get('profile') for item in candidates} != set(vit_plan['profiles'])):
        raise ValueError('Independent ViT selection lacks all scheduled candidates')
    for item in candidates:
        candidate_run = vit_training/('pretrain_'+item['profile'])
        candidate_summary = _read(candidate_run/'summary.json')
        if (candidate_summary.get('status') != 'complete'
                or candidate_summary.get('completed_epochs') != vit_plan['vit_epochs']
                or candidate_summary.get('best_selection') != item.get('best_selection')
                or file_hash(_regular(candidate_run/'best.pt')) != item.get('checkpoint_sha256')):
            raise ValueError('ViT candidate search was incomplete or a candidate changed')
        paths += [candidate_run/'summary.json', candidate_run/'best.pt']
    winner = max(candidates, key=lambda item: (item['best_selection']['accuracy'],
        -item['best_selection']['nll'], -item['declaration_index']))
    if winner['profile'] != profile:
        raise ValueError('Selected ViT is not the declared highest-scoring candidate')
    if vit_source.name == DEFAULT_VIT_SOURCE.name and vit_hash != VIT_CHECKPOINT_SHA256:
        raise ValueError('Known strongest independent ViT checkpoint bytes changed')

    previous_joint = None
    previous_run = vit_training/'new_joint_concat'
    if previous_run.exists() or 'new_joint_concat' in vit_plan.get('jobs', []):
        previous_state, previous_summary, previous_metadata = _validate_state(
            previous_run, vit_recorded, VIT_TRAINING_PROTOCOL, 'new_joint_concat', 'concat')
        original_cnn_hash = vit_plan.get('source_provenance', {}).get('parent_checkpoint_sha256')
        if (not original_cnn_hash
                or previous_metadata.get('parent_checkpoint_sha256') != original_cnn_hash
                or previous_metadata.get('vit_checkpoint_sha256') != vit_hash
                or previous_summary['completed_epochs'] != vit_plan.get('fusion_epochs', 30)):
            raise ValueError('Previous joint reference used a different CNN/ViT dependency or horizon')
        previous_joint = dict(state=previous_state, summary=previous_summary,
            checkpoint=str((previous_run/'best.pt').resolve()),
            checkpoint_sha256=file_hash(previous_run/'best.pt'))
        paths += [previous_run/'best.pt', previous_run/'metadata.json', previous_run/'summary.json']

    paths += [cnn_run/'metadata.json', cnn_run/'summary.json', cnn_run/'best.pt',
              vit_run/'metadata.json', vit_run/'summary.json', vit_run/'best.pt', selected_path]
    cnn_diagnostic = cnn_source/'feature_analysis/summary.json'
    if cnn_diagnostic.is_file():
        diagnostic = _read(cnn_diagnostic)
        if (diagnostic.get('status') != 'complete'
                or diagnostic.get('identity', {}).get('checkpoints', {}).get(CNN_JOB, {}).get('best.pt') != cnn_hash):
            raise ValueError('CNN checkpoint changed since its completed feature analysis')
        paths.append(cnn_diagnostic)
    vit_diagnostic = vit_source/'feature_analysis/summary.json'
    if vit_diagnostic.is_file():
        diagnostic = _read(vit_diagnostic)
        if (diagnostic.get('status') != 'complete'
                or diagnostic.get('identity', {}).get('selected_vit') != selected_record):
            raise ValueError('ViT selection changed since its completed feature analysis')
        if (previous_joint is not None and diagnostic.get('identity', {}).get('checkpoints', {})
                .get('new_joint_concat', {}).get('best.pt') != previous_joint['checkpoint_sha256']):
            raise ValueError('Previous joint checkpoint changed since its completed feature analysis')
        paths.append(vit_diagnostic)
    _validate_models(cnn_weights, vit['model'], profile, cnn_plan['seed'])
    files = _historical_files((cnn_plan, cnn_recorded, vit_plan, vit_recorded))
    files.update({str(path.resolve()): file_hash(path) for path in paths})
    checkpoint_cnn_epochs = cnn_metadata['augmentation_epoch_offset']+cnn['epoch']
    provenance = dict(protocol='tiny-best-independent-pair-sources-v1',
        cnn_suite=str(cnn_source), vit_suite=str(vit_source),
        parent_run=str(cnn_run), parent_checkpoint_sha256=cnn_hash,
        vit_checkpoint_sha256=vit_hash, vit_profile=profile,
        source_execution_environment=cnn_plan['execution_environment'],
        source_files=files, source_cnn_accuracy=cnn['selection']['accuracy'],
        source_vit_accuracy=vit['selection']['accuracy'],
        cnn_best_selection=cnn['selection'], vit_best_selection=vit['selection'],
        source_cnn_best_epoch=cnn['epoch'], source_cnn_best_source=cnn['source'],
        source_cnn_best_adaptation_epoch=cnn_summary['best_global_adaptation_epoch'],
        source_cnn_checkpoint_training_epochs=checkpoint_cnn_epochs,
        source_cnn_completed_training_epochs=cnn_metadata['augmentation_epoch_offset']+cnn_summary['completed_epochs'],
        source_vit_best_epoch=vit['epoch'], source_vit_best_source=vit['source'],
        source_vit_completed_epochs=vit_summary['completed_epochs'],
        source_cnn_stage_train_seconds=cnn_summary.get('total_train_seconds'),
        source_vit_train_seconds=vit_summary.get('total_train_seconds'),
        source_vit_search_train_seconds=sum(_read(vit_training/('pretrain_'+p)/'summary.json').get('total_train_seconds', 0.)
                                            for p in vit_plan['profiles']),
        previous_joint_info=(dict(checkpoint=previous_joint['checkpoint'],
            checkpoint_sha256=previous_joint['checkpoint_sha256'],
            best_epoch=previous_joint['state']['epoch'], best_source=previous_joint['state']['source'],
            best_selection=previous_joint['state']['selection'],
            completed_epochs=previous_joint['summary']['completed_epochs']) if previous_joint is not None else None),
        original_source_suite=cnn_plan.get('source_suite'),
        original_cnn_checkpoint_sha256=cnn_metadata.get('parent_checkpoint_sha256'),
        initialization='selected independent model weights only; fresh joint optimizer and schedule')
    return dict(cnn_suite=str(cnn_source), vit_suite=str(vit_source), source_suite=str(vit_source),
        source_training_dir=str(vit_training), parent_run=str(cnn_run), parent_model=cnn_weights,
        selected=dict(selected_record, state=vit), config=requested, seed=cnn_plan['seed'],
        split_hash=cnn_plan['split_hash'], parent_completed_epochs=checkpoint_cnn_epochs,
        cnn_summary=cnn_summary, vit_summary=vit_summary,
        previous_joint=previous_joint,
        source_files=files, provenance=provenance)
