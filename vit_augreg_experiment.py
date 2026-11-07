"""Finite AugReg ViT training, followed by matched CNN/ViT fusion experiments.

All standalone candidates start from random weights, run the full recorded
horizon, and use only the original training/development partitions.  Selection
is explicit and immutable; the audit and official validation sets are unused.
This module has no email or VM control side effects.
"""
import copy
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import time
import traceback

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import yaml

from correction_experiment import file_hash
from datasets.resnet_tiny import prepare_datasets
from datasets.utility_tiny import loader
from fusion30_experiment import paired_predictions, _selection_curve_png
from generalization_experiment import (_loss, update_ema as _ordinary_ema,
    _recover_best, _clean_best, _isolated_evaluate, _diagnose_at)
from literature_experiment import environment, recipe_hash, save_json
from models.attention_correction import state_digest
from models.spatial_preact import build_spatial_model
from spatial_experiment import _probe, restore_rng, rng_state
from utility_experiment import runtime, sync
from vit_followup_experiment import (atomic_save, _save_history, _save_best,
    _save_trained_best, _recover_trained_best, _clean_trained_best, _cost,
    _make_reference)


PROTOCOL = 'tiny-vit-augreg-training-v1'
EMA_DECAY = .999
FUSION_JOBS = ('cnn_frozen_head', 'new_frozen_concat', 'new_joint_concat',
    'new_frozen_normalized', 'new_frozen_residual_linear',
    'new_frozen_residual', 'new_frozen_cross_patch', 'new_frozen_mlp_patch')


def load_sources(source_suite, config=None):
    from fusion_continuation_sources import load_sources as load
    return load(source_suite, config=config)


def split_job(job):
    if job.startswith('pretrain_'):
        from models.vit_augreg import PROFILES
        profile = job[len('pretrain_'):]
        if profile not in PROFILES:
            raise ValueError(f'Unknown ViT profile: {profile}')
        return 'pretrain', profile
    if job not in FUSION_JOBS:
        raise ValueError(f'Unknown AugReg job: {job}')
    return 'fusion', ('cnn_only' if job == 'cnn_frozen_head' else
                      job.removeprefix('new_frozen_').removeprefix('new_joint_'))


class FrozenCNNHeadControl(nn.Module):
    """Fixed CNN representation and BN buffers; train its original linear head."""
    variant = 'cnn_frozen_head'
    frozen_backbones = True
    analysis_modes = ('learned',)

    def __init__(self, state, seed=42):
        super().__init__()
        self.encoder = build_spatial_model('cnn8', seed)
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.requires_grad_(False)
        self.encoder.head.dense.requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.encoder.head.dense.train(mode)
        return self

    def forward(self, images):
        return self.encoder(images)

    def forward_decomposed(self, images):
        from models.vit_followup import _cnn_features, _cnn_pool
        _, deep = _cnn_features(self.encoder, images)
        features = _cnn_pool(self.encoder, deep)
        return dict(logits=self.encoder.head.dense(features),
                    features_before=features, features_after=features,
                    retrieval_maps={})

    @property
    def attention_flops_per_image(self):
        return 0


def job_config(plan, job):
    scope, profile = split_job(job)
    if scope == 'pretrain':
        result = copy.deepcopy(plan['profile_recipes'][profile])
        result['training']['epochs'] = plan['vit_epochs']
    else:
        result = copy.deepcopy(plan['config'])
        result['training'].update(epochs=plan['fusion_epochs'],
            encoder_learning_rate=5e-5, vit_learning_rate=5e-5,
            fusion_learning_rate=5e-4, learning_rate=5e-5,
            accumulation_steps=1, early_stopping_patience=0)
        result['training']['scheduler'] = dict(name='ratio-preserving-cosine',
            warmup_epochs=min(1, plan['fusion_epochs']-1),
            warmup_start_factor=.1, min_ratio=.02)
    result['training']['seed'] = plan['seed']
    if plan.get('smoke'):
        result['training']['epochs'] = 1
        result['training']['accumulation_steps'] = 1
        result['training']['scheduler']['warmup_epochs'] = 0
    return result


def build_job_model(job, sources, selected=None, seed=None):
    from models.vit_augreg import build_vit, AugRegFusion
    scope, variant = split_job(job)
    seed = sources['seed'] if seed is None else seed
    if scope == 'pretrain':
        return build_vit(variant, seed)
    if job == 'cnn_frozen_head':
        return FrozenCNNHeadControl(sources['parent_model'], seed)
    if selected is None or 'state' not in selected:
        raise ValueError('Fusion requires the completed selected ViT checkpoint')
    encoder = build_spatial_model('cnn8', seed)
    encoder.load_state_dict(sources['parent_model'], strict=True)
    vit = build_vit(selected['profile'], seed)
    vit.load_state_dict(selected['state']['model'], strict=True)
    return AugRegFusion(encoder, vit, variant=variant,
                        frozen=job != 'new_joint_concat')


def parameter_role(model, parameter, standalone=False):
    if standalone:
        return 'vit'
    if isinstance(model, FrozenCNNHeadControl):
        if any(parameter is p for p in model.encoder.head.dense.parameters()):
            return 'fusion'
    if hasattr(model, 'encoder') and any(parameter is p for p in model.encoder.parameters()):
        return 'encoder'
    if hasattr(model, 'vit') and any(parameter is p for p in model.vit.parameters()):
        return 'vit'
    return 'fusion'


def make_optimizer(model, config, standalone=False):
    groups = {(role, decay): [] for role in ('encoder', 'vit', 'fusion') for decay in (True, False)}
    seen = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            raise ValueError('Duplicate trainable parameter')
        seen.add(id(parameter))
        if not parameter.requires_grad:
            continue
        role = parameter_role(model, parameter, standalone)
        decay = parameter.ndim > 1 and not any(t in name for t in
            ('position_embedding', 'class_token', 'pos_embed', 'cls_token'))
        groups[(role, decay)].append(parameter)
    training = config['training']
    supplied = [dict(params=parameters, role=role,
        weight_decay=training['weight_decay'] if decay else 0.,
        lr=training.get(role+'_learning_rate', training['learning_rate']))
        for (role, decay), parameters in groups.items() if parameters]
    if not supplied:
        raise ValueError('No trainable parameters')
    return torch.optim.AdamW(supplied,
        betas=tuple(training.get('betas', (.9, .999))),
        eps=training.get('eps', 1e-8))


def make_scheduler(optimizer, config):
    training, schedule = config['training'], config['training']['scheduler']
    horizon, warmup = training['epochs'], schedule['warmup_epochs']
    peak = training['learning_rate']
    floor = schedule.get('min_ratio', schedule.get('min_lr', 1e-6)/peak)
    start = schedule.get('warmup_start_factor', .01)
    if horizon < 1 or not 0 <= warmup < horizon or not 0 < floor <= 1 or not 0 < start <= 1:
        raise ValueError('Invalid full-horizon warmup/cosine schedule')
    def factor(step):
        if step < warmup:
            return start+(1-start)*step/warmup
        progress = min(1., max(0., (step-warmup)/max(1, horizon-warmup)))
        return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def _frozen_state(model):
    if not getattr(model, 'frozen_backbones', False):
        return {}
    result = {name: p.detach() for name, p in model.named_parameters() if not p.requires_grad}
    result.update({name: value.detach() for name, value in model.named_buffers()})
    return result


def frozen_digest(model):
    digest = hashlib.sha256()
    state = _frozen_state(model)
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest() if state else None


@torch.no_grad()
def update_ema(ema, model):
    _ordinary_ema(ema, model, decay=EMA_DECAY)
    live, averaged = model.state_dict(), ema.state_dict()
    for name in _frozen_state(model):
        averaged[name].copy_(live[name])


def train_one_epoch(model, ema, batches, config, seed, data_epoch,
                    device, amp, dtype, optimizer, scaler):
    """Sample-weighted accumulation, including a short final microbatch/window.

    Buffer only the CPU microbatches of one accumulation window so its actual
    example count is known before backward. Each mean loss is weighted by its
    fraction of that window, avoiding both a short-window learning-rate change
    and artificially enlarged FP16 scaled gradients. Accumulation1 is exactly
    the original single-microbatch mean-loss backward operation.
    """
    from resnet_experiment import mix_batch
    accumulation = int(config['training'].get('accumulation_steps', 1))
    if accumulation < 1:
        raise ValueError('Accumulation steps must be positive')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = count = skipped = updates = 0
    number_batches = len(batches)
    iterator = iter(tqdm(batches, desc=f'Train data epoch {data_epoch}', leave=False))
    for window_start in range(0, number_batches, accumulation):
        window = list(itertools.islice(iterator, accumulation))
        window_count = sum(len(batch[1]) for batch in window)
        if not window_count:
            raise ValueError('Empty accumulation window')
        for local_index, (images, labels, _) in enumerate(window):
            batch_index = window_start+local_index
            images, labels = images.to(device), labels.to(device)
            images, targets = mix_batch(images, labels, config['training'], seed, data_epoch, batch_index)
            with torch.autocast(device, enabled=amp, dtype=dtype):
                loss = _loss(model, images, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite AugReg training loss')
            size = len(labels)
            scaler.scale(loss*(size/window_count)).backward()
            total += float(loss.detach())*size
            count += size
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
            config['training']['gradient_clip_norm'], error_if_nonfinite=not amp)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            skipped += 1
        else:
            update_ema(ema, model)
            updates += 1
        optimizer.zero_grad(set_to_none=True)
    if not count:
        raise ValueError('Empty AugReg training loader')
    return dict(train_soft_target_loss=total/count, skipped_amp_steps=skipped,
                optimizer_steps=updates, accumulation_steps=accumulation)


def _rates(optimizer):
    return {group['role']+'_lr': group['lr'] for group in optimizer.param_groups}


def _validate(state, metadata, config):
    if state.get('metadata') != metadata or recipe_hash(state['config']) != recipe_hash(config):
        raise ValueError('AugReg checkpoint source, architecture, recipe or dependency changed')
    if not 0 <= state['epoch'] <= config['training']['epochs']:
        raise ValueError('Checkpoint outside recorded training horizon')


def _failure_identity(plan, job):
    return dict(protocol=PROTOCOL, job=job, source_hashes=plan['source_hashes'],
        source_provenance=plan['source_provenance'], split_hash=plan['split_hash'],
        seed=plan['seed'], recipe_hash=recipe_hash(job_config(plan, job)))


def _owned_failure(plan, job, run):
    path = Path(run)/'failure.json'
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise ValueError('Invalid failure record')
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('identity') != _failure_identity(plan, job):
        raise ValueError('Failure record belongs to another experiment')
    return True


def _record_failure(plan, job, run, failure):
    run = Path(run)
    if run.is_symlink():
        return False
    if run.exists():
        entries = {p.name for p in run.iterdir()}
        try:
            owns = _owned_failure(plan, job, run)
        except (OSError, ValueError):
            return False
        if entries and not (owns and entries == {'failure.json'}):
            metadata_path = run/'metadata.json'
            if not metadata_path.is_file() or metadata_path.is_symlink():
                return False
            saved = json.loads(metadata_path.read_text(encoding='utf-8'))
            if any(saved.get(k) != v for k, v in _failure_identity(plan, job).items()):
                return False
    else:
        run.mkdir(parents=True)
    save_json(run/'failure.json', dict(failure, identity=_failure_identity(plan, job)))
    return True


def _clear_failure(plan, job, run):
    if _owned_failure(plan, job, run):
        (Path(run)/'failure.json').unlink()


def _metadata(plan, job, run, sources, model, config, device, amp, dtype, selected):
    from models.vit_augreg import description
    scope, variant = split_job(job)
    profile = variant if scope == 'pretrain' else (selected['profile'] if selected else None)
    identity = _failure_identity(plan, job)
    return dict(**identity, scope=scope, variant=variant, model_variant=variant,
        profile=profile, suite_directory=str(Path(run).resolve().parent),
        source_files=plan['source_files'], parent_run=plan['parent_run'],
        parent_checkpoint_sha256=sources['provenance']['parent_checkpoint_sha256'],
        vit_checkpoint_sha256=None if selected is None else selected['checkpoint_sha256'],
        environment=environment(device, amp, dtype),
        architecture=description(profile) if scope == 'pretrain' else dict(
            fusion_variant=variant, vit=description(profile) if profile else None,
            independent_branches=True, frozen_backbones=bool(getattr(model, 'frozen_backbones', False)),
            fusion=model.fusion_description() if hasattr(model, 'fusion_description') else dict(
                variant='cnn_frozen_head', original_cnn_classifier_retrained=True,
                cnn_backbone_and_bn_buffers_fixed=True)),
        initial_state_sha256=state_digest(model), params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        encoder_params=sum(p.numel() for p in model.encoder.parameters()) if hasattr(model, 'encoder') else 0,
        frozen_initial_digest=frozen_digest(model),
        encoder_frozen=bool(getattr(model, 'frozen_backbones', False)),
        augmentation_epoch_offset=0 if scope == 'pretrain' else sources['parent_completed_epochs'],
        effective_batch_size=config['dataset']['batch_size']*config['training'].get('accumulation_steps', 1),
        fresh_optimizer=True, external_pretraining=False, ema_decay=EMA_DECAY)


def _train_job(plan, job, run, sources, selected=None):
    config, seed = job_config(plan, job), plan['seed']
    horizon = config['training']['epochs']
    device, amp, dtype, scaler = runtime(config, seed)
    supplied, manifest, fingerprint = prepare_datasets(config, seed)
    if fingerprint != plan['split_hash']:
        raise ValueError('AugReg training data split changed')
    model = build_job_model(job, sources, selected, seed).to(device)
    metadata = _metadata(plan, job, run, sources, model, config, device, amp, dtype, selected)
    run = Path(run)
    if run.is_symlink():
        raise ValueError('Refusing symlink training directory')
    owns_failure = _owned_failure(plan, job, run)
    if (run/'summary.json').is_file():
        summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
        if any(summary.get(k) != value for k, value in metadata.items()):
            raise ValueError('Completed AugReg job differs from this request')
        if summary.get('status') == 'complete' and summary['completed_epochs'] == horizon:
            state = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
            _validate(state, metadata, config)
            if state['epoch'] != summary['best_epoch'] or state['selection'] != summary['best_selection']:
                raise ValueError('Completed best checkpoint differs from summary')
            _clear_failure(plan, job, run)
            return summary
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = make_optimizer(model, config, split_job(job)[0] == 'pretrain')
    scheduler = make_scheduler(optimizer, config)
    began, elapsed = time.perf_counter(), 0.
    if (run/'last.pt').is_file():
        state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        _validate(state, metadata, config)
        if [row['epoch'] for row in state['history']] != list(range(state['epoch']+1)):
            raise ValueError('Inconsistent AugReg checkpoint history')
        model.load_state_dict(state['model'], strict=True)
        ema.load_state_dict(state['ema_model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'], device)
        history, elapsed = state['history'], state['elapsed_wall_seconds']
        best, best_epoch, best_source = _recover_best(run, history, metadata, config)
        trained_best, trained_epoch, trained_source = _recover_trained_best(run, history, metadata, config)
        del state
    else:
        if run.exists():
            entries = {p.name for p in run.iterdir()}
            failure_only = owns_failure and entries == {'failure.json'}
            metadata_file = run/'metadata.json'
            if not failure_only and (not metadata_file.is_file() or json.loads(metadata_file.read_text()) != metadata):
                raise ValueError('Refusing unrelated AugReg run directory')
            allowed = {'metadata.json', 'config.yaml', 'split_manifest.json', 'best.pt', 'best.tmp',
                       'last.tmp', 'epochs.csv.tmp'}
            if owns_failure:
                allowed.add('failure.json')
            if any(name not in allowed and not re.fullmatch(r'best_epoch0_(?:raw|ema)\.(?:pt|tmp)', name) for name in entries):
                raise ValueError('Refusing unrelated files in unstarted run')
        else:
            run.mkdir(parents=True)
        save_json(run/'metadata.json', metadata)
        save_json(run/'split_manifest.json', manifest)
        (run/'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        raw = _isolated_evaluate(model, supplied['selection'], config, seed, device, dtype)
        averaged = _isolated_evaluate(ema, supplied['selection'], config, seed, device, dtype)
        probe = _probe(model, supplied['train_probe'], config, seed, device, dtype)
        history = [dict(epoch=0, augmentation_epoch=metadata['augmentation_epoch_offset'], train_seconds=0.,
            **_rates(optimizer), **{f'selection_{k}': v for k, v in raw.items()},
            **{f'ema_selection_{k}': v for k, v in averaged.items()},
            **{f'train_probe_{k}': v for k, v in probe.items()})]
        best_source, candidate, selected_score = ('ema', ema, averaged) if averaged['accuracy'] > raw['accuracy'] else ('raw', model, raw)
        best, best_epoch = selected_score['accuracy'], 0
        trained_best, trained_epoch, trained_source = -1., None, None
        _save_best(run, dict(model=candidate.state_dict(), epoch=0, source=best_source,
            selection=selected_score, metadata=metadata, config=config))
        atomic_save(run/'last.pt', dict(model=model.state_dict(), ema_model=ema.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            history=history, epoch=0, elapsed_wall_seconds=0., metadata=metadata,
            config=config, rng=rng_state(device)))
        _save_history(run, history)
    for epoch in range(history[-1]['epoch']+1, horizon+1):
        sync(device)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rates = _rates(optimizer)
        data_epoch = metadata['augmentation_epoch_offset']+epoch
        losses = train_one_epoch(model, ema,
            loader(supplied['train'], config['dataset'], seed, data_epoch),
            config, seed, data_epoch, device, amp, dtype, optimizer, scaler)
        sync(device)
        row = dict(epoch=epoch, augmentation_epoch=data_epoch,
            train_seconds=time.perf_counter()-started,
            peak_training_memory_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
            **rates, **losses)
        raw = _isolated_evaluate(model, supplied['selection'], config, seed, device, dtype)
        averaged = _isolated_evaluate(ema, supplied['selection'], config, seed, device, dtype)
        row.update({f'selection_{k}': v for k, v in raw.items()})
        row.update({f'ema_selection_{k}': v for k, v in averaged.items()})
        if epoch == 1 or epoch % 5 == 0 or epoch == horizon:
            probe = _probe(model, supplied['train_probe'], config, seed, device, dtype)
            row.update({f'train_probe_{k}': v for k, v in probe.items()})
            row['clean_accuracy_gap_pp'] = 100*(probe['accuracy']-raw['accuracy'])
        for source, candidate, score in (('raw', model, raw), ('ema', ema, averaged)):
            if score['accuracy'] > trained_best:
                trained_best, trained_epoch, trained_source = score['accuracy'], epoch, source
                _save_trained_best(run, dict(model=candidate.state_dict(), epoch=epoch, source=source,
                    selection=score, metadata=metadata, config=config))
            if score['accuracy'] > best:
                best, best_epoch, best_source = score['accuracy'], epoch, source
                _save_best(run, dict(model=candidate.state_dict(), epoch=epoch, source=source,
                    selection=score, metadata=metadata, config=config))
        history.append(row)
        scheduler.step()
        atomic_save(run/'last.pt', dict(model=model.state_dict(), ema_model=ema.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            history=history, epoch=epoch, elapsed_wall_seconds=elapsed+time.perf_counter()-began,
            metadata=metadata, config=config, rng=rng_state(device)))
        _clean_best(run, best_epoch, best_source)
        _clean_trained_best(run, trained_epoch, trained_source)
        _save_history(run, history)
        print(f'{job} {epoch}/{horizon}: raw={100*raw["accuracy"]:.2f}% '
              f'EMA={100*averaged["accuracy"]:.2f}% best={100*best:.2f}% {best_source}@{best_epoch}', flush=True)
    unchanged = frozen_digest(model) == metadata['frozen_initial_digest']
    if not unchanged:
        raise RuntimeError('Frozen encoder parameters or normalization buffers changed')
    model.zero_grad(set_to_none=True)
    del optimizer, scheduler, scaler, ema
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    last_dir, last_scores = _diagnose_at(model, supplied, config, seed, run, plan['examples'], 'last_analysis')
    state = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    _validate(state, metadata, config)
    model.load_state_dict(state['model'], strict=True)
    best_dir, best_scores = _diagnose_at(model, supplied, config, seed, run, plan['examples'], 'best_analysis')
    result = dict(**metadata, status='complete', completed_epochs=history[-1]['epoch'], planned_epochs=horizon,
        best_epoch=state['epoch'], best_source=state['source'], best_selection=state['selection'],
        initial_selection={k: history[0][f'selection_{k}'] for k in ('accuracy', 'top5_accuracy', 'nll')},
        initial_ema_selection={k: history[0][f'ema_selection_{k}'] for k in ('accuracy', 'top5_accuracy', 'nll')},
        best_trained_selection=dict(accuracy=trained_best, epoch=trained_epoch, source=trained_source),
        best_trained_checkpoint='best_trained.pt',
        clean_scores=best_scores, diagnostic_directory=best_dir,
        last_clean_scores=last_scores, last_diagnostic_directory=last_dir,
        last_epoch_selection={k: history[-1][f'selection_{k}'] for k in ('accuracy', 'top5_accuracy', 'nll')},
        last_ema_selection={k: history[-1][f'ema_selection_{k}'] for k in ('accuracy', 'top5_accuracy', 'nll')},
        total_train_seconds=sum(row['train_seconds'] for row in history),
        total_experiment_seconds=elapsed+time.perf_counter()-began,
        peak_training_memory_bytes=max(row.get('peak_training_memory_bytes', 0) for row in history),
        frozen_backbones_unchanged=unchanged, audit_partition_evaluated=False,
        official_validation_evaluated=False, **_cost(model, device))
    save_json(run/'summary.json', result)
    _clear_failure(plan, job, run)
    return result


def select_vit(plan, output):
    """Select only after every predeclared candidate finished its full horizon."""
    output = Path(output)
    candidates = []
    for index, profile in enumerate(plan['profiles']):
        job, run = 'pretrain_'+profile, output/('pretrain_'+profile)
        summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
        if summary.get('status') != 'complete' or summary.get('completed_epochs') != plan['vit_epochs']:
            raise ValueError('All predeclared ViT candidates must complete before model selection')
        state = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
        config = job_config(plan, job)
        metadata = state['metadata']
        if any(metadata.get(k) != v for k, v in _failure_identity(plan, job).items()):
            raise ValueError('Candidate provenance changed')
        _validate(state, metadata, config)
        if (state['epoch'] != summary['best_epoch'] or state['source'] != summary['best_source']
                or state['selection'] != summary['best_selection']
                or any(summary.get(k) != v for k, v in metadata.items())):
            raise ValueError('Candidate best checkpoint differs from completed summary')
        candidates.append(dict(profile=profile, job=job, declaration_index=index,
            checkpoint=str((run/'best.pt').resolve()), checkpoint_sha256=file_hash(run/'best.pt'),
            best_epoch=state['epoch'], best_source=state['source'], best_selection=state['selection']))
    winner = max(candidates, key=lambda c: (c['best_selection']['accuracy'],
        -c['best_selection']['nll'], -c['declaration_index']))
    result = dict(protocol=PROTOCOL, **{k: v for k, v in winner.items() if k != 'declaration_index'},
        complete=True, candidates=candidates,
        selection_policy='All complete; highest development Top-1, lower NLL, then declared profile order')
    path = output/'selected_vit.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != result:
        raise ValueError('Previously selected ViT dependency changed')
    save_json(path, result)
    return result


def _load_selected(plan, output):
    selected = select_vit(plan, output)
    state = torch.load(selected['checkpoint'], map_location='cpu', weights_only=False)
    if file_hash(selected['checkpoint']) != selected['checkpoint_sha256']:
        raise ValueError('Selected ViT checkpoint bytes changed')
    return dict(selected, state=state)


def load_analysis_model(job, run_dir, plan, sources, checkpoint='best.pt'):
    """Reconstruct the exact architecture and validate a best/last checkpoint."""
    run_dir = Path(run_dir)
    if checkpoint in ('best', 'last', 'best_trained'):
        checkpoint += '.pt'
    if checkpoint not in ('best.pt', 'last.pt', 'best_trained.pt'):
        raise ValueError('Unsupported analysis checkpoint')
    selected = None if split_job(job)[0] == 'pretrain' or job == 'cnn_frozen_head' else _load_selected(plan, run_dir.parent)
    model = build_job_model(job, sources, selected, plan['seed'])
    state = torch.load(run_dir/checkpoint, map_location='cpu', weights_only=False)
    config = job_config(plan, job)
    metadata = json.loads((run_dir/'metadata.json').read_text(encoding='utf-8'))
    _validate(state, metadata, config)
    if any(metadata.get(k) != v for k, v in _failure_identity(plan, job).items()):
        raise ValueError('Analysis source identity changed')
    if selected is not None and (metadata.get('profile') != selected['profile']
            or metadata.get('vit_checkpoint_sha256') != selected['checkpoint_sha256']):
        raise ValueError('Analysis selected ViT differs from training')
    model.load_state_dict(state['model'], strict=True)
    return model, state


def comparison(output, plan):
    output = Path(output)
    reference = json.loads((output/'parent_reference.json').read_text(encoding='utf-8'))
    parent_predictions = output/reference['directory']/'predictions.npz'
    rows, summaries, paired = [], {}, {}
    for job in plan['jobs']:
        run = output/job
        if not (run/'summary.json').is_file():
            continue
        item = json.loads((run/'summary.json').read_text(encoding='utf-8'))
        if item.get('status') != 'complete':
            continue
        summaries[job] = item
        clean, val = item['clean_scores']['train_clean'], item['clean_scores']['selection']
        rows.append(dict(job=job, scope=item['scope'], profile=item['profile'], variant=item['variant'],
            completed_epochs=item['completed_epochs'], planned_epochs=item['planned_epochs'],
            best_top1_pct=100*val['accuracy'], best_epoch=item['best_epoch'], best_source=item['best_source'],
            best_train_top1_pct=100*clean['accuracy'], gap_pp=100*(clean['accuracy']-val['accuracy']),
            best_nll=val['nll'], initial_top1_pct=100*item['initial_selection']['accuracy'],
            last_top1_pct=100*item['last_epoch_selection']['accuracy'],
            last_ema_top1_pct=100*item['last_ema_selection']['accuracy'],
            params=item['params'], trainable_params=item['trainable_params'],
            gflops=item['forward_flops_estimate']/1e9, train_minutes=item['total_train_seconds']/60))
        for label, directory in (('best', item['diagnostic_directory']), ('last', item['last_diagnostic_directory'])):
            name = f'{job}_{label}_vs_original_cnn'
            paired[name] = paired_predictions(parent_predictions,
                run/directory/'selection/predictions.npz', output/f'paired_{name}.csv')
    for candidate, reference_job in (
            ('new_frozen_concat', 'cnn_frozen_head'),
            ('new_joint_concat', 'new_frozen_concat'),
            ('new_frozen_normalized', 'new_frozen_concat'),
            ('new_frozen_residual', 'new_frozen_residual_linear'),
            ('new_frozen_cross_patch', 'new_frozen_mlp_patch')):
        if candidate in summaries and reference_job in summaries:
            a, b = summaries[candidate], summaries[reference_job]
            name = candidate+'_vs_'+reference_job
            paired[name] = paired_predictions(
                output/reference_job/b['diagnostic_directory']/'selection/predictions.npz',
                output/candidate/a['diagnostic_directory']/'selection/predictions.npz',
                output/f'paired_{name}.csv')
    if rows:
        pd.DataFrame(rows).to_csv(output/'comparison.csv', index=False)
    save_json(output/'paired_summary.json', paired)
    figures = []
    for job in summaries:
        histories = {job: pd.read_csv(output/job/'epochs.csv')}
        for column, label in (('selection_accuracy', 'raw'), ('ema_selection_accuracy', 'ema')):
            name = f'{job}_{label}_top1.png'
            _selection_curve_png(output, histories, column, name, f'AugReg study: {job} {label} development Top-1 (%)')
            figures.append(name)
    lines = ['# AugReg ViT and matched CNN/ViT fusion study', '',
        'All standalone candidates start from scratch on the same80K training images and run '
        'their complete predeclared schedules. Profile selection uses development Top-1, lower NLL '
        'and declaration order. This is a paper-informed Tiny ImageNet adaptation, not an exact '
        'reproduction of ImageNet training or evidence of method novelty.', '',
        '|Job|Epochs|Best Top-1|Best checkpoint|Last raw|Last EMA|Trainable params|Training minutes|',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'|{row["job"]}|{row["completed_epochs"]}/{row["planned_epochs"]}|'
            f'{row["best_top1_pct"]:.2f}%|{row["best_source"]}@{row["best_epoch"]}|'
            f'{row["last_top1_pct"]:.2f}%|{row["last_ema_top1_pct"]:.2f}%|'
            f'{row["trainable_params"]}|{row["train_minutes"]:.2f}|')
    lines += ['', 'All downstream models use the same original CNN checkpoint, unchanged old '
        'fusion augmentation and30-epoch schedule. Frozen encoders also keep BN buffers and '
        'dropout behavior fixed. Independent ViT training costs and candidate-search costs are '
        'additional to fusion costs. Audit and official validation are not evaluated.', '']
    lines += [f'![{name}]({name})' for name in figures]
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return dict(rows=rows, paired=paired)


def run_augreg_study(plan, output_dir, examples=8, resume=True):
    plan = copy.deepcopy(plan)
    plan['examples'] = examples
    if examples < 0 or plan['vit_epochs'] < 1 or plan['fusion_epochs'] < 1:
        raise ValueError('Positive training horizons and nonnegative galleries required')
    expected_jobs = ['pretrain_'+profile for profile in plan['profiles']]+list(FUSION_JOBS)
    if plan['jobs'] != expected_jobs or len(set(expected_jobs)) != len(expected_jobs):
        raise ValueError('AugReg queue must contain every candidate followed by the matched fusion jobs')
    for job in expected_jobs:
        split_job(job)
    sources = load_sources(plan['source_suite'], config=plan['config'])
    if sources['provenance'] != plan['source_provenance'] or sources['source_files'] != plan['source_files']:
        raise ValueError('Original source metadata or checkpoint identity changed')
    if sources['split_hash'] != plan['split_hash'] or sources['seed'] != plan['seed']:
        raise ValueError('Original seed or data partition changed')
    output = Path(output_dir).resolve()
    for protected in (Path(sources['source_suite']).resolve(), Path(sources['parent_run']).resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('New outputs must be separate from immutable source runs')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the formal AugReg study')
    if (output/'suite.json').is_file():
        if not resume or json.loads((output/'suite.json').read_text(encoding='utf-8')) != plan:
            raise ValueError('Resume requires unchanged source, recipes, profiles, horizons and data')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Refusing unrelated nonempty training output')
        output.mkdir(parents=True, exist_ok=True)
        save_json(output/'suite.json', plan)
    reference_plan = dict(plan, parent_checkpoint_sha256=sources['provenance']['parent_checkpoint_sha256'])
    _make_reference(reference_plan, output, sources['parent_model'])
    outcomes, failures, skipped = {}, {}, []
    for job in expected_jobs:
        selected = None
        try:
            if split_job(job)[0] != 'pretrain' and job != 'cnn_frozen_head':
                selected = _load_selected(plan, output)
            result = _train_job(plan, job, output/job, sources, selected)
            outcomes[job] = dict(status=result['status'], completed_epochs=result['completed_epochs'])
        except Exception as exc:
            failure = dict(error_type=type(exc).__name__, traceback=traceback.format_exc())
            failures[job] = failure
            failure['job_record_written'] = _record_failure(plan, job, output/job, failure)
            print(f'AugReg job {job} failed ({type(exc).__name__}); independent jobs will continue.', flush=True)
        finally:
            del selected
            gc.collect()
            torch.cuda.empty_cache()
    immutable = all(Path(path).is_file() and file_hash(path) == digest for path, digest in plan['source_files'].items())
    if not immutable:
        failures['immutable_sources'] = dict(error_type='SourceMutation', message='An immutable source file changed')
    try:
        comparison(output, plan)
    except Exception as exc:
        failures['comparison'] = dict(error_type=type(exc).__name__, traceback=traceback.format_exc())
    result = dict(protocol=PROTOCOL, status='failed' if failures else 'complete', jobs=outcomes,
        failures=failures, skipped_jobs=skipped, source_files_unchanged=immutable,
        candidate_search_model_epochs=len(plan['profiles'])*plan['vit_epochs'],
        fusion_model_epochs=len(FUSION_JOBS)*plan['fusion_epochs'],
        selected_vit=json.loads((output/'selected_vit.json').read_text(encoding='utf-8')) if (output/'selected_vit.json').is_file() else None,
        audit_partition_evaluated=False, official_validation_evaluated=False)
    save_json(output/'completion.json', result)
    if failures:
        raise RuntimeError('AugReg study has incomplete jobs; all available results retained')
    return result
