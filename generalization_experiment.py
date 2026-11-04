"""Finite recipe, readout, depth and normalization controls for generalization.

The study distinguishes readout supervision, augmentation mismatch and a
sharpness-aware objective. Fresh jobs are compared only with their matched
fresh control; warm jobs share the original100-epoch parent and short schedule.
No audit/final-validation labels are read. This module never sends mail/stops.
"""
import copy
import gc
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch import nn
from tqdm import tqdm
import yaml

from baseline_diagnostics import class_labels, clean_datasets
from correction_experiment import file_hash
from datasets.resnet_tiny import prepare_datasets
from datasets.utility_tiny import loader
from fusion30_experiment import paired_predictions, _selection_curve_png
from literature_experiment import ROOT, environment, evaluate, new_name, recipe_hash, save_json
from main import build_scheduler
from mechanism_diagnostics import load_spatial_parent
from models.attention_correction import state_digest
from models.generalization_preact import (VARIANTS, build_generalization_model, description,
                                         initialization_family, soft_loss)
from models.spatial_preact import build_spatial_model
from resnet_experiment import atomic_save, mix_batch
from spatial_experiment import (_diagnose, _probe, _save_history, execution_environment,
                                restore_rng, rng_state, source_hashes as spatial_sources)
from utility_experiment import runtime, sync
from utils.baseline_diagnostics import evaluate_clean


PROTOCOL = 'tiny-generalization-remedies-v1'
DEFAULT_JOBS = VARIANTS
EMA_DECAY = .999
SAM_RHO = .05
NEW_SOURCES = ('generalization_experiment.py', 'models/generalization_preact.py',
               'mechanism_diagnostics.py', 'fusion30_experiment.py')


def source_hashes():
    return {**spatial_sources(), **{name: file_hash(ROOT/name) for name in NEW_SOURCES}}


def job_config(base, variant, epochs):
    result = copy.deepcopy(base)
    warm = variant.startswith('warm_')
    result['training'].update(epochs=epochs, early_stopping_patience=0, probe_per_class=20,
                              learning_rate=1e-4 if warm else 5e-4)
    result['training']['scheduler'].update(name='cosine', warmup_epochs=min(1 if warm else 2, epochs-1),
                                          min_lr=1e-6 if warm else 1e-5)
    if variant.endswith('_gentle'):
        result['augmentation'].update(crop_scale=[.85, 1.], randaugment_ops=1,
                                      randaugment_magnitude=3)
        result['training'].update(cutmix_alpha=0., mixup_alpha=.2, mix_probability=.5)
    if variant.endswith('_fp32_control') or variant.endswith('_sam'):
        result['training']['mixed_precision'] = False
    if variant == 'fresh_no_cutmix':
        result['training']['cutmix_alpha'] = 0.
    if variant == 'fresh_no_randaugment':
        result['augmentation']['randaugment_ops'] = 0
    if variant == 'fresh_wd_low':
        result['training']['weight_decay'] = .01
    if variant == 'fresh_wd_high':
        result['training']['weight_decay'] = .1
    if variant == 'fresh_sgd':
        # A reasonable SGD-family recipe, not an attempt to isolate optimizer
        # name while holding its unsuitable AdamW hyperparameters constant.
        result['training'].update(learning_rate=.1, weight_decay=5e-4)
        result['training']['scheduler'].update(warmup_epochs=min(5, epochs-1), min_lr=1e-4)
    return result


@torch.no_grad()
def update_ema(ema, model, decay=EMA_DECAY):
    if not 0 <= decay < 1:
        raise ValueError('EMA decay must be in [0,1)')
    ema_state, live_state = ema.state_dict(), model.state_dict()
    if ema_state.keys() != live_state.keys():
        raise ValueError('EMA state layout differs from model')
    for name, value in ema_state.items():
        source = live_state[name].detach()
        if value.is_floating_point():
            value.mul_(decay).add_(source, alpha=1-decay)
        else:
            value.copy_(source)


def _loss(model, images, targets):
    return model.training_loss(images, targets) if hasattr(model, 'training_loss') else soft_loss(model(images), targets)


def sam_step(model, images, targets, optimizer, *, rho=SAM_RHO, clip=5.):
    """FP32 SAM/AdamW step, with one BN-buffer update and exact restoration.

    AdamW performs the descent on second-pass gradients. No AMP GradScaler is
    involved; the identical already-mixed batch/soft targets feed both passes.
    """
    if rho <= 0:
        raise ValueError('SAM rho must be positive')
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(images.device.type, enabled=False):
        first = _loss(model, images.float(), targets.float())
    if not torch.isfinite(first):
        raise RuntimeError('Non-finite first SAM loss')
    first.backward()
    selected = [p for p in model.parameters() if p.grad is not None]
    if not selected:
        raise RuntimeError('Empty first SAM gradient')
    norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm() for p in selected]))
    if not torch.isfinite(norm):
        raise RuntimeError('Non-finite SAM gradient norm')
    backups = [(p, p.detach().clone()) for p in selected]
    bn = [(module, module.track_running_stats) for module in model.modules()
          if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    try:
        with torch.no_grad():
            for parameter, _ in backups:
                parameter.add_(parameter.grad, alpha=float(rho/(norm+1e-12)))
        # Keep batch normalization in training mode, but do not update running
        # moments or counters for a second time at the perturbed weights.
        for module, _ in bn:
            module.track_running_stats = False
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(images.device.type, enabled=False):
            second = _loss(model, images.float(), targets.float())
        if not torch.isfinite(second):
            raise RuntimeError('Non-finite perturbed SAM loss')
        second.backward()
    finally:
        with torch.no_grad():
            for parameter, original in backups:
                parameter.copy_(original)
        for module, tracking in bn:
            module.track_running_stats = tracking
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return float(first.detach()), float(second.detach())


def train_one_epoch(model, ema, batches, config, seed, data_epoch, device, amp, dtype,
                    optimizer, scaler, *, sam=False):
    model.train()
    total, perturbed, count, skipped = 0., 0., 0, 0
    for batch_index, (images, labels, _) in enumerate(tqdm(batches, desc=f'Train data epoch {data_epoch}', leave=False)):
        images, labels = images.to(device), labels.to(device)
        images, targets = mix_batch(images, labels, config['training'], seed, data_epoch, batch_index)
        if sam:
            loss, second = sam_step(model, images, targets, optimizer,
                clip=config['training']['gradient_clip_norm'])
            update_ema(ema, model)
            perturbed += second*len(labels)
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, enabled=amp, dtype=dtype):
                loss_tensor = _loss(model, images, targets)
            if not torch.isfinite(loss_tensor):
                raise RuntimeError('Non-finite training loss')
            scaler.scale(loss_tensor).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'],
                                          error_if_nonfinite=not amp)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped += 1
            else:
                update_ema(ema, model)
            loss = float(loss_tensor.detach())
        total += loss*len(labels)
        count += len(labels)
    if not count:
        raise ValueError('Empty training loader')
    return dict(train_soft_target_loss=total/count,
                sam_perturbed_loss=perturbed/count if sam else None, skipped_amp_steps=skipped)


def _save_best(run, value):
    atomic_save(run/f'best_epoch{value["epoch"]}_{value["source"]}.pt', value)
    atomic_save(run/'best.pt', value)


def _validate_checkpoint(state, metadata, config):
    if state.get('metadata') != metadata or recipe_hash(state['config']) != recipe_hash(config):
        raise ValueError('Checkpoint differs from initialized model, parent, source, recipe or environment')
    if not 0 <= state['epoch'] <= config['training']['epochs']:
        raise ValueError('Checkpoint epoch outside saved horizon')


def _recover_best(run, history, metadata, config):
    winner = max(((row['selection_accuracy'], -row['epoch'], 1, row['epoch'], 'raw') for row in history),
                 default=None)
    winner_ema = max(((row['ema_selection_accuracy'], -row['epoch'], 0, row['epoch'], 'ema') for row in history),
                     default=None)
    winner = max(winner, winner_ema)
    epoch, source = winner[-2:]
    path = run/f'best_epoch{epoch}_{source}.pt'
    if not path.is_file():
        raise ValueError('Committed best checkpoint version is missing')
    value = torch.load(path, map_location='cpu', weights_only=False)
    _validate_checkpoint(value, metadata, config)
    if value['epoch'] != epoch or value['source'] != source or value['selection']['accuracy'] != winner[0]:
        raise ValueError('Best checkpoint does not match committed history')
    atomic_save(run/'best.pt', value)
    _clean_best(run, epoch, source)
    return winner[0], epoch, source


def _clean_best(run, epoch, source):
    keep = f'best_epoch{epoch}_{source}.pt'
    for path in run.glob('best_epoch*.pt'):
        if re.fullmatch(r'best_epoch\d+_(?:raw|ema)\.pt', path.name) and path.name != keep:
            path.unlink()


def _isolated_evaluate(model, dataset, config, seed, device, dtype):
    state = rng_state(device)
    try:
        return evaluate(model, loader(dataset, config['dataset'], seed), device, False, dtype)
    finally:
        restore_rng(state, device)


def _diagnose_at(model, supplied, config, seed, run, examples, label):
    target = run/label
    if target.exists():
        target = run/new_name(label+'_retry')
    target.mkdir()
    diagnostic, scores = _diagnose(model, supplied, config, seed, target, examples)
    return diagnostic.relative_to(run).as_posix(), scores


def _train_job(plan, variant, run, parent_state, deadline=None):
    run = Path(run)
    horizon = plan['epochs'] if variant.startswith('warm_') else plan['fresh_epochs']
    config = job_config(plan['config'], variant, horizon)
    seed = plan['seed']
    device, amp, dtype, scaler = runtime(config, seed)
    supplied, manifest, fingerprint = prepare_datasets(config, seed)
    if fingerprint != plan['split_hash']:
        raise ValueError('Training data split differs from parent')
    model = build_generalization_model(variant, parent_state, seed).to(device)
    metadata = dict(protocol=PROTOCOL, variant=variant, source_hashes=plan['source_hashes'],
        parent_checkpoint_sha256=plan['parent_checkpoint_sha256'], parent_run=plan['parent_run'],
        split_hash=fingerprint, seed=seed, recipe_hash=recipe_hash(config),
        initial_state_sha256=state_digest(model), environment=environment(device, amp, dtype),
        params=sum(p.numel() for p in model.parameters()), encoder_frozen=False,
        optimizer_family='SGD-momentum-Nesterov' if variant == 'fresh_sgd' else 'AdamW',
        architecture=description(variant), ema_decay=EMA_DECAY, sam_rho=SAM_RHO if variant.endswith('_sam') else None,
        fresh_optimizer=True, parent_optimizer_resumed=False,
        augmentation_epoch_offset=plan['parent_completed_epochs'] if variant.startswith('warm_') else 0)
    completed = run/'summary.json'
    if completed.exists():
        prior = json.loads(completed.read_text(encoding='utf-8'))
        if any(prior.get(key) != value for key, value in metadata.items()):
            raise ValueError('Saved run metadata differs from the requested study')
        if prior['completed_epochs'] == horizon:
            return prior
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    groups = [dict(params=decay, weight_decay=config['training']['weight_decay']),
              dict(params=no_decay, weight_decay=0.)]
    optimizer = (torch.optim.SGD(groups, lr=config['training']['learning_rate'], momentum=.9, nesterov=True)
                 if variant == 'fresh_sgd' else
                 torch.optim.AdamW(groups, lr=config['training']['learning_rate']))
    scheduler = build_scheduler(optimizer, config['training'])
    began, elapsed = time.perf_counter(), 0.
    if (run/'last.pt').exists():
        state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        _validate_checkpoint(state, metadata, config)
        if [row['epoch'] for row in state['history']] != list(range(state['epoch']+1)):
            raise ValueError('Checkpoint history chronology is inconsistent')
        model.load_state_dict(state['model'], strict=True)
        ema.load_state_dict(state['ema_model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'], device)
        history, elapsed = state['history'], state['elapsed_wall_seconds']
        best, best_epoch, best_source = _recover_best(run, history, metadata, config)
        del state
    else:
        if run.exists():
            if not (run/'metadata.json').exists() or json.loads((run/'metadata.json').read_text()) != metadata:
                raise ValueError('Refusing an unrecognized/inconsistent unstarted directory')
            allowed = {'metadata.json', 'config.yaml', 'split_manifest.json', 'best.pt', 'best.tmp', 'last.tmp', 'epochs.csv.tmp'}
            if any(path.name not in allowed and not re.fullmatch(r'best_epoch0_(?:raw|ema)\.(?:pt|tmp)', path.name)
                   for path in run.iterdir()):
                raise ValueError('Refusing to overwrite unrelated unstarted files')
        else:
            run.mkdir(parents=True)
        save_json(run/'metadata.json', metadata)
        save_json(run/'split_manifest.json', manifest)
        (run/'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        selected = _isolated_evaluate(model, supplied['selection'], config, seed, device, dtype)
        probe = _probe(model, supplied['train_probe'], config, seed, device, dtype)
        history = [dict(epoch=0, lr=optimizer.param_groups[0]['lr'], train_seconds=0.,
            augmentation_epoch=metadata['augmentation_epoch_offset'],
            **{f'selection_{k}': v for k, v in selected.items()},
            **{f'ema_selection_{k}': v for k, v in selected.items()},
            **{f'train_probe_{k}': v for k, v in probe.items()},
            clean_accuracy_gap_pp=100*(probe['accuracy']-selected['accuracy']))]
        best, best_epoch, best_source = selected['accuracy'], 0, 'raw'
        _save_best(run, dict(model=model.state_dict(), epoch=0, source='raw', selection=selected,
                            metadata=metadata, config=config))
        atomic_save(run/'last.pt', dict(model=model.state_dict(), ema_model=ema.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            history=history, epoch=0, elapsed_wall_seconds=0., metadata=metadata, config=config,
            rng=rng_state(device)))
        _save_history(run, history)
    budget_exhausted = False
    for epoch in range(history[-1]['epoch']+1, horizon+1):
        estimated = (np.mean([r['train_seconds'] for r in history[-3:] if r['epoch']])
                     if len(history) > 1 else plan.get('estimated_epoch_seconds', 90.)*(2 if variant.endswith('_sam') else 1))
        if deadline is not None and time.monotonic()+1.1*estimated+15 >= deadline:
            budget_exhausted = True
            break
        sync(device)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        data_epoch = metadata['augmentation_epoch_offset']+epoch
        lr = optimizer.param_groups[0]['lr']
        losses = train_one_epoch(model, ema, loader(supplied['train'], config['dataset'], seed, data_epoch),
            config, seed, data_epoch, device, amp, dtype, optimizer, scaler, sam=variant.endswith('_sam'))
        sync(device)
        row = dict(epoch=epoch, augmentation_epoch=data_epoch, lr=lr,
            train_seconds=time.perf_counter()-started, **losses)
        raw_score = _isolated_evaluate(model, supplied['selection'], config, seed, device, dtype)
        ema_score = _isolated_evaluate(ema, supplied['selection'], config, seed, device, dtype)
        row.update({f'selection_{k}': v for k, v in raw_score.items()})
        row.update({f'ema_selection_{k}': v for k, v in ema_score.items()})
        if epoch == 1 or epoch % 5 == 0 or epoch == horizon:
            probe = _probe(model, supplied['train_probe'], config, seed, device, dtype)
            row.update({f'train_probe_{k}': v for k, v in probe.items()})
            row['clean_accuracy_gap_pp'] = 100*(probe['accuracy']-raw_score['accuracy'])
        for source, candidate, score in (('raw', model, raw_score), ('ema', ema, ema_score)):
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
        _save_history(run, history)
        print(f'{variant} epoch {epoch}/{horizon}: raw={100*raw_score["accuracy"]:.2f}% '
              f'EMA={100*ema_score["accuracy"]:.2f}%; best={100*best:.2f}% ({best_source}@{best_epoch})', flush=True)
    _save_history(run, history)
    # Save a final clean probe even if the budget stopped at an off-cycle epoch.
    if 'train_probe_accuracy' not in history[-1]:
        probe = _probe(model, supplied['train_probe'], config, seed, device, dtype)
        history[-1].update({f'train_probe_{k}': v for k, v in probe.items()})
        history[-1]['clean_accuracy_gap_pp'] = 100*(probe['accuracy']-history[-1]['selection_accuracy'])
        _save_history(run, history)
    model.zero_grad(set_to_none=True)
    del optimizer, scheduler, scaler, ema
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    # The last RAW model is evaluated separately from the automatically chosen
    # best. Its errors remain available when best stays at epoch zero.
    last_directory, last_scores = _diagnose_at(model, supplied, config, seed, run, plan['examples'], 'last_analysis')
    state = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    _validate_checkpoint(state, metadata, config)
    model.load_state_dict(state['model'], strict=True)
    best_directory, best_scores = _diagnose_at(model, supplied, config, seed, run, plan['examples'], 'best_analysis')
    result = dict(**metadata, status='budget_exhausted' if budget_exhausted else 'complete',
        completed_epochs=history[-1]['epoch'], planned_epochs=horizon,
        best_epoch=state['epoch'], best_source=state['source'], best_selection=state['selection'],
        clean_scores=best_scores, diagnostic_directory=best_directory,
        last_clean_scores=last_scores, last_diagnostic_directory=last_directory,
        last_epoch_selection={k: history[-1][f'selection_{k}'] for k in ('accuracy', 'top5_accuracy', 'nll')},
        last_ema_selection={k: history[-1][f'ema_selection_{k}'] for k in ('accuracy', 'top5_accuracy', 'nll')},
        initial_selection={k: history[0][f'selection_{k}'] for k in ('accuracy', 'top5_accuracy', 'nll')},
        total_train_seconds=sum(row['train_seconds'] for row in history),
        total_experiment_seconds=elapsed+time.perf_counter()-began,
        audit_partition_evaluated=False, official_validation_evaluated=False)
    save_json(run/'summary.json', result)
    return result


def _make_reference(plan, output, parent_state):
    marker = output/'parent_reference.json'
    if marker.exists():
        return json.loads(marker.read_text())
    clean, selection, _, _, _ = clean_datasets(plan['config'], plan['seed'], plan['split_hash'], 0)
    model = build_spatial_model('cnn8', plan['seed'])
    model.load_state_dict(parent_state, strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).float().eval()
    directory = output/new_name('parent_selection')
    scores = evaluate_clean(model, selection, device=device,
        batch_size=plan['config']['dataset']['batch_size'], num_workers=plan['config']['dataset']['num_workers'],
        output_dir=directory, class_names=class_labels(clean), example_count=0)
    result = dict(directory=directory.relative_to(output).as_posix(), metrics=scores,
                  parent_checkpoint_sha256=plan['parent_checkpoint_sha256'])
    save_json(marker, result)
    return result


def comparison(output, plan):
    rows, paired = [], {}
    reference = json.loads((output/'parent_reference.json').read_text())
    parent_predictions = output/reference['directory']/'predictions.npz'
    summaries = {}
    for variant in plan['jobs']:
        run = output/variant
        if not (run/'summary.json').exists():
            continue
        item = json.loads((run/'summary.json').read_text())
        summaries[variant] = item
        score, fitted = item['clean_scores']['selection'], item['clean_scores']['train_clean']
        last = item['last_clean_scores']
        rows.append(dict(variant=variant, completed_epochs=item['completed_epochs'],
            planned_epochs=item['planned_epochs'], status=item['status'], best_epoch=item['best_epoch'],
            best_source=item['best_source'], best_selection_top1_pct=100*score['accuracy'],
            best_train_top1_pct=100*fitted['accuracy'], best_gap_pp=100*(fitted['accuracy']-score['accuracy']),
            best_selection_nll=score['nll'], last_raw_top1_pct=100*last['selection']['accuracy'],
            last_train_top1_pct=100*last['train_clean']['accuracy'],
            last_gap_pp=100*(last['train_clean']['accuracy']-last['selection']['accuracy']),
            last_ema_top1_pct=100*item['last_ema_selection']['accuracy'],
            train_minutes=item['total_train_seconds']/60, params=item['params']))
        for state, directory in (('best', item['diagnostic_directory']), ('last', item['last_diagnostic_directory'])):
            predictions = run/directory/'selection/predictions.npz'
            paired[f'{variant}_{state}_vs_parent'] = paired_predictions(parent_predictions, predictions,
                output/f'paired_{variant}_{state}_vs_parent.csv')
    for scope, control in (('warm', 'warm_control'), ('fresh', 'fresh_control')):
        if control not in summaries:
            continue
        left = summaries[control]
        for variant, item in summaries.items():
            if variant == control or not variant.startswith(scope+'_'):
                continue
            paired[f'{variant}_best_vs_{control}'] = paired_predictions(
                output/control/left['diagnostic_directory']/'selection/predictions.npz',
                output/variant/item['diagnostic_directory']/'selection/predictions.npz',
                output/f'paired_{variant}_best_vs_{control}.csv')
    for scope in ('warm', 'fresh'):
        sam, control = scope+'_sam', scope+'_fp32_control'
        if sam in summaries and control in summaries:
            left, right = summaries[control], summaries[sam]
            key = f'{sam}_best_vs_{control}'
            paired[key] = paired_predictions(
                output/control/left['diagnostic_directory']/'selection/predictions.npz',
                output/sam/right['diagnostic_directory']/'selection/predictions.npz',
                output/f'paired_{key}.csv')
    if rows:
        pd.DataFrame(rows).to_csv(output/'comparison.csv', index=False)
    curve_files = []
    for scope in ('warm', 'fresh'):
        names = [name for name in plan['jobs'] if name in summaries and name.startswith(scope+'_')]
        # The inherited plotting helper uses three colors. Keep each panel to
        # three matched jobs, retaining readable legends and a zoomed axis.
        for offset in range(0, len(names), 3):
            histories = {name: pd.read_csv(output/name/'epochs.csv') for name in names[offset:offset+3]}
            for column, label in (('selection_accuracy', 'raw'), ('ema_selection_accuracy', 'ema')):
                filename = f'{scope}_{offset//3+1}_{label}_top1.png'
                _selection_curve_png(output, histories, column, filename,
                    f'{scope} selection Top1 (%), {label}; compare matched horizons')
                curve_files.append(filename)
    save_json(output/'paired_summary.json', paired)
    lines = ['# Generalization correction study', '',
        'All jobs keep seed42 and the original80K/10K development split. Warm jobs start from the same '
        '100-epoch CNN8. Parameter/buffer-matched fresh jobs use identical random initial weights and must be compared with '
        'their matched fresh control. Selection chooses raw/EMA checkpoints; '
        'this is a development screen, not a final test estimate.', '',
        '| Job | Epochs | Best source/epoch | Best selection | Last raw | Last EMA | Best gap pp |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["variant"]} | {row["completed_epochs"]}/{row["planned_epochs"]} | '
            f'{row["best_source"]}@{row["best_epoch"]} | {row["best_selection_top1_pct"]:.3f}% | '
            f'{row["last_raw_top1_pct"]:.3f}% | {row["last_ema_top1_pct"]:.3f}% | {row["best_gap_pp"]:.3f} |')
    lines += ['', '## Read the controls correctly', '',
        '- A smaller train-selection gap is useful only when selection accuracy/NLL also improve.',
        '- Best and last raw models have separate full clean80K/selection10K class/error reports. '
        'Epoch0 being selected no longer hides what the actual trained model changed.',
        '- Full8 retrains the readout; phase-balanced adds task supervision at all four grids with no extra CNN parameters. '
        'These tests differ from changing pooling only at inference.',
        '- Gentle augmentation tests semantic damage/mismatch: crop scale.85..1, RandAugment1/M3, '
        'no CutMix, otherwise retained Mixup/label smoothing/weight decay. It is a recipe bundle, not isolated-factor proof.',
        '- SAM uses two FP32 passes with the same mixed batch, rho.05, AdamW, and only one BN running-stat update. '
        'EMA decay.999 averages parameters and floating buffers; integer counters are copied. Compare warm SAM '
        'primarily to the corresponding warm/fresh FP32 control, which has the same precision and learning rate.',
        '- Shallow26 changes depth at fixed widths/stride/head. GN changes normalization while retaining '
        'convolution/linear initialization. Their full state hashes intentionally differ from full-depth BN.',
        '- The SGD arm changes optimizer-family LR/weight-decay recipe as well as optimizer. '
        'No-CutMix/no-RandAugment and low/high WD are explicit individual-factor controls.',
        '- A budget-exhausted job is partial, checkpointed, and resumable. Compare fresh arms at matched epochs '
        'if one was stopped early. The deadline is checked between epochs, not a hard wall-clock kill.',
        '- There is no promise that any regularizer improves this already-regularized model. '
        'No audit/official-validation partition was evaluated.', '',
        'See comparison.csv, paired_summary.json, each epochs.csv, and best_analysis/last_analysis.', '']
    lines += ['', '## Learning curves', '']
    lines += [f'![{name}]({name})' for name in curve_files]
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return dict(rows=rows, paired=paired)


def run_training_study(parent_run, output_dir, config=None, epochs=30, deadline=None,
                       previous_fusion=None, examples=8, smoke=False, resume=True, jobs=None,
                       fresh_epochs=100):
    if not 1 <= epochs <= 30 or not 1 <= fresh_epochs <= 100 or examples < 0:
        raise ValueError('Need warm1..30/fresh1..100 epochs and nonnegative examples')
    jobs = list(VARIANTS if jobs is None else jobs)
    if not jobs or len(set(jobs)) != len(jobs) or any(job not in VARIANTS for job in jobs):
        raise ValueError('Need unique known training jobs')
    parent, summary, state, digest = load_spatial_parent(parent_run)
    requested = copy.deepcopy(state['config'] if config is None else config)
    _, _, _, split, _ = clean_datasets(requested, summary['seed'], summary['split_hash'], 0)
    output = Path(output_dir).resolve()
    if output == parent or parent in output.parents:
        raise ValueError('Training output must be outside the original checkpoint run')
    plan = dict(protocol=PROTOCOL, config=requested, jobs=jobs, epochs=1 if smoke else epochs,
        fresh_epochs=1 if smoke else fresh_epochs,
        seed=summary['seed'], split_hash=split, parent_run=str(parent),
        parent_completed_epochs=summary['completed_epochs'], parent_checkpoint_sha256=digest,
        source_hashes=source_hashes(), execution_environment=execution_environment(requested, summary['seed']),
        smoke=bool(smoke), examples=examples, ema_decay=EMA_DECAY, sam_rho=SAM_RHO,
        estimated_epoch_seconds=summary.get('total_train_seconds', 9000)/summary['completed_epochs'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing the full training study on CPU')
    if (output/'suite.json').exists():
        if not resume or json.loads((output/'suite.json').read_text()) != plan:
            raise ValueError('Resume requires the same sources, parent, config, jobs and environment')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Refusing an unrelated nonempty study output')
        output.mkdir(parents=True, exist_ok=True)
        save_json(output/'suite.json', plan)
    _make_reference(plan, output, state['model'])
    outcomes, failures, initial = {}, {}, {}
    for variant in jobs:
        if deadline is not None and time.monotonic() >= deadline:
            break
        try:
            result = _train_job(plan, variant, output/variant, state['model'], deadline)
            outcomes[variant] = dict(status=result['status'], completed_epochs=result['completed_epochs'])
            initial[variant] = result['initial_state_sha256']
            comparison(output, plan)
            if result['status'] == 'budget_exhausted':
                break
        except Exception as exc:
            failure = dict(error_type=type(exc).__name__, traceback=traceback.format_exc())
            failures[variant] = failure
            directory = output/variant
            directory.mkdir(exist_ok=True)
            save_json(directory/'failure.json', failure)
            print(f'Generalization arm {variant} failed ({type(exc).__name__}); attempting remaining arms.', flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    for family in {initialization_family(key) for key in initial if key.startswith('fresh_')}:
        digests = {value for key, value in initial.items()
                   if key.startswith('fresh_') and initialization_family(key) == family}
        if len(digests) > 1:
            raise RuntimeError('Layout-matched fresh controls did not share exactly the same initial weights')
    if file_hash(parent/'best.pt') != digest:
        raise RuntimeError('The original parent checkpoint changed')
    budget = len(outcomes)+len(failures) < len(jobs) or any(v['status'] == 'budget_exhausted' for v in outcomes.values())
    result = dict(status='failed' if failures else 'budget_exhausted' if budget else 'complete',
        protocol=PROTOCOL, jobs=outcomes, failures=failures, skipped_jobs=[j for j in jobs if j not in outcomes and j not in failures],
        warm_epochs_per_group=plan['epochs'], fresh_epochs_per_group=plan['fresh_epochs'], parent_checkpoint_unchanged=True,
        audit_partition_evaluated=False, official_validation_evaluated=False)
    save_json(output/'completion.json', result)
    comparison(output, plan)
    if failures:
        raise RuntimeError('One or more generalization training arms failed; partial results retained')
    return result
