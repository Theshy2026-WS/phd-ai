"""Controlled CNN8/attention feature-fusion training and comparable reports.

Warm-start screens use a10x learning-rate ratio for new fusion parameters.
Fresh controls test selected mechanisms before memorization. Both best and
actual last checkpoints are retained and evaluated. No mail or VM control.
"""
import copy
import gc
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

from baseline_diagnostics import clean_datasets
from correction_experiment import file_hash
from datasets.resnet_tiny import prepare_datasets
from datasets.utility_tiny import loader
from fusion30_experiment import paired_predictions, _selection_curve_png
from generalization_experiment import (update_ema, train_one_epoch, _save_best,
    _recover_best, _clean_best, _isolated_evaluate, _diagnose_at)
from literature_experiment import ROOT, environment, new_name, recipe_hash, save_json
from mechanism_diagnostics import load_spatial_parent, _relocation_only
from models.attention_correction import state_digest
from models.attention_fusion import build_attention_model, description
from models.spatial_preact import build_spatial_model
from resnet_experiment import atomic_save
from spatial_experiment import (_probe, _save_history, execution_environment,
    restore_rng, rng_state, source_hashes as spatial_sources)
from utility_experiment import runtime, sync
from utils.baseline_diagnostics import evaluate_clean
from utils.literature_profiling import estimate_flops


PROTOCOL = 'tiny-attention-feature-fusion-v1'
WARM_VARIANTS = ('control', 'conv_cross', 'cross16', 'self8', 'conv16',
                 'self16', 'cross32', 'bidir16', 'gated_cross16')
FRESH_VARIANTS = ('control', 'conv_cross', 'self8', 'cross16')
DEFAULT_JOBS = tuple('warm_'+name for name in WARM_VARIANTS)+tuple('fresh_'+name for name in FRESH_VARIANTS)
EMA_DECAY = .999
NEW_SOURCES = ('attention_fusion_experiment.py', 'models/attention_fusion.py',
               'generalization_experiment.py', 'models/generalization_preact.py',
               'fusion30_experiment.py', 'mechanism_diagnostics.py')


def source_hashes():
    return {**spatial_sources(), **{name: file_hash(ROOT/name) for name in NEW_SOURCES}}


def split_job(job):
    if job not in DEFAULT_JOBS:
        raise ValueError(f'Unknown attention training job: {job}')
    return tuple(job.split('_', 1))


def job_config(base, job, epochs):
    scope, _ = split_job(job)
    result = copy.deepcopy(base)
    warm = scope == 'warm'
    encoder_lr = 5e-5 if warm else 5e-4
    result['training'].update(epochs=epochs, early_stopping_patience=0,
        probe_per_class=20, learning_rate=encoder_lr, encoder_learning_rate=encoder_lr,
        fusion_learning_rate=5e-4)
    result['training']['scheduler'] = dict(name='ratio-preserving-cosine',
        warmup_epochs=min(1 if warm else 2, epochs-1), min_ratio=.02, warmup_start_factor=.1)
    return result


def make_optimizer(model, config):
    if not hasattr(model, 'encoder'):
        raise ValueError('Attention models must expose their shared CNN encoder')
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    groups = {(role, decay): [] for role in ('encoder', 'fusion') for decay in (True, False)}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            raise ValueError('The feature-fusion study requires all model parameters trainable')
        groups[('encoder' if id(parameter) in encoder_ids else 'fusion', parameter.ndim > 1)].append(parameter)
    supplied = []
    for (role, decay), parameters in groups.items():
        if parameters:
            supplied.append(dict(params=parameters, role=role,
                weight_decay=config['training']['weight_decay'] if decay else 0.,
                lr=config['training'][role+'_learning_rate']))
    return torch.optim.AdamW(supplied)


def make_scheduler(optimizer, config):
    training = config['training']
    schedule = training['scheduler']
    horizon, warmup = training['epochs'], schedule['warmup_epochs']
    floor, start = schedule['min_ratio'], schedule['warmup_start_factor']
    if horizon < 1 or not 0 <= warmup < horizon or not 0 < floor <= 1:
        raise ValueError('Invalid ratio-preserving schedule')
    def factor(step):
        if step < warmup:
            return start+(1-start)*step/warmup
        progress = min(1., max(0., (step-warmup)/max(1, horizon-warmup)))
        return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def _rates(optimizer):
    values = {}
    for group in optimizer.param_groups:
        key = group['role']+'_lr'
        if key in values and values[key] != group['lr']:
            raise ValueError('Decay/no-decay groups must share their role learning rate')
        values[key] = group['lr']
    return values


def _fusion_state(model):
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    new_parameters = [(name, p.detach().float()) for name, p in model.named_parameters()
                      if id(p) not in encoder_ids]
    norm = sum(float(value.square().sum()) for _, value in new_parameters)**.5
    gammas = {name: value.cpu().reshape(-1).tolist() for name, value in new_parameters
              if 'gamma' in name and value.numel() <= 16}
    return dict(fusion_parameter_l2=norm, fusion_gamma_json=json.dumps(gammas, sort_keys=True))


def _cost(model, device):
    estimate = estimate_flops(model, device)
    extra = getattr(model, 'attention_flops_per_image', 0)
    extra = extra() if callable(extra) else extra
    return dict(forward_flops_estimate=int(estimate+extra),
        attention_matmul_flops=int(extra),
        flops_note='Conv/Linear plus explicitly declared QK/AV arithmetic; normalization/softmax/elementwise work excluded')


def _validate(state, metadata, config):
    if state.get('metadata') != metadata or recipe_hash(state['config']) != recipe_hash(config):
        raise ValueError('Attention checkpoint differs from source, parent, split, initialization, recipe or environment')
    if not 0 <= state['epoch'] <= config['training']['epochs']:
        raise ValueError('Attention checkpoint epoch outside the recorded horizon')


def _trained_winner(history):
    candidates = [(row[f'{prefix}selection_accuracy'], -row['epoch'], source == 'raw',
                   row['epoch'], source)
                  for row in history if row['epoch'] > 0
                  for prefix, source in (('', 'raw'), ('ema_', 'ema'))]
    return max(candidates) if candidates else None


def _save_trained_best(run, value):
    atomic_save(run/f'trained_best_epoch{value["epoch"]}_{value["source"]}.pt', value)
    atomic_save(run/'best_trained.pt', value)


def _clean_trained_best(run, epoch, source):
    import re
    keep = f'trained_best_epoch{epoch}_{source}.pt' if epoch is not None else None
    for path in run.glob('trained_best_epoch*.pt'):
        if re.fullmatch(r'trained_best_epoch\d+_(?:raw|ema)\.pt', path.name) and path.name != keep:
            path.unlink()
    if epoch is None and (run/'best_trained.pt').exists():
        (run/'best_trained.pt').unlink()


def _recover_trained_best(run, history, metadata, config):
    winner = _trained_winner(history)
    if winner is None:
        _clean_trained_best(run, None, None)
        return -1., None, None
    score, _, _, epoch, source = winner
    path = run/f'trained_best_epoch{epoch}_{source}.pt'
    if not path.is_file():
        raise ValueError('Committed best trained checkpoint version is missing')
    value = torch.load(path, map_location='cpu', weights_only=False)
    _validate(value, metadata, config)
    if value['epoch'] != epoch or value['source'] != source or value['selection']['accuracy'] != score:
        raise ValueError('Best trained checkpoint differs from committed history')
    atomic_save(run/'best_trained.pt', value)
    _clean_trained_best(run, epoch, source)
    return score, epoch, source


def _train_job(plan, job, run, parent_state, deadline=None):
    """Inner job supports CPU fixtures; the full public study requires CUDA."""
    scope, variant = split_job(job)
    horizon = plan['warm_epochs'] if scope == 'warm' else plan['fresh_epochs']
    config, seed = job_config(plan['config'], job, horizon), plan['seed']
    device, amp, dtype, scaler = runtime(config, seed)
    supplied, manifest, fingerprint = prepare_datasets(config, seed)
    if fingerprint != plan['split_hash']:
        raise ValueError('Attention study split differs from the reference')
    model = build_attention_model(variant, parent_state if scope == 'warm' else None, seed=seed).to(device)
    metadata = dict(protocol=PROTOCOL, job=job, scope=scope, variant=variant, model_variant=variant,
        seed=seed, split_hash=fingerprint, parent_run=plan['parent_run'],
        parent_checkpoint_sha256=plan['parent_checkpoint_sha256'], source_hashes=plan['source_hashes'],
        recipe_hash=recipe_hash(config), environment=environment(device, amp, dtype),
        initial_state_sha256=state_digest(model), encoder_initial_state_sha256=state_digest(model.encoder),
        architecture=description(variant), params=sum(p.numel() for p in model.parameters()),
        encoder_params=sum(p.numel() for p in model.encoder.parameters()), encoder_frozen=False,
        fresh_optimizer=True, parent_optimizer_resumed=False, ema_decay=EMA_DECAY,
        augmentation_epoch_offset=plan['parent_completed_epochs'] if scope == 'warm' else 0,
        encoder_peak_lr=config['training']['encoder_learning_rate'],
        fusion_peak_lr=config['training']['fusion_learning_rate'],
        initial_fusion=_fusion_state(model))
    run = Path(run)
    if (run/'summary.json').exists():
        saved = json.loads((run/'summary.json').read_text(encoding='utf-8'))
        if any(saved.get(key) != value for key, value in metadata.items()):
            raise ValueError('Completed attention job differs from this initialized plan')
        if saved['completed_epochs'] == horizon:
            return saved
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    began, elapsed = time.perf_counter(), 0.
    if (run/'last.pt').exists():
        state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        _validate(state, metadata, config)
        if [row['epoch'] for row in state['history']] != list(range(state['epoch']+1)):
            raise ValueError('Attention checkpoint history chronology is inconsistent')
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
            if not (run/'metadata.json').exists() or json.loads((run/'metadata.json').read_text()) != metadata:
                raise ValueError('Refusing an unrelated or inconsistent attention job directory')
            allowed = {'metadata.json', 'config.yaml', 'split_manifest.json', 'best.pt',
                       'best.tmp', 'last.tmp', 'epochs.csv.tmp'}
            import re
            if any(p.name not in allowed and not re.fullmatch(r'best_epoch0_(?:raw|ema)\.(?:pt|tmp)', p.name)
                   for p in run.iterdir()):
                raise ValueError('Refusing to overwrite unrelated unstarted attention files')
        else:
            run.mkdir(parents=True)
        save_json(run/'metadata.json', metadata)
        save_json(run/'split_manifest.json', manifest)
        (run/'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        selected = _isolated_evaluate(model, supplied['selection'], config, seed, device, dtype)
        probe = _probe(model, supplied['train_probe'], config, seed, device, dtype)
        history = [dict(epoch=0, train_seconds=0., augmentation_epoch=metadata['augmentation_epoch_offset'],
            **_rates(optimizer), **_fusion_state(model),
            **{f'selection_{k}': v for k, v in selected.items()},
            **{f'ema_selection_{k}': v for k, v in selected.items()},
            **{f'train_probe_{k}': v for k, v in probe.items()},
            clean_accuracy_gap_pp=100*(probe['accuracy']-selected['accuracy']))]
        best, best_epoch, best_source = selected['accuracy'], 0, 'raw'
        trained_best, trained_epoch, trained_source = -1., None, None
        _save_best(run, dict(model=model.state_dict(), epoch=0, source='raw', selection=selected,
                            metadata=metadata, config=config))
        atomic_save(run/'last.pt', dict(model=model.state_dict(), ema_model=ema.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            history=history, epoch=0, elapsed_wall_seconds=0., metadata=metadata, config=config,
            rng=rng_state(device)))
        _save_history(run, history)
    partial = False
    for epoch in range(history[-1]['epoch']+1, horizon+1):
        estimate = (np.mean([row['train_seconds'] for row in history[-3:] if row['epoch']])
                    if len(history) > 1 else plan.get('estimated_epoch_seconds', 100.))
        if deadline is not None and time.monotonic()+1.1*estimate+15 >= deadline:
            partial = True
            break
        sync(device)
        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rates = _rates(optimizer)
        data_epoch = metadata['augmentation_epoch_offset']+epoch
        losses = train_one_epoch(model, ema, loader(supplied['train'], config['dataset'], seed, data_epoch),
            config, seed, data_epoch, device, amp, dtype, optimizer, scaler)
        sync(device)
        row = dict(epoch=epoch, augmentation_epoch=data_epoch, train_seconds=time.perf_counter()-started,
            peak_training_memory_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
            **rates, **losses, **_fusion_state(model))
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
        print(f'{job} epoch {epoch}/{horizon}: raw={100*raw["accuracy"]:.2f}% '
              f'EMA={100*averaged["accuracy"]:.2f}%; best={100*best:.2f}% ({best_source}@{best_epoch})', flush=True)
    _save_history(run, history)
    if 'train_probe_accuracy' not in history[-1]:
        probe = _probe(model, supplied['train_probe'], config, seed, device, dtype)
        history[-1].update({f'train_probe_{k}': v for k, v in probe.items()})
        history[-1]['clean_accuracy_gap_pp'] = 100*(probe['accuracy']-history[-1]['selection_accuracy'])
        _save_history(run, history)
    final_fusion = _fusion_state(model)
    model.zero_grad(set_to_none=True)
    del optimizer, scheduler, scaler, ema
    gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
    last_dir, last_scores = _diagnose_at(model, supplied, config, seed, run, plan['examples'], 'last_analysis')
    state = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    _validate(state, metadata, config)
    model.load_state_dict(state['model'], strict=True)
    best_dir, best_scores = _diagnose_at(model, supplied, config, seed, run, plan['examples'], 'best_analysis')
    candidates = [(row[f'{prefix}selection_accuracy'], -row['epoch'], source, row)
        for row in history[1:] for prefix, source in (('', 'raw'), ('ema_', 'ema'))]
    trained = max(candidates, key=lambda value: (value[0], value[1], value[2] == 'raw')) if candidates else None
    result = dict(**metadata, status='budget_exhausted' if partial else 'complete',
        completed_epochs=history[-1]['epoch'], planned_epochs=horizon,
        best_epoch=state['epoch'], best_source=state['source'], best_selection=state['selection'],
        clean_scores=best_scores, diagnostic_directory=best_dir,
        last_clean_scores=last_scores, last_diagnostic_directory=last_dir,
        last_epoch_selection={k: history[-1][f'selection_{k}'] for k in ('accuracy','top5_accuracy','nll')},
        last_ema_selection={k: history[-1][f'ema_selection_{k}'] for k in ('accuracy','top5_accuracy','nll')},
        initial_selection={k: history[0][f'selection_{k}'] for k in ('accuracy','top5_accuracy','nll')},
        best_trained_selection=None if trained is None else dict(accuracy=trained[0], epoch=trained[3]['epoch'],
            source=trained[2], nll=trained[3][('ema_' if trained[2]=='ema' else '')+'selection_nll'],
            top5_accuracy=trained[3][('ema_' if trained[2]=='ema' else '')+'selection_top5_accuracy']),
        best_trained_checkpoint='best_trained.pt' if trained is not None else None,
        final_fusion=final_fusion, total_train_seconds=sum(row['train_seconds'] for row in history),
        total_experiment_seconds=elapsed+time.perf_counter()-began,
        peak_training_memory_bytes=max(row.get('peak_training_memory_bytes',0) for row in history),
        audit_partition_evaluated=False, official_validation_evaluated=False,
        **_cost(model, device))
    save_json(run/'summary.json', result)
    return result


def _make_reference(plan, output, parent_state):
    marker = output/'parent_reference.json'
    if marker.exists(): return json.loads(marker.read_text())
    from baseline_diagnostics import class_labels
    clean, selected, _, _, _ = clean_datasets(plan['config'], plan['seed'], plan['split_hash'], 0)
    model = build_spatial_model('cnn8', plan['seed'])
    model.load_state_dict(parent_state, strict=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).float().eval()
    directory = output/new_name('parent_selection')
    scores = evaluate_clean(model, selected, device=device,
        batch_size=plan['config']['dataset']['batch_size'], num_workers=plan['config']['dataset']['num_workers'],
        output_dir=directory, class_names=class_labels(clean), example_count=0)
    result = dict(directory=directory.relative_to(output).as_posix(), metrics=scores,
                  parent_checkpoint_sha256=plan['parent_checkpoint_sha256'])
    save_json(marker, result)
    return result


def comparison(output, plan):
    output = Path(output)
    reference = json.loads((output/'parent_reference.json').read_text())
    parent_predictions = output/reference['directory']/'predictions.npz'
    rows, summaries, paired = [], {}, {}
    for job in plan['jobs']:
        run = output/job
        if not (run/'summary.json').exists(): continue
        item = json.loads((run/'summary.json').read_text())
        summaries[job] = item
        clean, val = item['clean_scores']['train_clean'], item['clean_scores']['selection']
        last = item['last_clean_scores']
        trained = item['best_trained_selection']
        rows.append(dict(job=job, scope=item['scope'], variant=item['variant'],
            completed_epochs=item['completed_epochs'], planned_epochs=item['planned_epochs'], status=item['status'],
            initial_top1_pct=100*item['initial_selection']['accuracy'],
            best_top1_pct=100*val['accuracy'], best_epoch=item['best_epoch'], best_source=item['best_source'],
            best_post_training_top1_pct=None if trained is None else 100*trained['accuracy'],
            best_post_training_epoch=None if trained is None else trained['epoch'],
            best_train_top1_pct=100*clean['accuracy'], gap_pp=100*(clean['accuracy']-val['accuracy']),
            best_nll=val['nll'], last_top1_pct=100*last['selection']['accuracy'],
            last_train_top1_pct=100*last['train_clean']['accuracy'],
            last_ema_top1_pct=100*item['last_ema_selection']['accuracy'], params=item['params'],
            encoder_params=item['encoder_params'], gflops=item['forward_flops_estimate']/1e9,
            train_minutes=item['total_train_seconds']/60))
        for label, directory in (('best', item['diagnostic_directory']), ('last', item['last_diagnostic_directory'])):
            paired[f'{job}_{label}_vs_parent'] = paired_predictions(parent_predictions,
                run/directory/'selection/predictions.npz', output/f'paired_{job}_{label}_vs_parent.csv')
    for scope in ('warm','fresh'):
        for control_variant in ('control','conv_cross','conv16'):
            control = scope+'_'+control_variant
            if control not in summaries: continue
            left = summaries[control]
            for job, item in summaries.items():
                if item['scope'] != scope or job == control: continue
                if control_variant == 'conv16' and item['variant'] != 'self16': continue
                name = f'{job}_best_vs_{control}'
                paired[name] = paired_predictions(output/control/left['diagnostic_directory']/'selection/predictions.npz',
                    output/job/item['diagnostic_directory']/'selection/predictions.npz', output/f'paired_{name}.csv')
    if rows: pd.DataFrame(rows).to_csv(output/'comparison.csv', index=False)
    save_json(output/'paired_summary.json', paired)
    figures = []
    for scope in ('warm','fresh'):
        jobs = [job for job in plan['jobs'] if job in summaries and job.startswith(scope+'_')]
        for offset in range(0,len(jobs),3):
            histories = {job:pd.read_csv(output/job/'epochs.csv') for job in jobs[offset:offset+3]}
            for column,label in (('selection_accuracy','raw'),('ema_selection_accuracy','ema')):
                filename = f'{scope}_{offset//3+1}_{label}_top1.png'
                _selection_curve_png(output,histories,column,filename,
                    f'Attention fusion: {scope} {label} selection Top1 (%)')
                figures.append(filename)
    lines = ['# CNN8 attention feature-fusion study', '',
        'The reference is the original CNN8. All variants keep its phase00 classifier. '
        'Warm jobs share the same trained parent; fresh jobs share the same random encoder. '
        'CNN and fusion parameters are jointly optimized. Fusion modifies features, not class logits.', '',
        '| Job | Epochs | Initial Top1 | Best Top1 | Best source/epoch | Post-training best | Last raw | Params |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for row in rows:
        post = '-' if row['best_post_training_top1_pct'] is None else f'{row["best_post_training_top1_pct"]:.3f}%'
        lines.append(f'|{row["job"]}|{row["completed_epochs"]}/{row["planned_epochs"]}|'
            f'{row["initial_top1_pct"]:.3f}%|{row["best_top1_pct"]:.3f}%|'
            f'{row["best_source"]}@{row["best_epoch"]}|{post}|{row["last_top1_pct"]:.3f}%|{row["params"]}|')
    lines += ['', '## Interpretation', '',
        '- Initial gamma is nonzero, so an attention epoch0 model need not equal the parent. '
        'Report initialization and best post-training scores separately; an epoch0 selection is not a learned improvement.',
        '- Warm encoder peak LR5e-5 and new fusion peak LR5e-4 retain a10x ratio throughout warmup/cosine. '
        'Fresh groups use5e-4 for both roles. This addresses the previous uniformly slow new-branch adaptation.',
        '- Conv_cross is the static/multiscale capacity control for late attention fusion; Conv16 controls early '
        'self16 insertion. Parameters/FLOPs are reported, not assumed exactly equal.',
        '- Compare warm only with warm and fresh only with fresh. Best and actual last models have separate '
        'full clean80K/selection10K error reports. Audit/official validation remains unused.',
        '- Raw/EMA model selection uses the development selection set. Attention maps and post-training branch '
        'removal are supporting diagnostics, not proof of a causal explanation or final-test performance.', '',
        'See comparison.csv, paired_summary.json, each run/epochs.csv, best_analysis and last_analysis.', '',
        '## Curves', '']
    lines += [f'![{name}]({name})' for name in figures]
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return dict(rows=rows,paired=paired)


def run_attention_study(parent_run, output_dir, config=None, warm_epochs=30, fresh_epochs=100,
                        examples=8, smoke=False, resume=True, deadline=None, jobs=None):
    if not 1 <= warm_epochs <= 30 or not 1 <= fresh_epochs <= 100 or examples < 0:
        raise ValueError('Need warm1..30/fresh1..100 epochs and nonnegative examples')
    jobs = list(DEFAULT_JOBS if jobs is None else jobs)
    if not jobs or len(set(jobs)) != len(jobs): raise ValueError('Need nonempty unique jobs')
    for job in jobs: split_job(job)
    parent, summary, state, digest = load_spatial_parent(parent_run)
    requested = copy.deepcopy(state['config'] if config is None else config)
    _relocation_only(state['config'],requested)
    _,_,_,fingerprint,_ = clean_datasets(requested,summary['seed'],summary['split_hash'],0)
    output = Path(output_dir).resolve()
    if output == parent or parent in output.parents:
        raise ValueError('Attention output must be outside the original run')
    plan = dict(protocol=PROTOCOL,config=requested,jobs=jobs,warm_epochs=1 if smoke else warm_epochs,
        fresh_epochs=1 if smoke else fresh_epochs,examples=examples,smoke=bool(smoke),seed=summary['seed'],
        split_hash=fingerprint,parent_run=str(parent),parent_checkpoint_sha256=digest,
        parent_completed_epochs=summary['completed_epochs'],source_hashes=source_hashes(),
        execution_environment=execution_environment(requested,summary['seed']),
        estimated_epoch_seconds=summary.get('total_train_seconds',10000)/summary['completed_epochs'])
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for the full attention study')
    if (output/'suite.json').exists():
        if not resume or json.loads((output/'suite.json').read_text()) != plan:
            raise ValueError('Attention resume requires identical source, parent, jobs, horizons and environment')
    else:
        if output.exists() and any(output.iterdir()): raise ValueError('Refusing unrelated nonempty output')
        output.mkdir(parents=True,exist_ok=True);save_json(output/'suite.json',plan)
    _make_reference(plan,output,state['model'])
    outcomes,failures,initial = {},{},{}
    for job in jobs:
        if deadline is not None and time.monotonic() >= deadline: break
        try:
            result = _train_job(plan,job,output/job,state['model'],deadline)
            outcomes[job] = dict(status=result['status'],completed_epochs=result['completed_epochs'])
            initial[job] = result['encoder_initial_state_sha256']
            comparison(output,plan)
            if result['status'] == 'budget_exhausted': break
        except Exception as exc:
            failure = dict(error_type=type(exc).__name__,traceback=traceback.format_exc())
            failures[job] = failure
            (output/job).mkdir(exist_ok=True);save_json(output/job/'failure.json',failure)
            print(f'Attention job {job} failed ({type(exc).__name__}); attempting remaining jobs.',flush=True)
        gc.collect();torch.cuda.empty_cache()
    for scope in ('warm','fresh'):
        if len({value for job,value in initial.items() if job.startswith(scope+'_')}) > 1:
            raise RuntimeError(f'{scope} attention jobs did not share identical encoder initialization')
    if file_hash(parent/'best.pt') != digest: raise RuntimeError('Original parent checkpoint changed')
    partial = len(outcomes)+len(failures)<len(jobs) or any(v['status']=='budget_exhausted' for v in outcomes.values())
    result = dict(status='failed' if failures else 'budget_exhausted' if partial else 'complete',
        protocol=PROTOCOL,jobs=outcomes,failures=failures,
        skipped_jobs=[job for job in jobs if job not in outcomes and job not in failures],
        parent_checkpoint_unchanged=True,audit_partition_evaluated=False,official_validation_evaluated=False)
    save_json(output/'completion.json',result);comparison(output,plan)
    if failures: raise RuntimeError('One or more attention jobs failed; partial results retained')
    return result
