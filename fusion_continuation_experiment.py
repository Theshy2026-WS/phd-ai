"""Continue the completed CNN/ViT fusion and run inexpensive matched controls.

Old runs are immutable inputs. Continuation retains AdamW moments, EMA, scaler
and RNG, while starting an explicitly recorded new learning-rate phase. This
module never sends mail, stops a VM, or runs native64 global attention.
"""
import copy
import gc
import json
import math
from pathlib import Path
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
from generalization_experiment import (_loss, update_ema as _update_ema,
    _recover_best, _clean_best, _isolated_evaluate, _diagnose_at)
from literature_experiment import ROOT, environment, recipe_hash, save_json
from models.attention_correction import state_digest
from models.spatial_preact import build_spatial_model
from models.vit_followup import build_followup_model, DualFeatureFusion, _cnn_features, _cnn_pool
from resnet_experiment import mix_batch
from spatial_experiment import _probe, execution_environment, restore_rng, rng_state
from utility_experiment import runtime, sync
from vit_followup_experiment import (_cost, _make_reference, _save_history, atomic_save,
    _save_best, _save_trained_best, _clean_trained_best, _recover_trained_best,
    source_hashes as followup_sources)


PROTOCOL = 'tiny-fusion-continuation-training-v1'
DEFAULT_JOBS = ('dual_continue', 'cnn_head_control', 'cnn_head_continue', 'dual_frozen_head')
CONTINUED_JOBS = {'dual_continue', 'cnn_head_continue'}
EMA_DECAY = .999


def load_sources(source_suite, config=None):
    # Lazy import lets the validated loader and CPU transaction fixtures evolve
    # independently without importing a training module from the loader.
    from fusion_continuation_sources import load_sources as supplied_loader
    return supplied_loader(source_suite, config=config)


def source_hashes():
    names = ('fusion_continuation_experiment.py', 'fusion_continuation_sources.py',
             'fusion_continuation_diagnostics.py')
    return {**followup_sources(), **{name: file_hash(ROOT/name) for name in names}}


def split_job(job):
    if job not in DEFAULT_JOBS:
        raise ValueError(f'Unknown continuation job: {job}')
    return ('continuation' if job in CONTINUED_JOBS else 'control',
            'cnn_head_control' if job.startswith('cnn_') else 'dual_fusion')


class CNNHeadControl(nn.Module):
    """Unmodified CNN function, with its classifier assigned the head LR role."""
    variant = 'cnn_head_control'
    analysis_modes = ('learned',)

    def __init__(self, state=None, seed=42):
        super().__init__()
        self.encoder = build_spatial_model('cnn8', seed)
        if state is not None:
            self.encoder.load_state_dict(state, strict=True)
        self.analysis_mode = 'learned'

    def forward(self, images):
        return self.encoder(images)

    def forward_decomposed(self, images):
        _, deep = _cnn_features(self.encoder, images)
        features = _cnn_pool(self.encoder, deep)
        logits = self.encoder.head.dense(features)
        return dict(logits=logits, features_before=features, features_after=features,
                    retrieval_maps={})

    @property
    def attention_flops_per_image(self):
        return 0


class FrozenDualFeatureFusion(DualFeatureFusion):
    """Same state layout/function as dual fusion; only its classifier trains."""
    frozen_backbones = True

    def __init__(self, model):
        nn.Module.__init__(self)
        self.encoder, self.vit = model.encoder, model.vit
        self.fusion_classifier = model.fusion_classifier
        self.analysis_mode = 'learned'
        self.encoder.requires_grad_(False)
        self.vit.requires_grad_(False)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.vit.eval()
        return self


def build_job_model(job, sources, seed=None):
    _, variant = split_job(job)
    seed = sources['seed'] if seed is None else seed
    if variant == 'cnn_head_control':
        return CNNHeadControl(sources['parent_model'], seed)
    model = build_followup_model('dual_fusion', parent_state=sources['parent_model'],
        vit_state=sources['vit_model'], seed=seed)
    if job == 'dual_continue':
        model.load_state_dict(sources['dual_last']['model'], strict=True)
    elif job == 'dual_frozen_head':
        model = FrozenDualFeatureFusion(model)
    return model


def parameter_role(model, parameter):
    if hasattr(model, 'fusion_classifier'):
        head = model.fusion_classifier
    else:
        head = model.encoder.head.dense
    if any(parameter is p for p in head.parameters()):
        return 'fusion'
    if any(parameter is p for p in model.encoder.parameters()):
        return 'encoder'
    if hasattr(model, 'vit') and any(parameter is p for p in model.vit.parameters()):
        return 'vit'
    raise ValueError('Parameter is outside the declared backbone/head roles')


def job_config(base, job, epochs):
    split_job(job)
    result = copy.deepcopy(base)
    result['training'].update(epochs=epochs, early_stopping_patience=0, probe_per_class=20,
        learning_rate=5e-5, encoder_learning_rate=5e-5,
        vit_learning_rate=5e-5, fusion_learning_rate=5e-4)
    continued = job in CONTINUED_JOBS
    result['training']['scheduler'] = dict(name='explicit-continuation-cosine' if continued else 'head-control-cosine',
        warmup_epochs=min(2 if continued else 1, epochs-1), min_ratio=.02,
        warmup_start_factor=.02 if continued else .1)
    return result


def make_optimizer(model, config):
    groups = {(role, decay): [] for role in ('encoder', 'vit', 'fusion') for decay in (True, False)}
    seen = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            raise ValueError('Duplicated parameter')
        seen.add(id(parameter))
        if not parameter.requires_grad:
            if not getattr(model, 'frozen_backbones', False):
                raise ValueError('Unexpected frozen parameter in a jointly trained model')
            continue
        decay = parameter.ndim > 1 and not any(part in name for part in ('position_embedding', 'class_token'))
        groups[(parameter_role(model, parameter), decay)].append(parameter)
    supplied = [dict(params=params, role=role,
        weight_decay=config['training']['weight_decay'] if decay else 0.,
        lr=config['training'][role+'_learning_rate'])
        for (role, decay), params in groups.items() if params]
    if not supplied:
        raise ValueError('No trainable parameters')
    return torch.optim.AdamW(supplied)


def make_scheduler(optimizer, config):
    training, schedule = config['training'], config['training']['scheduler']
    horizon, warmup = training['epochs'], schedule['warmup_epochs']
    floor, start = schedule['min_ratio'], schedule['warmup_start_factor']
    if horizon < 1 or not 0 <= warmup < horizon or not 0 < floor <= 1:
        raise ValueError('Invalid continuation schedule')
    def factor(step):
        if step < warmup:
            return start+(1-start)*step/warmup
        progress = min(1., max(0., (step-warmup)/max(1, horizon-warmup)))
        return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def restore_inherited_optimizer(optimizer, checkpoint, config):
    """Preserve AdamW history only after matching roles/order/tensor shapes."""
    saved = checkpoint['optimizer']
    if len(saved['param_groups']) != len(optimizer.param_groups):
        raise ValueError('Inherited optimizer group count changed')
    for current, prior in zip(optimizer.param_groups, saved['param_groups']):
        if (prior.get('role') != current['role'] or len(prior['params']) != len(current['params'])
                or prior['weight_decay'] != current['weight_decay']):
            raise ValueError('Inherited optimizer role/order/decay changed')
        for parameter, identifier in zip(current['params'], prior['params']):
            state = saved['state'].get(identifier, {})
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in state and state[key].shape != parameter.shape:
                    raise ValueError('Inherited optimizer tensor shape changed')
    # On CPU, optimizer casting can retain tensor storage from the supplied
    # state dictionary. Never let a training step alter an in-memory source.
    optimizer.load_state_dict(copy.deepcopy(saved))
    for group in optimizer.param_groups:
        peak = config['training'][group['role']+'_learning_rate']
        group['lr'] = peak
        group['initial_lr'] = peak


def update_ema(ema, model):
    _update_ema(ema, model, decay=EMA_DECAY)
    if getattr(model, 'frozen_backbones', False):
        # Avoid even rounding drift from EMA arithmetic on fixed representations.
        with torch.no_grad():
            ema.encoder.load_state_dict(model.encoder.state_dict(), strict=True)
            ema.vit.load_state_dict(model.vit.state_dict(), strict=True)


def train_one_epoch(model, ema, batches, config, seed, data_epoch, device, amp, dtype, optimizer, scaler):
    model.train()
    total = count = skipped = 0
    for batch_index, (images, labels, _) in enumerate(tqdm(batches, desc=f'Train data epoch {data_epoch}', leave=False)):
        images, labels = images.to(device), labels.to(device)
        images, targets = mix_batch(images, labels, config['training'], seed, data_epoch, batch_index)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device, enabled=amp, dtype=dtype):
            loss = _loss(model, images, targets)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'], error_if_nonfinite=not amp)
        previous = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous:
            skipped += 1
        else:
            update_ema(ema, model)
        total += float(loss.detach())*len(labels)
        count += len(labels)
    if not count:
        raise ValueError('Empty training loader')
    return dict(train_soft_target_loss=total/count, skipped_amp_steps=skipped)


def _rates(optimizer):
    values = {}
    for group in optimizer.param_groups:
        key = group['role']+'_lr'
        if key in values and values[key] != group['lr']:
            raise ValueError('Decay/no-decay role learning rates differ')
        values[key] = group['lr']
    return values


def _frozen_digests(model):
    return ({'encoder': state_digest(model.encoder), 'vit': state_digest(model.vit)}
            if getattr(model, 'frozen_backbones', False) else None)


def _validate(state, metadata, config):
    if state.get('metadata') != metadata or recipe_hash(state['config']) != recipe_hash(config):
        raise ValueError('Continuation checkpoint source/initialization/recipe changed')
    if not 0 <= state['epoch'] <= config['training']['epochs']:
        raise ValueError('Continuation checkpoint is outside its saved phase')


def _failure_identity(plan, job, run):
    horizon = plan['continue_epochs'] if job in CONTINUED_JOBS else plan['control_epochs']
    return dict(protocol=PROTOCOL, job=job, suite_directory=str(Path(run).resolve().parent),
        source_hashes=plan['source_hashes'], source_provenance=plan['source_provenance'],
        split_hash=plan['split_hash'], seed=plan['seed'],
        recipe_hash=recipe_hash(job_config(plan['config'], job, horizon)))


def _owned_failure(plan, job, run):
    path = Path(run)/'failure.json'
    if not path.exists(): return False
    if path.is_symlink() or not path.is_file():
        raise ValueError('Invalid failure record')
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('identity') != _failure_identity(plan, job, run):
        raise ValueError('Failure belongs to another experiment')
    return True


def _record_failure(plan, job, run, failure):
    run = Path(run)
    if run.is_symlink(): return False
    if run.exists():
        try:
            owned = _owned_failure(plan, job, run)
        except (ValueError, OSError):
            return False
        entries = {p.name for p in run.iterdir()}
        if entries and not (owned and entries == {'failure.json'}):
            path = run/'metadata.json'
            if not path.is_file() or path.is_symlink(): return False
            saved = json.loads(path.read_text(encoding='utf-8'))
            expected = _failure_identity(plan, job, run)
            if any(saved.get(key) != value for key, value in expected.items()): return False
    else:
        run.mkdir(parents=True)
    save_json(run/'failure.json', dict(failure, identity=_failure_identity(plan, job, run)))
    return True


def _clear_failure(plan, job, run):
    if _owned_failure(plan, job, run): (Path(run)/'failure.json').unlink()


def _load_cnn_dependency(plan, output):
    run = Path(output)/'cnn_head_control'
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'complete' or summary['completed_epochs'] != plan['control_epochs']:
        raise ValueError('CNN continuation requires its completed head-LR control')
    state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
    if (state['epoch'] != plan['control_epochs'] or state['metadata'] !=
            {key: summary[key] for key in state['metadata']}):
        raise ValueError('CNN dependency checkpoint differs from its summary')
    if (state['metadata']['source_provenance'] != plan['source_provenance']
            or state['metadata']['source_hashes'] != plan['source_hashes']
            or state['metadata']['split_hash'] != plan['split_hash']
            or recipe_hash(state['config']) != recipe_hash(job_config(plan['config'], 'cnn_head_control', plan['control_epochs']))):
        raise ValueError('CNN dependency provenance differs')
    return state, file_hash(run/'last.pt')


def _train_job(plan, job, run, sources, inherited=None, inherited_hash=None):
    scope, variant = split_job(job)
    horizon = plan['continue_epochs'] if job in CONTINUED_JOBS else plan['control_epochs']
    config, seed = job_config(plan['config'], job, horizon), plan['seed']
    device, amp, dtype, scaler = runtime(config, seed)
    supplied, manifest, fingerprint = prepare_datasets(config, seed)
    if fingerprint != plan['split_hash']:
        raise ValueError('Training data split differs from source')
    model = build_job_model(job, sources, seed).to(device)
    if inherited is not None:
        model.load_state_dict(inherited['model'], strict=True)
    inherited_epochs = (sources['dual_completed_epochs'] if job == 'dual_continue'
                        else plan['control_epochs'] if job == 'cnn_head_continue' else 0)
    offset = sources['parent_completed_epochs']+inherited_epochs
    metadata = dict(protocol=PROTOCOL, job=job, scope=scope, variant=variant, model_variant=variant,
        suite_directory=str(Path(run).resolve().parent),
        seed=seed, split_hash=fingerprint, source_hashes=plan['source_hashes'],
        source_provenance=plan['source_provenance'], source_files=plan['source_files'],
        recipe_hash=recipe_hash(config), environment=environment(device, amp, dtype),
        parent_run=plan['parent_run'], parent_checkpoint_sha256=plan['parent_checkpoint_sha256'],
        inherited_checkpoint_sha256=inherited_hash, inherited_adaptation_epochs=inherited_epochs,
        inherited_joint_epochs=inherited_epochs, augmentation_epoch_offset=offset,
        optimizer_restored_from_source=inherited is not None,
        ema_restored_from_source=inherited is not None,
        scaler_restored_from_source=inherited is not None,
        rng_restored_from_source=inherited is not None,
        source_scheduler_reused=False, source_scheduler_state=None if inherited is None else inherited.get('scheduler'),
        new_schedule=config['training']['scheduler'],
        initial_state_sha256=state_digest(model), params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        encoder_params=sum(p.numel() for p in model.encoder.parameters()),
        frozen_backbone_initial_digests=_frozen_digests(model),
        encoder_frozen=bool(getattr(model, 'frozen_backbones', False)),
        ema_decay=EMA_DECAY, encoder_peak_lr=5e-5, vit_peak_lr=5e-5, fusion_peak_lr=5e-4)
    run = Path(run)
    owns_failure = _owned_failure(plan, job, run)
    if (run/'summary.json').exists():
        summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
        if any(summary.get(k) != v for k, v in metadata.items()):
            raise ValueError('Completed job differs from this request')
        if summary['completed_epochs'] == horizon:
            _clear_failure(plan, job, run)
            return summary
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = make_optimizer(model, config)
    began, elapsed = time.perf_counter(), 0.
    if (run/'last.pt').exists():
        # The local phase checkpoint supersedes inherited state on resume.
        scheduler = make_scheduler(optimizer, config)
        state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        _validate(state, metadata, config)
        if [r['epoch'] for r in state['history']] != list(range(state['epoch']+1)):
            raise ValueError('Inconsistent continuation history')
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
            path = run/'metadata.json'
            if not failure_only and (not path.is_file() or json.loads(path.read_text()) != metadata):
                raise ValueError('Refusing unrelated run directory')
            allowed = {'metadata.json','config.yaml','split_manifest.json','best.pt','best.tmp','last.tmp','epochs.csv.tmp'}
            if owns_failure: allowed.add('failure.json')
            import re
            if any(name not in allowed and not re.fullmatch(r'best_epoch0_(?:raw|ema)\.(?:pt|tmp)',name) for name in entries):
                raise ValueError('Refusing unrelated unstarted files')
        else:
            run.mkdir(parents=True)
        if inherited is not None:
            restore_inherited_optimizer(optimizer, inherited, config)
            ema.load_state_dict(inherited['ema_model'], strict=True)
            scaler.load_state_dict(inherited['scaler'])
            restore_rng(inherited['rng'], device)
        scheduler = make_scheduler(optimizer, config)
        save_json(run/'metadata.json', metadata)
        save_json(run/'split_manifest.json', manifest)
        (run/'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        raw = _isolated_evaluate(model, supplied['selection'], config, seed, device, dtype)
        averaged = _isolated_evaluate(ema, supplied['selection'], config, seed, device, dtype)
        probe = _probe(model, supplied['train_probe'], config, seed, device, dtype)
        history = [dict(epoch=0, global_adaptation_epoch=inherited_epochs, augmentation_epoch=offset,
            train_seconds=0., **_rates(optimizer),
            **{f'selection_{k}':v for k,v in raw.items()},
            **{f'ema_selection_{k}':v for k,v in averaged.items()},
            **{f'train_probe_{k}':v for k,v in probe.items()})]
        best_source, candidate, selected = ('ema', ema, averaged) if averaged['accuracy'] > raw['accuracy'] else ('raw', model, raw)
        best, best_epoch = selected['accuracy'], 0
        trained_best, trained_epoch, trained_source = -1., None, None
        _save_best(run, dict(model=candidate.state_dict(), epoch=0, source=best_source,
                            selection=selected, metadata=metadata, config=config))
        atomic_save(run/'last.pt', dict(model=model.state_dict(), ema_model=ema.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            history=history, epoch=0, elapsed_wall_seconds=0., metadata=metadata, config=config, rng=rng_state(device)))
        _save_history(run, history)
    for epoch in range(history[-1]['epoch']+1, horizon+1):
        sync(device)
        if device == 'cuda': torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rates = _rates(optimizer)
        data_epoch = offset+epoch
        losses = train_one_epoch(model, ema, loader(supplied['train'], config['dataset'], seed, data_epoch),
            config, seed, data_epoch, device, amp, dtype, optimizer, scaler)
        sync(device)
        row = dict(epoch=epoch, global_adaptation_epoch=inherited_epochs+epoch,
            augmentation_epoch=data_epoch, train_seconds=time.perf_counter()-started,
            peak_training_memory_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
            **rates, **losses)
        raw = _isolated_evaluate(model, supplied['selection'], config, seed, device, dtype)
        averaged = _isolated_evaluate(ema, supplied['selection'], config, seed, device, dtype)
        row.update({f'selection_{k}':v for k,v in raw.items()})
        row.update({f'ema_selection_{k}':v for k,v in averaged.items()})
        if epoch == 1 or epoch % 5 == 0 or epoch == horizon:
            probe = _probe(model, supplied['train_probe'], config, seed, device, dtype)
            row.update({f'train_probe_{k}':v for k,v in probe.items()})
            row['clean_accuracy_gap_pp'] = 100*(probe['accuracy']-raw['accuracy'])
        for source, candidate, score in (('raw',model,raw),('ema',ema,averaged)):
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
        _clean_best(run,best_epoch,best_source)
        _clean_trained_best(run,trained_epoch,trained_source)
        _save_history(run,history)
        print(f'{job} phase {epoch}/{horizon} (adaptation {inherited_epochs+epoch}): '
              f'raw={100*raw["accuracy"]:.2f}% EMA={100*averaged["accuracy"]:.2f}% '
              f'best={100*best:.2f}% {best_source}@{best_epoch}',flush=True)
    frozen_unchanged = _frozen_digests(model) == metadata['frozen_backbone_initial_digests']
    if not frozen_unchanged: raise RuntimeError('Frozen backbone parameters or buffers changed')
    model.zero_grad(set_to_none=True)
    del optimizer,scheduler,scaler,ema
    gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
    last_dir,last_scores = _diagnose_at(model,supplied,config,seed,run,plan['examples'],'last_analysis')
    state = torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    _validate(state,metadata,config)
    model.load_state_dict(state['model'],strict=True)
    best_dir,best_scores = _diagnose_at(model,supplied,config,seed,run,plan['examples'],'best_analysis')
    trained_rows = [(row[f'{prefix}selection_accuracy'],-row['epoch'],source,row)
        for row in history[1:] for prefix,source in (('','raw'),('ema_','ema'))]
    trained = max(trained_rows,key=lambda v:(v[0],v[1],v[2]=='raw')) if trained_rows else None
    result = dict(**metadata,status='complete',completed_epochs=history[-1]['epoch'],planned_epochs=horizon,
        total_adaptation_epochs=inherited_epochs+history[-1]['epoch'],
        best_epoch=state['epoch'],best_global_adaptation_epoch=inherited_epochs+state['epoch'],
        best_source=state['source'],best_selection=state['selection'],
        initial_selection={k:history[0][f'selection_{k}'] for k in ('accuracy','top5_accuracy','nll')},
        initial_ema_selection={k:history[0][f'ema_selection_{k}'] for k in ('accuracy','top5_accuracy','nll')},
        best_trained_selection=None if trained is None else dict(accuracy=trained[0],epoch=trained[3]['epoch'],
            global_adaptation_epoch=inherited_epochs+trained[3]['epoch'],source=trained[2]),
        best_trained_checkpoint='best_trained.pt' if trained is not None else None,
        clean_scores=best_scores,diagnostic_directory=best_dir,last_clean_scores=last_scores,last_diagnostic_directory=last_dir,
        last_epoch_selection={k:history[-1][f'selection_{k}'] for k in ('accuracy','top5_accuracy','nll')},
        last_ema_selection={k:history[-1][f'ema_selection_{k}'] for k in ('accuracy','top5_accuracy','nll')},
        total_train_seconds=sum(row['train_seconds'] for row in history),
        total_experiment_seconds=elapsed+time.perf_counter()-began,
        peak_training_memory_bytes=max(row.get('peak_training_memory_bytes',0) for row in history),
        frozen_backbones_unchanged=frozen_unchanged,
        audit_partition_evaluated=False,official_validation_evaluated=False,**_cost(model,device))
    save_json(run/'summary.json',result)
    _clear_failure(plan,job,run)
    return result


def comparison(output,plan):
    output = Path(output)
    reference = json.loads((output/'parent_reference.json').read_text(encoding='utf-8'))
    parent_predictions = output/reference['directory']/'predictions.npz'
    summaries,rows,paired = {},[],{}
    for job in plan['jobs']:
        run = output/job
        if not (run/'summary.json').is_file(): continue
        item = json.loads((run/'summary.json').read_text(encoding='utf-8'))
        summaries[job] = item
        clean,val = item['clean_scores']['train_clean'],item['clean_scores']['selection']
        rows.append(dict(job=job,completed_epochs=item['completed_epochs'],planned_epochs=item['planned_epochs'],
            inherited_adaptation_epochs=item['inherited_adaptation_epochs'],total_adaptation_epochs=item['total_adaptation_epochs'],
            initial_top1_pct=100*item['initial_selection']['accuracy'],initial_ema_top1_pct=100*item['initial_ema_selection']['accuracy'],
            best_top1_pct=100*val['accuracy'],best_epoch=item['best_epoch'],
            best_global_adaptation_epoch=item['best_global_adaptation_epoch'],best_source=item['best_source'],
            best_train_top1_pct=100*clean['accuracy'],gap_pp=100*(clean['accuracy']-val['accuracy']),
            best_nll=val['nll'],last_top1_pct=100*item['last_epoch_selection']['accuracy'],
            last_ema_top1_pct=100*item['last_ema_selection']['accuracy'],
            params=item['params'],trainable_params=item['trainable_params'],gflops=item['forward_flops_estimate']/1e9,
            train_minutes=item['total_train_seconds']/60))
        for label,directory in (('best',item['diagnostic_directory']),('last',item['last_diagnostic_directory'])):
            name=f'{job}_{label}_vs_original_cnn'
            paired[name]=paired_predictions(parent_predictions,run/directory/'selection/predictions.npz',output/f'paired_{name}.csv')
    if rows: pd.DataFrame(rows).to_csv(output/'comparison.csv',index=False)
    if 'dual_continue' in summaries and 'cnn_head_continue' in summaries:
        a,b=summaries['dual_continue'],summaries['cnn_head_continue']
        paired['continued_dual_best_vs_matched_continued_cnn']=paired_predictions(
            output/'cnn_head_continue'/b['diagnostic_directory']/'selection/predictions.npz',
            output/'dual_continue'/a['diagnostic_directory']/'selection/predictions.npz',
            output/'paired_continued_dual_best_vs_matched_continued_cnn.csv')
    save_json(output/'paired_summary.json',paired)
    figures=[]
    for job in summaries:
        histories={job:pd.read_csv(output/job/'epochs.csv')}
        for column,label in (('selection_accuracy','raw'),('ema_selection_accuracy','ema')):
            name=f'{job}_{label}_top1.png'
            _selection_curve_png(output,histories,column,name,f'Fusion continuation {job}: {label} phase Top1 (%)')
            figures.append(name)
    lines=['# Fusion continuation and matched head-LR controls','',
        'The original30-epoch fusion is immutable. Its model/AdamW/EMA/scaler/RNG initialize '
        'a new100-epoch learning-rate phase, giving130 total joint adaptation epochs. '
        'CNN controls match classifier learning rates and the30+100 phase structure. '
        'Frozen-head fusion starts from the two original independent pretrained encoders.','',
        '| Job | New epochs | Prior adaptation | Best | Best phase/global | Last raw | Last EMA |',
        '|---|---:|---:|---:|---|---:|---:|']
    for row in rows:
        lines.append(f'|{row["job"]}|{row["completed_epochs"]}/{row["planned_epochs"]}|{row["inherited_adaptation_epochs"]}|'
            f'{row["best_top1_pct"]:.3f}%|{row["best_epoch"]}/{row["best_global_adaptation_epoch"]} '
            f'{row["best_source"]}|{row["last_top1_pct"]:.3f}%|{row["last_ema_top1_pct"]:.3f}%|')
    lines += ['', 'This is a staged30+100 protocol, not a fresh130-epoch cosine schedule. '
        'Best phase0 is retained as a valid inherited candidate, not called new learning. '
        'All scores use development selection; audit and official validation are not evaluated.', '']
    lines += [f'![{name}]({name})' for name in figures]
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return dict(rows=rows,paired=paired)


def run_continuation_study(source_suite,output_dir,config=None,continue_epochs=100,control_epochs=30,
                           examples=8,smoke=False,resume=True,jobs=None):
    if not 1<=continue_epochs<=100 or not 1<=control_epochs<=30 or examples<0:
        raise ValueError('Need continuation1..100/control1..30 and nonnegative examples')
    jobs=list(DEFAULT_JOBS if jobs is None else jobs)
    if not jobs or len(set(jobs))!=len(jobs): raise ValueError('Need unique nonempty jobs')
    for job in jobs: split_job(job)
    if 'cnn_head_continue' in jobs and ('cnn_head_control' not in jobs or jobs.index('cnn_head_control')>jobs.index('cnn_head_continue')):
        raise ValueError('CNN continuation requires its control phase first')
    sources=load_sources(source_suite,config=config)
    output=Path(output_dir).resolve()
    source_path=Path(sources['source_suite']).resolve()
    if output==source_path or source_path in output.parents or output in source_path.parents:
        raise ValueError('Continuation outputs must be separate from the source suite')
    parent_hash=sources['source_files'].get(str(Path(sources['parent_run'])/'best.pt'))
    if parent_hash is None:
        parent_hash=sources['provenance'].get('parent_checkpoint_sha256')
    plan=dict(protocol=PROTOCOL,config=copy.deepcopy(sources['config']),jobs=jobs,
        continue_epochs=1 if smoke else continue_epochs,control_epochs=1 if smoke else control_epochs,
        smoke=bool(smoke),examples=examples,seed=sources['seed'],split_hash=sources['split_hash'],
        parent_run=sources['parent_run'],parent_checkpoint_sha256=parent_hash,
        parent_completed_epochs=sources['parent_completed_epochs'],dual_completed_epochs=sources['dual_completed_epochs'],
        source_suite=sources['source_suite'],source_training_dir=sources['source_training_dir'],
        source_provenance=sources['provenance'],source_files=sources['source_files'],
        source_hashes=source_hashes(),execution_environment=execution_environment(sources['config'],sources['seed']))
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for the full continuation study')
    if (output/'suite.json').is_file():
        if not resume or json.loads((output/'suite.json').read_text(encoding='utf-8'))!=plan:
            raise ValueError('Continuation resume requires identical source, data, environment and phase schedule')
    else:
        if output.exists() and any(output.iterdir()): raise ValueError('Refusing unrelated nonempty output')
        output.mkdir(parents=True,exist_ok=True)
        save_json(output/'suite.json',plan)
    _make_reference(plan,output,sources['parent_model'])
    outcomes,failures={},{}
    for job in jobs:
        try:
            inherited,inherited_hash=(sources['dual_last'],sources['provenance'].get('dual_last_checkpoint_sha256')) if job=='dual_continue' else (
                _load_cnn_dependency(plan,output) if job=='cnn_head_continue' else (None,None))
            if job=='dual_continue' and inherited_hash is None:
                inherited_hash=sources['source_files'].get(str(Path(sources['source_training_dir'])/'dual_fusion/last.pt'))
            result=_train_job(plan,job,output/job,sources,inherited,inherited_hash)
            outcomes[job]=dict(status=result['status'],completed_epochs=result['completed_epochs'])
            comparison(output,plan)
        except Exception as exc:
            failure=dict(error_type=type(exc).__name__,traceback=traceback.format_exc())
            failures[job]=failure
            failure['job_record_written']=_record_failure(plan,job,output/job,failure)
            print(f'Continuation job {job} failed ({type(exc).__name__}); attempting independent jobs.',flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    if any(file_hash(Path(path))!=digest for path,digest in plan['source_files'].items()):
        raise RuntimeError('An immutable source checkpoint or report changed')
    result=dict(protocol=PROTOCOL,status='failed' if failures else 'complete',jobs=outcomes,failures=failures,
        skipped_jobs=[job for job in jobs if job not in outcomes and job not in failures],source_files_unchanged=True,
        audit_partition_evaluated=False,official_validation_evaluated=False)
    save_json(output/'completion.json',result)
    comparison(output,plan)
    if failures: raise RuntimeError('One or more continuation jobs failed; independent results retained')
    return result
