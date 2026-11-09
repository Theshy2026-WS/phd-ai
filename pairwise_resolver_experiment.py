"""Four matched, frozen-anchor candidate resolvers; no mail or VM control.

Only a small new resolver is trained.  Clean frozen features are cached once;
augmented training images still pass through the same immutable full anchor.
Interrupted epochs are replayed from the last completed epoch on resume.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import signal
import time
import traceback

import pandas as pd
import torch
import torch.nn.functional as F

from best_pair_sources import DEFAULT_CNN_SOURCE, DEFAULT_VIT_SOURCE, load_best_sources
from correction_experiment import file_hash
from datasets.resnet_tiny import prepare_datasets, mix_batch
from datasets.utility_tiny import loader
from literature_experiment import ROOT, new_name, recipe_hash, save_json as _save_json
from models.attention_correction import state_digest
from spatial_experiment import execution_environment, rng_state, restore_rng
from utility_experiment import runtime, sync
from vit_followup_experiment import atomic_save


PROTOCOL = 'tiny-pairwise-resolver-suite-v1'
JOBS = ('pairwise', 'static_query', 'mlp', 'mean_tokens')
EMA_DECAY = .999
RESOLVER_POLICY = dict(topk=5,width=128,heads=4,intermediate_block=6,initial_eta=.1)
OPTIMIZER_POLICY = dict(name='AdamW',learning_rate=5e-4,weight_decay=.05,ema_decay=EMA_DECAY,
    schedule='1epoch warmup followed by cosine; floor.02; fresh optimizer each arm')
SOURCES = ('pairwise_resolver_experiment.py', 'pairwise_resolver_model.py',
    'pairwise_resolver_diagnostics.py', 'run_pairwise_resolver.py')


class BudgetExceeded(RuntimeError):
    pass


def save_json(path,value):
    for attempt in range(5):
        try:return _save_json(path,value)
        except PermissionError:
            if attempt==4:raise
            time.sleep(.05*(attempt+1))


def check_deadline(deadline_unix):
    if deadline_unix is not None and time.time() >= deadline_unix:
        raise BudgetExceeded('Compute deadline reached; incomplete epoch will replay on resume')


def source_hashes():
    from best_pair_joint import source_hashes as inherited
    return {**inherited(), **{name: file_hash(ROOT/name) for name in SOURCES}}


def _complete_best_pair(path, sources):
    """Validate a completed new joint run before considering its score."""
    from best_pair_joint import validate_resume, load_best_pair_model
    path = Path(path).resolve()
    completion = json.loads((path/'completion.json').read_text(encoding='utf-8'))
    if completion.get('status') != 'complete':
        raise ValueError('Anchor joint suite is not complete')
    plan = validate_resume(path)
    if plan.get('smoke') or plan['seed'] != sources['seed'] or plan['split_hash'] != sources['split_hash']:
        raise ValueError('Anchor must be a complete non-smoke joint run on the same split/seed')
    model, state = load_best_pair_model(plan, path/'training', sources=sources)
    del model
    run = path/'training/new_joint_concat'
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if (summary.get('status') != 'complete' or summary['completed_epochs'] != plan['epochs']
            or summary['best_epoch'] != state['epoch'] or summary['best_selection'] != state['selection']
            or summary['best_source'] != state['source']):
        raise ValueError('Completed strongest-pair anchor summary differs from checkpoint')
    paths = [path/'suite.json', path/'completion.json', path/'training/completion.json',
        run/'metadata.json', run/'summary.json', run/'best.pt']
    return dict(suite=str(path), suite_type='best_pair_joint',
        checkpoint=str(run/'best.pt'), checkpoint_sha256=file_hash(run/'best.pt'),
        best_epoch=state['epoch'], best_source=state['source'], best_selection=state['selection'],
        profile=sources['selected']['profile']), state, paths


def load_anchor_sources(cnn_suite=DEFAULT_CNN_SOURCE, vit_suite=DEFAULT_VIT_SOURCE,
                        base_suite=None, config=None):
    """Select once, then pin exact full joint weights; never substitute encoders."""
    sources = load_best_sources(cnn_suite, vit_suite, config=config)
    previous = sources.get('previous_joint')
    if previous is None:
        raise ValueError('Completed original joint concat anchor is required')
    original = dict(suite=str(Path(vit_suite).resolve()), suite_type='vit_augreg',
        checkpoint=previous['checkpoint'], checkpoint_sha256=previous['checkpoint_sha256'],
        best_epoch=previous['state']['epoch'], best_source=previous['state']['source'],
        best_selection=previous['state']['selection'], profile=sources['selected']['profile'])
    candidates = [(original, previous['state'], [])]
    ignored = []
    if base_suite is not None:
        if Path(base_suite).resolve() != Path(vit_suite).resolve():
            candidates = [_complete_best_pair(base_suite, sources)]
    else:
        for path in sorted((ROOT/'runs/best_pair_joint_tiny').glob('suite_*')):
            marker = path/'completion.json'
            if not marker.is_file():
                ignored.append(dict(suite=str(path), reason='no completion record'))
                continue
            record = json.loads(marker.read_text(encoding='utf-8'))
            if record.get('status') != 'complete':
                ignored.append(dict(suite=str(path), reason='incomplete; never used as anchor'))
                continue
            candidates.append(_complete_best_pair(path, sources))
    selected, state, paths = max(candidates, key=lambda item:
        (item[0]['best_selection']['accuracy'], -item[0]['best_selection']['nll']))
    files = dict(sources['source_files'])
    files.update({str(p.resolve()): file_hash(p) for p in paths})
    return dict(sources, anchor=selected, anchor_state=state,
        source_files=files, anchor_candidates=[item[0] for item in candidates],
        ignored_anchor_suites=ignored)


def _separate_output(output, sources):
    output = Path(output).resolve()
    for source in (sources['cnn_suite'], sources['vit_suite'], sources['anchor']['suite']):
        source = Path(source).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('New output must be separate from immutable source suites')


def _validate_data(config, sources):
    datasets, manifest, fingerprint = prepare_datasets(config, sources['seed'])
    if (fingerprint != sources['split_hash'] or len(datasets['train']) != 80000
            or len(datasets['selection']) != 10000 or len(datasets['train_probe']) != 4000
            or set(manifest['train']) & set(manifest['selection'])):
        raise ValueError('Require original disjoint80K/10K split and fixed4K clean train probe')
    environment = execution_environment(config, sources['seed'])
    if environment != sources['provenance']['source_execution_environment']:
        raise ValueError('Resolver comparison requires recorded GPU/software/precision environment')
    return environment


def make_plan(args):
    epochs = 1 if args.smoke else (30 if args.epochs is None else args.epochs)
    if not 1 <= epochs <= 30 or args.examples < 0:
        raise ValueError('Need1..30 epochs and nonnegative examples')
    sources = load_anchor_sources(args.cnn_suite, args.vit_suite, args.base_suite)
    config = copy.deepcopy(sources['config'])
    data = Path(args.data_dir or config['dataset']['data_dir']).expanduser()
    config['dataset']['data_dir'] = str((data if data.is_absolute() else ROOT/data).resolve())
    config['dataset']['download'] = bool(args.allow_download)
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    if config['dataset']['num_workers'] < 0 or config['dataset']['batch_size'] != 64:
        raise ValueError('Require nonnegative workers and matched batch64')
    config['training'].update(probe_per_class=20)
    _separate_output(args.output, sources)
    environment = _validate_data(config, sources)
    return dict(protocol=PROTOCOL, cnn_suite=sources['cnn_suite'], vit_suite=sources['vit_suite'],
        anchor=sources['anchor'], anchor_candidates=sources['anchor_candidates'],
        ignored_anchor_suites=sources['ignored_anchor_suites'],
        anchor_selection_policy=('Explicit completed suite' if args.base_suite else
            'Highest developmentTop1 then lowerNLL among validated completed original/new joint anchors'),
        source_files=sources['source_files'], source_hashes=source_hashes(),
        execution_environment=environment, config=config, config_recipe_hash=recipe_hash(config),
        seed=sources['seed'], split_hash=sources['split_hash'], jobs=list(JOBS),
        epochs=epochs, total_new_model_epochs=epochs*len(JOBS), smoke=bool(args.smoke),
        examples=args.examples, fusion_augmentation_epoch_offset=100,
        resolver=copy.deepcopy(RESOLVER_POLICY),optimizer=copy.deepcopy(OPTIMIZER_POLICY),
        stages=['training', 'feature_analysis'], shutdown_after_completion=False,
        audit_partition_evaluated=False, official_validation_evaluated=False,
        partial_resume_policy='Replay an interrupted epoch from its last completed epoch checkpoint',
        timing_policy='External runner compute deadline includes all four jobs and analysis; maximum total10h')


def create_suite(args, plan=None):
    plan = make_plan(args) if plan is None else plan
    root = Path(args.output).resolve()
    sources = dict(cnn_suite=plan['cnn_suite'], vit_suite=plan['vit_suite'], anchor=plan['anchor'])
    _separate_output(root, sources)
    suite = root/new_name('smoke' if plan['smoke'] else 'suite')
    suite.mkdir(parents=True, exist_ok=False)
    save_json(suite/'suite.json', plan)
    print(f'Pairwise resolver suite: {suite}', flush=True)
    print(f'Frozen complete joint anchor: {plan["anchor"]["checkpoint"]} '
        f'(recorded development Top1={100*plan["anchor"]["best_selection"]["accuracy"]:.2f}%, '
        f'SHA256={plan["anchor"]["checkpoint_sha256"]})', flush=True)
    print(f'New training: {len(plan["jobs"])} resolver arms x {plan["epochs"]} epochs; '
        'both encoders and the existing joint classifier remain fixed.', flush=True)
    return suite


def _load_pinned_sources(plan):
    sources = load_anchor_sources(plan['cnn_suite'], plan['vit_suite'],
        plan['anchor']['suite'], config=plan['config'])
    if sources['anchor'] != plan['anchor'] or sources['source_files'] != plan['source_files']:
        raise ValueError('Pinned complete anchor or immutable source identities changed')
    return sources


def validate_resume(path):
    plan = json.loads((Path(path)/'suite.json').read_text(encoding='utf-8'))
    if plan.get('protocol') != PROTOCOL or plan.get('source_hashes') != source_hashes():
        raise ValueError('Resume requires unchanged resolver protocol and source implementation')
    if (plan.get('jobs') != list(JOBS) or not 1 <= plan['epochs'] <= 30
            or plan['total_new_model_epochs'] != len(JOBS)*plan['epochs']
            or plan.get('shutdown_after_completion') is not False
            or plan.get('fusion_augmentation_epoch_offset') != 100
            or plan.get('resolver') != RESOLVER_POLICY or plan.get('optimizer') != OPTIMIZER_POLICY
            or (plan.get('smoke') and plan['epochs']!=1)
            or plan.get('audit_partition_evaluated') is not False
            or plan.get('official_validation_evaluated') is not False
            or recipe_hash(plan['config']) != plan['config_recipe_hash']):
        raise ValueError('Recorded four-arm resolver recipe or no-shutdown policy changed')
    sources = _load_pinned_sources(plan)
    if _validate_data(plan['config'], sources) != plan['execution_environment']:
        raise ValueError('Resume environment changed')
    return plan


def job_config(plan, job):
    if job not in JOBS:
        raise ValueError('Unknown resolver arm')
    config = copy.deepcopy(plan['config'])
    config['training'].update(epochs=plan['epochs'], learning_rate=5e-4,
        weight_decay=.05, accumulation_steps=1, early_stopping_patience=0)
    config['training']['scheduler'] = dict(name='cosine', warmup_epochs=min(1,plan['epochs']-1),
        warmup_start_factor=.1, min_ratio=.02)
    return config


def build_resolver(plan, job, sources=None):
    from pairwise_resolver_model import PairwiseResolver
    from vit_augreg_experiment import build_job_model
    if job not in JOBS:
        raise ValueError('Unknown resolver arm')
    sources = sources or _load_pinned_sources(plan)
    anchor = build_job_model('new_joint_concat', sources, sources['selected'], plan['seed'])
    anchor.load_state_dict(sources['anchor_state']['model'], strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(plan['seed']+73001)
        return PairwiseResolver(anchor, variant=job, **plan['resolver'])


def resolver_state(model):
    return {name: value.detach().cpu().clone() for name,value in model.state_dict().items()
            if not name.startswith('anchor.')}


def load_resolver_state(model, state):
    expected = {name for name in model.state_dict() if not name.startswith('anchor.')}
    if set(state) != expected or any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError('Resolver-only checkpoint keys or finite values differ')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.startswith('anchor.') for name in missing):
        raise ValueError('Resolver-only checkpoint unexpectedly changed anchor layout')


def _cache_identity(plan):
    return dict(protocol=PROTOCOL, anchor_sha256=plan['anchor']['checkpoint_sha256'],
        split_hash=plan['split_hash'], seed=plan['seed'], intermediate_block=plan['resolver']['intermediate_block'],
        extraction='FP32 clean images; frozen anchor eval; patch tokens excludeCLS')


def load_feature_cache(plan, path):
    path=Path(path)
    digest_path=path.with_suffix('.sha256.json')
    if digest_path.is_file():
        recorded=json.loads(digest_path.read_text(encoding='utf-8'))
        if recorded.get('sha256')!=file_hash(path):
            raise ValueError('Frozen feature cache bytes changed')
    cache = torch.load(path, map_location='cpu', weights_only=False)
    identity=cache.get('identity',{})
    if any(identity.get(key)!=value for key,value in _cache_identity(plan).items()):
        raise ValueError('Frozen feature cache identity differs from pinned anchor')
    n = len(cache['labels'])
    if (identity.get('split') not in ('selection','train_probe')
            or n != (10000 if identity.get('split')=='selection' else 4000)
            or len(cache['image_ids']) != n or len(set(cache['image_ids'])) != n
            or cache['features'].shape != (n,2432) or cache['patch_tokens'].shape != (n,64,384)
            or cache['anchor_logits'].shape != (n,200) or len(cache['class_names']) != 200
            or cache['labels'].dtype != torch.long or cache['labels'].ndim != 1
            or int(cache['labels'].min())<0 or int(cache['labels'].max())>=200
            or any(not torch.isfinite(cache[key]).all() for key in ('features','patch_tokens','anchor_logits'))):
        raise ValueError('Frozen feature cache dimensions, IDs or finite values are invalid')
    return cache


@torch.no_grad()
def extract_cache(plan, model, dataset, path, device, deadline_unix=None):
    if Path(path).is_file():
        cache=load_feature_cache(plan,path)
        expected_ids=[Path(dataset.folder.samples[i][0]).relative_to(dataset.folder.root).as_posix()
            for i in dataset.indices]
        expected_labels=torch.tensor([dataset.folder.samples[i][1] for i in dataset.indices])
        if cache['image_ids']!=expected_ids or not torch.equal(cache['labels'],expected_labels):
            raise ValueError('Cached frozen features no longer match exact clean dataset ordering')
        return cache
    rows = {key: [] for key in ('features','patch_tokens','anchor_logits','labels')}
    image_ids = []
    model.eval()
    for images,labels,ids in loader(dataset,plan['config']['dataset'],plan['seed']):
        check_deadline(deadline_unix)
        with torch.autocast(device,enabled=False):
            values = model.extract_anchor(images.to(device).float())
        for target,key in (('features','features'),('patch_tokens','patch_tokens'),('anchor_logits','logits')):
            rows[target].append(values[key].detach().float().cpu())
        rows['labels'].append(labels.cpu())
        image_ids.extend(ids)
    cache = {key: torch.cat(values) for key,values in rows.items()}
    cache.update(image_ids=image_ids,class_names=list(dataset.folder.classes),
        identity=dict(_cache_identity(plan),split=dataset.split))
    atomic_save(Path(path),cache)
    save_json(Path(path).with_suffix('.sha256.json'),dict(sha256=file_hash(path),identity=cache['identity']))
    return load_feature_cache(plan,path)


@torch.no_grad()
def evaluate_cached(model, cache, device, batch_size=128, deadline_unix=None):
    model.eval()
    losses = correct = top5 = count = 0
    for start in range(0,len(cache['labels']),batch_size):
        check_deadline(deadline_unix)
        end = start+batch_size
        logits = model.forward_cached(cache['features'][start:end].to(device),
            cache['patch_tokens'][start:end].to(device),cache['anchor_logits'][start:end].to(device))
        labels = cache['labels'][start:end].to(device)
        losses += float(F.cross_entropy(logits.float(),labels,reduction='sum'))
        correct += int((logits.argmax(1)==labels).sum())
        top5 += int((logits.topk(5,dim=1).indices==labels[:,None]).any(1).sum())
        count += len(labels)
    return dict(accuracy=correct/count,top5_accuracy=top5/count,nll=losses/count,n=count)


def make_optimizer(model, config):
    groups = []
    for decay in (True,False):
        values = [p for p in model.parameters() if p.requires_grad and (p.ndim>1)==decay]
        if values:
            groups.append(dict(params=values,weight_decay=config['training']['weight_decay'] if decay else 0.))
    return torch.optim.AdamW(groups,lr=config['training']['learning_rate'])


def _metadata(plan,job,model):
    return dict(protocol=PROTOCOL,job=job,seed=plan['seed'],split_hash=plan['split_hash'],
        source_hashes=plan['source_hashes'],anchor=plan['anchor'],recipe_hash=recipe_hash(job_config(plan,job)),
        params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_anchor_digest=state_digest(model.anchor),frozen_anchor=True,
        resolver=plan['resolver'],optimizer=plan['optimizer'],
        mean_tokens_source='Fixed4K clean training probe only; no development/audit fitting',
        checkpoint_layout='resolver_state only; immutable full anchor rebuilt and hash validated')


def _validate_checkpoint(state,metadata,config):
    if (state.get('metadata') != metadata or recipe_hash(state['config']) != recipe_hash(config)
            or not 0 <= state['epoch'] <= config['training']['epochs']):
        raise ValueError('Resolver checkpoint metadata, recipe or epoch differs')


def _restore_best_snapshots(run,last,metadata,config):
    """last.pt is the atomic commit record; best files are recoverable indexes."""
    best=last['best_snapshot'];trained=last.get('best_trained_snapshot')
    for name,snapshot in (('best.pt',best),('best_trained.pt',trained)):
        if snapshot is None:continue
        _validate_checkpoint(snapshot,metadata,config)
        if snapshot['epoch']>last['epoch'] or (name=='best_trained.pt' and snapshot['epoch']<1):
            raise ValueError('Committed best snapshot lies outside completed epoch history')
        existing=Path(run)/name
        rewrite=not existing.is_file()
        if not rewrite:
            saved=torch.load(existing,map_location='cpu',weights_only=False)
            rewrite=(saved.get('epoch')!=snapshot['epoch'] or saved.get('source')!=snapshot['source']
                or saved.get('selection')!=snapshot['selection'] or saved.get('metadata')!=metadata
                or set(saved.get('resolver_state',{}))!=set(snapshot['resolver_state'])
                or any(not torch.equal(saved['resolver_state'][key].cpu(),value.cpu())
                    for key,value in snapshot['resolver_state'].items()))
        if rewrite:atomic_save(existing,snapshot)
    return best,trained


@torch.no_grad()
def _update_ema(ema,model):
    parameters=dict(model.named_parameters())
    for name,value in model.state_dict().items():
        if name.startswith('anchor.'):continue
        if value.is_floating_point() and name in parameters:
            ema[name].mul_(EMA_DECAY).add_(value,alpha=1-EMA_DECAY)
        else:
            ema[name].copy_(value)


def train_one_epoch(model,batches,config,seed,data_epoch,device,amp,dtype,optimizer,scaler,ema,
                    deadline_unix=None):
    model.train()
    total = count = skipped = 0
    gradient_l2=[];correction_rms=[];entropy=[];peak=[]
    for index,(images,labels,_) in enumerate(batches):
        check_deadline(deadline_unix)
        images,labels = images.to(device),labels.to(device)
        images,targets = mix_batch(images,labels,config['training'],seed,data_epoch,index)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device,enabled=amp,dtype=dtype):
            detail=model(images,return_details=index==0)
            logits=detail['logits'] if isinstance(detail,dict) else detail
            loss = -(targets.float()*F.log_softmax(logits.float(),dim=-1)).sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite resolver loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
            config['training'].get('gradient_clip_norm',5.),error_if_nonfinite=not amp)
        gradient_l2.append(float(norm))
        if index==0 and isinstance(detail,dict):
            correction_rms.append(float(detail['delta'].float().square().mean().sqrt().detach()))
            for weights in detail.get('retrieval_maps',{}).values():
                if isinstance(weights,torch.Tensor) and weights.ndim==4 and weights.min()>=0:
                    entropy.append(float(-(weights*weights.clamp_min(1e-12).log()).sum(-1).mean().detach()))
                    peak.append(float(weights.amax(-1).mean().detach()))
        previous = scaler.get_scale()
        scaler.step(optimizer);scaler.update()
        if scaler.get_scale() >= previous:
            _update_ema(ema,model)
        else:
            skipped += 1
        total += float(loss.detach())*len(labels);count += len(labels)
    if not count:
        raise ValueError('Empty resolver training dataset')
    return dict(train_soft_target_loss=total/count,skipped_amp_steps=skipped,
        mean_trainable_gradient_l2=sum(gradient_l2)/len(gradient_l2),
        first_batch_correction_rms=correction_rms[0] if correction_rms else 0.,
        first_batch_attention_entropy=sum(entropy)/len(entropy) if entropy else None,
        first_batch_attention_peak=sum(peak)/len(peak) if peak else None,
        eta=float(model.eta.detach()))


def _write_history(run,history):
    temporary = run/'epochs.csv.tmp'
    pd.DataFrame(history).to_csv(temporary,index=False)
    for attempt in range(5):
        try:
            temporary.replace(run/'epochs.csv');break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.05*(attempt+1))


def train_job(plan,job,training,sources,caches,deadline_unix=None):
    from vit_augreg_experiment import make_scheduler
    run = Path(training)/job
    if run.is_symlink():
        raise ValueError('Refusing symlink resolver directory')
    config = job_config(plan,job)
    device,amp,dtype,scaler = runtime(config,plan['seed'])
    model = build_resolver(plan,job,sources).to(device)
    model.set_train_patch_mean(caches['train_probe']['patch_tokens'].mean(0).to(device))
    metadata = _metadata(plan,job,model)
    if run.exists() and not (run/'metadata.json').is_file():
        raise ValueError('Refusing unrelated resolver run directory')
    run.mkdir(exist_ok=True)
    if (run/'metadata.json').is_file() and json.loads((run/'metadata.json').read_text())!=metadata:
        raise ValueError('Existing resolver run identity differs')
    if not (run/'metadata.json').is_file():save_json(run/'metadata.json',metadata)
    if (run/'config.json').is_file():
        if json.loads((run/'config.json').read_text())!=config:
            raise ValueError('Existing resolver config differs from recorded recipe')
    else:save_json(run/'config.json',config)
    optimizer = make_optimizer(model,config)
    scheduler = make_scheduler(optimizer,config)
    if (run/'last.pt').is_file():
        state = torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        _validate_checkpoint(state,metadata,config)
        if [row['epoch'] for row in state['history']]!=list(range(state['epoch']+1)):
            raise ValueError('Resolver checkpoint history chronology differs')
        load_resolver_state(model,state['resolver_state'])
        ema={name:value.to(device) for name,value in state['ema_resolver_state'].items()};history=state['history']
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']);restore_rng(state['rng'],device)
        best_state,trained_state=_restore_best_snapshots(run,state,metadata,config)
        best=best_state['selection']['accuracy']
    else:
        initial=evaluate_cached(model,caches['selection'],device,deadline_unix=deadline_unix)
        history=[dict(epoch=0,augmentation_epoch=100,train_seconds=0.,
            **{f'selection_{k}':v for k,v in initial.items()},
            **{f'ema_selection_{k}':v for k,v in initial.items()})]
        ema={name:value.to(device) for name,value in resolver_state(model).items()};best=initial['accuracy']
        trained_state=None
        best_state=dict(resolver_state=resolver_state(model),epoch=0,source='raw',selection=initial,
            metadata=metadata,config=config)
        atomic_save(run/'best.pt',best_state)
        atomic_save(run/'last.pt',dict(best_state,ema_resolver_state=ema,history=history,
            best_snapshot=best_state,best_trained_snapshot=None,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(device)))
        _write_history(run,history)
    if history[-1]['epoch']>=plan['epochs'] and (run/'summary.json').is_file():
        result=json.loads((run/'summary.json').read_text())
        if result.get('status')=='complete' and all(result.get(k)==v for k,v in metadata.items()):
            return result
    datasets,_,fingerprint=prepare_datasets(config,plan['seed'])
    if fingerprint!=plan['split_hash']:
        raise ValueError('Resolver augmented training split changed')
    status='complete'
    try:
        for epoch in range(history[-1]['epoch']+1,plan['epochs']+1):
            check_deadline(deadline_unix)
            sync(device);started=time.perf_counter()
            lr=optimizer.param_groups[0]['lr']
            losses=train_one_epoch(model,loader(datasets['train'],config['dataset'],plan['seed'],100+epoch),
                config,plan['seed'],100+epoch,device,amp,dtype,optimizer,scaler,ema,deadline_unix)
            sync(device);seconds=time.perf_counter()-started
            raw=evaluate_cached(model,caches['selection'],device,deadline_unix=deadline_unix)
            raw_state=resolver_state(model)
            load_resolver_state(model,ema)
            averaged=evaluate_cached(model,caches['selection'],device,deadline_unix=deadline_unix)
            load_resolver_state(model,raw_state)
            row=dict(epoch=epoch,augmentation_epoch=100+epoch,train_seconds=seconds,learning_rate=lr,**losses,
                **{f'selection_{k}':v for k,v in raw.items()},
                **{f'ema_selection_{k}':v for k,v in averaged.items()})
            if epoch==1 or epoch%5==0 or epoch==plan['epochs']:
                probe=evaluate_cached(model,caches['train_probe'],device,deadline_unix=deadline_unix)
                row.update({f'train_probe_{k}':v for k,v in probe.items()})
            for source,weights,score in (('raw',raw_state,raw),('ema',ema,averaged)):
                candidate=dict(resolver_state=copy.deepcopy(weights),epoch=epoch,source=source,
                    selection=score,metadata=metadata,config=config)
                if trained_state is None or score['accuracy']>trained_state['selection']['accuracy']:
                    trained_state=candidate
                if score['accuracy']>best:
                    best=score['accuracy']
                    best_state=candidate
            history.append(row);scheduler.step()
            atomic_save(run/'last.pt',dict(resolver_state=raw_state,ema_resolver_state=ema,epoch=epoch,
                source='raw',selection=raw,metadata=metadata,config=config,history=history,
                best_snapshot=best_state,best_trained_snapshot=trained_state,
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(device)))
            # Best checkpoints are written only AFTER the epoch transaction commits.
            # Resume reconstructs either index from the snapshots if interrupted here.
            atomic_save(run/'best.pt',best_state)
            atomic_save(run/'best_trained.pt',trained_state)
            _write_history(run,history)
            print(f'{job} {epoch}/{plan["epochs"]}: raw={raw["accuracy"]*100:.2f}% '
                f'EMA={averaged["accuracy"]*100:.2f}% best={best*100:.2f}% '
                f'{best_state["source"]}@{best_state["epoch"]}',flush=True)
    except BudgetExceeded:
        status='budget_exhausted'
        committed=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        _validate_checkpoint(committed,metadata,config)
        history=committed['history']
        best_state,trained_state=_restore_best_snapshots(run,committed,metadata,config)
        best=best_state['selection']['accuracy']
    unchanged=state_digest(model.anchor)==metadata['frozen_anchor_digest']
    if not unchanged:
        raise RuntimeError('Frozen complete anchor parameters or buffers changed')
    result=dict(metadata,status=status,completed_epochs=history[-1]['epoch'],planned_epochs=plan['epochs'],
        best_epoch=best_state['epoch'],best_source=best_state['source'],best_selection=best_state['selection'],
        best_trained_selection=(trained_state['selection'] if trained_state else None),
        best_trained_epoch=(trained_state['epoch'] if trained_state else None),
        best_trained_source=(trained_state['source'] if trained_state else None),
        initial_selection={key:history[0]['selection_'+key] for key in ('accuracy','top5_accuracy','nll','n')},
        last_epoch_selection={key:history[-1]['selection_'+key] for key in ('accuracy','top5_accuracy','nll','n')},
        last_ema_selection={key:history[-1]['ema_selection_'+key] for key in ('accuracy','top5_accuracy','nll','n')},
        total_train_seconds=sum(row['train_seconds'] for row in history),frozen_anchor_unchanged=unchanged,
        interrupted_epoch_replays_on_resume=status!='complete',audit_partition_evaluated=False,official_validation_evaluated=False)
    save_json(run/'summary.json',result)
    del model,optimizer,scheduler,scaler;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return result


def load_resolver_model(plan,training_dir,job,checkpoint='best.pt'):
    if checkpoint in ('best','last','best_trained'):checkpoint+='.pt'
    if checkpoint not in ('best.pt','last.pt','best_trained.pt'):
        raise ValueError('Unsupported resolver checkpoint')
    model=build_resolver(plan,job)
    run=Path(training_dir)/job
    state=torch.load(run/checkpoint,map_location='cpu',weights_only=False)
    metadata=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    _validate_checkpoint(state,metadata,job_config(plan,job))
    # The fixed train mean lives in resolver_state; set it before metadata checks.
    load_resolver_state(model,state['resolver_state'])
    if _metadata(plan,job,model)!=metadata:
        raise ValueError('Analysis resolver architecture or pinned source differs')
    return model,state


def training_report(plan,training):
    training=Path(training)
    rows=[]
    for job in JOBS:
        path=training/job/'summary.json'
        if not path.is_file():continue
        summary=json.loads(path.read_text())
        rows.append(dict(job=job,status=summary['status'],completed_epochs=summary['completed_epochs'],
            planned_epochs=plan['epochs'],best_top1_pct=100*summary['best_selection']['accuracy'],
            best_nll=summary['best_selection']['nll'],best_epoch=summary['best_epoch'],best_source=summary['best_source'],
            best_trained_top1_pct=(100*summary['best_trained_selection']['accuracy']
                if summary.get('best_trained_selection') else None),
            best_trained_epoch=summary.get('best_trained_epoch'),best_trained_source=summary.get('best_trained_source'),
            last_top1_pct=100*summary['last_epoch_selection']['accuracy'],
            last_ema_top1_pct=100*summary['last_ema_selection']['accuracy'],trainable_params=summary['trainable_params'],
            train_minutes=summary['total_train_seconds']/60))
    pd.DataFrame(rows).to_csv(training/'comparison.csv',index=False)
    (training/'report.md').write_text('# Frozen joint-anchor candidate resolvers\n\n'
        f'Anchor source: {plan["anchor"]["suite"]}; recorded selectionTop1 '
        f'{100*plan["anchor"]["best_selection"]["accuracy"]:.2f}%. Both encoders and original '
        'fusion classifier remain fixed. Four matched new modules use identical augmented '
        'training views and30-epoch schedules; only the small new resolver is updated.\n\n'
        'All checkpoint selection uses the10K development split. Per-position token means '
        'are fitted only on the fixed4K clean training probe. No audit or official validation '
        'evaluation is performed. Interrupted epochs replay from the last completed epoch.\n\n'
        'See comparison.csv for completed horizons, raw/EMA scores, parameter counts and '
        'actual training time; best_trained excludes the epoch-zero initialization, which '
        'can remain the overall selected best. Missing arms were not completed.\n',encoding='utf-8')
    return rows


def run_suite(suite,deadline_unix=None):
    suite=Path(suite).resolve();plan=validate_resume(suite)
    started=time.perf_counter();training=suite/'training';training.mkdir(exist_ok=True)
    outcomes={};status='complete';failure=None
    old_handler=None
    def interrupted(signum,frame):
        raise BudgetExceeded('Termination requested; retaining last completed epoch')
    if hasattr(signal,'SIGTERM'):
        old_handler=signal.signal(signal.SIGTERM,interrupted)
    try:
        check_deadline(deadline_unix)
        sources=_load_pinned_sources(plan)
        device,_,_,_=runtime(plan['config'],plan['seed'])
        datasets,manifest,fingerprint=prepare_datasets(plan['config'],plan['seed'])
        if fingerprint!=plan['split_hash']:raise ValueError('Data split changed')
        save_json(training/'split_manifest.json',manifest)
        model=build_resolver(plan,'pairwise',sources).to(device)
        caches={name:extract_cache(plan,model,datasets[name],training/(name+'_cache.pt'),device,deadline_unix)
            for name in ('selection','train_probe')}
        del model;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        mean=caches['train_probe']['patch_tokens'].mean(0)
        atomic_save(training/'train_patch_mean.pt',dict(mean=mean,identity=_cache_identity(plan),
            image_ids=caches['train_probe']['image_ids'],n=len(caches['train_probe']['labels'])))
        for job in plan['jobs']:
            check_deadline(deadline_unix)
            save_json(suite/'status.json',dict(protocol=PROTOCOL,status='running',active_job=job,outcomes=outcomes))
            result=train_job(plan,job,training,sources,caches,deadline_unix)
            outcomes[job]=dict(status=result['status'],completed_epochs=result['completed_epochs'])
            training_report(plan,training)
            if result['status']!='complete':
                status='budget_exhausted';break
        if status=='complete':
            from pairwise_resolver_diagnostics import run_pairwise_diagnostics
            result=run_pairwise_diagnostics(plan,training,suite/'feature_analysis',
                examples=plan['examples'],deadline_unix=deadline_unix)
            outcomes['feature_analysis']=dict(status=result.get('status','complete'))
            if result.get('status')!='complete':status=result.get('status','failed')
    except BudgetExceeded as exc:
        status='budget_exhausted';failure=dict(error_type=type(exc).__name__,message=str(exc))
    except Exception as exc:
        status='failed';failure=dict(error_type=type(exc).__name__,traceback=traceback.format_exc())
        save_json(suite/'failure.json',failure)
    finally:
        if old_handler is not None:signal.signal(signal.SIGTERM,old_handler)
    training_report(plan,training)
    unchanged=all(Path(p).is_file() and file_hash(p)==digest for p,digest in plan['source_files'].items())
    if not unchanged:status='failed'
    result=dict(protocol=PROTOCOL,status=status,outcomes=outcomes,failure=failure,
        elapsed_active_hours=(time.perf_counter()-started)/3600,
        source_checkpoints_unchanged=unchanged,shutdown_after_completion=False,
        audit_partition_evaluated=False,official_validation_evaluated=False,smoke=plan['smoke'])
    save_json(suite/'completion.json',result);save_json(suite/'status.json',result)
    (suite/'report.md').write_text('# Candidate-specific evidence retrieval\n\n'
        f'Status: **{status}**. Four frozen-anchor resolver arms, each planned for{plan["epochs"]}epochs.\n\n'
        '- [Training](training/report.md)\n- [Feature analysis](feature_analysis/report.md)\n\n'
        'Training and analysis obey the runner compute deadline. Partial results are retained '
        'and can resume without repeating completed epochs. The instance remains running.\n',encoding='utf-8')
    return result


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cnn-suite',default=str(DEFAULT_CNN_SOURCE))
    parser.add_argument('--vit-suite',default=str(DEFAULT_VIT_SOURCE))
    parser.add_argument('--base-suite',help='Completed joint anchor suite; default highest validated completed known joint run')
    parser.add_argument('--epochs',type=int,help='Matched new-module epochs per arm, default30')
    parser.add_argument('--data-dir');parser.add_argument('--num-workers',type=int)
    parser.add_argument('--allow-download',action='store_true')
    parser.add_argument('--examples',type=int,default=8);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output','--output-dir',default=str(ROOT/'runs/pairwise_resolver_tiny'))
    parser.add_argument('--resume');parser.add_argument('--deadline-unix',type=float)
    return parser


def main(argv=None):
    args=arguments().parse_args(argv)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no resolver suite created')
    if args.resume:
        plan=validate_resume(args.resume)
        if args.smoke or (args.epochs is not None and args.epochs!=plan['epochs']):
            raise ValueError('Resume retains saved horizon and protocol')
        suite=Path(args.resume)
    else:suite=create_suite(args)
    result=run_suite(suite,args.deadline_unix)
    return 0 if result['status']=='complete' else 2


if __name__=='__main__':raise SystemExit(main())
