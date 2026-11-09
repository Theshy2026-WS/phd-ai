"""Three targeted controls for the winning CNN GAP + ViT CLS linear fusion.

Only these new controls train. Completed independent sources and the winning
joint reference are immutable. All participating encoder weights update.
This child experiment has no SMTP or instance-control side effects.
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
from torch.nn import functional as F

from best_pair_sources import DEFAULT_CNN_SOURCE, DEFAULT_VIT_SOURCE, load_best_sources
from correction_experiment import file_hash
from datasets.resnet_tiny import prepare_datasets, mix_batch
from datasets.utility_tiny import loader
from literature_experiment import ROOT, new_name, recipe_hash, save_json as _save_json
from models.attention_correction import state_digest
from spatial_experiment import execution_environment, rng_state, restore_rng
from utility_experiment import runtime, sync
from vit_augreg_experiment import make_scheduler
from vit_followup_experiment import atomic_save


PROTOCOL = 'tiny-fusion-mechanism-suite-v1'
JOBS = ('random_vit_cls_linear', 'reset_cnn_head_cls_linear', 'patch_mean_linear')
DEFAULT_REFERENCE = ROOT/'runs/best_pair_joint_tiny/suite_20261109T014414_c55052'
EMA_DECAY = .999
MODEL_POLICY = dict(base='winning CLS+Linear concatenation', all_encoders_trainable=True,
    head_initialization_seed=42, reset_cnn_head_seed_offset=91001,
    readout={'random_vit_cls_linear':'cls', 'reset_cnn_head_cls_linear':'cls',
             'patch_mean_linear':'normalized_patch_mean'},
    only_change={'random_vit_cls_linear':'replace ViT weights by original random initialization',
        'reset_cnn_head_cls_linear':'reset CNN classifier columns and bias only',
        'patch_mean_linear':'replace normalized CLS readout with normalized patch-token mean'})
OPTIMIZER_POLICY = dict(name='AdamW',encoder_learning_rate=5e-5,vit_learning_rate=5e-5,
    fusion_learning_rate=5e-4,weight_decay=.05,ema_decay=EMA_DECAY,
    schedule='1epoch warmup and ratio-preserving cosine, floor.02')
SOURCES = ('fusion_mechanism_experiment.py','fusion_mechanism_diagnostics.py',
    'run_fusion_mechanism.py','best_pair_sources.py','best_pair_joint.py',
    'models/vit_augreg.py','vit_augreg_experiment.py',
    'hcsi_tiny_diagnostics.py','hcsi_tiny_experiment.py',
    'datasets/resnet_tiny.py','datasets/utility_tiny.py')


class BudgetExceeded(RuntimeError):
    pass


def check_deadline(deadline_unix):
    if deadline_unix is not None and time.time() >= deadline_unix:
        raise BudgetExceeded('Compute deadline reached; replay incomplete epoch on resume')


def save_json(path,value):
    for attempt in range(5):
        try:
            return _save_json(path,value)
        except PermissionError:
            if attempt==4:
                raise
            time.sleep(.05*(attempt+1))


def source_hashes():
    from best_pair_joint import source_hashes as inherited
    return {**inherited(),**{name:file_hash(ROOT/name) for name in SOURCES}}


def _reference_sources(path,sources):
    from best_pair_joint import validate_resume,load_best_pair_model
    path=Path(path).resolve()
    completion=json.loads((path/'completion.json').read_text(encoding='utf-8'))
    plan=validate_resume(path)
    if (completion.get('status')!='complete' or plan.get('smoke')
            or plan['seed']!=sources['seed'] or plan['split_hash']!=sources['split_hash']
            or plan['source_provenance']['parent_checkpoint_sha256']!=sources['provenance']['parent_checkpoint_sha256']
            or plan['selected_vit']['checkpoint_sha256']!=sources['selected']['checkpoint_sha256']):
        raise ValueError('Reference must be the completed same-source joint run')
    # The reference owns its original narrower dependency scope.
    model,state=load_best_pair_model(plan,path/'training')
    del model
    run=path/'training/new_joint_concat'
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if (summary.get('status')!='complete' or summary['completed_epochs']!=plan['epochs']
            or summary['best_selection']!=state['selection'] or summary['best_epoch']!=state['epoch']
            or summary['best_source']!=state['source']):
        raise ValueError('Reference summary differs from completed checkpoint')
    paths=[path/'suite.json',path/'completion.json',path/'training/completion.json',
        run/'metadata.json',run/'summary.json',run/'best.pt']
    record=dict(suite=str(path),checkpoint=str(run/'best.pt'),checkpoint_sha256=file_hash(run/'best.pt'),
        epoch=state['epoch'],source=state['source'],selection=state['selection'],
        initial_state_sha256=json.loads((run/'metadata.json').read_text(encoding='utf-8'))['initial_state_sha256'],
        reference_only=True,repeat_training=False,
        representation='CNN GAP + ViT normalized CLS -> concatenation -> Linear200')
    return dict(record=record,state=state,plan=plan),paths


def load_sources(plan=None, *, cnn_suite=DEFAULT_CNN_SOURCE,vit_suite=DEFAULT_VIT_SOURCE,
                 reference_suite=DEFAULT_REFERENCE,config=None):
    if plan is not None:
        cnn_suite,vit_suite=plan['cnn_suite'],plan['vit_suite']
        reference_suite=plan['reference']['suite']
        config=plan['config']
    sources=load_best_sources(cnn_suite,vit_suite,config=config)
    reference,paths=_reference_sources(reference_suite,sources)
    files=dict(sources['source_files'])
    files.update({str(p.resolve()):file_hash(p) for p in paths})
    result=dict(sources,reference=reference,source_files=files)
    if plan is not None and (files!=plan['source_files'] or reference['record']!=plan['reference']
            or sources['provenance']!=plan['source_provenance']
            or sources['seed']!=plan['seed'] or sources['split_hash']!=plan['split_hash']
            or {k:v for k,v in sources['selected'].items() if k!='state'}!=plan['selected_vit']):
        raise ValueError('Pinned independent weights or completed reference changed')
    return result


def load_reference_model(plan,sources=None):
    from best_pair_joint import validate_resume,load_best_pair_model
    sources=sources or load_sources(plan)
    reference=sources['reference']
    if reference['record']!=plan['reference']:
        raise ValueError('Reference identity changed')
    canonical=validate_resume(reference['record']['suite'])
    return load_best_pair_model(canonical,Path(reference['record']['suite'])/'training')


def _separate_output(output,sources):
    output=Path(output).resolve()
    for source in (sources['cnn_suite'],sources['vit_suite'],sources['reference']['record']['suite']):
        source=Path(source).resolve()
        if output==source or source in output.parents or output in source.parents:
            raise ValueError('Output must be separate from immutable source suites')


def _validate_data(config,sources):
    datasets,manifest,fingerprint=prepare_datasets(config,sources['seed'])
    if (fingerprint!=sources['split_hash'] or len(datasets['train'])!=80000
            or len(datasets['selection'])!=10000 or len(datasets['train_probe'])!=4000
            or set(manifest['train']) & set(manifest['selection'])):
        raise ValueError('Require original80K/10K split and fixed4K training probe')
    environment=execution_environment(config,sources['seed'])
    if environment!=sources['provenance']['source_execution_environment']:
        raise ValueError('Matched comparison requires recorded GPU/software/precision environment')
    return environment


def make_plan(args):
    epochs=1 if args.smoke else (30 if args.epochs is None else args.epochs)
    if not 1<=epochs<=30 or args.examples<0:
        raise ValueError('Need1..30 epochs and nonnegative example count')
    sources=load_sources(cnn_suite=args.cnn_suite,vit_suite=args.vit_suite,reference_suite=args.reference_suite)
    config=copy.deepcopy(sources['config'])
    data=Path(args.data_dir or config['dataset']['data_dir']).expanduser()
    config['dataset']['data_dir']=str((data if data.is_absolute() else ROOT/data).resolve())
    config['dataset']['download']=bool(args.allow_download)
    if args.num_workers is not None:
        config['dataset']['num_workers']=args.num_workers
    if config['dataset']['num_workers']<0 or config['dataset']['batch_size']!=64:
        raise ValueError('Require nonnegative workers and source batch64')
    config['training']['probe_per_class']=20
    _separate_output(args.output,sources)
    environment=_validate_data(config,sources)
    return dict(protocol=PROTOCOL,cnn_suite=sources['cnn_suite'],vit_suite=sources['vit_suite'],
        source_provenance=sources['provenance'],source_files=sources['source_files'],
        reference=sources['reference']['record'],source_hashes=source_hashes(),
        selected_vit={key:value for key,value in sources['selected'].items() if key!='state'},
        config=config,config_recipe_hash=recipe_hash(config),execution_environment=environment,
        seed=sources['seed'],split_hash=sources['split_hash'],epochs=epochs,fusion_epochs=epochs,jobs=list(JOBS),
        total_new_model_epochs=epochs*len(JOBS),smoke=bool(args.smoke),examples=args.examples,
        model_policy=copy.deepcopy(MODEL_POLICY),optimizer_policy=copy.deepcopy(OPTIMIZER_POLICY),
        fusion_augmentation_epoch_offset=100,
        initialization='Exact winning initial model; change only the declared factor for each control',
        inherited_cnn_training_epochs=sources['parent_completed_epochs'],
        stages=['training','feature_analysis'],shutdown_after_completion=False,
        audit_partition_evaluated=False,official_validation_evaluated=False,
        complete_all=True,max_hours=None,
        partial_resume_policy='Atomic completed epoch state; replay an interrupted epoch',
        timing_policy='Finish all declared controls; wrapper emails once then stops managed instance')


def create_suite(args,plan=None):
    plan=make_plan(args) if plan is None else plan
    root=Path(args.output).resolve()
    _separate_output(root,dict(cnn_suite=plan['cnn_suite'],vit_suite=plan['vit_suite'],
        reference=dict(record=plan['reference'])))
    suite=root/new_name('smoke' if plan['smoke'] else 'suite')
    suite.mkdir(parents=True,exist_ok=False)
    save_json(suite/'suite.json',plan)
    print(f'Fusion mechanism suite: {suite}',flush=True)
    print(f'Independent CNN: {100*plan["source_provenance"]["source_cnn_accuracy"]:.2f}%; '
        f'ViT: {100*plan["source_provenance"]["source_vit_accuracy"]:.2f}%. '
        f'Existing joint reference: {100*plan["reference"]["selection"]["accuracy"]:.2f}%; not retrained.',flush=True)
    print('All CNN/ViT weights update. Core trains/analyzes only; run_fusion_mechanism.py '
          'sends one final email and stops the instance after all jobs.',flush=True)
    return suite


def validate_resume(path):
    plan=json.loads((Path(path)/'suite.json').read_text(encoding='utf-8'))
    if plan.get('protocol')!=PROTOCOL or plan.get('source_hashes')!=source_hashes():
        raise ValueError('Resume requires unchanged source and fusion mechanism protocol')
    if (plan.get('jobs')!=list(JOBS) or not 1<=plan['epochs']<=30 or plan.get('fusion_epochs')!=plan['epochs']
            or plan['total_new_model_epochs']!=len(JOBS)*plan['epochs']
            or plan.get('model_policy')!=MODEL_POLICY or plan.get('optimizer_policy')!=OPTIMIZER_POLICY
            or plan.get('fusion_augmentation_epoch_offset')!=100
            or plan.get('shutdown_after_completion') is not False
            or plan.get('audit_partition_evaluated') is not False
            or plan.get('official_validation_evaluated') is not False
            or plan.get('complete_all') is not True or plan.get('max_hours') is not None
            or (plan.get('smoke') and plan['epochs']!=1)
            or recipe_hash(plan['config'])!=plan['config_recipe_hash']):
        raise ValueError('Recorded fusion mechanism recipe, horizon or child delivery policy changed')
    sources=load_sources(plan)
    if _validate_data(plan['config'],sources)!=plan['execution_environment']:
        raise ValueError('Resume environment changed')
    return plan


def job_config(plan,job):
    from vit_augreg_experiment import job_config as canonical_config
    if job not in JOBS:
        raise ValueError('Unknown fusion mechanism job')
    return canonical_config(dict(plan,fusion_epochs=plan['epochs']), 'new_joint_concat')


from models.vit_augreg import AugRegFusion


class MechanismFusion(AugRegFusion):
    """State-layout-compatible winning model with one explicit readout control."""
    def extract_features(self,images):
        cnn,cls,patches=super().extract_features(images)
        return cnn,(patches.mean(1) if self.readout_policy=='normalized_patch_mean' else cls),patches

    def fusion_description(self):
        return dict(super().fusion_description(),control_job=self.control_job,
            readout_policy=self.readout_policy,control_only_change=MODEL_POLICY['only_change'][self.control_job])


def build_model(plan,job,sources=None):
    from vit_augreg_experiment import build_job_model
    from models.vit_augreg import build_vit
    if job not in JOBS:
        raise ValueError('Unknown fusion mechanism job')
    sources=sources or load_sources(plan)
    # Isolated canonical construction gives every arm precisely the winner's
    # classifier columns. Extra counterfactual draws cannot alter epoch RNG.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(plan['seed'])
        model=build_job_model('new_joint_concat',sources,sources['selected'],plan['seed'])
        post_construction_rng=torch.get_rng_state().clone()
    canonical_digest=state_digest(model)
    expected=sources.get('reference',{}).get('record',{}).get('initial_state_sha256')
    if expected is not None and canonical_digest!=expected:
        raise ValueError('Canonical initial model no longer matches the winning reference')
    model.__class__=MechanismFusion
    model.control_job=job
    model.readout_policy=plan['model_policy']['readout'][job]
    model.canonical_initial_state_sha256=canonical_digest
    model.canonical_training_rng_cpu=post_construction_rng
    if job=='random_vit_cls_linear':
        # Original independent ViT seed, not a newly chosen initialization.
        random_vit=build_vit(sources['selected']['profile'],plan['seed'])
        random_vit.classifier=torch.nn.Identity()
        model.vit.load_state_dict(random_vit.state_dict(),strict=True)
        model.vit_initialization='original random independent-ViT initialization'
    else:
        model.vit_initialization='selected independently trained ViT checkpoint'
    if job=='reset_cnn_head_cls_linear':
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(plan['seed']+MODEL_POLICY['reset_cnn_head_seed_offset'])
            replacement=torch.nn.Linear(2048,model.num_classes)
        with torch.no_grad():
            model.fusion_classifier.weight[:,:2048].copy_(replacement.weight)
            model.fusion_classifier.bias.copy_(replacement.bias)
    model.vit_initialization_digest=state_digest(model.vit)
    model.requires_grad_(True)
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError('All participating encoder and fusion parameters must update')
    return model


def make_optimizer(model,config):
    from vit_augreg_experiment import make_optimizer as canonical_optimizer
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen encoder parameters are excluded from this joint comparison')
    return canonical_optimizer(model,config,standalone=False)


def cpu_state(model):
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}


def validate_weights(model,state):
    if set(state)!=set(model.state_dict()) or any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError('Checkpoint model layout or finite values differ')
    model.load_state_dict(state,strict=True)


@torch.no_grad()
def update_ema(ema,model):
    for name,value in model.state_dict().items():
        if value.is_floating_point():
            ema[name].mul_(EMA_DECAY).add_(value,alpha=1-EMA_DECAY)
        else:
            ema[name].copy_(value)


@torch.no_grad()
def _evaluate_model(model,dataset,config,seed,device,deadline_unix=None):
    model.eval()
    count=correct=top5=0
    loss=0.
    for images,labels,_ in loader(dataset,config['dataset'],seed):
        check_deadline(deadline_unix)
        images,labels=images.to(device),labels.to(device)
        with torch.autocast(device,enabled=False):
            logits=model(images.float()).float()
        if not torch.isfinite(logits).all():
            raise RuntimeError('Non-finite clean evaluation logits')
        loss+=float(F.cross_entropy(logits,labels,reduction='sum'))
        correct+=int((logits.argmax(1)==labels).sum())
        top5+=int((logits.topk(min(5,logits.shape[1]),dim=1).indices==labels[:,None]).any(1).sum())
        count+=len(labels)
    if not count:
        raise ValueError('Empty clean evaluation split')
    return dict(accuracy=correct/count,top5_accuracy=top5/count,nll=loss/count,n=count)


def evaluate_model(model,dataset,config,seed,device,deadline_unix=None):
    previous=rng_state(device)
    try:
        return _evaluate_model(model,dataset,config,seed,device,deadline_unix)
    finally:
        restore_rng(previous,device)


def train_one_epoch(model,batches,config,seed,data_epoch,device,amp,dtype,optimizer,scaler,ema,deadline_unix=None):
    model.train()
    total=count=skipped=steps=0
    gradient_norms=[]
    for index,(images,labels,_) in enumerate(batches):
        check_deadline(deadline_unix)
        images,labels=images.to(device),labels.to(device)
        images,targets=mix_batch(images,labels,config['training'],seed,data_epoch,index)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device,enabled=amp,dtype=dtype):
            logits=model(images)
            loss=-(targets.float()*F.log_softmax(logits.float(),dim=-1)).sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite fusion mechanism training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['training'].get('gradient_clip_norm',5.),error_if_nonfinite=not amp)
        gradient_norms.append(float(norm))
        previous=scaler.get_scale()
        scaler.step(optimizer);scaler.update()
        if scaler.get_scale()>=previous:
            update_ema(ema,model);steps+=1
        else:
            skipped+=1
        total+=float(loss.detach())*len(labels);count+=len(labels)
    if not count:
        raise ValueError('Empty augmented training split')
    return dict(train_soft_target_loss=total/count,optimizer_steps=steps,skipped_amp_steps=skipped,
        mean_trainable_gradient_l2=sum(gradient_norms)/len(gradient_norms))


def _metadata(plan,job,model):
    params=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable!=params:
        raise ValueError('All participating encoder parameters must update')
    return dict(protocol=PROTOCOL,job=job,seed=plan['seed'],split_hash=plan['split_hash'],
        source_hashes=plan['source_hashes'],source_provenance=plan['source_provenance'],
        source_files=plan['source_files'],reference=plan['reference'],
        recipe_hash=recipe_hash(job_config(plan,job)),model_policy=plan['model_policy'],
        optimizer_policy=plan['optimizer_policy'],params=params,trainable_params=trainable,
        encoder_frozen=False,vit_frozen=False,augmentation_epoch_offset=100,
        vit_initialization=getattr(model,'vit_initialization',None),
        vit_initialization_digest=getattr(model,'vit_initialization_digest',None),
        canonical_initial_state_sha256=getattr(model,'canonical_initial_state_sha256',None),
        initial_state_sha256=state_digest(model),
        readout_policy=plan['model_policy']['readout'][job],
        only_change=plan['model_policy']['only_change'][job],
        initialization='Exact winning initialization with only declared control factor changed',
        fresh_optimizer=True,matched_training_views=True)


def _validate_checkpoint(state,metadata,config):
    if (state.get('metadata')!=metadata or recipe_hash(state['config'])!=recipe_hash(config)
            or not 0<=state['epoch']<=config['training']['epochs']):
        raise ValueError('Checkpoint metadata, recipe or epoch changed')


def _restore_best_snapshots(run,last,metadata,config):
    best=last['best_snapshot'];trained=last.get('best_trained_snapshot')
    for name,snapshot in (('best.pt',best),('best_trained.pt',trained)):
        if snapshot is None:
            continue
        _validate_checkpoint(snapshot,metadata,config)
        if snapshot['epoch']>last['epoch'] or (name=='best_trained.pt' and snapshot['epoch']<1):
            raise ValueError('Best snapshot is outside committed epoch history')
        # last.pt is the atomic transaction; these files are recoverable indexes.
        atomic_save(Path(run)/name,snapshot)
    return best,trained


def _write_history(run,history):
    temporary=Path(run)/'epochs.csv.tmp'
    pd.DataFrame(history).to_csv(temporary,index=False)
    for attempt in range(5):
        try:
            temporary.replace(Path(run)/'epochs.csv');return
        except PermissionError:
            if attempt==4:
                raise
            time.sleep(.05*(attempt+1))


def train_job(plan,job,training,sources,datasets=None,deadline_unix=None):
    run=Path(training)/job
    if run.is_symlink():
        raise ValueError('Refusing symlink training directory')
    config=job_config(plan,job)
    device,amp,dtype,scaler=runtime(config,plan['seed'])
    model=build_model(plan,job,sources).to(device)
    if hasattr(model,'canonical_training_rng_cpu'):
        torch.set_rng_state(model.canonical_training_rng_cpu)
    metadata=_metadata(plan,job,model)
    if run.exists() and not (run/'metadata.json').is_file():
        if any(run.iterdir()):
            raise ValueError('Refusing unrelated training directory')
    run.mkdir(exist_ok=True)
    if not (run/'last.pt').is_file():
        allowed={'metadata.json','config.json','last.tmp'}
        if any(entry.name not in allowed for entry in run.iterdir()):
            raise ValueError('Refusing unrelated files before the first atomic checkpoint')
    for filename,value in (('metadata.json',metadata),('config.json',config)):
        path=run/filename
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8'))!=value:
                raise ValueError('Existing run identity or recipe differs')
        else:
            save_json(path,value)
    if datasets is None:
        datasets,_,fingerprint=prepare_datasets(config,plan['seed'])
        if fingerprint!=plan['split_hash']:
            raise ValueError('Training split changed')
    optimizer=make_optimizer(model,config)
    scheduler=make_scheduler(optimizer,config)
    if (run/'last.pt').is_file():
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        _validate_checkpoint(state,metadata,config)
        history=state['history']
        if [row['epoch'] for row in history]!=list(range(state['epoch']+1)):
            raise ValueError('Checkpoint history chronology differs')
        validate_weights(model,state['model'])
        ema={name:value.to(device) for name,value in state['ema'].items()}
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']);restore_rng(state['rng'],device)
        best_state,trained_state=_restore_best_snapshots(run,state,metadata,config)
    else:
        initial=evaluate_model(model,datasets['selection'],config,plan['seed'],device,deadline_unix)
        history=[dict(epoch=0,augmentation_epoch=100,train_seconds=0.,
            **{'selection_'+k:v for k,v in initial.items()},**{'ema_selection_'+k:v for k,v in initial.items()})]
        weights=cpu_state(model)
        ema={name:value.to(device).clone() for name,value in weights.items()}
        best_state=dict(model=weights,epoch=0,source='raw',selection=initial,metadata=metadata,config=config)
        trained_state=None
        atomic_save(run/'last.pt',dict(best_state,ema={n:v.cpu() for n,v in ema.items()},history=history,
            best_snapshot=best_state,best_trained_snapshot=None,optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(device)))
        atomic_save(run/'best.pt',best_state)
        _write_history(run,history)
    if history[-1]['epoch']>=plan['epochs'] and (run/'summary.json').is_file():
        result=json.loads((run/'summary.json').read_text(encoding='utf-8'))
        if (result.get('status')=='complete' and result.get('completed_epochs')==plan['epochs']
                and all(result.get(k)==v for k,v in metadata.items())
                and result.get('best_epoch')==best_state['epoch']
                and result.get('best_source')==best_state['source']
                and result.get('best_selection')==best_state['selection']):
            return result
    status='complete'
    try:
        for epoch in range(history[-1]['epoch']+1,plan['epochs']+1):
            check_deadline(deadline_unix)
            sync(device);started=time.perf_counter()
            rates={group['role']+'_lr':group['lr'] for group in optimizer.param_groups}
            losses=train_one_epoch(model,loader(datasets['train'],config['dataset'],plan['seed'],100+epoch),
                config,plan['seed'],100+epoch,device,amp,dtype,optimizer,scaler,ema,deadline_unix)
            sync(device);seconds=time.perf_counter()-started
            raw=evaluate_model(model,datasets['selection'],config,plan['seed'],device,deadline_unix)
            raw_weights=cpu_state(model)
            validate_weights(model,ema)
            averaged=evaluate_model(model,datasets['selection'],config,plan['seed'],device,deadline_unix)
            validate_weights(model,raw_weights)
            row=dict(epoch=epoch,augmentation_epoch=100+epoch,train_seconds=seconds,**rates,**losses,
                **{'selection_'+k:v for k,v in raw.items()},**{'ema_selection_'+k:v for k,v in averaged.items()})
            if epoch==1 or epoch%5==0 or epoch==plan['epochs']:
                probe=evaluate_model(model,datasets['train_probe'],config,plan['seed'],device,deadline_unix)
                row.update({'train_probe_'+k:v for k,v in probe.items()})
            for source,weights,score in (('raw',raw_weights,raw),('ema',{n:v.detach().cpu().clone() for n,v in ema.items()},averaged)):
                candidate=dict(model=weights,epoch=epoch,source=source,selection=score,metadata=metadata,config=config)
                if trained_state is None or score['accuracy']>trained_state['selection']['accuracy']:
                    trained_state=candidate
                if score['accuracy']>best_state['selection']['accuracy']:
                    best_state=candidate
            history.append(row);scheduler.step()
            atomic_save(run/'last.pt',dict(model=raw_weights,ema={n:v.detach().cpu() for n,v in ema.items()},
                epoch=epoch,source='raw',selection=raw,metadata=metadata,config=config,history=history,
                best_snapshot=best_state,best_trained_snapshot=trained_state,optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(device)))
            atomic_save(run/'best.pt',best_state);atomic_save(run/'best_trained.pt',trained_state)
            _write_history(run,history)
            print(f'{job} {epoch}/{plan["epochs"]}: raw={raw["accuracy"]*100:.2f}% '
                f'EMA={averaged["accuracy"]*100:.2f}% best={best_state["selection"]["accuracy"]*100:.2f}% '
                f'{best_state["source"]}@{best_state["epoch"]}; train={seconds/60:.2f}min',flush=True)
    except BudgetExceeded:
        status='budget_exhausted'
        committed=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        _validate_checkpoint(committed,metadata,config)
        history=committed['history']
        best_state,trained_state=_restore_best_snapshots(run,committed,metadata,config)
    result=dict(metadata,status=status,completed_epochs=history[-1]['epoch'],planned_epochs=plan['epochs'],
        best_epoch=best_state['epoch'],best_source=best_state['source'],best_selection=best_state['selection'],
        best_trained_epoch=trained_state['epoch'] if trained_state else None,
        best_trained_source=trained_state['source'] if trained_state else None,
        best_trained_selection=trained_state['selection'] if trained_state else None,
        initial_selection={k:history[0]['selection_'+k] for k in ('accuracy','top5_accuracy','nll','n')},
        last_epoch_selection={k:history[-1]['selection_'+k] for k in ('accuracy','top5_accuracy','nll','n')},
        last_ema_selection={k:history[-1]['ema_selection_'+k] for k in ('accuracy','top5_accuracy','nll','n')},
        total_train_seconds=sum(row['train_seconds'] for row in history),
        interrupted_epoch_replays_on_resume=status!='complete',audit_partition_evaluated=False,
        official_validation_evaluated=False)
    save_json(run/'summary.json',result)
    del model,optimizer,scheduler,scaler,ema;gc.collect()
    if device=='cuda':
        torch.cuda.empty_cache()
    return result


def load_model(plan,training_dir,job,checkpoint='best.pt',sources=None):
    if checkpoint in ('best','last','best_trained'):
        checkpoint+='.pt'
    if checkpoint not in ('best.pt','last.pt','best_trained.pt','last_ema'):
        raise ValueError('Unsupported fusion mechanism checkpoint')
    actual='last.pt' if checkpoint=='last_ema' else checkpoint
    run=Path(training_dir)/job
    state=torch.load(run/actual,map_location='cpu',weights_only=False)
    model=build_model(plan,job,sources)
    metadata=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    _validate_checkpoint(state,metadata,job_config(plan,job))
    if _metadata(plan,job,model)!=metadata:
        raise ValueError('Analysis architecture or source identity changed')
    if checkpoint=='last_ema':
        state=dict(state,model=state['ema'],source='ema',selection=state['history'][-1] and
            {k:state['history'][-1]['ema_selection_'+k] for k in ('accuracy','top5_accuracy','nll','n')})
    validate_weights(model,state['model'])
    return model,state


def training_report(plan,training):
    training=Path(training);rows=[]
    for job in JOBS:
        path=training/job/'summary.json'
        if not path.is_file():
            continue
        summary=json.loads(path.read_text(encoding='utf-8'))
        rows.append(dict(job=job,status=summary['status'],completed_epochs=summary['completed_epochs'],
            planned_epochs=plan['epochs'],best_top1_pct=100*summary['best_selection']['accuracy'],
            best_epoch=summary['best_epoch'],best_source=summary['best_source'],best_nll=summary['best_selection']['nll'],
            best_trained_top1_pct=100*summary['best_trained_selection']['accuracy'] if summary['best_trained_selection'] else None,
            best_trained_epoch=summary['best_trained_epoch'],best_trained_source=summary['best_trained_source'],
            initial_top1_pct=100*summary['initial_selection']['accuracy'],
            last_top1_pct=100*summary['last_epoch_selection']['accuracy'],
            last_ema_top1_pct=100*summary['last_ema_selection']['accuracy'],
            params=summary['params'],trainable_params=summary['trainable_params'],
            train_minutes=summary['total_train_seconds']/60))
    pd.DataFrame(rows).to_csv(training/'comparison.csv',index=False)
    (training/'report.md').write_text('# Winning fusion: targeted mechanism controls\n\n'
        f'Three controls, each {plan["epochs"]} epochs. All CNN and ViT parameters update. '
        'The completed CLS+Linear winner is reused, never retrained. Each control changes '
        'one declared factor: independent ViT weights, inherited CNN classifier columns/bias, '
        'or final ViT readout. New ViT classifier columns and canonical post-construction RNG '
        'are shared exactly.\n\n'
        'Epoch-zero, best-trained, actual last raw and last EMA are distinct. '
        'Interrupted epochs replay from the last atomic completed epoch. All training '
        'views match indices101..130. Audit and official validation remain unused.\n',encoding='utf-8')
    return rows


def _archive_owned_failure(path,job=None):
    path=Path(path)
    if not path.exists():
        return
    record=json.loads(path.read_text(encoding='utf-8'))
    if record.get('protocol')!=PROTOCOL or (job is not None and record.get('job')!=job):
        raise ValueError('Refusing to resolve a foreign failure marker')
    archive=path.parent/'recovered_failures'
    archive.mkdir(exist_ok=True)
    path.replace(archive/(new_name(path.stem)+'.json'))


def run_suite(suite,deadline_unix=None):
    suite=Path(suite).resolve();plan=validate_resume(suite)
    started=time.perf_counter();training=suite/'training'
    if training.is_symlink():
        raise ValueError('Refusing symlink training root')
    marker=training/'suite.json'
    if marker.is_file():
        if json.loads(marker.read_text(encoding='utf-8'))!=plan:
            raise ValueError('Training root manifest differs from the recorded suite plan')
    elif training.exists() and any(training.iterdir()):
        raise ValueError('Refusing unrelated training root without a suite manifest')
    training.mkdir(exist_ok=True)
    save_json(marker,plan)
    outcomes={};status='complete';failure=None
    old_handler=None
    def interrupted(signum,frame):
        raise BudgetExceeded('Termination requested; retaining the last completed epoch')
    if hasattr(signal,'SIGTERM'):
        old_handler=signal.signal(signal.SIGTERM,interrupted)
    try:
        check_deadline(deadline_unix)
        sources=load_sources(plan)
        datasets,manifest,fingerprint=prepare_datasets(plan['config'],plan['seed'])
        if fingerprint!=plan['split_hash']:
            raise ValueError('Recorded split changed')
        save_json(training/'split_manifest.json',manifest)
        for job in JOBS:
            check_deadline(deadline_unix)
            save_json(suite/'status.json',dict(protocol=PROTOCOL,status='running',active_job=job,outcomes=outcomes))
            try:
                result=train_job(plan,job,training,sources,datasets,deadline_unix)
            except BudgetExceeded:
                raise
            except Exception as exc:
                status='failed'
                detail=dict(protocol=PROTOCOL,job=job,status='failed',
                    error_type=type(exc).__name__,traceback=traceback.format_exc())
                outcomes[job]=detail
                # Keep failure evidence outside a possibly not-yet-owned run dir,
                # so an initial construction failure does not block a safe retry.
                save_json(training/(job+'_failure.json'),detail)
                print(f'{job} failed ({type(exc).__name__}); continuing independent controls.',flush=True)
                continue
            outcomes[job]=dict(status=result['status'],completed_epochs=result['completed_epochs'])
            if result['status']=='complete':
                _archive_owned_failure(training/(job+'_failure.json'),job)
            training_report(plan,training)
            if result['status']!='complete':
                status='budget_exhausted';break
        if status!='budget_exhausted':
            from fusion_mechanism_diagnostics import run_fusion_mechanism_diagnostics
            result=run_fusion_mechanism_diagnostics(plan,training,suite/'feature_analysis',examples=plan['examples'],deadline_unix=deadline_unix)
            outcomes['feature_analysis']=dict(status=result.get('status','complete'))
            if result.get('status')!='complete' and status=='complete':
                status=result.get('status','failed')
    except BudgetExceeded as exc:
        status='budget_exhausted';failure=dict(error_type=type(exc).__name__,message=str(exc))
    except Exception as exc:
        status='failed';failure=dict(protocol=PROTOCOL,error_type=type(exc).__name__,traceback=traceback.format_exc())
        save_json(suite/'failure.json',failure)
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGTERM,old_handler)
    training_report(plan,training)
    unchanged=all(Path(p).is_file() and file_hash(p)==digest for p,digest in plan['source_files'].items())
    if not unchanged:
        status='failed'
    if status=='complete':
        if (any(outcomes.get(job,{}).get('status')!='complete'
                or outcomes.get(job,{}).get('completed_epochs')!=plan['epochs'] for job in JOBS)
                or outcomes.get('feature_analysis',{}).get('status')!='complete'):
            status='failed'
            failure=dict(protocol=PROTOCOL,error_type='IncompleteSuite',
                message='All declared training horizons and feature analysis must complete')
        else:
            _archive_owned_failure(suite/'failure.json')
    result=dict(protocol=PROTOCOL,status=status,outcomes=outcomes,failure=failure,
        elapsed_active_hours=(time.perf_counter()-started)/3600,source_checkpoints_unchanged=unchanged,
        shutdown_after_completion=False,audit_partition_evaluated=False,official_validation_evaluated=False,
        smoke=plan['smoke'])
    save_json(suite/'completion.json',result);save_json(suite/'status.json',result)
    (suite/'report.md').write_text('# Fusion mechanism controls\n\n'
        f'Status: **{status}**. Three targeted controls; no repeated winner training.\n\n'
        '- [Training](training/report.md)\n- [Feature evidence](feature_analysis/report.md)\n\n'
        'This experiment child performs no email or instance operation. '
        'The run_fusion_mechanism.py wrapper handles one combined delivery followed by managed STOP.\n',encoding='utf-8')
    return result


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cnn-suite',default=str(DEFAULT_CNN_SOURCE))
    parser.add_argument('--vit-suite',default=str(DEFAULT_VIT_SOURCE))
    parser.add_argument('--reference-suite',default=str(DEFAULT_REFERENCE))
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--data-dir');parser.add_argument('--num-workers',type=int)
    parser.add_argument('--allow-download',action='store_true')
    parser.add_argument('--examples',type=int,default=8);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output','--output-dir',default=str(ROOT/'runs/fusion_mechanism_tiny'))
    parser.add_argument('--resume');parser.add_argument('--deadline-unix',type=float)
    return parser


def main(argv=None):
    args=arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no fusion mechanism run started')
    if args.resume:
        plan=validate_resume(args.resume)
        if args.smoke or (args.epochs is not None and args.epochs!=plan['epochs']):
            raise ValueError('Resume retains its recorded horizon and smoke/full protocol')
        suite=Path(args.resume)
    else:
        suite=create_suite(args)
    result=run_suite(suite,args.deadline_unix)
    return 0 if result['status']=='complete' else 2


if __name__=='__main__':
    raise SystemExit(main())
