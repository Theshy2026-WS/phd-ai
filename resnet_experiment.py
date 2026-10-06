"""Controlled ResNet-18 vs ResNet-18 + MHSA on native-resolution Tiny ImageNet."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import timm
import yaml
from tqdm import tqdm

from datasets.resnet_tiny import prepare_datasets, mix_batch
from datasets.utility_tiny import loader
from models.audit_resnet18 import AuditResNet18
from main import build_scheduler
from utility_experiment import runtime, sync, dump, sample_metrics, occlude
from utils.profiling import estimate_model_flops, benchmark_inference

PROTOCOL = 'tiny-resnet18-v1-80-10-10'
ROOT = Path(__file__).resolve().parent


def source_hashes():
    paths = ['resnet_experiment.py', 'models/audit_resnet18.py', 'datasets/resnet_tiny.py',
             'datasets/utility_tiny.py', 'datasets/tiny_imagenet.py', 'main.py',
             'utility_experiment.py', 'utils/profiling.py']
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def make_model(config, variant, seed):
    return AuditResNet18(variant, seed, heads=config['model']['attention_heads'],
                        residual_scale=config['model']['attention_residual_scale'])


def atomic_save(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def train_epoch(model, batches, config, seed, epoch, device, amp, dtype, optimizer, scaler):
    model.train()
    total, count = 0., 0
    for batch_index, (x, y, _) in enumerate(tqdm(batches, desc=f'Train {epoch}', leave=False)):
        x, y = x.to(device), y.to(device)
        x, targets = mix_batch(x, y, config['training'], seed, epoch, batch_index)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device, enabled=amp, dtype=dtype):
            logits = model(x)
            loss = -(targets * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'],
                                       error_if_nonfinite=not amp)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        count += len(y)
    return total/count


@torch.no_grad()
def evaluate(model, batches, device, amp, dtype, area=0., predictions=False):
    model.eval()
    totals = np.zeros(4)
    rows = []
    for x, y, ids in tqdm(batches, desc=f'Eval occlusion={area}', leave=False):
        x, y = occlude(x.to(device), area), y.to(device)
        with torch.autocast(device, enabled=amp, dtype=dtype):
            logits = model(x)
        m = sample_metrics(logits, y)
        if not torch.isfinite(m['nll']).all():
            raise RuntimeError('Non-finite evaluation loss')
        totals += [m['nll'].sum().item(), m['correct'].sum().item(),
                   m['top5_correct'].sum().item(), len(y)]
        if predictions:
            columns = {k: v.cpu().tolist() for k, v in m.items()}
            targets = y.cpu().tolist()
            rows.extend(dict(sample_id=sample_id, target=targets[i], occlusion=area,
                             **{k: v[i] for k, v in columns.items()}) for i, sample_id in enumerate(ids))
    return dict(nll=totals[0]/totals[3], accuracy=totals[1]/totals[3],
                top5_accuracy=totals[2]/totals[3]), pd.DataFrame(rows)


def train(args):
    resumed = getattr(args, 'resume', None)
    if resumed:
        run = Path(resumed)
        saved = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        config, metadata = saved['config'], saved['metadata']
        if metadata['source_hashes'] != source_hashes():
            raise ValueError('Source changed since checkpoint; restore its source before resuming')
        if args.epochs is not None and args.epochs != config['training']['epochs']:
            raise ValueError('Resume preserves the original total epochs and LR schedule')
        variant, seed = metadata['variant'], metadata['seed']
    else:
        config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
        if args.epochs is not None:
            config['training']['epochs'] = args.epochs
            config['training']['scheduler']['warmup_epochs'] = min(5, max(0, args.epochs-1))
        variant, seed = args.variant, args.seed
        if variant not in ('cnn', 'attention'):
            raise ValueError('Specify --variant cnn or attention')
        name = f'{variant}_seed{seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}'
        run = Path(args.output)/name
    t = config['training']
    if (t['epochs'] < 1 or config['dataset']['batch_size'] < 2
            or config['dataset']['batch_size'] % 2 or t['probe_every'] < 1
            or t['probe_per_class'] < 1):
        raise ValueError('Need positive epochs/probe interval and an even batch size for Mixup')
    device, amp, dtype, scaler = runtime(config, seed)
    datasets, manifest, fingerprint = prepare_datasets(config, seed)
    model = make_model(config, variant, seed).to(device)
    # No weight decay on normalization parameters or biases, for BOTH models.
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim > 1 else no_decay).append(p)
    optimizer = torch.optim.SGD([{'params': decay, 'weight_decay': t['weight_decay']},
                                 {'params': no_decay, 'weight_decay': 0.}],
                                lr=t['learning_rate'], momentum=t['momentum'], nesterov=True)
    scheduler = build_scheduler(optimizer, t)
    if resumed:
        if fingerprint != metadata['split_hash']:
            raise ValueError('Dataset split differs from checkpoint')
        current = dict(torch=str(torch.__version__), torchvision=torchvision.__version__,
                       timm=timm.__version__, amp=amp, dtype=str(dtype),
                       device=torch.cuda.get_device_name() if device=='cuda' else 'cpu')
        if any(metadata[k] != v for k,v in current.items()):
            raise ValueError('Resume requires the original software, GPU and precision settings')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        random.setstate(saved['rng']['python'])
        np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'])
        if device == 'cuda' and saved['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        history, best, start = saved['history'], saved['best'], saved['epoch'] + 1
    else:
        run.mkdir(parents=True, exist_ok=False)
        (run/'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        dump(run/'split_manifest.json', manifest)
        probe = datasets['train_probe']
        dump(run/'train_probe_ids.json', [Path(probe.folder.samples[i][0]).relative_to(
             probe.folder.root).as_posix() for i in probe.indices])
        try:
            commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
            dirty = bool(subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = 'unknown', None
        metadata = dict(protocol=PROTOCOL, variant=variant, seed=seed, split_hash=fingerprint,
                        git_commit=commit, git_dirty=dirty, source_hashes=source_hashes(),
                        torch=str(torch.__version__), torchvision=torchvision.__version__,
                        timm=timm.__version__, device=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
                        amp=amp, dtype=str(dtype), params=sum(p.numel() for p in model.parameters()),
                        addon_params=sum(p.numel() for p in model.addon.parameters()),
                        forward_flops_estimate=estimate_model_flops(model, (3,64,64), device))
        dump(run/'metadata.json', metadata)
        history, best, start = [], -1., 1
    print(f'Run: {run}', flush=True)
    print(json.dumps(metadata, indent=2), flush=True)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start, t['epochs']+1):
        sync(device); began = time.perf_counter()
        lr = optimizer.param_groups[0]['lr']
        loss = train_epoch(model, loader(datasets['train'], config['dataset'], seed, epoch),
                           config, seed, epoch, device, amp, dtype, optimizer, scaler)
        sync(device); train_seconds = time.perf_counter()-began
        began = time.perf_counter()
        selection, _ = evaluate(model, loader(datasets['selection'],config['dataset'],seed),device,amp,dtype)
        sync(device); selection_seconds = time.perf_counter()-began
        row = dict(epoch=epoch, lr=lr, train_soft_target_loss=loss,
                   train_seconds=train_seconds, selection_seconds=selection_seconds,
                   **{f'selection_{k}':v for k,v in selection.items()})
        if epoch == 1 or epoch % t['probe_every'] == 0 or epoch == t['epochs']:
            began = time.perf_counter()
            probe, _ = evaluate(model, loader(datasets['train_probe'],config['dataset'],seed),device,amp,dtype)
            sync(device)
            row.update({f'train_probe_{k}':v for k,v in probe.items()})
            row['probe_seconds'] = time.perf_counter()-began
            row['clean_accuracy_gap'] = probe['accuracy']-selection['accuracy']
        row['peak_training_memory_bytes'] = torch.cuda.max_memory_allocated() if device=='cuda' else 0
        improved = selection['accuracy'] > best
        if improved:
            best = selection['accuracy']
            atomic_save(run/'best.pt', dict(model=model.state_dict(), epoch=epoch,
                                           selection=selection, metadata=metadata, config=config))
        history.append(row)
        scheduler.step()
        atomic_save(run/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                    best=best, config=config, metadata=metadata, history=history,
                    rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                             cuda=torch.cuda.get_rng_state_all() if device=='cuda' else None)))
        pd.DataFrame(history).to_csv(run/'epochs.csv', index=False)
        print(row, flush=True)
    best_state = torch.load(run/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best_state['model'])
    profile = benchmark_inference(model, (3,64,64), device, config['dataset']['batch_size'],
                                   amp_enabled=amp, amp_dtype=dtype, **config['profiling'])
    summary = dict(**metadata, best_epoch=best_state['epoch'], best_selection=best_state['selection'],
                   total_train_seconds=sum(r['train_seconds'] for r in history),
                   total_selection_seconds=sum(r['selection_seconds'] for r in history),
                   total_probe_seconds=sum(r.get('probe_seconds',0) for r in history),
                   peak_training_memory_bytes=max(r['peak_training_memory_bytes'] for r in history),
                   inference=profile, completed_epochs=len(history))
    dump(run/'summary.json', summary)
    print(f'Completed: {run}\nBest selection accuracy: {best:.4f}', flush=True)
    return run


def pair_predictions(cnn, attention):
    keys = ['sample_id','occlusion']
    for frame in (cnn, attention):
        if frame.duplicated(keys).any():
            raise ValueError('Duplicate sample/occlusion keys')
    left, right = (f.set_index(keys).sort_index() for f in (cnn,attention))
    if not left.index.equals(right.index) or not left.target.equals(right.target):
        raise ValueError('Paired predictions must have identical sample IDs and labels')
    paired = left.add_suffix('_cnn').join(right.add_suffix('_attention')).reset_index()
    paired['utility_nll'] = paired.nll_cnn-paired.nll_attention
    paired['rescued'] = ~paired.correct_cnn.astype(bool) & paired.correct_attention.astype(bool)
    paired['harmed'] = paired.correct_cnn.astype(bool) & ~paired.correct_attention.astype(bool)
    return paired


def audit(cnn_run, attention_run, output):
    runs = [Path(cnn_run), Path(attention_run)]
    states = [torch.load(r/'best.pt', map_location='cpu', weights_only=False) for r in runs]
    for variant, state, run in zip(('cnn','attention'),states,runs):
        if state['metadata']['variant'] != variant or not (run/'summary.json').exists():
            raise ValueError(f'Need a completed {variant} run: {run}')
    for key in ('seed','split_hash','protocol','source_hashes','torch','torchvision','timm','amp','dtype','device'):
        if states[0]['metadata'][key] != states[1]['metadata'][key]:
            raise ValueError(f'Incompatible comparison: {key}')
    if states[0]['config'] != states[1]['config']:
        raise ValueError('Comparison requires identical configurations')
    if states[0]['metadata']['source_hashes'] != source_hashes():
        raise ValueError('Source changed since training; restore it before auditing')
    config = copy.deepcopy(states[0]['config'])
    seed = states[0]['metadata']['seed']
    device, amp, dtype, _ = runtime(config,seed)
    datasets, _, fingerprint = prepare_datasets(config,seed)
    if fingerprint != states[0]['metadata']['split_hash']:
        raise ValueError('Audit split mismatch')
    destination = Path(output)/f'audit_seed{seed}_{uuid.uuid4().hex[:8]}'
    destination.mkdir(parents=True,exist_ok=False)
    frames, rows = [], []
    for variant, state in zip(('cnn','attention'),states):
        model = make_model(config,variant,seed).to(device)
        model.load_state_dict(state['model'])
        predictions = []
        for area in config['audit']['occlusion']:
            metrics, frame = evaluate(model,loader(datasets['audit'],config['dataset'],seed),
                                      device,amp,dtype,area,predictions=True)
            predictions.append(frame)
            rows.append(dict(variant=variant,occlusion=area,**metrics))
        frame = pd.concat(predictions,ignore_index=True)
        frame.to_csv(destination/f'{variant}_predictions.csv',index=False)
        frames.append(frame)
        del model
    paired = pair_predictions(*frames)
    paired.to_csv(destination/'paired_utility.csv',index=False)
    pd.DataFrame(rows).to_csv(destination/'metrics.csv',index=False)
    paired.groupby('occlusion').agg(samples=('rescued','size'), rescued=('rescued','sum'),
                                   harmed=('harmed','sum'), mean_nll_utility=('utility_nll','mean')).to_csv(
                                       destination/'paired_summary.csv')
    dump(destination/'provenance.json',dict(protocol=PROTOCOL, seed=seed,split_hash=fingerprint,
          runs=[str(r.resolve()) for r in runs], source_hashes=source_hashes(),
          checkpoint_sha256=[hashlib.sha256((r/'best.pt').read_bytes()).hexdigest() for r in runs],
          note='Exploratory audit; official validation remains untouched. Occlusion is centered masking, not general robustness.'))
    print(f'Audit saved: {destination}',flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for command in ('train','suite'):
        p = sub.add_parser(command)
        p.add_argument('--config',default=str(ROOT/'configs/tiny_imagenet_resnet18.yaml'))
        p.add_argument('--output',default='runs/resnet18_utility')
        p.add_argument('--seed',type=int,default=42)
        p.add_argument('--epochs',type=int)
        if command == 'train':
            p.add_argument('--variant',choices=['cnn','attention'])
            p.add_argument('--resume',help='Run directory; restores saved config and epoch boundary state')
        else:
            p.add_argument('--cnn-run',help='Reuse a completed CNN run with the same configuration')
    p = sub.add_parser('audit')
    p.add_argument('--cnn',required=True)
    p.add_argument('--attention',required=True)
    p.add_argument('--output',default='runs/resnet18_utility')
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'audit':
        audit(args.cnn,args.attention,args.output)
    else:
        args.variant = 'cnn'
        if args.cnn_run:
            prior = torch.load(Path(args.cnn_run)/'best.pt',map_location='cpu',weights_only=False)
            expected = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
            if args.epochs is not None:
                expected['training']['epochs'] = args.epochs
                expected['training']['scheduler']['warmup_epochs'] = min(5,max(0,args.epochs-1))
            expected['training']['seed'] = args.seed
            if (prior['config'] != expected or prior['metadata']['variant'] != 'cnn'
                    or prior['metadata']['source_hashes'] != source_hashes()
                    or not (Path(args.cnn_run)/'summary.json').exists()):
                raise ValueError('--cnn-run must be a completed CNN run with this config, seed and source')
        cnn = args.cnn_run or train(args)
        args.variant = 'attention'
        attention = train(args)
        audit(cnn,attention,args.output)


if __name__ == '__main__':
    main()
