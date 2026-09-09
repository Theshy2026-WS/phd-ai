"""Tiny ImageNet mechanistic baselines and held-out model utility audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
import torchvision
import yaml
from tqdm import tqdm

from main import configure_runtime, configure_mixed_precision, build_scheduler
from models.audit_repvit import AuditRepViT
from datasets.utility_tiny import make_datasets, loader
from utils.profiling import estimate_model_flops, benchmark_inference


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def runtime(config, seed):
    config['training']['seed'] = seed
    configure_runtime(config['training'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enabled, dtype, _, scaler = configure_mixed_precision(config['training'], device)
    return device, enabled, dtype, scaler


def pass_epoch(model, batches, device, amp, dtype, smoothing=0., optimizer=None, scaler=None):
    model.train(optimizer is not None)
    totals = np.zeros(4, dtype=float)
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, _ in tqdm(batches, leave=False):
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=dtype, enabled=amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; stop before saving invalid checkpoint')
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scores = logits.float()
            totals += [loss.item() * len(y),
                       F.cross_entropy(scores, y, reduction='sum').item(),
                       (scores.argmax(1) == y).sum().item(), len(y)]
    return dict(loss=totals[0]/totals[3], nll=totals[1]/totals[3],
                accuracy=totals[2]/totals[3])


def train(args):
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
        config['training']['scheduler']['warmup_epochs'] = min(1, args.epochs - 1)
    if config['training']['epochs'] < 1:
        raise ValueError('epochs must be positive')
    device, amp, dtype, scaler = runtime(config, args.seed)
    datasets, manifest, fingerprint = make_datasets(config['dataset'], args.seed)
    name = f'{args.variant}_seed{args.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}'
    run = Path(args.output) / name
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    dump(run / 'split_manifest.json', manifest)
    model = AuditRepViT(args.variant, args.seed).to(device)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = 'unknown', None
    metadata = dict(variant=args.variant, seed=args.seed, split_hash=fingerprint,
                    git_commit=commit, git_dirty=dirty, torch=str(torch.__version__),
                    torchvision=torchvision.__version__, timm=timm.__version__,
                    device=torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
                    amp=amp, dtype=str(dtype), protocol='tiny-utility-v1-80-10-10',
                    params=sum(p.numel() for p in model.parameters()),
                    addon_params=sum(p.numel() for p in model.addon.parameters()),
                    forward_flops_estimate=estimate_model_flops(model, (3,64,64), device))
    dump(run / 'metadata.json', metadata)
    print(f'Run: {run}\n{json.dumps(metadata, indent=2)}', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                                 weight_decay=config['training']['weight_decay'])
    scheduler = build_scheduler(optimizer, config['training'])
    history, best = [], -1.
    start = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(config['training']['epochs']):
        sync(device)
        t = time.perf_counter()
        train_metrics = pass_epoch(model, loader(datasets['train'], config['dataset'], args.seed, epoch),
                                   device, amp, dtype, config['training']['label_smoothing'], optimizer, scaler)
        sync(device)
        train_seconds = time.perf_counter() - t
        t = time.perf_counter()
        selection = pass_epoch(model, loader(datasets['selection'], config['dataset'], args.seed),
                               device, amp, dtype)
        sync(device)
        row = dict(epoch=epoch+1, lr=optimizer.param_groups[0]['lr'], train_seconds=train_seconds,
                   selection_seconds=time.perf_counter()-t,
                   **{'train_'+k:v for k,v in train_metrics.items()},
                   **{'selection_'+k:v for k,v in selection.items()})
        history.append(row)
        scheduler.step()
        checkpoint = dict(model=model.state_dict(), config=config, metadata=metadata,
                          epoch=epoch+1, selection=selection)
        if selection['accuracy'] > best:
            best = selection['accuracy']
            torch.save(checkpoint, run / 'best.pt')
        torch.save(checkpoint, run / 'last.pt')
        pd.DataFrame(history).to_csv(run / 'epochs.csv', index=False)
        print(row, flush=True)
    peak = torch.cuda.max_memory_allocated() if device == 'cuda' else None
    training_wall = time.perf_counter()-start
    # Remove optimizer and gradients before measuring standalone inference allocations.
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler, checkpoint
    if device == 'cuda':
        torch.cuda.empty_cache()
    profile = {}
    for batch in (1, config['dataset']['batch_size']):
        profile[str(batch)] = benchmark_inference(
            model, (3,64,64), device, batch,
            config['profiling']['warmup_iterations'], config['profiling']['benchmark_iterations'], amp, dtype)
    dump(run / 'summary.json', dict(best_selection_accuracy=best,
         training_and_selection_wall_seconds=training_wall,
         peak_training_and_selection_bytes=peak, inference=profile,
         note='Unfused timm implementation; synthetic forward only; no data transfer included.'))
    print(f'Completed {run}. Audit uses best.pt, selected without audit labels.', flush=True)
    return run


def sample_metrics(logits, labels):
    scores = logits.float()
    logp = scores.log_softmax(1)
    p = logp.exp()
    top = p.topk(2, dim=1).values
    other = scores.clone()
    other.scatter_(1, labels[:, None], float('-inf'))
    return dict(pred=scores.argmax(1), confidence=top[:,0], entropy=-(p*logp).sum(1),
                uncertainty_margin=top[:,0]-top[:,1],
                true_margin=scores.gather(1,labels[:,None]).squeeze(1)-other.max(1).values,
                nll=F.nll_loss(logp, labels, reduction='none'),
                correct=scores.argmax(1).eq(labels),
                top5_correct=p.topk(5,dim=1).indices.eq(labels[:,None]).any(1))


def occlude(x, area):
    if not 0 <= area <= 1:
        raise ValueError('Occlusion area must be between 0 and 1')
    x = x.clone()
    side = round(x.shape[-1] * area**0.5)
    lo = (x.shape[-1]-side)//2
    x[...,lo:lo+side,lo:lo+side] = 0  # normalized dataset-mean fill
    return x


def audit(args):
    runs = [Path(x) for x in (args.cnn, args.conv, args.attention)]
    checkpoints = [torch.load(r/'best.pt', map_location='cpu', weights_only=False) for r in runs]
    variants = ['cnn','conv','attention']
    base = checkpoints[0]
    for variant, ck in zip(variants, checkpoints):
        if ck['metadata']['variant'] != variant:
            raise ValueError(f'Expected {variant} checkpoint')
        if ck['config'] != base['config'] or any(ck['metadata'][k] != base['metadata'][k]
                for k in ('seed','split_hash','protocol','timm','torch','torchvision','git_commit','amp','dtype')):
            raise ValueError('Checkpoint protocol/config/seed/code/runtime mismatch')
    config, seed = base['config'], base['metadata']['seed']
    device, amp, dtype, _ = runtime(config, seed)
    datasets, _, fingerprint = make_datasets(config['dataset'], seed)
    if fingerprint != base['metadata']['split_hash']:
        raise ValueError('Dataset manifest changed')
    output = Path(args.output) / f'audit_seed{seed}_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    for variant, ck, run in zip(variants, checkpoints, runs):
        model = AuditRepViT(variant, seed).to(device)
        model.load_state_dict(ck['model'])
        model.eval()
        rows = []
        raw_logits = []
        with torch.inference_mode():
            for area in config['audit']['occlusion']:
                for x, y, ids in tqdm(loader(datasets['audit'], config['dataset'], seed), desc=f'{variant} area={area}'):
                    x, y = occlude(x, area).to(device), y.to(device)
                    with torch.autocast(device, dtype=dtype, enabled=amp):
                        logits = model(x)
                    raw_logits.append(logits.float().cpu().numpy())
                    metrics = {k:v.cpu().tolist() for k,v in sample_metrics(logits,y).items()}
                    labels = y.cpu().tolist()
                    for i, sample_id in enumerate(ids):
                        rows.append(dict(sample_id=sample_id, target=labels[i], occlusion=area,
                                         **{k:v[i] for k,v in metrics.items()}))
        frame = pd.DataFrame(rows)
        frame.to_csv(output/f'{variant}_predictions.csv', index=False)
        np.savez_compressed(output/f'{variant}_logits.npz',
                            logits=np.concatenate(raw_logits),
                            sample_id=frame.sample_id.to_numpy(dtype=str),
                            target=frame.target.to_numpy(), occlusion=frame.occlusion.to_numpy())
        frames.append(frame)
        del model
    paired = pair_predictions(frames)
    paired.to_csv(output/'paired_utility.csv',index=False)
    groups = []
    calibration = []
    for variant, frame in zip(variants, frames):
        for area, group in frame.groupby('occlusion'):
            ece = 0.
            bins = np.minimum((group.confidence.to_numpy()*15).astype(int),14)
            for b in range(15):
                selected = group[bins == b]
                if len(selected):
                    ece += len(selected)/len(group)*abs(selected.confidence.mean()-selected.correct.mean())
            calibration.append(dict(variant=variant, occlusion=area, ece_15_bins=ece,
                                    nll=group.nll.mean(), accuracy=group.correct.mean(),
                                    top5=group.top5_correct.mean()))
    pd.DataFrame(calibration).to_csv(output/'calibration.csv', index=False)
    for area, group in paired.groupby('occlusion'):
        for reference in ('cnn','conv'):
            a, b = group['attention_correct'].astype(bool), group[f'{reference}_correct'].astype(bool)
            groups.append(dict(occlusion=area, reference=reference, count=len(group),
                attention_accuracy=a.mean(), reference_accuracy=b.mean(),
                corrected=(a & ~b).mean(), harmed=(~a & b).mean(), both_correct=(a & b).mean(),
                both_wrong=(~a & ~b).mean(), mean_delta_nll=group[f'utility_vs_{reference}'].mean(),
                oracle_accuracy=(a|b).mean()))
    pd.DataFrame(groups).to_csv(output/'groups.csv',index=False)
    # Fixed entropy bins (not fitted labels); exploratory, not a trained router.
    paired['entropy_bin'] = pd.cut(paired.cnn_entropy, np.linspace(0,np.log(200)+1e-6,6), include_lowest=True)
    paired.groupby(['occlusion','entropy_bin'], observed=True).agg(
        count=('sample_id','size'), utility_vs_cnn=('utility_vs_cnn','mean'),
        utility_vs_conv=('utility_vs_conv','mean')).to_csv(output/'uncertainty_groups.csv')
    dump(output/'provenance.json', dict(runs=[str(r.resolve()) for r in runs],
        checkpoint_sha256=[hashlib.sha256((r/'best.pt').read_bytes()).hexdigest() for r in runs],
        config=config, metadata=base['metadata'],
        interpretation='Between-model utility, not causal branch contribution. Oracle uses labels and is not deployable.'))
    print(f'Audit saved: {output}', flush=True)
    return output


def pair_predictions(frames):
    keys = ['sample_id','target','occlusion']
    out = None
    for variant, f in zip(('cnn','conv','attention'), frames):
        if f.duplicated(keys).any():
            raise ValueError('Duplicate sample keys')
        f = f.rename(columns={c:f'{variant}_{c}' for c in f.columns if c not in keys})
        if out is None:
            out = f
        else:
            out = out.merge(f, on=keys, how='outer', validate='one_to_one', indicator=True)
            if not (out['_merge'] == 'both').all():
                raise ValueError('Sample/label/occlusion mismatch')
            out = out.drop(columns='_merge')
    for reference in ('cnn','conv'):
        out[f'utility_vs_{reference}'] = out[f'{reference}_nll'] - out.attention_nll
        out[f'margin_gain_vs_{reference}'] = out.attention_true_margin - out[f'{reference}_true_margin']
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--config',default='configs/tiny_imagenet_utility.yaml')
    t.add_argument('--variant',choices=['cnn','conv','attention'],required=True)
    t.add_argument('--seed',type=int,default=42)
    t.add_argument('--epochs',type=int)
    t.add_argument('--output',default='runs/tiny_utility')
    a = sub.add_parser('audit')
    for name in ('cnn','conv','attention'):
        a.add_argument('--'+name, required=True, help='Run directory containing best.pt')
    a.add_argument('--output',default='runs/tiny_utility')
    s = sub.add_parser('suite', help='Train all three variants sequentially, then audit each seed')
    s.add_argument('--config',default='configs/tiny_imagenet_utility.yaml')
    s.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    s.add_argument('--epochs',type=int)
    s.add_argument('--output',default='runs/tiny_utility')
    args = p.parse_args()
    if args.command == 'suite':
        all_groups = []
        for seed in args.seeds:
            runs = {}
            for variant in ('cnn','conv','attention'):
                runs[variant] = str(train(argparse.Namespace(config=args.config,variant=variant,
                    seed=seed,epochs=args.epochs,output=args.output)))
            destination = audit(argparse.Namespace(**runs,output=args.output))
            groups = pd.read_csv(destination/'groups.csv')
            groups['seed'] = seed
            all_groups.append(groups)
        combined = pd.concat(all_groups, ignore_index=True)
        prefix = Path(args.output)/f'suite_{uuid.uuid4().hex[:8]}'
        combined.to_csv(str(prefix)+'_seeds.csv',index=False)
        combined.groupby(['occlusion','reference']).agg({
            k:['mean','std'] for k in ('attention_accuracy','reference_accuracy',
                                      'mean_delta_nll','corrected','harmed')
        }).to_csv(str(prefix)+'_mean_std.csv')
    else:
        (train if args.command == 'train' else audit)(args)


if __name__ == '__main__':
    main()
