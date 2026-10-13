"""Literature-derived ResNet/AlterNet/BoTNet screening on Tiny ImageNet."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import time
import uuid

import numpy as np
import pandas as pd
import torch
import torchvision
import timm
import yaml
from tqdm import tqdm

from datasets.resnet_tiny import prepare_datasets
from datasets.utility_tiny import loader
from main import build_scheduler
from models.literature_resnet import VARIANTS, DEFAULT_VARIANTS, CONTROLS, build_model, architecture_description
from utils.literature_coatnet_compat import is_pre_coatnet_source
from utils.literature_fusion_compat import is_pre_fusion_source, is_pre_twostage_source
from resnet_experiment import atomic_save
from utils.literature_training import train_epoch
from utility_experiment import runtime, sync
from utils.profiling import benchmark_inference
from utils.literature_profiling import estimate_flops
from utils.literature_legacy import is_legacy_source
from utils.literature_eval_migration import is_attention_fixed_source

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'tiny-literature-v1-80-10-10'
DEFAULT_CONFIG = ROOT/'configs/tiny_imagenet_literature.yaml'


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def source_hashes():
    paths = ['literature_experiment.py', 'models/literature_resnet.py',
             'models/literature_alternet.py', 'utils/literature_profiling.py',
             'resnet_experiment.py', 'datasets/resnet_tiny.py', 'datasets/utility_tiny.py',
             'datasets/tiny_imagenet.py', 'main.py', 'utility_experiment.py',
             'utils/profiling.py', 'requirements-literature.txt', 'utils/literature_legacy.py',
             'utils/literature_eval_migration.py', 'models/literature_coatnet.py',
             'utils/literature_coatnet_compat.py', 'models/literature_acmix.py',
             'models/conformer_author.py', 'models/literature_conformer.py',
             'utils/literature_training.py', 'utils/literature_fusion_compat.py',
             'models/literature_twostage.py']
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}


def new_name(prefix):
    return f'{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}'


def load_config(args):
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    t = config['training']
    t['scheduler']['warmup_epochs'] = min(t['scheduler']['warmup_epochs'], max(0, t['epochs']-1))
    if t['epochs'] < 1 or config['dataset']['batch_size'] < 2 or config['dataset']['batch_size'] % 2:
        raise ValueError('Need positive epochs and an even batch size >=2')
    return config


def environment(device, amp, dtype):
    import einops
    return dict(torch=str(torch.__version__), torchvision=torchvision.__version__,
                timm=timm.__version__, einops=einops.__version__, amp=amp, dtype=str(dtype),
                device=torch.cuda.get_device_name() if device == 'cuda' else 'cpu')


def recipe_hash(config):
    recipe = copy.deepcopy(config)
    recipe['training'].pop('seed', None)
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def evaluate(model, batches, device, amp, dtype):
    """All models use full FP32 evaluation, independently of training AMP."""
    model.eval()
    loss, correct, top5, count = 0., 0, 0, 0
    for x, y, _ in tqdm(batches, desc='Evaluate classification', leave=False):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, enabled=False):
            logits = model(x.float())
        logits = logits.float()
        nll = torch.nn.functional.cross_entropy(logits, y, reduction='sum')
        if not torch.isfinite(nll):
            bad_state = [name for name, value in model.state_dict().items()
                         if value.is_floating_point() and not torch.isfinite(value).all()]
            raise RuntimeError('Non-finite FP32 evaluation loss; '
                               f'nonfinite state tensors={bad_state[:8]}; '
                               'No checkpoint is saved for this epoch.')
        loss += nll.item()
        correct += (logits.argmax(1) == y).sum().item()
        top5 += (logits.topk(5, dim=1).indices == y[:, None]).any(1).sum().item()
        count += len(y)
    if not count:
        raise ValueError('Empty evaluation dataset')
    return dict(nll=loss/count, accuracy=correct/count, top5_accuracy=top5/count)


def train_run(config, variant, seed, run, resume=False, deadline=None):
    """Return completion status; a time budget stops only at epoch boundaries."""
    began_run = time.perf_counter()
    run = Path(run)
    saved = None
    if resume:
        saved = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        config, metadata = saved['config'], saved['metadata']
        variant, seed = metadata['variant'], metadata['seed']
        if metadata['source_hashes'] != source_hashes():
            raise ValueError('Source changed: restore the recorded source before resuming')
    else:
        config = copy.deepcopy(config)
    device, amp, dtype, scaler = runtime(config, seed)
    datasets, manifest, fingerprint = prepare_datasets(config, seed)
    model = build_model(variant, seed).to(device)
    t = config['training']
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        # Positional embeddings, biases and normalization parameters have no decay.
        group = no_decay if p.ndim <= 1 or any(k in name for k in ('pos_embedding', 'rel_h', 'rel_w', 'rel_pos', 'cls_token')) else decay
        group.append(p)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': t['weight_decay']},
                                   {'params': no_decay, 'weight_decay': 0.}],
                                  lr=t['learning_rate'])
    scheduler = build_scheduler(optimizer, t)
    if saved:
        if fingerprint != metadata['split_hash'] or environment(device, amp, dtype) != metadata['environment']:
            raise ValueError('Resume requires the same split, software, device and precision')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        random.setstate(saved['rng']['python'])
        np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        history, best = saved['history'], saved['best']
        previous_wall = saved['elapsed_wall_seconds']
        if metadata.get('revalidate_best_fp32', False):
            # Re-score the saved candidate without retraining or touching optimizer
            # state. Earlier unretained checkpoints cannot be retrospectively ranked.
            candidate = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(candidate['model'])
            candidate['selection'] = evaluate(model, loader(datasets['selection'], config['dataset'], seed),
                                              device, False, dtype)
            best = candidate['selection']['accuracy']
            metadata['revalidate_best_fp32'] = False
            candidate['metadata'] = copy.deepcopy(metadata)
            atomic_save(run/'best.pt', candidate)
            model.load_state_dict(saved['model'])
            saved.update(best=best, metadata=metadata)
            atomic_save(run/'last.pt', saved)
            save_json(run/'metadata.json', metadata)
            random.setstate(saved['rng']['python'])
            np.random.set_state(saved['rng']['numpy'])
            torch.set_rng_state(saved['rng']['torch'])
            if device == 'cuda':
                torch.cuda.set_rng_state_all(saved['rng']['cuda'])
            del candidate
            print(f'Revalidated saved best using FP32: {best:.4f}; resume at epoch {len(history)+1}', flush=True)
        del saved
    else:
        run.mkdir(parents=True, exist_ok=True)
        if (run/'last.pt').exists() or (run/'summary.json').exists():
            raise ValueError('Refusing to overwrite a run; use --resume')
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = 'unknown', None
        metadata = dict(protocol=PROTOCOL, variant=variant, seed=seed, split_hash=fingerprint,
                        source_hashes=source_hashes(), git_commit=commit, git_dirty=dirty,
                        environment=environment(device, amp, dtype), recipe_hash=recipe_hash(config),
                        architecture=architecture_description(variant), evaluation_precision='float32',
                        inference_precision='float32',
                        params=sum(p.numel() for p in model.parameters()),
                        forward_flops_estimate=estimate_flops(model, device),
                        flops_convention='1 MAC = 2 FLOPs; Conv/Linear/QK/AV/relative dot products; excludes norm, softmax and elementwise ops')
        (run/'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        save_json(run/'metadata.json', metadata)
        save_json(run/'split_manifest.json', manifest)
        history, best, previous_wall = [], -1., 0.
    print(f'Run: {run}\nVariant: {variant}; epochs: {t["epochs"]}; params: {metadata["params"]:,}; '
          f'forward FLOPs (estimate): {metadata["forward_flops_estimate"]:,}', flush=True)
    if t['epochs'] <= 15:
        print('This short schedule is screening, not a converged paper reproduction.', flush=True)
    for epoch in range(len(history)+1, t['epochs']+1):
        if deadline is not None and history:
            estimate = np.mean([r['train_seconds']+r['selection_seconds'] for r in history[-3:]])*1.15
            if time.perf_counter()+estimate > deadline:
                print(f'Paused at epoch {epoch-1}; checkpoint: {run}/last.pt', flush=True)
                return False
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        sync(device)
        started = time.perf_counter()
        lr = optimizer.param_groups[0]['lr']
        loss = train_epoch(model, loader(datasets['train'], config['dataset'], seed, epoch),
                           config, seed, epoch, device, amp, dtype, optimizer, scaler)
        sync(device)
        train_seconds = time.perf_counter()-started
        peak = torch.cuda.max_memory_allocated() if device == 'cuda' else 0
        started = time.perf_counter()
        selection = evaluate(model, loader(datasets['selection'], config['dataset'], seed), device, amp, dtype)
        sync(device)
        row = dict(epoch=epoch, lr=lr, train_soft_target_loss=loss, train_seconds=train_seconds,
                   evaluation_precision='float32',
                   selection_seconds=time.perf_counter()-started, peak_training_memory_bytes=peak,
                   **{f'selection_{k}': v for k, v in selection.items()})
        if selection['accuracy'] > best:
            best = selection['accuracy']
            atomic_save(run/'best.pt', dict(model=model.state_dict(), epoch=epoch, selection=selection,
                                           metadata=metadata, config=config))
        history.append(row)
        scheduler.step()
        atomic_save(run/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), history=history,
                    best=best, config=config, metadata=metadata, epoch=epoch,
                    elapsed_wall_seconds=previous_wall+time.perf_counter()-began_run,
                    rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                             cuda=torch.cuda.get_rng_state_all() if device == 'cuda' else None)))
        pd.DataFrame(history).to_csv(run/'epochs.csv', index=False)
        remaining = (t['epochs']-epoch)*np.mean([r['train_seconds']+r['selection_seconds'] for r in history[-3:]])
        print(f'Epoch {epoch}/{t["epochs"]}: selection Top-1={selection["accuracy"]:.4f}, '
              f'Top-5={selection["top5_accuracy"]:.4f}, train={train_seconds:.1f}s, '
              f'approx remaining training+selection={remaining/60:.1f} min', flush=True)

    # Release optimizer state before measuring inference; never count it as model memory.
    model.zero_grad(set_to_none=True)
    del optimizer, scheduler, scaler
    best_state = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best_state['model'])
    best_epoch, best_selection = best_state['epoch'], best_state['selection']
    del best_state
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    started = time.perf_counter()
    heldout = evaluate(model, loader(datasets['audit'], config['dataset'], seed), device, amp, dtype)
    sync(device)
    heldout_seconds = time.perf_counter()-started
    profiles = {}
    for batch_size in (1, config['dataset']['batch_size']):
        profiles[str(batch_size)] = benchmark_inference(model, (3, 64, 64), device, batch_size,
                            amp_enabled=False, amp_dtype=dtype, **config['profiling'])
    summary = dict(**metadata, best_epoch=best_epoch, best_selection=best_selection,
                   heldout=heldout, completed_epochs=len(history),
                   total_train_seconds=sum(r['train_seconds'] for r in history),
                   total_selection_seconds=sum(r['selection_seconds'] for r in history),
                   heldout_seconds=heldout_seconds,
                   total_experiment_seconds=previous_wall+time.perf_counter()-began_run,
                   peak_training_memory_bytes=max(r['peak_training_memory_bytes'] for r in history),
                   inference=profiles, inference_batch_size=config['dataset']['batch_size'])
    save_json(run/'summary.json', summary)
    print(f'Completed: {run}\nHeld-out Top-1={heldout["accuracy"]:.4f}; '
          f'train time={summary["total_train_seconds"]:.1f}s', flush=True)
    return True


def comparable_source(summary):
    hashes = summary['source_hashes']
    if summary['variant'] in DEFAULT_VARIANTS + ('coatnet0', 'acmix50', 'conformer_tiny') and is_pre_twostage_source(hashes):
        return source_hashes()
    if summary['variant'] in DEFAULT_VARIANTS + ('coatnet0',) and is_pre_fusion_source(hashes):
        return source_hashes()
    # Additive CoAtNet registration leaves the four completed 4506e0d models
    # unchanged. Comparison only: never relax checkpoint resume checks.
    if summary['variant'] in DEFAULT_VARIANTS and is_pre_coatnet_source(hashes):
        return source_hashes()
    # The precision repair does not alter CNN forward/training or its recipe.
    # Preserve the original provenance; never rewrite an old run as current.
    if (summary['variant'] in ('resnet50', 'preact50')
            and is_legacy_source(hashes)):
        return source_hashes()
    return hashes


def compare(runs, output):
    summaries = [(Path(r), json.loads((Path(r)/'summary.json').read_text())) for r in runs]
    if not summaries:
        return
    for _, s in summaries:
        if s['protocol'] != PROTOCOL:
            raise ValueError('Wrong experiment protocol')
        for key in ('split_hash', 'recipe_hash', 'environment', 'completed_epochs',
                    'evaluation_precision', 'inference_precision'):
            if s.get(key, 'legacy AMP') != summaries[0][1].get(key, 'legacy AMP'):
                raise ValueError(f'Incompatible runs: {key}')
        if comparable_source(s) != comparable_source(summaries[0][1]):
            raise ValueError('Incompatible runs: source_hashes')
    pairs = [(s['variant'], s['seed']) for _, s in summaries]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Only one run per variant/seed may be compared')
    scores = {(s['variant'], s['seed']): s['heldout']['accuracy'] for _, s in summaries}
    rows = []
    for run, s in summaries:
        control = CONTROLS.get(s['variant'])
        base = scores.get((control, s['seed']))
        rows.append(dict(variant=s['variant'], seed=s['seed'], epochs=s['completed_epochs'],
                         best_epoch=s['best_epoch'], selection_top1_pct=100*s['best_selection']['accuracy'],
                         heldout_top1_pct=100*s['heldout']['accuracy'],
                         heldout_top5_pct=100*s['heldout']['top5_accuracy'],
                         control=control, delta_vs_control_pp=None if base is None else 100*(s['heldout']['accuracy']-base),
                         params=s['params'], forward_gflops_estimate=s['forward_flops_estimate']/1e9,
                         train_minutes=s['total_train_seconds']/60,
                         experiment_minutes=s['total_experiment_seconds']/60,
                         latency_batch1_ms=s['inference']['1']['latency_per_batch_ms'],
                         throughput_images_per_second=s['inference'][str(s['inference_batch_size'])]['throughput_images_per_second'],
                         throughput_batch_size=s['inference_batch_size'],
                         git_commit=s['git_commit'],
                         evaluation_precision=s.get('evaluation_precision', 'legacy AMP'),
                         inference_precision=s.get('inference_precision', 'legacy AMP'),
                         resumed_legacy_training=bool(s.get('evaluation_migration')),
                         attention_precision=s['architecture'].get('attention_precision', 'outer autocast'),
                         reused_legacy_cnn=is_legacy_source(s['source_hashes']), run=str(run.resolve())))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(frame.drop(columns=['run']).to_string(index=False), flush=True)
    print(f'Comparison saved: {output}', flush=True)


def repair_suite(args, deadline):
    """New suite, preserved old files; reuse only completed, verified CNN controls."""
    original = Path(args.from_suite).resolve()
    old = json.loads((original/'suite.json').read_text())
    if not is_legacy_source(old['source_hashes']):
        raise ValueError('Repair only supports the exact ff14146 source snapshot; no unknown edits')
    jobs = []
    for job in old['jobs']:
        if job['variant'] not in VARIANTS:
            raise ValueError('Unknown variant in original suite')
        run = (original/job['directory']).resolve()
        if not run.is_relative_to(original):
            raise ValueError('Original run must be inside its suite directory')
        updated = dict(job, directory=f'{job["variant"]}_seed{job["seed"]}')
        if job['variant'] in ('resnet50', 'preact50') and (run/'summary.json').exists():
            summary = json.loads((run/'summary.json').read_text())
            if (summary['protocol'] != PROTOCOL or not is_legacy_source(summary['source_hashes'])
                    or summary['variant'] != job['variant'] or summary['seed'] != job['seed']
                    or summary['recipe_hash'] != recipe_hash(old['config'])
                    or summary['completed_epochs'] != old['config']['training']['epochs']):
                raise ValueError(f'Incompatible CNN result: {run}')
            updated['reuse_run'] = str(run)
        jobs.append(updated)
    if not jobs:
        raise ValueError('Original suite has no jobs')
    if len({(j['variant'], j['seed']) for j in jobs}) != len(jobs):
        raise ValueError('Duplicate jobs in original suite')
    # Reject environment/split changes before spending time training the next model.
    device, amp, dtype, _ = runtime(copy.deepcopy(old['config']), jobs[0]['seed'])
    _, _, fingerprint = prepare_datasets(old['config'], jobs[0]['seed'])
    for job in jobs:
        if 'reuse_run' in job:
            summary = json.loads((Path(job['reuse_run'])/'summary.json').read_text())
            if summary['environment'] != environment(device, amp, dtype) or summary['split_hash'] != fingerprint:
                raise ValueError('Reusing CNN requires its original environment and dataset split')
    suite = Path(args.output)/new_name('suite_fp32fix')
    suite.mkdir(parents=True, exist_ok=False)
    save_json(suite/'suite.json', dict(config=old['config'], source_hashes=source_hashes(), jobs=jobs,
              repair=dict(original_suite=str(original), original_source_hashes=old['source_hashes'],
                          note='Reuse verified completed CNNs unchanged; restart attention models from epoch 1 with precision fix')))
    for job in jobs:
        print(f'{job["variant"]}: ' + (f'reuse {job["reuse_run"]}' if 'reuse_run' in job else 'restart from epoch 1'), flush=True)
    args.resume, args.epochs = str(suite), None
    run_suite(args, deadline)


def job_path(suite, job):
    return Path(job['reuse_run']) if 'reuse_run' in job else suite/job['directory']


def migrate_fp32_evaluation(args, deadline):
    """Clone a known suite and resume unchanged training with FP32 evaluation."""
    original = Path(args.from_suite).resolve()
    old = json.loads((original/'suite.json').read_text())
    if not is_attention_fixed_source(old['source_hashes']):
        raise ValueError('FP32 evaluation migration requires the exact 85dd1e7 suite source')
    jobs = old['jobs']
    if not jobs or len({(j['variant'], j['seed']) for j in jobs}) != len(jobs):
        raise ValueError('Empty or duplicate jobs')
    config = old['config']
    device, amp, dtype, _ = runtime(copy.deepcopy(config), jobs[0]['seed'])
    _, _, fingerprint = prepare_datasets(config, jobs[0]['seed'])
    # Verify every saved source and recipe before writing anything.
    origins = []
    for job in jobs:
        if job['variant'] not in VARIANTS or not isinstance(job['seed'], int):
            raise ValueError('Invalid job')
        run = job_path(original, job).resolve()
        if (run/'last.pt').exists():
            state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
            meta = state['metadata']
            known = is_attention_fixed_source(meta['source_hashes']) or (
                job['variant'] in ('resnet50', 'preact50') and is_legacy_source(meta['source_hashes']))
            if (not known or meta['protocol'] != PROTOCOL or meta['variant'] != job['variant']
                    or meta['seed'] != job['seed'] or recipe_hash(state['config']) != recipe_hash(config)
                    or meta['environment'] != environment(device, amp, dtype) or meta['split_hash'] != fingerprint
                    or state['epoch'] != len(state['history']) or not (run/'best.pt').exists()):
                raise ValueError(f'Incompatible checkpoint: {run}')
            if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state['model'].values()):
                raise ValueError(f'Checkpoint has non-finite model state: {run}')
            del state
        elif (run/'summary.json').exists():
            raise ValueError(f'Completed run is missing last.pt: {run}')
        origins.append(run)
    suite = Path(args.output)/new_name('suite_fp32eval')
    suite.mkdir(parents=True, exist_ok=False)
    migrated_jobs = []
    current_hashes = source_hashes()
    current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    current_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    for job, origin in zip(jobs, origins):
        migrated = dict(variant=job['variant'], seed=job['seed'], directory=f'{job["variant"]}_seed{job["seed"]}')
        migrated_jobs.append(migrated)
        if not (origin/'last.pt').exists():
            continue
        destination = suite/migrated['directory']
        destination.mkdir()
        for name in ('config.yaml', 'epochs.csv', 'split_manifest.json'):
            if (origin/name).exists():
                shutil.copy2(origin/name, destination/name)
        last = torch.load(origin/'last.pt', map_location='cpu', weights_only=False)
        origin_metadata = copy.deepcopy(last['metadata'])
        save_json(destination/'origin_metadata.json', origin_metadata)
        meta = copy.deepcopy(origin_metadata)
        meta.update(source_hashes=current_hashes, git_commit=current_commit, git_dirty=current_dirty,
                    evaluation_precision='float32', inference_precision='float32', revalidate_best_fp32=True,
                    evaluation_migration=dict(original_run=str(origin), original_git_commit=origin_metadata['git_commit'],
                                              original_source_hashes=origin_metadata['source_hashes'],
                                              completed_epochs_before_migration=last['epoch'],
                                              selection_note='Earlier epochs selected under AMP; saved best is re-scored in FP32. Unretained checkpoints cannot be re-ranked. Training unchanged.'))
        last['metadata'] = meta
        for row in last['history']:
            row.setdefault('evaluation_precision', 'legacy AMP')
        atomic_save(destination/'last.pt', last)
        save_json(destination/'metadata.json', meta)
        del last
        best = torch.load(origin/'best.pt', map_location='cpu', weights_only=False)
        best['metadata'] = copy.deepcopy(meta)
        atomic_save(destination/'best.pt', best)
        del best
        print(f'{job["variant"]}: preserved checkpoint from {origin}; no training restart', flush=True)
    save_json(suite/'suite.json', dict(config=config, source_hashes=current_hashes, jobs=migrated_jobs,
                                     migration=dict(original_suite=str(original), evaluation_precision='float32')))
    args.resume, args.epochs = str(suite), None
    run_suite(args, deadline)


def run_suite(args, deadline):
    if args.resume:
        suite = Path(args.resume)
        plan = json.loads((suite/'suite.json').read_text())
        if plan['source_hashes'] != source_hashes():
            if is_attention_fixed_source(plan['source_hashes']):
                raise ValueError('Use migrate-evaluation --from-suite '
                                 f'"{suite}" to preserve completed training and switch evaluation to FP32')
            if is_legacy_source(plan['source_hashes']):
                raise ValueError('Old precision version: use repair-suite --from-suite '
                                 f'"{suite}" to preserve CNN results and restart attention safely')
            raise ValueError('Source changed since suite creation')
        if args.epochs is not None and args.epochs != plan['config']['training']['epochs']:
            raise ValueError('Resume cannot change the original epoch schedule')
    else:
        suite = Path(args.output)/new_name('suite')
        suite.mkdir(parents=True, exist_ok=False)
        plan = dict(config=load_config(args), source_hashes=source_hashes(),
                    jobs=[dict(variant=v, seed=s, directory=f'{v}_seed{s}')
                          for s in args.seeds for v in args.variants])
        save_json(suite/'suite.json', plan)
    print(f'Suite: {suite}', flush=True)
    for job in plan['jobs']:
        run = job_path(suite, job)
        if 'reuse_run' in job and not (run/'summary.json').exists():
            raise ValueError(f'Reused CNN was moved or removed: {run}')
        if (run/'summary.json').exists():
            continue
        if deadline is not None and time.perf_counter() >= deadline:
            break
        complete = train_run(plan['config'], job['variant'], job['seed'], run,
                             resume=(run/'last.pt').exists(), deadline=deadline)
        completed = [job_path(suite, j) for j in plan['jobs']
                     if (job_path(suite, j)/'summary.json').exists()]
        compare(completed, suite/'comparison.csv')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not complete:
            break
    done = sum((job_path(suite, j)/'summary.json').exists() for j in plan['jobs'])
    compare([job_path(suite, j) for j in plan['jobs']
             if (job_path(suite, j)/'summary.json').exists()], suite/'comparison.csv')
    print(f'Suite completed {done}/{len(plan["jobs"])} models. '
          f'Resume: python literature_experiment.py suite --resume "{suite}"', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'suite'):
        p = sub.add_parser(command)
        p.add_argument('--config', default=str(DEFAULT_CONFIG))
        p.add_argument('--output', default='runs/literature_tiny')
        p.add_argument('--epochs', type=int, help='Override config epochs; fixed total schedule on resume')
        p.add_argument('--resume', help='Existing run directory (train) or suite directory (suite)')
        p.add_argument('--max-minutes', type=float, help='Soft wall-clock budget; pause at epoch boundaries')
        if command == 'train':
            p.add_argument('--variant', choices=VARIANTS)
            p.add_argument('--seed', type=int, default=42)
        else:
            p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(DEFAULT_VARIANTS))
            p.add_argument('--seeds', '--seed', nargs='+', type=int, default=[42])
    p = sub.add_parser('compare')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--output', default='runs/literature_tiny/comparison.csv')
    p = sub.add_parser('repair-suite', help='Recover ff14146 suite after the FP16 attention fix')
    p.add_argument('--from-suite', required=True)
    p.add_argument('--output', default='runs/literature_tiny')
    p.add_argument('--max-minutes', type=float)
    p = sub.add_parser('migrate-evaluation', help='Resume 85dd1e7 training with complete FP32 evaluation')
    p.add_argument('--from-suite', required=True)
    p.add_argument('--output', default='runs/literature_tiny')
    p.add_argument('--max-minutes', type=float)
    args = parser.parse_args()
    minutes = getattr(args, 'max_minutes', None)
    if minutes is not None and minutes <= 0:
        parser.error('--max-minutes must be positive')
    deadline = time.perf_counter()+minutes*60 if minutes is not None else None
    try:
        if args.command == 'compare':
            compare(args.runs, args.output)
        elif args.command == 'repair-suite':
            parser.error('repair-suite is superseded. For an existing suite_fp32fix_... from 85dd1e7, '
                         'use migrate-evaluation --from-suite PATH; for fresh runs use suite.')
        elif args.command == 'migrate-evaluation':
            migrate_fp32_evaluation(args, deadline)
        elif args.command == 'suite':
            if len(set(args.variants)) != len(args.variants) or len(set(args.seeds)) != len(args.seeds):
                parser.error('Duplicate variants or seeds')
            run_suite(args, deadline)
        else:
            if not args.resume and not args.variant:
                parser.error('train needs --variant or --resume')
            config = load_config(args) if not args.resume else None
            if args.resume and args.epochs is not None:
                original = yaml.safe_load((Path(args.resume)/'config.yaml').read_text())
                if args.epochs != original['training']['epochs']:
                    parser.error('Resume cannot change the original epoch schedule')
            run = Path(args.resume) if args.resume else Path(args.output)/new_name(f'{args.variant}_seed{args.seed}')
            if (run/'summary.json').exists():
                print(f'Already complete: {run}', flush=True)
                return
            train_run(config, args.variant, args.seed, run, resume=bool(args.resume), deadline=deadline)
            print(f'Resume if paused: python literature_experiment.py train --resume "{run}"', flush=True)
    except KeyboardInterrupt:
        print('\nInterrupted. Use the printed run/suite directory with --resume. '
              'The last completed epoch is saved; the current incomplete epoch will repeat.', flush=True)
        raise SystemExit(130)


if __name__ == '__main__':
    main()
