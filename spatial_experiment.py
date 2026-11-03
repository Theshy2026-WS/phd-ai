"""Paired native64 CNN geometry study with clean training probes.

Both models train from scratch. Only selection labels choose checkpoints.
Neither the audit partition nor official validation is evaluated.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import yaml

from baseline_diagnostics import load_run, clean_datasets, class_labels, class_comparison
from correction_experiment import file_hash
from datasets.resnet_tiny import prepare_datasets
from datasets.utility_tiny import UtilityDataset, loader
from literature_experiment import (ROOT, new_name, save_json, recipe_hash, environment,
                                  evaluate, source_hashes as literature_sources)
from main import build_scheduler
from models.attention_correction import state_digest
from models.spatial_preact import build_spatial_model, spatial_description, SPATIAL_VARIANTS
from resnet_experiment import atomic_save
from utility_experiment import runtime, sync
from utils.baseline_diagnostics import evaluate_clean
from utils.literature_profiling import estimate_flops
from utils.literature_training import train_epoch
from utils.profiling import benchmark_inference


PROTOCOL = 'tiny-spatial-cnn-v1-80-10-10'
NEW_SOURCES = ('spatial_experiment.py', 'models/spatial_preact.py',
               'baseline_diagnostics.py', 'utils/baseline_diagnostics.py')


def source_hashes():
    return {**literature_sources(), **{name: file_hash(ROOT/name) for name in NEW_SOURCES}}


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda'])


def execution_environment(config, seed):
    device, amp, dtype, _ = runtime(copy.deepcopy(config), seed)
    return environment(device, amp, dtype)


def make_plan(args):
    if args.probe_every < 1 or not 1 <= args.probe_per_class <= 400 or args.examples < 0:
        raise ValueError('Need positive probe interval, 1..400 probe images/class and nonnegative examples')
    epochs = 1 if args.smoke else (args.epochs if args.epochs is not None else 100)
    if epochs < 1 or len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        raise ValueError('Need positive epochs and unique seeds/variants')
    if any(seed < 0 for seed in args.seeds):
        raise ValueError('Seeds must be nonnegative')
    parent, summary, state, checkpoint_hash = load_run(args.base_run)
    if summary['completed_epochs'] != 100:
        raise ValueError('Reference must be the original completed 100-epoch CNN')
    output = Path(args.output).resolve()
    if output == parent or parent in output.parents:
        raise ValueError('Output must be outside the reference training run')
    config = copy.deepcopy(state['config'])
    data = Path(args.data_dir or config['dataset']['data_dir']).expanduser()
    config['dataset']['data_dir'] = str(data.resolve() if data.is_absolute() else (ROOT/data).resolve())
    config['dataset']['download'] = bool(args.allow_download)
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    if config['dataset']['num_workers'] < 0:
        raise ValueError('num_workers must be nonnegative')
    config['training'].update(epochs=epochs, early_stopping_patience=0,
                              probe_per_class=args.probe_per_class)
    config['training']['scheduler']['warmup_epochs'] = min(
        config['training']['scheduler']['warmup_epochs'], epochs-1)
    # This checks the original complete 80/10/10 split before starting compute.
    _, _, _, fingerprint, _ = clean_datasets(config, summary['seed'], summary['split_hash'], 0)
    return dict(protocol=PROTOCOL, config=config, source_hashes=source_hashes(),
        execution_environment=execution_environment(config, args.seeds[0]),
        base_run=str(parent), parent_checkpoint_sha256=checkpoint_hash,
        parent_best_epoch=summary['best_epoch'], historical_reference=summary['best_selection'],
        split_hash=fingerprint, fresh_training=True, smoke=args.smoke,
        probe_every=args.probe_every, probe_per_class=args.probe_per_class, examples=args.examples,
        jobs=[dict(variant=variant, seed=seed, directory=f'{variant}_seed{seed}')
              for seed in args.seeds for variant in args.variants],
        interpretation='Geometry/downsampling-timing intervention, not a novelty or attention claim')


def create_suite(args, plan=None):
    plan = make_plan(args) if plan is None else plan
    root = Path(args.output).resolve()
    parent = Path(plan['base_run']).resolve()
    if root == parent or parent in root.parents:
        raise ValueError('Output must be outside the reference training run')
    suite = root/new_name('smoke' if plan['smoke'] else 'suite')
    suite.mkdir(parents=True, exist_ok=False)
    save_json(suite/'suite.json', plan)
    print(f'Suite: {suite}', flush=True)
    return suite


def validate_resume(path):
    suite = Path(path).resolve()
    plan = json.loads((suite/'suite.json').read_text(encoding='utf-8'))
    if plan.get('protocol') != PROTOCOL or plan.get('source_hashes') != source_hashes():
        raise ValueError('Resume requires the recorded spatial protocol and source files')
    if plan['execution_environment'] != execution_environment(plan['config'], plan['jobs'][0]['seed']):
        raise ValueError('Resume requires the recorded software, GPU and precision environment')
    clean_datasets(plan['config'], plan['jobs'][0]['seed'], plan['split_hash'], 0)
    for job in plan['jobs']:
        run = suite/job['directory']
        for name in ('last.pt', 'best.pt'):
            if not (run/name).exists():
                continue
            state = torch.load(run/name, map_location='cpu', weights_only=False)
            meta = state['metadata']
            if (meta['source_hashes'] != plan['source_hashes'] or meta['split_hash'] != plan['split_hash']
                    or meta['variant'] != job['variant'] or meta['seed'] != job['seed']
                    or recipe_hash(state['config']) != recipe_hash(plan['config'])
                    or not 1 <= state['epoch'] <= plan['config']['training']['epochs']):
                raise ValueError(f'Checkpoint does not match saved plan: {run/name}')
    return plan


def _probe(model, dataset, config, seed, device, dtype):
    state = rng_state(device)
    try:
        return evaluate(model, loader(dataset, config['dataset'], seed), device, False, dtype)
    finally:
        restore_rng(state, device)


def _save_history(run, history):
    temporary = run/'epochs.csv.tmp'
    pd.DataFrame(history).to_csv(temporary, index=False)
    temporary.replace(run/'epochs.csv')


def _allow_unstarted_restart(run, config, variant, seed, fingerprint, manifest, probe_ids, model, device, amp, dtype):
    """Restart an interrupted first epoch only in this runner's known directory."""
    required = {'metadata.json', 'config.yaml', 'split_manifest.json', 'train_probe_ids.json'}
    if {path.name for path in run.iterdir()} != required:
        raise ValueError('Refusing to overwrite an incomplete directory without last.pt')
    metadata = json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    saved_config = yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
    saved_manifest = json.loads((run/'split_manifest.json').read_text(encoding='utf-8'))
    if (metadata.get('protocol') != PROTOCOL or metadata.get('variant') != variant
            or metadata.get('seed') != seed or metadata.get('source_hashes') != source_hashes()
            or metadata.get('split_hash') != fingerprint or saved_manifest != manifest
            or json.loads((run/'train_probe_ids.json').read_text(encoding='utf-8')) != probe_ids
            or recipe_hash(saved_config) != recipe_hash(config)
            or metadata.get('environment') != environment(device, amp, dtype)
            or metadata.get('initial_state_sha256') != state_digest(model)):
        raise ValueError('Unstarted run does not match the requested source, data and initialization')
    print('Verified interrupted first epoch; restarting this run at epoch 1.', flush=True)


def _diagnose(model, datasets, config, seed, run, examples):
    output = run/'diagnostics'
    if output.exists():
        output = run/new_name('diagnostics_retry')
    output.mkdir()
    base = datasets['train']
    clean = UtilityDataset(base.folder, base.indices, 'train_clean', seed)
    names = class_labels(clean)
    before = state_digest(model)
    scores = {}
    for name, dataset in (('train_clean', clean), ('selection', datasets['selection'])):
        scores[name] = evaluate_clean(model, dataset, device=next(model.parameters()).device.type,
            batch_size=config['dataset']['batch_size'], num_workers=config['dataset']['num_workers'],
            output_dir=output/name, class_names=names, example_count=examples)
    if state_digest(model) != before:
        raise RuntimeError('Model parameters/buffers changed during clean diagnosis')
    class_comparison(output)
    save_json(output/'summary.json', dict(status='complete', scores=scores,
        generalization_gap_pp=100*(scores['train_clean']['accuracy']-scores['selection']['accuracy']),
        official_validation_evaluated=False, audit_partition_evaluated=False))
    return output, scores


def train_run(plan, job, run):
    """Train/resume one selected geometry; completed summaries are immutable."""
    run = Path(run)
    if (run/'summary.json').exists():
        return json.loads((run/'summary.json').read_text(encoding='utf-8'))
    config = copy.deepcopy(plan['config'])
    variant, seed = job['variant'], job['seed']
    began = time.perf_counter()
    device, amp, dtype, scaler = runtime(config, seed)
    if environment(device, amp, dtype) != plan['execution_environment']:
        raise ValueError('All paired jobs require the same recorded execution environment')
    datasets, manifest, fingerprint = prepare_datasets(config, seed)
    if fingerprint != plan['split_hash']:
        raise ValueError('Spatial study split differs from the reference')
    probe = datasets['train_probe']
    probe_ids = [Path(probe.folder.samples[i][0]).relative_to(probe.folder.root).as_posix()
                 for i in probe.indices]
    model = build_spatial_model(variant, seed).to(device)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        group = no_decay if parameter.ndim <= 1 else decay
        group.append(parameter)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': config['training']['weight_decay']},
                                   {'params': no_decay, 'weight_decay': 0.}],
                                  lr=config['training']['learning_rate'])
    scheduler = build_scheduler(optimizer, config['training'])
    saved = None
    if (run/'last.pt').exists():
        saved = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
        metadata = saved['metadata']
        if (metadata['source_hashes'] != source_hashes() or metadata['split_hash'] != fingerprint
                or metadata['variant'] != variant or metadata['seed'] != seed
                or recipe_hash(saved['config']) != recipe_hash(config)
                or metadata['environment'] != environment(device, amp, dtype)):
            raise ValueError('Resume requires identical source, split, recipe, software, GPU and precision')
        if saved['epoch'] != len(saved['history']) or not 1 <= saved['epoch'] <= config['training']['epochs']:
            raise ValueError('Resume checkpoint epoch/history chronology is inconsistent')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng'], device)
        history, best, elapsed = saved['history'], saved['best'], saved['elapsed_wall_seconds']
        del saved
    else:
        if run.exists():
            _allow_unstarted_restart(run, config, variant, seed, fingerprint, manifest, probe_ids, model, device, amp, dtype)
        else:
            run.mkdir(parents=True)
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = 'unknown'
        metadata = dict(protocol=PROTOCOL, variant=variant, seed=seed, split_hash=fingerprint,
            source_hashes=source_hashes(), recipe_hash=recipe_hash(config), git_commit=commit,
            environment=environment(device, amp, dtype), architecture=spatial_description(variant),
            initial_state_sha256=state_digest(model), params=sum(p.numel() for p in model.parameters()),
            forward_flops_estimate=estimate_flops(model, device), evaluation_precision='float32',
            fresh_training=True, parent_checkpoint_used_for_initialization=False)
        save_json(run/'metadata.json', metadata)
        save_json(run/'split_manifest.json', manifest)
        save_json(run/'train_probe_ids.json', probe_ids)
        (run/'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        history, best, elapsed = [], -1., 0.
    print(f'Run: {run}; {variant}; next epoch={len(history)+1}; total epochs={config["training"]["epochs"]}', flush=True)
    for epoch in range(len(history)+1, config['training']['epochs']+1):
        sync(device)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        lr = optimizer.param_groups[0]['lr']
        loss = train_epoch(model, loader(datasets['train'], config['dataset'], seed, epoch),
                           config, seed, epoch, device, amp, dtype, optimizer, scaler)
        sync(device)
        train_seconds = time.perf_counter()-started
        peak = torch.cuda.max_memory_allocated() if device == 'cuda' else 0
        started = time.perf_counter()
        selected = evaluate(model, loader(datasets['selection'], config['dataset'], seed), device, False, dtype)
        sync(device)
        row = dict(epoch=epoch, lr=lr, train_soft_target_loss=loss, train_seconds=train_seconds,
            selection_seconds=time.perf_counter()-started, peak_training_memory_bytes=peak,
            **{f'selection_{key}': value for key, value in selected.items()})
        if epoch == 1 or epoch % plan['probe_every'] == 0 or epoch == config['training']['epochs']:
            started = time.perf_counter()
            probe = _probe(model, datasets['train_probe'], config, seed, device, dtype)
            row.update({f'train_probe_{key}': value for key, value in probe.items()})
            row['clean_accuracy_gap_pp'] = 100*(probe['accuracy']-selected['accuracy'])
            row['probe_seconds'] = time.perf_counter()-started
        if selected['accuracy'] > best:
            best = selected['accuracy']
            atomic_save(run/'best.pt', dict(model=model.state_dict(), epoch=epoch, selection=selected,
                                           metadata=metadata, config=config))
        history.append(row)
        scheduler.step()
        atomic_save(run/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), history=history, best=best,
            config=config, metadata=metadata, epoch=epoch,
            elapsed_wall_seconds=elapsed+time.perf_counter()-began, rng=rng_state(device)))
        _save_history(run, history)
        print(f'Epoch {epoch}/{config["training"]["epochs"]}: selection={100*selected["accuracy"]:.2f}% '
              f'NLL={selected["nll"]:.5f}; train={train_seconds:.1f}s', flush=True)
    _save_history(run, history)  # Recover CSV even if interrupted after last.pt was saved.
    model.zero_grad(set_to_none=True)
    del optimizer, scheduler, scaler
    state = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    best_epoch, best_selection = state['epoch'], state['selection']
    del state
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    diagnostic_dir, scores = _diagnose(model, datasets, config, seed, run, plan['examples'])
    profiles = {str(batch): benchmark_inference(model, (3, 64, 64), device, batch,
        amp_enabled=False, **config['profiling']) for batch in (1, config['dataset']['batch_size'])}
    summary = dict(**metadata, completed_epochs=len(history), best_epoch=best_epoch,
        best_selection=best_selection, clean_scores=scores,
        diagnostic_directory=diagnostic_dir.relative_to(run).as_posix(), inference=profiles,
        total_train_seconds=sum(row['train_seconds'] for row in history),
        total_selection_seconds=sum(row['selection_seconds'] for row in history),
        total_probe_seconds=sum(row.get('probe_seconds', 0.) for row in history),
        total_experiment_seconds=elapsed+time.perf_counter()-began,
        peak_training_memory_bytes=max(row['peak_training_memory_bytes'] for row in history),
        official_validation_evaluated=False, audit_partition_evaluated=False)
    save_json(run/'summary.json', summary)
    return summary


def paired_selection(left_file, right_file, output):
    """Pair identical evaluated images, using saved logit-derived decisions."""
    with np.load(left_file, allow_pickle=False) as left, np.load(right_file, allow_pickle=False) as right:
        if (not np.array_equal(left['image_ids'], right['image_ids']) or
                not np.array_equal(left['targets'], right['targets']) or
                not np.array_equal(left['class_names'], right['class_names'])):
            raise ValueError('Paired predictions require identical ordered IDs, labels and class mapping')
        ids, labels = left['image_ids'], left['targets']
        if labels.ndim != 1 or len(labels) == 0 or ids.shape != labels.shape:
            raise ValueError('Need nonempty one-dimensional paired IDs/targets')
        if len(set(ids.tolist())) != len(ids):
            raise ValueError('Duplicate paired image IDs')
        a, b = left['predicted_labels'], right['predicted_labels']
        p, q = left['probabilities'], right['probabilities']
        if (p.ndim != 2 or p.shape != q.shape or p.shape[0] != len(labels)
                or a.shape != labels.shape or b.shape != labels.shape
                or not np.issubdtype(labels.dtype, np.integer)
                or not np.issubdtype(a.dtype, np.integer) or not np.issubdtype(b.dtype, np.integer)
                or p.shape[1] != len(left['class_names'])
                or any(np.any(values < 0) or np.any(values >= p.shape[1]) for values in (labels, a, b))
                or not np.isfinite(p).all() or not np.isfinite(q).all()
                or np.any(p < 0) or np.any(q < 0)
                or not np.allclose(p.sum(1), 1., atol=1e-5)
                or not np.allclose(q.sum(1), 1., atol=1e-5)):
            raise ValueError('Invalid paired probability arrays')
        ix = np.arange(len(labels))
        smallest = float(np.nextafter(np.float32(0), np.float32(1)))
        loss_a = -np.log(np.maximum(p[ix, labels].astype(np.float64), smallest))
        loss_b = -np.log(np.maximum(q[ix, labels].astype(np.float64), smallest))
        correct_a, correct_b = a == labels, b == labels
        corrected, harmed = ~correct_a & correct_b, correct_a & ~correct_b
        pd.DataFrame(dict(image_id=ids, target=labels, prediction_cnn4=a, prediction_cnn8=b,
            correct_cnn4=correct_a, correct_cnn8=correct_b, corrected=corrected, harmed=harmed,
            nll_cnn4=loss_a, nll_cnn8=loss_b, nll_improvement=loss_a-loss_b)).to_csv(output, index=False)
        return dict(n=len(labels), corrected=int(corrected.sum()), harmed=int(harmed.sum()),
            delta_accuracy_pp=100*float((correct_b.astype(float)-correct_a.astype(float)).mean()),
            mean_nll_improvement=float((loss_a-loss_b).mean()),
            nll_note='Computed from saved float32 probabilities; zeros clipped to smallest positive float32')


def comparison(suite, plan):
    rows, summaries = [], {}
    for job in plan['jobs']:
        run = suite/job['directory']
        if not (run/'summary.json').exists():
            continue
        summary = json.loads((run/'summary.json').read_text())
        summaries[(job['seed'], job['variant'])] = (run, summary)
        rows.append(dict(variant=job['variant'], seed=job['seed'], best_epoch=summary['best_epoch'],
            completed_epochs=summary['completed_epochs'],
            selection_top1_pct=100*summary['clean_scores']['selection']['accuracy'],
            train_clean_top1_pct=100*summary['clean_scores']['train_clean']['accuracy'],
            selection_nll=summary['clean_scores']['selection']['nll'], params=summary['params'],
            gflops=summary['forward_flops_estimate']/1e9,
            train_minutes=summary['total_train_seconds']/60,
            batch1_ms=summary['inference']['1']['latency_per_batch_ms'],
            batch64_images_per_second=summary['inference'][str(plan['config']['dataset']['batch_size'])]['throughput_images_per_second']))
    if rows:
        pd.DataFrame(rows).to_csv(suite/'comparison.csv', index=False)
    paired = {}
    for seed in sorted({job['seed'] for job in plan['jobs']}):
        if (seed, 'cnn4') not in summaries or (seed, 'cnn8') not in summaries:
            continue
        left, a = summaries[(seed, 'cnn4')]
        right, b = summaries[(seed, 'cnn8')]
        if a['initial_state_sha256'] != b['initial_state_sha256']:
            raise ValueError('Paired geometry models did not share the same initial weights')
        paired[str(seed)] = paired_selection(
            left/a['diagnostic_directory']/'selection/predictions.npz',
            right/b['diagnostic_directory']/'selection/predictions.npz',
            suite/f'paired_selection_seed{seed}.csv')
    save_json(suite/'paired_summary.json', paired)


def run_suite(path):
    suite = Path(path).resolve()
    plan = validate_resume(suite)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing full spatial training on CPU')
    for job in plan['jobs']:
        train_run(plan, job, suite/job['directory'])
        comparison(suite, plan)
        gc.collect()
        torch.cuda.empty_cache()
    save_json(suite/'completion.json', dict(status='complete', protocol=PROTOCOL,
        completed_runs=len(plan['jobs']), smoke=plan['smoke'],
        note='Development study; official validation and audit partition were not evaluated'))
    return True


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run')
    parser.add_argument('--data-dir')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--epochs', type=int, help='Default100; resume cannot change the saved horizon')
    parser.add_argument('--smoke', action='store_true', help='Separate fresh one-epoch suite, never extended to100')
    parser.add_argument('--probe-every', type=int, default=5)
    parser.add_argument('--probe-per-class', type=int, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--variants', choices=SPATIAL_VARIANTS, nargs='+', default=list(SPATIAL_VARIANTS))
    parser.add_argument('--resume')
    parser.add_argument('--output', default=str(ROOT/'runs/spatial_tiny'))
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--examples', type=int, default=24)
    return parser


def main():
    args = arguments().parse_args()
    if args.resume:
        plan = validate_resume(args.resume)
        if args.smoke or args.epochs is not None and args.epochs != plan['config']['training']['epochs']:
            raise ValueError('Resume keeps the saved smoke/full protocol and total horizon')
        suite = Path(args.resume)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no spatial suite created')
        suite = create_suite(args)
    run_suite(suite)
    print(f'Completed spatial suite: {suite}', flush=True)


if __name__ == '__main__':
    main()
