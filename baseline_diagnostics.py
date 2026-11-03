"""Clean Tiny ImageNet baseline diagnosis and bounded, fresh training controls.

Uses trusted local checkpoints. Never evaluates official validation, emails,
stops a VM, modifies an existing run, or resumes a 100-epoch job as 200 epochs.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import tarfile

import torch

from correction_experiment import file_hash, find_parent
from datasets.utility_tiny import UtilityDataset, make_datasets
from literature_experiment import ROOT, PROTOCOL, new_name, save_json, train_run, recipe_hash
from main import configure_runtime
from models.attention_correction import state_digest
from models.literature_resnet import build_model
from utils.baseline_diagnostics import evaluate_clean


DIAGNOSTIC_PROTOCOL = 'tiny-clean-baseline-v1'
CONTROL_VARIANTS = ('longer200', 'no_mix100')
MODEL_SOURCES = ('models/literature_twostage.py', 'models/literature_alternet.py')
DIAGNOSTIC_SOURCES = ('baseline_diagnostics.py', 'utils/baseline_diagnostics.py',
                      'datasets/utility_tiny.py', 'datasets/tiny_imagenet.py')


def diagnostic_sources():
    return {name: file_hash(ROOT/name) for name in DIAGNOSTIC_SOURCES}


def load_run(path=None):
    """Require a completed native64 PreAct parent and verify checkpoint identity."""
    if path is None:
        path, _, _ = find_parent()
    run = Path(path).resolve()
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if (summary.get('variant') != 'resnet50_preact' or
            summary.get('protocol') != PROTOCOL or
            summary.get('completed_epochs', 0) < 1):
        raise ValueError('Need a completed literature-protocol resnet50_preact run')
    checkpoint_path = run/'best.pt'
    checkpoint_hash = file_hash(checkpoint_path)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    for key in ('variant', 'seed', 'split_hash', 'source_hashes', 'recipe_hash'):
        if state['metadata'][key] != summary[key]:
            raise ValueError(f'Summary/checkpoint mismatch: {key}')
    if state['epoch'] != summary['best_epoch']:
        raise ValueError('Summary/checkpoint best epoch mismatch')
    if not 1 <= state['epoch'] <= summary['completed_epochs'] <= state['config']['training']['epochs']:
        raise ValueError('Checkpoint/completed/configured epoch chronology is invalid')
    if state['selection'] != summary['best_selection']:
        raise ValueError('Summary/checkpoint selection metrics mismatch')
    if recipe_hash(state['config']) != summary['recipe_hash']:
        raise ValueError('Checkpoint configuration differs from its recorded recipe hash')
    for name in MODEL_SOURCES:
        if summary['source_hashes'].get(name) != file_hash(ROOT/name):
            raise ValueError(f'Parent model implementation differs: {name}')
    return run, summary, state, checkpoint_hash


def evaluation_config(state, args):
    config = copy.deepcopy(state['config'])
    data = Path(args.data_dir or config['dataset']['data_dir']).expanduser()
    config['dataset']['data_dir'] = str(data.resolve() if data.is_absolute() else (ROOT/data).resolve())
    config['dataset']['download'] = bool(args.allow_download)
    if args.batch_size is not None:
        config['dataset']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    if config['dataset']['batch_size'] < 1 or config['dataset']['num_workers'] < 0:
        raise ValueError('Need batch_size >= 1 and num_workers >= 0')
    return config


def clean_datasets(config, seed, expected_hash, train_per_class):
    """Reuse the original split, then explicitly disable train augmentation."""
    if train_per_class < 0 or train_per_class > 400:
        raise ValueError('train_per_class must be 0 (all) or 1..400')
    datasets, manifest, fingerprint = make_datasets(config['dataset'], seed)
    if fingerprint != expected_hash:
        raise ValueError('Dataset split/class mapping differs from trained checkpoint')
    base = datasets['train']
    indices = list(base.indices)
    if train_per_class:
        by_class = {}
        for index in indices:
            by_class.setdefault(base.folder.samples[index][1], []).append(index)
        generator = torch.Generator().manual_seed(config['dataset']['split_seed'] + 90001)
        indices = []
        for label in sorted(by_class):
            candidates = by_class[label]
            order = torch.randperm(len(candidates), generator=generator).tolist()
            indices.extend(candidates[j] for j in order[:train_per_class])
    # UtilityDataset augments only split == 'train'. A new clean view is essential.
    train = UtilityDataset(base.folder, indices, 'train_clean', seed)
    selection = datasets['selection']
    evaluation_manifest = dict(
        train_clean=[Path(base.folder.samples[i][0]).relative_to(base.folder.root).as_posix()
                     for i in indices], selection=manifest['selection'])
    names = list(base.folder.classes)
    return train, selection, names, fingerprint, evaluation_manifest


def class_labels(train):
    """Add human-readable words without replacing the original synset mapping."""
    root = Path(train.folder.root).parent
    descriptions = {}
    words = root/'words.txt'
    if words.exists():
        for line in words.read_text(encoding='utf-8').splitlines():
            fields = line.split('\t', 1)
            if len(fields) == 2:
                descriptions[fields[0]] = fields[1]
    return [f'{synset}: {descriptions[synset]}' if synset in descriptions else synset
            for synset in train.folder.classes]


def bundle_report(output):
    """Small analysis bundle; full probability arrays remain on the VM."""
    path = output/'diagnostics_report.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        for item in sorted(output.rglob('*')):
            if item.is_file() and item != path and item.suffix != '.npz':
                archive.add(item, arcname=item.relative_to(output))
    return path


def class_comparison(output):
    """Expose class-specific train-selection gaps, with denominators intact."""
    with (output/'train_clean/per_class.csv').open(encoding='utf-8', newline='') as stream:
        train = {row['label']: row for row in csv.DictReader(stream)}
    with (output/'selection/per_class.csv').open(encoding='utf-8', newline='') as stream:
        selection = list(csv.DictReader(stream))
    rows = []
    for selected in selection:
        fitted = train[selected['label']]
        rows.append(dict(label=selected['label'], class_name=selected['class_name'],
            train_n=int(fitted['n']), selection_n=int(selected['n']),
            train_recall=float(fitted['recall']), selection_recall=float(selected['recall']),
            gap_pp=100*(float(fitted['recall'])-float(selected['recall'])),
            train_nll=float(fitted['nll']), selection_nll=float(selected['nll'])))
    rows.sort(key=lambda row: (row['selection_recall'], int(row['label'])))
    with (output/'class_comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(args):
    run, summary, state, checkpoint_hash = load_run(args.base_run)
    config = evaluation_config(state, args)
    seed = summary['seed']
    train, selection, synsets, fingerprint, manifest = clean_datasets(
        config, seed, summary['split_hash'], args.train_per_class)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly for CPU diagnosis')
    configure_runtime(config['training'])
    model = build_model('resnet50_preact', seed)
    model.load_state_dict(state['model'], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()
           if value.is_floating_point()):
        raise ValueError('Non-finite parent checkpoint')
    del state
    model = model.to(args.device).float().eval()
    before = state_digest(model)
    output_root = Path(args.output).resolve() if args.output else ROOT/'runs/baseline_diagnostics'
    output = output_root/new_name('diagnosis')
    if output == run or run in output.parents:
        raise ValueError('Diagnostic output must be outside the original training run')
    output.mkdir(parents=True, exist_ok=False)
    labels = class_labels(train)
    save_json(output/'evaluation_manifest.json', manifest)
    save_json(output/'class_mapping.json', dict(enumerate(synsets)))
    print(f'Parent: {run}\nCheckpoint epoch: {summary["best_epoch"]}\n'
          f'Clean training images: {len(train)}; selection images: {len(selection)}\n'
          f'Output: {output}\nNo training or final-test evaluation requested.', flush=True)
    scores = {}
    for split, dataset in (('train_clean', train), ('selection', selection)):
        scores[split] = evaluate_clean(model, dataset, device=args.device,
                batch_size=config['dataset']['batch_size'],
                num_workers=config['dataset']['num_workers'], output_dir=output/split,
                class_names=labels, example_count=args.examples)
        print(f'{split}: Top-1={100*scores[split]["accuracy"]:.2f}% '
              f'Top-5={100*scores[split]["top5_accuracy"]:.2f}% '
              f'NLL={scores[split]["nll"]:.5f}', flush=True)
    after = state_digest(model)
    if after != before or file_hash(run/'best.pt') != checkpoint_hash:
        raise RuntimeError('Model state or source checkpoint changed during diagnosis')
    class_comparison(output)
    gap = dict(accuracy_pp=100*(scores['train_clean']['accuracy']-scores['selection']['accuracy']),
               nll_selection_minus_train=scores['selection']['nll']-scores['train_clean']['nll'])
    replay = dict(recorded=summary['best_selection'], observed={
        key: scores['selection'][key] for key in ('accuracy', 'top5_accuracy', 'nll')},
        accuracy_delta_pp=100*(scores['selection']['accuracy']-summary['best_selection']['accuracy']),
        top5_delta_pp=100*(scores['selection']['top5_accuracy']-summary['best_selection']['top5_accuracy']),
        nll_delta=scores['selection']['nll']-summary['best_selection']['nll'])
    report = dict(protocol=DIAGNOSTIC_PROTOCOL, status='complete',
        created_utc=datetime.now(timezone.utc).isoformat(), base_run=str(run),
        checkpoint_sha256=checkpoint_hash, checkpoint_epoch=summary['best_epoch'],
        model_variant=summary['variant'], seed=seed, split_hash=fingerprint,
        train_per_class=args.train_per_class, train_scope='all' if not args.train_per_class else 'stratified_probe',
        scores=scores, generalization_gap=gap, selection_replay=replay,
        model_state_unchanged=True, source_hashes=diagnostic_sources(),
        training_source_hashes=summary['source_hashes'],
        training_environment=summary.get('environment'),
        evaluation=dict(device=args.device, device_name=torch.cuda.get_device_name() if args.device=='cuda' else 'cpu',
            precision='float32; autocast disabled', torch=str(torch.__version__),
            tf32=config['training'].get('enable_tf32', True),
            batch_size=config['dataset']['batch_size'], num_workers=config['dataset']['num_workers'],
            preprocessing='native64 RGB, fixed normalization; no random transform or label mixing',
            official_validation_evaluated=False, audit_partition_evaluated=False))
    save_json(output/'summary.json', report)
    lines = [f'# Baseline diagnosis: checkpoint epoch {summary["best_epoch"]}', '',
        '| Split | Images | Top-1 | Top-5 | NLL |', '|---|---:|---:|---:|---:|']
    for name, score in scores.items():
        lines.append(f'| {name} | {score["n"]} | {100*score["accuracy"]:.2f}% | '
                     f'{100*score["top5_accuracy"]:.2f}% | {score["nll"]:.5f} |')
    lines += ['', f'Clean training minus selection Top-1: {gap["accuracy_pp"]:.2f} percentage points.',
        f'Selection replay delta: {replay["accuracy_delta_pp"]:+.4f} percentage points.', '',
        f'Selection replay NLL delta: {replay["nll_delta"]:+.8f}.', '',
        'Use per_class.csv, confusion_pairs.csv and galleries to form hypotheses. '
        'These results do not automatically diagnose underfitting, overfitting, or an information bottleneck.',
        'Training images are in-sample. Selection has been used for model development. '
        'Official validation and the audit partition were not evaluated.',
        'The analysis archive excludes predictions.npz; full probability arrays remain in split folders.']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    archive = bundle_report(output)
    print(f'Clean train-selection gap: {gap["accuracy_pp"]:.2f} pp\n'
          f'Selection replay delta: {replay["accuracy_delta_pp"]:+.4f} pp\n'
          f'Report bundle: {archive}', flush=True)
    del model
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    return output


def control_config(config, variant):
    if variant not in CONTROL_VARIANTS:
        raise ValueError(f'Unknown control {variant}')
    result = copy.deepcopy(config)
    result['training']['epochs'] = 200 if variant == 'longer200' else 100
    result['training']['early_stopping_patience'] = 0
    if variant == 'no_mix100':
        result['training']['mix_probability'] = 0.0
    return result


def run_controls(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for controlled retraining')
    parent, summary, state, digest = load_run(args.base_run)
    if summary['completed_epochs'] != 100:
        raise ValueError('Controls must start from the original 100-epoch reference recipe')
    config = evaluation_config(state, args)
    # Controls retain the original training loader/batch policy exactly.
    if config['dataset']['batch_size'] != state['config']['dataset']['batch_size']:
        raise ValueError('Do not change batch size in the controlled comparison')
    if len(args.variants) != len(set(args.variants)):
        raise ValueError('Duplicate controls requested')
    # Validate the complete original split before spending any training compute.
    clean_datasets(config, summary['seed'], summary['split_hash'], 0)
    output_root = Path(args.output).resolve() if args.output else ROOT/'runs/baseline_controls'
    suite = output_root/new_name('controls')
    if suite == parent or parent in suite.parents:
        raise ValueError('Controls must be outside the original training run')
    suite.mkdir(parents=True, exist_ok=False)
    save_json(suite/'plan.json', dict(protocol='tiny-baseline-controls-v1', base_run=str(parent),
        parent_checkpoint_sha256=digest, seed=summary['seed'], split_hash=summary['split_hash'],
        fresh_training=True, variants=args.variants, source_hashes=diagnostic_sources(),
        changes={'longer200': '200-epoch cosine schedule from scratch; all other training factors retained',
                 'no_mix100': '100 epochs from scratch; mix_probability=0; smoothing/crop/RandAugment retained'}))
    del state
    for variant in args.variants:
        configured = control_config(config, variant)
        child = suite/variant
        print(f'Fresh control: {variant}; never resumes or initializes from the parent weights.', flush=True)
        train_run(configured, 'resnet50_preact', summary['seed'], child)
        diagnostic_args = copy.copy(args)
        diagnostic_args.base_run = str(child)
        diagnostic_args.output = str(suite/'diagnostics')
        diagnostic_args.device = 'cuda'
        evaluate_run(diagnostic_args)
    print(f'Controlled runs and clean diagnoses: {suite}', flush=True)
    return suite


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('evaluate', 'controls'):
        command = sub.add_parser(name)
        command.add_argument('--base-run', help='Completed resnet50_preact run; auto-detect only if unique')
        command.add_argument('--data-dir', help='Override archived data directory without changing the split')
        command.add_argument('--allow-download', action='store_true', help='Explicitly permit missing dataset download')
        command.add_argument('--batch-size', type=int)
        command.add_argument('--num-workers', type=int)
        command.add_argument('--train-per-class', type=int, default=0, help='0 = full 80K clean training set; 20 = 4K probe')
        command.add_argument('--examples', type=int, default=24, help='Images per gallery, 0 disables images')
        command.add_argument('--output', help='Parent folder for a new timestamped output directory')
    sub.choices['evaluate'].add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    sub.choices['controls'].add_argument('--variants', choices=CONTROL_VARIANTS, nargs='+', required=True,
        help='Opt-in retraining only; choose after inspecting the clean diagnosis')
    return parser


def main():
    args = arguments().parse_args()
    if args.examples < 0:
        raise ValueError('examples must be nonnegative')
    if args.train_per_class < 0 or args.train_per_class > 400:
        raise ValueError('train_per_class must be 0 (all) or 1..400')
    if args.command == 'evaluate':
        evaluate_run(args)
    else:
        run_controls(args)


if __name__ == '__main__':
    main()
