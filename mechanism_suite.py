"""Bounded Tiny ImageNet mechanism diagnosis and matched 30-epoch adaptation.

The saved spatial CNN8 is immutable. Diagnostics and adaptation have separate
reports. Only the selection partition selects checkpoints; audit and official
validation are never evaluated. This module never emails or stops an instance.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import traceback

import torch

from baseline_diagnostics import clean_datasets
from correction_experiment import file_hash
from literature_experiment import ROOT, new_name, recipe_hash, save_json
from spatial_experiment import execution_environment, source_hashes as spatial_sources


PROTOCOL = 'tiny-mechanism-fusion30-v1'
DEFAULT_PARENT = ROOT/'runs/spatial_tiny/suite_20261103T175622_a19c86/cnn8_seed42'
NEW_SOURCES = ('mechanism_suite.py', 'mechanism_diagnostics.py',
               'fusion30_experiment.py', 'models/spatial_fusion.py',
               'run_mechanism_overnight.py')


def source_hashes():
    return {**spatial_sources(), **{name: file_hash(ROOT/name) for name in NEW_SOURCES}}


def _outside_parent(output, parent):
    if output == parent or parent in output.parents:
        raise ValueError('Output must be outside the source checkpoint run')


def make_plan(args):
    from mechanism_diagnostics import load_spatial_parent
    epochs = 1 if args.smoke else (30 if args.epochs is None else args.epochs)
    if not 1 <= epochs <= 30:
        raise ValueError('This bounded study permits 1..30 additional epochs per group')
    if args.probe_every < 1 or args.examples < 0:
        raise ValueError('Need a positive probe interval and nonnegative example count')
    parent, summary, state, checkpoint_hash = load_spatial_parent(args.parent_run)
    config = copy.deepcopy(state['config'])
    del state
    location = Path(args.data_dir or config['dataset']['data_dir']).expanduser()
    config['dataset']['data_dir'] = str((location if location.is_absolute() else ROOT/location).resolve())
    config['dataset']['download'] = bool(args.allow_download)
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    if config['dataset']['num_workers'] < 0:
        raise ValueError('num_workers must be nonnegative')
    # Equal calibration batches avoid weighting a short last batch equally.
    batch = config['dataset']['batch_size']
    if batch < 2 or batch % 2 or 80000 % batch:
        raise ValueError('Need an even batch size dividing 80000 for matched training and BN calibration')
    output = Path(args.output).resolve()
    _outside_parent(output, parent)
    train, selection, _, fingerprint, _ = clean_datasets(
        config, summary['seed'], summary['split_hash'], 0)
    if len(train) != 80000 or len(selection) != 10000:
        raise ValueError('Expected the complete 80000/10000 development partitions')
    return dict(protocol=PROTOCOL, parent_run=str(parent),
        parent_checkpoint_sha256=checkpoint_hash, parent_best_epoch=summary['best_epoch'],
        parent_completed_epochs=summary['completed_epochs'], seed=summary['seed'],
        split_hash=fingerprint, config=config, config_recipe_hash=recipe_hash(config),
        source_hashes=source_hashes(),
        execution_environment=execution_environment(config, summary['seed']),
        epochs=epochs, smoke=bool(args.smoke), examples=args.examples,
        probe_every=args.probe_every, jobs=['cnn8_control', 'unet32', 'unet32_noise'],
        training_initialization='same original CNN8 best checkpoint; fresh optimizer for every group',
        noise=dict(probability=0.5, rgb_standard_deviation=0.02, training_only=True),
        reports=['mechanism', 'fusion30'],
        note='Short development screen, not a from-scratch benchmark or a novelty claim')


def create_suite(args, plan=None):
    plan = make_plan(args) if plan is None else plan
    root = Path(args.output).resolve()
    _outside_parent(root, Path(plan['parent_run']).resolve())
    suite = root/new_name('smoke' if plan['smoke'] else 'suite')
    suite.mkdir(parents=True, exist_ok=False)
    save_json(suite/'suite.json', plan)
    print(f'Suite: {suite}', flush=True)
    return suite


def validate_resume(path):
    from mechanism_diagnostics import load_spatial_parent
    suite = Path(path).resolve()
    plan = json.loads((suite/'suite.json').read_text(encoding='utf-8'))
    _outside_parent(suite, Path(plan['parent_run']).resolve())
    if plan.get('protocol') != PROTOCOL or plan.get('source_hashes') != source_hashes():
        raise ValueError('Resume requires the recorded mechanism protocol and source files')
    if not 1 <= plan['epochs'] <= 30 or plan['jobs'] != ['cnn8_control', 'unet32', 'unet32_noise']:
        raise ValueError('Saved study schedule or model groups are invalid')
    if recipe_hash(plan['config']) != plan['config_recipe_hash']:
        raise ValueError('Saved dataset/training configuration changed')
    _, summary, state, checkpoint_hash = load_spatial_parent(plan['parent_run'])
    del state
    if (checkpoint_hash != plan['parent_checkpoint_sha256'] or
            summary['seed'] != plan['seed'] or summary['split_hash'] != plan['split_hash']):
        raise ValueError('Source checkpoint identity differs from the saved plan')
    if execution_environment(plan['config'], plan['seed']) != plan['execution_environment']:
        raise ValueError('Resume requires the recorded software, GPU and precision environment')
    train, selection, _, _, _ = clean_datasets(plan['config'], plan['seed'], plan['split_hash'], 0)
    if len(train) != 80000 or len(selection) != 10000:
        raise ValueError('Expected the complete development partitions')
    return plan


def run_suite(path):
    from mechanism_diagnostics import run_diagnostics
    from fusion30_experiment import train_fusion_suite
    suite = Path(path).resolve()
    plan = validate_resume(suite)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing full diagnosis/adaptation on CPU')
    stages = (
        ('mechanism', lambda: run_diagnostics(plan['parent_run'], suite/'mechanism',
            config=copy.deepcopy(plan['config']), examples=plan['examples'])),
        ('fusion30', lambda: train_fusion_suite(plan['parent_run'], suite/'fusion30',
            config=copy.deepcopy(plan['config']), epochs=plan['epochs'],
            probe_every=plan['probe_every'], examples=plan['examples'],
            smoke=plan['smoke'], resume=True)),
    )
    outcomes = {}
    for name, action in stages:
        print(f'\nStarting stage: {name}', flush=True)
        save_json(suite/'status.json', dict(protocol=PROTOCOL, active_stage=name, outcomes=outcomes))
        try:
            action()
            outcomes[name] = dict(status='complete')
        except Exception as exc:
            # A failed independent diagnostic must not discard the authorized
            # training work. The final process still exits nonzero and reports it.
            failure = dict(status='failed', error_type=type(exc).__name__, traceback=traceback.format_exc())
            outcomes[name] = failure
            directory = suite/name
            directory.mkdir(exist_ok=True)
            save_json(directory/'stage_failure.json', failure)
            print(f'Stage {name} failed ({type(exc).__name__}); details saved on disk.', flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    unchanged = file_hash(Path(plan['parent_run'])/'best.pt') == plan['parent_checkpoint_sha256']
    complete = unchanged and all(item['status'] == 'complete' for item in outcomes.values())
    result = dict(protocol=PROTOCOL, status='complete' if complete else 'failed',
        outcomes=outcomes, parent_checkpoint_unchanged=unchanged, smoke=plan['smoke'],
        additional_epochs_per_group=plan['epochs'],
        audit_partition_evaluated=False, official_validation_evaluated=False)
    save_json(suite/'status.json', result)
    if complete:
        save_json(suite/'completion.json', result)
    if not complete:
        raise RuntimeError('One or more stages failed; inspect the two analysis reports and stage_failure.json')
    print(f'Completed mechanism and adaptation suite: {suite}', flush=True)
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', default=str(DEFAULT_PARENT))
    parser.add_argument('--data-dir')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--epochs', type=int, help='Additional fine-tuning epochs per group; default30, maximum30')
    parser.add_argument('--probe-every', type=int, default=5)
    parser.add_argument('--examples', type=int, default=12)
    parser.add_argument('--smoke', action='store_true', help='Separate one-epoch adaptation screen; diagnosis remains complete')
    parser.add_argument('--output', default=str(ROOT/'runs/mechanism_tiny'))
    parser.add_argument('--resume')
    return parser


def main(argv=None):
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no suite created')
    if args.resume:
        plan = validate_resume(args.resume)
        if args.smoke or args.epochs is not None and args.epochs != plan['epochs']:
            raise ValueError('Resume retains the original smoke/full protocol and epoch horizon')
        suite = Path(args.resume).resolve()
    else:
        suite = create_suite(args)
    run_suite(suite)


if __name__ == '__main__':
    main()
