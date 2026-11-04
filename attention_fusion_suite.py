"""A finite attention-first experiment suite on the strong Tiny ImageNet CNN8.

This entry point runs feature-fusion comparisons and attention interventions.
It does not launch the separate19-job generalization queue, send mail or stop
an instance; run_attention_overnight.py adds notification and managed stop.
"""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import time
import traceback

import torch

from baseline_diagnostics import clean_datasets
from correction_experiment import file_hash
from generalization_suite import ActiveBudget, atomic_json
from literature_experiment import ROOT, new_name, recipe_hash, save_json
from mechanism_diagnostics import load_spatial_parent
from spatial_experiment import execution_environment, source_hashes as spatial_sources


PROTOCOL = 'tiny-attention-first-suite-v1'
DEFAULT_PARENT = ROOT/'runs/spatial_tiny/suite_20261103T175622_a19c86/cnn8_seed42'
SOURCES = ('attention_fusion_suite.py', 'attention_fusion_experiment.py',
           'attention_fusion_diagnostics.py', 'models/attention_fusion.py',
           'run_attention_overnight.py', 'generalization_suite.py',
           'generalization_experiment.py', 'fusion30_experiment.py')


def source_hashes():
    return {**spatial_sources(), **{name: file_hash(ROOT/name) for name in SOURCES}}


def make_plan(args):
    warm = 1 if args.smoke else (30 if args.warm_epochs is None else args.warm_epochs)
    fresh = 1 if args.smoke else (100 if args.fresh_epochs is None else args.fresh_epochs)
    hours = args.max_hours
    if not 1 <= warm <= 30 or not 1 <= fresh <= 100:
        raise ValueError('Need warm1..30 and fresh1..100 epochs')
    if hours is not None and (not math.isfinite(hours) or not .25 <= hours <= 48):
        raise ValueError('Optional max-hours must be finite and in0.25..48')
    if args.examples < 0:
        raise ValueError('Examples must be nonnegative')
    parent, summary, state, digest = load_spatial_parent(args.parent_run)
    config = copy.deepcopy(state['config'])
    del state
    data = Path(args.data_dir or config['dataset']['data_dir']).expanduser()
    config['dataset']['data_dir'] = str((data if data.is_absolute() else ROOT/data).resolve())
    config['dataset']['download'] = bool(args.allow_download)
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    if config['dataset']['num_workers'] < 0:
        raise ValueError('Workers must be nonnegative')
    output = Path(args.output).resolve()
    if output == parent or parent in output.parents:
        raise ValueError('New output must be outside the original CNN run')
    train, selection, _, split, manifest = clean_datasets(config, summary['seed'], summary['split_hash'], 0)
    if len(train) != 80000 or len(selection) != 10000 or set(manifest['train_clean']) & set(manifest['selection']):
        raise ValueError('Expected the original disjoint80K/10K development partitions')
    return dict(protocol=PROTOCOL, parent_run=str(parent), parent_checkpoint_sha256=digest,
        parent_best_epoch=summary['best_epoch'], seed=summary['seed'], split_hash=split,
        config=config, config_recipe_hash=recipe_hash(config), source_hashes=source_hashes(),
        execution_environment=execution_environment(config, summary['seed']),
        warm_epochs=warm, fresh_epochs=fresh, max_hours=hours, complete_all=hours is None,
        smoke=bool(args.smoke), examples=args.examples, width=128, heads=4,
        stages=['training', 'attention_analysis'],
        primary_question='Does content-dependent attention complement the strong CNN beyond added convolution/fusion?',
        attention_query='deep8x8 spatial semantics', attention_detail_sources=['middle16x16', 'shallow32x32'],
        audit_partition_evaluated=False, official_validation_evaluated=False)


def create_suite(args, plan=None):
    plan = make_plan(args) if plan is None else plan
    root, parent = Path(args.output).resolve(), Path(plan['parent_run']).resolve()
    if root == parent or parent in root.parents:
        raise ValueError('New output must be outside the original CNN run')
    suite = root/new_name('smoke' if plan['smoke'] else 'suite')
    suite.mkdir(parents=True, exist_ok=False)
    save_json(suite/'suite.json', plan)
    print(f'Attention suite: {suite}', flush=True)
    return suite


def validate_resume(path):
    suite = Path(path).resolve()
    plan = json.loads((suite/'suite.json').read_text(encoding='utf-8'))
    if plan.get('protocol') != PROTOCOL or plan.get('source_hashes') != source_hashes():
        raise ValueError('Resume requires the recorded attention protocol and source files')
    if recipe_hash(plan['config']) != plan['config_recipe_hash']:
        raise ValueError('Recorded data/training configuration changed')
    _, summary, state, digest = load_spatial_parent(plan['parent_run'])
    del state
    if (digest != plan['parent_checkpoint_sha256'] or summary['split_hash'] != plan['split_hash']
            or summary['seed'] != plan['seed']):
        raise ValueError('Original CNN checkpoint, split or seed changed')
    if execution_environment(plan['config'], plan['seed']) != plan['execution_environment']:
        raise ValueError('Resume requires the recorded GPU, software and precision environment')
    clean_datasets(plan['config'], plan['seed'], plan['split_hash'], 0)
    return plan


def run_suite(path):
    from attention_fusion_experiment import run_attention_study
    from attention_fusion_diagnostics import run_attention_diagnostics
    suite = Path(path).resolve()
    plan = validate_resume(suite)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; attention suite not started on CPU')
    outcomes = {}
    with ActiveBudget(suite, plan['max_hours']) as budget:
        # Complete-all is the intended mode. An explicit bound reserves45min
        # for interventions/mail rather than terminating a checkpoint write.
        reserve_seconds = 0 if plan['complete_all'] else min(45*60, plan['max_hours']*1800)
        train_deadline = None if plan['complete_all'] else time.monotonic()+max(0., budget.remaining()-reserve_seconds)
        analysis_deadline = None if plan['complete_all'] else time.monotonic()+budget.remaining()
        stages = (
            ('training', lambda: run_attention_study(plan['parent_run'], suite/'training',
                config=copy.deepcopy(plan['config']), warm_epochs=plan['warm_epochs'],
                fresh_epochs=plan['fresh_epochs'], examples=plan['examples'], smoke=plan['smoke'],
                resume=True, deadline=train_deadline)),
            ('attention_analysis', lambda: run_attention_diagnostics(plan['parent_run'], suite/'training',
                suite/'attention_analysis', config=copy.deepcopy(plan['config']),
                examples=plan['examples'], deadline=analysis_deadline)),
        )
        for name, action in stages:
            print(f'\nStarting attention stage: {name}', flush=True)
            atomic_json(suite/'status.json', dict(protocol=PROTOCOL, active_stage=name, outcomes=outcomes))
            try:
                result = action()
                outcomes[name] = dict(status=result.get('status', 'complete'))
            except TimeoutError:
                outcomes[name] = dict(status='budget_exhausted', reason='Explicit time bound; partial outputs retained')
            except Exception as exc:
                outcomes[name] = dict(status='failed', error_type=type(exc).__name__, traceback=traceback.format_exc())
                directory = suite/name; directory.mkdir(exist_ok=True)
                save_json(directory/'stage_failure.json', outcomes[name])
                print(f'Attention stage {name} failed ({type(exc).__name__}); available results will still be analyzed.', flush=True)
            gc.collect(); torch.cuda.empty_cache()
        unchanged = file_hash(Path(plan['parent_run'])/'best.pt') == plan['parent_checkpoint_sha256']
        failed = not unchanged or any(item['status'] == 'failed' for item in outcomes.values())
        partial = any(item['status'] != 'complete' for item in outcomes.values())
        status = 'failed' if failed else ('budget_exhausted' if partial else 'complete')
        result = dict(protocol=PROTOCOL, status=status, outcomes=outcomes, complete_all=plan['complete_all'],
            elapsed_active_hours=budget.elapsed()/3600, parent_checkpoint_unchanged=unchanged,
            warm_epochs=plan['warm_epochs'], fresh_epochs=plan['fresh_epochs'], smoke=plan['smoke'],
            audit_partition_evaluated=False, official_validation_evaluated=False)
        atomic_json(suite/'completion.json', result); atomic_json(suite/'status.json', result)
        (suite/'report.md').write_text('# CNN8 attention feature-fusion study\n\n'
            f'Status: **{status}**. This suite runs attention comparisons, not the19-job overfitting queue.\n\n'
            '- [Training models and matched controls](training/report.md)\n'
            '- [Uniform/removal interventions and feature-use diagnostics](attention_analysis/report.md)\n\n'
            'The current64.28% is a working reference, not a demonstrated Tiny ImageNet ceiling. '
            'A generic cross-attention block is not automatically a novel contribution.\n', encoding='utf-8')
        if failed:
            raise RuntimeError('Attention suite contains failures; partial reports have been retained')
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', default=str(DEFAULT_PARENT))
    parser.add_argument('--data-dir')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--warm-epochs', type=int, help='Default30 per warm screening model, maximum30')
    parser.add_argument('--fresh-epochs', type=int, help='Default100 per fresh primary control/model, maximum100')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--complete-all', action='store_true', help='Finish the finite attention queue without a time cutoff (default)')
    mode.add_argument('--max-hours', type=float, help='Optional active time bound; interruption preserves checkpoints')
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--smoke', action='store_true', help='Separate one-epoch check, never resumed as the formal schedule')
    parser.add_argument('--output', default=str(ROOT/'runs/attention_fusion_tiny'))
    parser.add_argument('--resume')
    return parser


def main(argv=None):
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no attention suite created')
    if args.resume:
        plan = validate_resume(args.resume)
        if args.smoke or any(getattr(args, key) is not None and getattr(args, key) != plan[key]
                            for key in ('warm_epochs', 'fresh_epochs', 'max_hours')):
            raise ValueError('Resume retains its saved attention horizon and time mode')
        if args.complete_all and not plan['complete_all']:
            raise ValueError('Resume retains its saved completion mode')
        suite = Path(args.resume)
    else:
        suite = create_suite(args)
    run_suite(suite)


if __name__ == '__main__':
    main()
