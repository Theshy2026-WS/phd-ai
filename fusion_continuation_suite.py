"""Independent100-epoch joint continuation with matched CNN and frozen-feature controls."""
import argparse
import copy
import gc
import json
from pathlib import Path
import traceback

import torch

from baseline_diagnostics import clean_datasets
from correction_experiment import file_hash
from fusion_continuation_sources import load_sources, DEFAULT_SOURCE
from generalization_suite import ActiveBudget, atomic_json
from literature_experiment import ROOT, new_name, recipe_hash, save_json
from spatial_experiment import execution_environment, source_hashes as spatial_sources


PROTOCOL = 'tiny-fusion-continuation-suite-v1'
SOURCES = ('fusion_continuation_sources.py', 'fusion_continuation_suite.py',
    'fusion_continuation_experiment.py', 'fusion_continuation_diagnostics.py',
    'run_fusion_continuation_overnight.py', 'models/vit_followup.py', 'models/vit.py',
    'vit_followup_experiment.py', 'vit_followup_diagnostics.py',
    'attention_fusion_diagnostics.py', 'generalization_suite.py',
    'generalization_experiment.py', 'fusion30_experiment.py',
    'run_notified.py', 'overnight_twostage.py')


def source_hashes():
    return {**spatial_sources(), **{name: file_hash(ROOT/name) for name in SOURCES}}


def _separate_output(output, source, parent):
    for protected in (source, parent):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Output must be separate from the original source suite and CNN parent')


def make_plan(args):
    extension = 1 if args.smoke else (100 if args.continue_epochs is None else args.continue_epochs)
    control = 1 if args.smoke else (30 if args.control_epochs is None else args.control_epochs)
    if not 1 <= extension <= 100 or not 1 <= control <= 30:
        raise ValueError('Need continuation1..100 and control1..30 epochs')
    if args.examples < 0:
        raise ValueError('Examples must be nonnegative')
    supplied = load_sources(args.source_suite)
    config = copy.deepcopy(supplied['config'])
    path = Path(args.data_dir or config['dataset']['data_dir']).expanduser()
    config['dataset']['data_dir'] = str((path if path.is_absolute() else ROOT/path).resolve())
    config['dataset']['download'] = bool(args.allow_download)
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    if config['dataset']['num_workers'] < 0:
        raise ValueError('Workers must be nonnegative')
    output = Path(args.output).resolve()
    _separate_output(output, Path(supplied['source_suite']), Path(supplied['parent_run']))
    train, selection, _, split, manifest = clean_datasets(config, supplied['seed'], supplied['split_hash'], 0)
    if (len(train) != 80000 or len(selection) != 10000
            or set(manifest['train_clean']) & set(manifest['selection'])):
        raise ValueError('Expected the original disjoint80K/10K development partitions')
    environment = execution_environment(config, supplied['seed'])
    if environment != supplied['provenance']['source_execution_environment']:
        raise ValueError('Continuation requires the original GPU, software and precision environment')
    return dict(protocol=PROTOCOL, source_suite=supplied['source_suite'],
        source_provenance=supplied['provenance'], source_files=supplied['source_files'],
        parent_run=supplied['parent_run'], seed=supplied['seed'], split_hash=split,
        config=config, config_recipe_hash=recipe_hash(config), source_hashes=source_hashes(),
        execution_environment=environment, continue_epochs=extension, control_epochs=control,
        complete_all=True, max_hours=None, smoke=bool(args.smoke), examples=args.examples,
        jobs=['dual_continue', 'cnn_head_control', 'cnn_head_continue', 'dual_frozen_head'],
        stages=['training', 'feature_analysis'],
        total_new_model_epochs=2*extension+2*control,
        source_joint_epochs=30, final_joint_epochs=30+extension,
        continuation_policy=dict(model='source dual last.pt', optimizer='preserved AdamW moments/steps',
            ema='preserved', scaler='preserved', rng='preserved',
            scheduler=f'new independent{extension}-epoch phase, not an edited original30-epoch schedule',
            warmup_epochs=min(2, extension-1), warmup_start_ratio=.02, min_ratio=.02,
            encoder_peak_lr=5e-5, vit_peak_lr=5e-5, head_peak_lr=5e-4,
            augmentation_epochs=[131, 130+extension]),
        controls=dict(cnn_head_control=f'Original CNN100; trunk5e-5/head5e-4 for{control}',
            cnn_head_continue=f'Continue the matched CNN{control} control for the same new{extension} phase',
            dual_frozen_head=f'Original independent CNN100/ViT100; fixed encoders and normalization buffers; head-only{control}'),
        diagnostics=['independent probability mixtures', 'three-way correctness intersections',
                     'branch masking', 'training-probe-mean ViT feature replacement',
                     'branch logit/margin contribution'],
        excluded_training_routes=['shallow64', '4096-token global attention', 'new ViT pretraining'],
        audit_partition_evaluated=False, official_validation_evaluated=False)


def create_suite(args, plan=None):
    plan = make_plan(args) if plan is None else plan
    root = Path(args.output).resolve()
    _separate_output(root, Path(plan['source_suite']), Path(plan['parent_run']))
    suite = root/new_name('smoke' if plan['smoke'] else 'suite')
    suite.mkdir(parents=True, exist_ok=False)
    save_json(suite/'suite.json', plan)
    print(f'Fusion continuation suite: {suite}', flush=True)
    return suite


def validate_resume(path):
    suite = Path(path).resolve()
    plan = json.loads((suite/'suite.json').read_text(encoding='utf-8'))
    if plan.get('protocol') != PROTOCOL or plan.get('source_hashes') != source_hashes():
        raise ValueError('Resume requires the recorded continuation protocol and source files')
    if recipe_hash(plan['config']) != plan['config_recipe_hash']:
        raise ValueError('Recorded training/data configuration changed')
    supplied = load_sources(plan['source_suite'], config=plan['config'])
    if supplied['provenance'] != plan['source_provenance'] or supplied['source_files'] != plan['source_files']:
        raise ValueError('Original source checkpoints or metadata changed')
    if execution_environment(plan['config'], plan['seed']) != plan['execution_environment']:
        raise ValueError('Resume requires the recorded GPU, software and precision environment')
    clean_datasets(plan['config'], plan['seed'], plan['split_hash'], 0)
    return plan


def run_suite(path):
    from fusion_continuation_experiment import run_continuation_study
    from fusion_continuation_diagnostics import run_continuation_diagnostics
    suite = Path(path).resolve()
    plan = validate_resume(suite)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; continuation not started on CPU')
    outcomes = {}
    with ActiveBudget(suite, None) as budget:
        stages = (
            ('training', lambda: run_continuation_study(plan['source_suite'], suite/'training',
                config=copy.deepcopy(plan['config']), continue_epochs=plan['continue_epochs'],
                control_epochs=plan['control_epochs'], examples=plan['examples'],
                smoke=plan['smoke'], resume=True)),
            ('feature_analysis', lambda: run_continuation_diagnostics(plan['source_suite'],
                suite/'training', suite/'feature_analysis', config=copy.deepcopy(plan['config']),
                examples=plan['examples'])),
        )
        for name, action in stages:
            print(f'\nStarting continuation stage: {name}', flush=True)
            atomic_json(suite/'status.json', dict(protocol=PROTOCOL, active_stage=name, outcomes=outcomes))
            try:
                result = action()
                outcomes[name] = dict(status=result.get('status', 'complete'))
            except Exception as exc:
                outcomes[name] = dict(status='failed', error_type=type(exc).__name__, traceback=traceback.format_exc())
                target = suite/name
                target.mkdir(exist_ok=True)
                save_json(target/'stage_failure.json', outcomes[name])
                print(f'Continuation stage {name} failed ({type(exc).__name__}); retaining partial results.', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        unchanged = all(Path(name).is_file() and file_hash(name) == digest for name, digest in plan['source_files'].items())
        status = 'complete' if unchanged and all(item['status'] == 'complete' for item in outcomes.values()) else 'failed'
        result = dict(protocol=PROTOCOL, status=status, outcomes=outcomes, complete_all=True,
            elapsed_active_hours=budget.elapsed()/3600, source_checkpoints_unchanged=unchanged,
            continue_epochs=plan['continue_epochs'], control_epochs=plan['control_epochs'],
            source_joint_epochs=30, final_joint_epochs=30+plan['continue_epochs'], smoke=plan['smoke'],
            excluded_training_routes=plan['excluded_training_routes'],
            audit_partition_evaluated=False, official_validation_evaluated=False)
        atomic_json(suite/'completion.json', result)
        atomic_json(suite/'status.json', result)
        (suite/'report.md').write_text('# CNN/ViT continuation and fusion controls\n\n'
            f'Status: **{status}**. Original joint30 remains unchanged; joint continuation adds'
            f' {plan["continue_epochs"]} epochs.\n\n'
            '- [Training comparisons](training/report.md)\n'
            '- [Probability, branch and error-intersection analysis](feature_analysis/report.md)\n\n'
            'The queue contains no native64 global-attention training. '
            'CNN-only classification-head learning rates and continuation horizons are matched. '
            'Frozen-feature fusion separates existing representations from joint adaptation. '
            'Results remain development-selection evidence; audit and official validation are unused.\n', encoding='utf-8')
        if status != 'complete':
            raise RuntimeError('Continuation suite contains failures; partial reports retained')
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-suite', default=str(DEFAULT_SOURCE))
    parser.add_argument('--data-dir')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--continue-epochs', type=int, help='Additional continuation epochs, default100, maximum100')
    parser.add_argument('--control-epochs', type=int, help='Matched CNN initial phase and frozen-head epochs, default30, maximum30')
    parser.add_argument('--complete-all', action='store_true', help='Finish the finite queue without a time cutoff (default)')
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', '--output-dir', default=str(ROOT/'runs/fusion_continuation_tiny'))
    parser.add_argument('--resume')
    return parser


def main(argv=None):
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no continuation suite created')
    if args.resume:
        plan = validate_resume(args.resume)
        if args.smoke or any(getattr(args, name) is not None and getattr(args, name) != plan[name]
                             for name in ('continue_epochs', 'control_epochs')):
            raise ValueError('Resume retains its recorded source/full-smoke protocol and horizons')
        suite = Path(args.resume)
    else:
        suite = create_suite(args)
    run_suite(suite)


if __name__ == '__main__':
    main()
