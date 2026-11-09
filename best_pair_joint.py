"""Jointly train the strongest completed CNN and independent ViT for30 epochs.

This new suite has one job, no pretraining/search, and no mail or VM control.
The existing tested AugReg training engine preserves raw/EMA/best/last and full
resume state. Source suites and their schedules remain immutable.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import traceback

import pandas as pd
import torch

from baseline_diagnostics import clean_datasets
from correction_experiment import file_hash
from fusion30_experiment import paired_predictions, _selection_curve_png
from generalization_suite import ActiveBudget, atomic_json
from literature_experiment import ROOT, new_name, recipe_hash, save_json
from spatial_experiment import execution_environment, source_hashes as spatial_sources


PROTOCOL = 'tiny-best-pair-joint-suite-v1'
JOB = 'new_joint_concat'
DEFAULT_CNN_SOURCE = ROOT/'runs/fusion_continuation_tiny/suite_20261106T194319_c2dd7a'
DEFAULT_VIT_SOURCE = ROOT/'runs/vit_augreg_tiny/suite_20261107T182435_87170e'
SOURCES = ('best_pair_joint.py', 'best_pair_sources.py', 'best_pair_diagnostics.py',
    'run_best_pair_joint.py', 'models/vit_augreg.py', 'vit_augreg_experiment.py',
    'vit_augreg_diagnostics.py', 'vit_followup_experiment.py',
    'vit_followup_diagnostics.py', 'models/vit_followup.py', 'models/vit.py',
    'fusion_continuation_sources.py', 'fusion_continuation_experiment.py',
    'fusion_continuation_diagnostics.py', 'attention_fusion_diagnostics.py',
    'generalization_experiment.py', 'generalization_suite.py', 'fusion30_experiment.py',
    'run_notified.py', 'datasets/resnet_tiny.py', 'datasets/utility_tiny.py')


def load_sources(cnn_suite, vit_suite, config=None):
    from best_pair_sources import load_best_sources
    return load_best_sources(cnn_suite, vit_suite, config=config)


def source_hashes():
    return {**spatial_sources(), **{name: file_hash(ROOT/name) for name in SOURCES}}


def _source_identity(sources):
    return sources.get('source_provenance', sources.get('provenance'))


def _separate_output(output, cnn_suite, vit_suite, parent):
    for protected in (cnn_suite, vit_suite, parent):
        protected = Path(protected).resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('New output must be separate from both immutable source suites and CNN run')


def _data_environment(config, sources):
    train, selected, _, split, manifest = clean_datasets(config, sources['seed'], sources['split_hash'], 0)
    if (len(train) != 80000 or len(selected) != 10000 or split != sources['split_hash']
            or set(manifest['train_clean']) & set(manifest['selection'])):
        raise ValueError('Expected the original disjoint80K/10K Tiny ImageNet partitions')
    environment = execution_environment(config, sources['seed'])
    if environment != _source_identity(sources)['source_execution_environment']:
        raise ValueError('Joint comparison requires the recorded GPU, software and precision environment')
    return environment


def make_plan(args):
    epochs = 1 if args.smoke else (30 if args.epochs is None else args.epochs)
    if not 1 <= epochs <= 30 or args.examples < 0:
        raise ValueError('Need1..30 joint epochs and nonnegative examples')
    sources = load_sources(args.cnn_suite, args.vit_suite)
    config = copy.deepcopy(sources['config'])
    data = Path(args.data_dir or config['dataset']['data_dir']).expanduser()
    config['dataset']['data_dir'] = str((data if data.is_absolute() else ROOT/data).resolve())
    config['dataset']['download'] = bool(args.allow_download)
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    if config['dataset']['num_workers'] < 0 or config['dataset']['batch_size'] != 64:
        raise ValueError('Need nonnegative workers and source microbatch64')
    cnn, vit = Path(args.cnn_suite).resolve(), Path(args.vit_suite).resolve()
    _separate_output(Path(args.output).resolve(), cnn, vit, sources['parent_run'])
    environment = _data_environment(config, sources)
    selected = {key: value for key, value in sources['selected'].items() if key != 'state'}
    provenance = _source_identity(sources)
    return dict(protocol=PROTOCOL, cnn_suite=str(cnn), vit_suite=str(vit),
        source_suite=str(vit), source_provenance=provenance, source_files=sources['source_files'],
        parent_run=sources['parent_run'], seed=sources['seed'], split_hash=sources['split_hash'],
        parent_completed_epochs=sources['parent_completed_epochs'],
        inherited_cnn_best_training_epochs=sources['parent_completed_epochs'],
        fusion_augmentation_epoch_offset=100,
        config=config, config_recipe_hash=recipe_hash(config), source_hashes=source_hashes(),
        execution_environment=environment, epochs=epochs, fusion_epochs=epochs,
        vit_epochs=300, profiles=[selected['profile']], profile_recipes={},
        selected_vit=selected, jobs=[JOB], total_new_model_epochs=epochs,
        complete_all=True, max_hours=None, smoke=bool(args.smoke), examples=args.examples,
        stages=['training', 'feature_analysis'],
        shutdown_after_completion=False,
        initialization='strongest completed pure CNN best + strongest independent ViT best; new concat head',
        optimizer_policy='Fresh AdamW and EMA for a new joint phase; no old joint model is resumed',
        fusion_policy=dict(structure='CNN2048 GAP + ViT384 CLS -> concat -> Linear200',
            independent_branch_parameters=True, both_encoders_trainable=True,
            encoder_lr=5e-5, vit_lr=5e-5, classifier_lr=5e-4,
            schedule='source30-epoch joint warmup/cosine, batch64, no accumulation',
            augmentation_epoch_offset=100, new_augmentation_epochs=[101, 100+epochs],
            augmentation_interpretation='Reuse the old joint30 training views; these indices do not count CNN pretraining epochs'),
        excluded_training_routes=['new CNN pretraining', 'new ViT pretraining', 'seed sweep',
                                  '4096-token global attention', 'instance stop'],
        audit_partition_evaluated=False, official_validation_evaluated=False)


def create_suite(args, plan=None):
    plan = make_plan(args) if plan is None else plan
    root = Path(args.output).resolve()
    _separate_output(root, plan['cnn_suite'], plan['vit_suite'], plan['parent_run'])
    suite = root/new_name('smoke' if plan['smoke'] else 'suite')
    suite.mkdir(parents=True, exist_ok=False)
    save_json(suite/'suite.json', plan)
    print(f'Best-pair joint suite: {suite}', flush=True)
    return suite


def _validate_source_identity(plan, sources):
    selected = {key: value for key, value in sources['selected'].items() if key != 'state'}
    if (_source_identity(sources) != plan['source_provenance']
            or sources['source_files'] != plan['source_files'] or selected != plan['selected_vit']
            or sources['parent_completed_epochs'] != plan['parent_completed_epochs']):
        raise ValueError('Strongest CNN/ViT sources, selected checkpoint or inherited history changed')


def validate_resume(path):
    suite = Path(path).resolve()
    plan = json.loads((suite/'suite.json').read_text(encoding='utf-8'))
    if plan.get('protocol') != PROTOCOL or plan.get('source_hashes') != source_hashes():
        raise ValueError('Resume requires the recorded best-pair protocol and source files')
    if (recipe_hash(plan['config']) != plan['config_recipe_hash']
            or plan['jobs'] != [JOB] or plan['epochs'] != plan['fusion_epochs']
            or plan['total_new_model_epochs'] != plan['epochs']
            or plan.get('shutdown_after_completion') is not False):
        raise ValueError('Recorded joint recipe, single-job horizon or no-shutdown policy changed')
    if not 1 <= plan['epochs'] <= 30 or (plan['smoke'] and plan['epochs'] != 1):
        raise ValueError('Invalid recorded joint horizon')
    if plan.get('fusion_augmentation_epoch_offset') != 100:
        raise ValueError('Joint phase must retain the matched augmentation views101..130')
    sources = load_sources(plan['cnn_suite'], plan['vit_suite'], config=plan['config'])
    _validate_source_identity(plan, sources)
    if _data_environment(plan['config'], sources) != plan['execution_environment']:
        raise ValueError('Recorded environment changed')
    return plan


def load_best_pair_model(plan, training_dir, checkpoint='best.pt', sources=None):
    """Rebuild one exact new-pair checkpoint without any new ViT selection."""
    from vit_augreg_experiment import build_job_model, job_config, _validate, _failure_identity
    if checkpoint in ('best', 'last', 'best_trained'):
        checkpoint += '.pt'
    if checkpoint not in ('best.pt', 'last.pt', 'best_trained.pt'):
        raise ValueError('Unsupported new-pair analysis checkpoint')
    sources = sources or load_sources(plan['cnn_suite'], plan['vit_suite'], config=plan['config'])
    _validate_source_identity(plan, sources)
    run = Path(training_dir)/JOB
    model = build_job_model(JOB, sources, sources['selected'], plan['seed'])
    state = torch.load(run/checkpoint, map_location='cpu', weights_only=False)
    metadata = json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    _validate(state, metadata, job_config(plan, JOB))
    if any(metadata.get(key) != value for key, value in _failure_identity(plan, JOB).items()):
        raise ValueError('New-pair analysis metadata differs from suite')
    if (metadata.get('parent_checkpoint_sha256') != plan['source_provenance']['parent_checkpoint_sha256']
            or metadata.get('vit_checkpoint_sha256') != plan['selected_vit']['checkpoint_sha256']):
        raise ValueError('New-pair analysis used different independent encoders')
    model.load_state_dict(state['model'], strict=True)
    return model, state


def training_report(plan, training, summary):
    training = Path(training)
    selected = summary['clean_scores']['selection']
    provenance = plan['source_provenance']
    row = dict(job=JOB, completed_epochs=summary['completed_epochs'], planned_epochs=plan['epochs'],
        best_top1_pct=100*selected['accuracy'], best_epoch=summary['best_epoch'],
        best_source=summary['best_source'], best_nll=selected['nll'],
        initial_top1_pct=100*summary['initial_selection']['accuracy'],
        last_top1_pct=100*summary['last_epoch_selection']['accuracy'],
        last_ema_top1_pct=100*summary['last_ema_selection']['accuracy'],
        params=summary['params'], trainable_params=summary['trainable_params'],
        gflops=summary['forward_flops_estimate']/1e9,
        train_minutes=summary['total_train_seconds']/60,
        source_cnn_top1_pct=100*provenance['cnn_best_selection']['accuracy'],
        source_vit_top1_pct=100*provenance['vit_best_selection']['accuracy'],
        inherited_cnn_best_training_epochs=plan['inherited_cnn_best_training_epochs'],
        inherited_vit_completed_epochs=provenance['source_vit_completed_epochs'],
        inherited_cnn_continuation_train_minutes=(provenance['source_cnn_stage_train_seconds']/60
            if provenance.get('source_cnn_stage_train_seconds') is not None else None),
        inherited_selected_vit_train_minutes=(provenance['source_vit_train_seconds']/60
            if provenance.get('source_vit_train_seconds') is not None else None),
        inherited_vit_search_train_minutes=(provenance['source_vit_search_train_seconds']/60
            if provenance.get('source_vit_search_train_seconds') is not None else None))
    pd.DataFrame([row]).to_csv(training/'comparison.csv', index=False)
    reference = json.loads((training/'parent_reference.json').read_text(encoding='utf-8'))
    parent = training/reference['directory']/'predictions.npz'
    paired = {}
    for label, directory in (('best', summary['diagnostic_directory']), ('last', summary['last_diagnostic_directory'])):
        paired[label+'_vs_strongest_cnn'] = paired_predictions(parent,
            training/JOB/directory/'selection/predictions.npz', training/f'paired_{label}_vs_strongest_cnn.csv')
    save_json(training/'paired_summary.json', paired)
    history = {JOB: pd.read_csv(training/JOB/'epochs.csv')}
    for column, label in (('selection_accuracy', 'raw'), ('ema_selection_accuracy', 'ema')):
        _selection_curve_png(training, history, column, f'joint_{label}_top1.png',
            f'Strongest independent CNN + ViT joint30: {label} development Top1 (%)')
    (training/'report.md').write_text('# Strongest independent CNN + ViT, joint training\n\n'
        'The CNN and ViT are independently selected source checkpoints. Their final features '
        'are concatenated and both encoders and the classifier are trained for the new phase. '
        'There is no new encoder pretraining.\n\n'
        f'Best: {row["best_top1_pct"]:.2f}% ({row["best_source"]}@{row["best_epoch"]}); '
        f'actual last raw/EMA: {row["last_top1_pct"]:.2f}%/{row["last_ema_top1_pct"]:.2f}%. '
        f'New training time: {row["train_minutes"]:.2f} minutes.\n\n'
        f'Source CNN: {row["source_cnn_top1_pct"]:.2f}%; source independent ViT: '
        f'{row["source_vit_top1_pct"]:.2f}%. Source CNN checkpoint follows '
        f'{plan["inherited_cnn_best_training_epochs"]} actual training epochs. The new phase '
        'reuses augmentation-view indices101..130 for a controlled comparison with the prior joint30; '
        'these indices do not count the source CNN training history.\n\n'
        'Source pretraining/search cost is inherited and recorded separately in suite provenance. '
        'Checkpoint selection remains development evidence. Audit and official validation are unused.\n', encoding='utf-8')
    return row


def run_training(plan, training):
    from vit_augreg_experiment import _train_job, _make_reference, _record_failure
    training = Path(training).resolve()
    sources = load_sources(plan['cnn_suite'], plan['vit_suite'], config=plan['config'])
    _validate_source_identity(plan, sources)
    _separate_output(training, plan['cnn_suite'], plan['vit_suite'], plan['parent_run'])
    if (training/'suite.json').is_file():
        if json.loads((training/'suite.json').read_text(encoding='utf-8')) != plan:
            raise ValueError('Training resume requires the recorded single joint phase')
    else:
        if training.exists() and any(training.iterdir()):
            raise ValueError('Refusing unrelated training output')
        training.mkdir(parents=True, exist_ok=True)
        save_json(training/'suite.json', plan)
    reference_plan = dict(plan, parent_checkpoint_sha256=plan['source_provenance']['parent_checkpoint_sha256'])
    failure = None
    try:
        marker = training/'parent_reference.json'
        if marker.exists():
            reference = json.loads(marker.read_text(encoding='utf-8'))
            relative = Path(reference['directory'])
            if (marker.is_symlink() or relative.is_absolute() or '..' in relative.parts
                    or reference.get('parent_checkpoint_sha256') != reference_plan['parent_checkpoint_sha256']
                    or not (training/relative/'predictions.npz').is_file()):
                raise ValueError('Saved strongest-CNN reference differs from the source checkpoint')
        _make_reference(reference_plan, training, sources['parent_model'])
        # The existing engine calls this field an epoch offset. Preserve actual
        # CNN history in provenance/plan, and use the old joint30 view indices
        # explicitly so only the CNN initialization changes in that comparison.
        training_sources = dict(sources, parent_completed_epochs=plan['fusion_augmentation_epoch_offset'])
        summary = _train_job(plan, JOB, training/JOB, training_sources, sources['selected'])
        training_report(plan, training, summary)
    except Exception as exc:
        failure = dict(error_type=type(exc).__name__, traceback=traceback.format_exc())
        _record_failure(plan, JOB, training/JOB, failure)
    immutable = all(Path(path).is_file() and file_hash(path) == digest for path, digest in plan['source_files'].items())
    result = dict(protocol=PROTOCOL, status='complete' if failure is None and immutable else 'failed',
        jobs={JOB: dict(status='complete', completed_epochs=plan['epochs'])} if failure is None else {},
        failures={JOB:failure} if failure else {}, source_files_unchanged=immutable,
        audit_partition_evaluated=False, official_validation_evaluated=False)
    save_json(training/'completion.json', result)
    if result['status'] != 'complete':
        raise RuntimeError('Best-pair joint training or source-integrity check failed; partial reports retained')
    return result


def run_suite(path):
    from best_pair_diagnostics import run_best_pair_diagnostics
    suite = Path(path).resolve()
    plan = validate_resume(suite)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no joint training started on CPU')
    outcomes = {}
    with ActiveBudget(suite, None) as budget:
        for name, action in (
            ('training', lambda: run_training(plan, suite/'training')),
            ('feature_analysis', lambda: run_best_pair_diagnostics(plan, suite/'training', suite/'feature_analysis', examples=plan['examples'])),
        ):
            atomic_json(suite/'status.json', dict(protocol=PROTOCOL, active_stage=name, outcomes=outcomes))
            try:
                result = action()
                outcomes[name] = dict(status=result.get('status', 'complete'))
            except Exception as exc:
                outcomes[name] = dict(status='failed', error_type=type(exc).__name__, traceback=traceback.format_exc())
                target = suite/name; target.mkdir(exist_ok=True)
                save_json(target/'stage_failure.json', outcomes[name])
                print(f'Best-pair stage {name} failed ({type(exc).__name__}); partial results retained.', flush=True)
            gc.collect(); torch.cuda.empty_cache()
        unchanged = all(Path(p).is_file() and file_hash(p) == digest for p, digest in plan['source_files'].items())
        status = 'complete' if unchanged and all(v['status'] == 'complete' for v in outcomes.values()) else 'failed'
        result = dict(protocol=PROTOCOL, status=status, outcomes=outcomes,
            epochs=plan['epochs'], total_new_model_epochs=plan['epochs'], smoke=plan['smoke'],
            elapsed_active_hours=budget.elapsed()/3600, source_checkpoints_unchanged=unchanged,
            shutdown_after_completion=False, audit_partition_evaluated=False, official_validation_evaluated=False)
        atomic_json(suite/'completion.json', result); atomic_json(suite/'status.json', result)
        (suite/'report.md').write_text('# Strongest CNN + strongest independent ViT\n\n'
            f'Status: **{status}**. One new joint phase of{plan["epochs"]} epochs.\n\n'
            '- [Training report](training/report.md)\n'
            '- [Feature evidence](feature_analysis/report.md)\n\n'
            'The reporting wrapper emails the analysis and leaves the instance running. '
            'Audit and official validation remain unused.\n', encoding='utf-8')
        if status != 'complete':
            raise RuntimeError('Best-pair suite contains incomplete work; reports retained')
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cnn-suite', default=str(DEFAULT_CNN_SOURCE))
    parser.add_argument('--vit-suite', default=str(DEFAULT_VIT_SOURCE))
    parser.add_argument('--epochs', '--fusion-epochs', dest='epochs', type=int, help='New joint epochs, default30')
    parser.add_argument('--data-dir')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', '--output-dir', default=str(ROOT/'runs/best_pair_joint_tiny'))
    parser.add_argument('--resume')
    return parser


def main(argv=None):
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no joint suite created')
    if args.resume:
        plan = validate_resume(args.resume)
        if args.smoke or (args.epochs is not None and args.epochs != plan['epochs']):
            raise ValueError('Resume keeps the recorded smoke/full protocol and horizon')
        suite = Path(args.resume)
    else:
        suite = create_suite(args)
    run_suite(suite)


if __name__ == '__main__': main()
