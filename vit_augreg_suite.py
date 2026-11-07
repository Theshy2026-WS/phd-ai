"""Finite, from-scratch AugReg ViT search followed by matched CNN/ViT fusion.

The Steiner et al. paper guides the search, not a claim of reproducing its
ImageNet experiments. Existing Tiny ImageNet partitions and sources are fixed.
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
from fusion_continuation_sources import DEFAULT_SOURCE, load_sources
from generalization_suite import ActiveBudget, atomic_json
from literature_experiment import ROOT, new_name, recipe_hash, save_json
from spatial_experiment import execution_environment, source_hashes as spatial_sources


PROTOCOL = 'tiny-vit-augreg-suite-v1'
DEFAULT_PROFILES = ('vit_tiny_p4_aug', 'vit_small_p8_aug', 'vit_small_p8_augreg')
DOWNSTREAM_JOBS = ('cnn_frozen_head', 'new_frozen_concat', 'new_joint_concat',
    'new_frozen_normalized', 'new_frozen_residual_linear', 'new_frozen_residual',
    'new_frozen_cross_patch', 'new_frozen_mlp_patch')
SOURCES = ('vit_augreg_suite.py', 'vit_augreg_experiment.py',
    'vit_augreg_diagnostics.py', 'run_vit_augreg_overnight.py', 'models/vit_augreg.py',
    'fusion_continuation_sources.py', 'fusion_continuation_experiment.py',
    'fusion_continuation_diagnostics.py', 'vit_followup_experiment.py',
    'vit_followup_diagnostics.py', 'models/vit_followup.py', 'models/vit.py',
    'attention_fusion_diagnostics.py', 'generalization_suite.py',
    'generalization_experiment.py', 'fusion30_experiment.py', 'run_notified.py',
    'overnight_twostage.py', 'datasets/resnet_tiny.py', 'datasets/utility_tiny.py')


def source_hashes():
    return {**spatial_sources(), **{name: file_hash(ROOT/name) for name in SOURCES}}


def profile_recipe(base, name, epochs, smoke=False):
    if name not in DEFAULT_PROFILES:
        raise ValueError('Unknown AugReg profile')
    config = copy.deepcopy(base)
    strong = name.endswith('augreg')
    config['augmentation'] = dict(crop_scale=[.5, 1.], randaugment_ops=2,
        randaugment_magnitude=9 if strong else 7)
    config['training'].update(epochs=epochs, learning_rate=5e-4,
        vit_learning_rate=5e-4, encoder_learning_rate=5e-4,
        fusion_learning_rate=5e-4, accumulation_steps=1 if smoke else 4,
        weight_decay=.03 if strong else .1,
        label_smoothing=.1, mixup_alpha=.5 if strong else .2,
        cutmix_alpha=0., mix_probability=1., gradient_clip_norm=1.,
        betas=[.9, .999], eps=1e-8, early_stopping_patience=0,
        probe_per_class=20)
    config['training']['scheduler'] = dict(name='augreg-full-horizon-cosine',
        warmup_epochs=min(20, epochs-1), warmup_start_factor=.01,
        min_lr=1e-6, min_ratio=.002)
    return config


def _separate_output(output, source, parent):
    for protected in (source, parent):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Output must be separate from the immutable source suite and CNN parent')


def make_plan(args):
    epochs = 1 if args.smoke else (300 if args.vit_epochs is None else args.vit_epochs)
    fusion = 1 if args.smoke else (30 if args.fusion_epochs is None else args.fusion_epochs)
    if not 1 <= epochs <= 600 or not 1 <= fusion <= 30:
        raise ValueError('Need ViT1..600 and fusion1..30 epochs')
    profiles = list(args.profiles or DEFAULT_PROFILES)
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in DEFAULT_PROFILES for p in profiles):
        raise ValueError('Need unique known AugReg profiles')
    if args.examples < 0:
        raise ValueError('Examples must be nonnegative')
    sources = load_sources(args.source_suite)
    config = copy.deepcopy(sources['config'])
    data = Path(args.data_dir or config['dataset']['data_dir']).expanduser()
    config['dataset']['data_dir'] = str((data if data.is_absolute() else ROOT/data).resolve())
    config['dataset']['download'] = bool(args.allow_download)
    if args.num_workers is not None:
        config['dataset']['num_workers'] = args.num_workers
    if config['dataset']['num_workers'] < 0:
        raise ValueError('Workers must be nonnegative')
    if config['dataset']['batch_size'] != 64:
        raise ValueError('Matched fusion requires the source microbatch64')
    output = Path(args.output).resolve()
    _separate_output(output, Path(sources['source_suite']), Path(sources['parent_run']))
    train, selection, _, split, manifest = clean_datasets(config, sources['seed'], sources['split_hash'], 0)
    if (len(train) != 80000 or len(selection) != 10000 or
            set(manifest['train_clean']) & set(manifest['selection']) or split != sources['split_hash']):
        raise ValueError('Expected the original disjoint80K/10K development partitions')
    environment = execution_environment(config, sources['seed'])
    if environment != sources['provenance']['source_execution_environment']:
        raise ValueError('Matched comparison requires the original GPU, software and precision environment')
    return dict(protocol=PROTOCOL, source_suite=sources['source_suite'],
        source_provenance=sources['provenance'], source_files=sources['source_files'],
        parent_run=sources['parent_run'], seed=sources['seed'], split_hash=split,
        config=config, config_recipe_hash=recipe_hash(config), source_hashes=source_hashes(),
        execution_environment=environment, vit_epochs=epochs, fusion_epochs=fusion,
        profiles=profiles, profile_recipes={p: profile_recipe(config, p, epochs, args.smoke) for p in profiles},
        jobs=[f'pretrain_{p}' for p in profiles]+list(DOWNSTREAM_JOBS),
        total_new_model_epochs=len(profiles)*epochs+len(DOWNSTREAM_JOBS)*fusion,
        complete_all=True, max_hours=None, smoke=bool(args.smoke), examples=args.examples,
        stages=['training', 'feature_analysis'],
        reference_paper='https://arxiv.org/abs/2106.10270',
        paper_scope='Native64 adaptation of AugReg principles, not an ImageNet reproduction',
        external_pretraining=False, independent_vit_initialization='from scratch, seed42',
        selection_policy='All scheduled candidates complete; highest best development Top1, then lower NLL, then profile order',
        fusion_policy=dict(cnn='same original CNN100 best for every job',
            augmentation='source recipe, batch64, no accumulation; augmentation epochs101..130',
            joint_lr=dict(encoder=5e-5, vit=5e-5, fusion=5e-4),
            frozen='encoders and BN buffers fixed; encoder dropout disabled',
            epoch0='Measured separately; normalization can change inherited-head calibration'),
        attention_policy=dict(query='one CNN pooled query',
            key_value='independent ViT patch tokens:64 for Small/8 or256 for Tiny/4',
            controls=['uniformattention', 'cross-image ViT mismatch', 'train-probe mean', 'MLP patch pooling']),
        excluded_training_routes=['shallow64', '4096-token global attention', 'external ImageNet pretrained weights', 'seed sweep'],
        audit_partition_evaluated=False, official_validation_evaluated=False)


def create_suite(args, plan=None):
    plan = make_plan(args) if plan is None else plan
    root = Path(args.output).resolve()
    _separate_output(root, Path(plan['source_suite']), Path(plan['parent_run']))
    suite = root/new_name('smoke' if plan['smoke'] else 'suite')
    suite.mkdir(parents=True, exist_ok=False)
    save_json(suite/'suite.json', plan)
    print(f'ViT AugReg suite: {suite}', flush=True)
    return suite


def validate_resume(path):
    suite = Path(path).resolve()
    plan = json.loads((suite/'suite.json').read_text(encoding='utf-8'))
    if plan.get('protocol') != PROTOCOL or plan.get('source_hashes') != source_hashes():
        raise ValueError('Resume requires the recorded AugReg protocol and source files')
    if recipe_hash(plan['config']) != plan['config_recipe_hash']:
        raise ValueError('Recorded training/data configuration changed')
    profiles = plan['profiles']
    expected = {p: profile_recipe(plan['config'], p, plan['vit_epochs'], plan['smoke']) for p in profiles}
    if expected != plan['profile_recipes']:
        raise ValueError('Recorded AugReg profile recipes changed')
    if plan['jobs'] != [f'pretrain_{p}' for p in profiles]+list(DOWNSTREAM_JOBS):
        raise ValueError('Recorded AugReg queue changed')
    sources = load_sources(plan['source_suite'], config=plan['config'])
    if sources['provenance'] != plan['source_provenance'] or sources['source_files'] != plan['source_files']:
        raise ValueError('Original source checkpoints or metadata changed')
    if execution_environment(plan['config'], plan['seed']) != plan['execution_environment']:
        raise ValueError('Resume requires the recorded GPU, software and precision environment')
    clean_datasets(plan['config'], plan['seed'], plan['split_hash'], 0)
    return plan


def run_suite(path):
    from vit_augreg_experiment import run_augreg_study
    from vit_augreg_diagnostics import run_augreg_diagnostics
    suite = Path(path).resolve()
    plan = validate_resume(suite)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; AugReg training not started on CPU')
    outcomes = {}
    with ActiveBudget(suite, None) as budget:
        stages = (
            ('training', lambda: run_augreg_study(plan, suite/'training', examples=plan['examples'], resume=True)),
            ('feature_analysis', lambda: run_augreg_diagnostics(plan, suite/'training', suite/'feature_analysis', examples=plan['examples'])),
        )
        for name, action in stages:
            print(f'\nStarting AugReg stage: {name}', flush=True)
            atomic_json(suite/'status.json', dict(protocol=PROTOCOL, active_stage=name, outcomes=outcomes))
            try:
                result = action()
                outcomes[name] = dict(status=result.get('status', 'complete'))
            except Exception as exc:
                outcomes[name] = dict(status='failed', error_type=type(exc).__name__, traceback=traceback.format_exc())
                target = suite/name
                target.mkdir(exist_ok=True)
                save_json(target/'stage_failure.json', outcomes[name])
                print(f'AugReg stage {name} failed ({type(exc).__name__}); retaining partial results.', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        unchanged = all(Path(name).is_file() and file_hash(name) == digest for name, digest in plan['source_files'].items())
        status = 'complete' if unchanged and all(v['status'] == 'complete' for v in outcomes.values()) else 'failed'
        result = dict(protocol=PROTOCOL, status=status, outcomes=outcomes, complete_all=True,
            elapsed_active_hours=budget.elapsed()/3600, source_checkpoints_unchanged=unchanged,
            vit_epochs=plan['vit_epochs'], fusion_epochs=plan['fusion_epochs'], profiles=plan['profiles'],
            total_new_model_epochs=plan['total_new_model_epochs'], smoke=plan['smoke'],
            external_pretraining=False, excluded_training_routes=plan['excluded_training_routes'],
            audit_partition_evaluated=False, official_validation_evaluated=False)
        atomic_json(suite/'completion.json', result)
        atomic_json(suite/'status.json', result)
        (suite/'report.md').write_text('# AugReg ViT and matched CNN/ViT fusion\n\n'
            f'Status: **{status}**. {len(plan["profiles"])} independent candidates, '
            f'{plan["vit_epochs"]} epochs each; eight fusion/control jobs, '
            f'{plan["fusion_epochs"]} epochs each.\n\n'
            '- [Training and candidate selection](training/report.md)\n'
            '- [Independent representations and fusion interventions](feature_analysis/report.md)\n\n'
            'This is an AugReg-informed Tiny ImageNet study, not a reproduction of the '
            'paper or a claim of architectural novelty. All reported selection scores '
            'remain development evidence. Audit and official validation are unused.\n', encoding='utf-8')
        if status != 'complete':
            raise RuntimeError('AugReg suite contains failures; partial reports retained')
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-suite', default=str(DEFAULT_SOURCE))
    parser.add_argument('--data-dir')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--vit-epochs', type=int, help='Independent candidate horizon, default300')
    parser.add_argument('--fusion-epochs', type=int, help='Each fusion/control horizon, default30')
    parser.add_argument('--profiles', nargs='+', choices=DEFAULT_PROFILES)
    parser.add_argument('--complete-all', action='store_true', help='Finish the complete finite queue (default)')
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', '--output-dir', default=str(ROOT/'runs/vit_augreg_tiny'))
    parser.add_argument('--resume')
    return parser


def main(argv=None):
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no AugReg suite created')
    if args.resume:
        plan = validate_resume(args.resume)
        if (args.smoke or any(getattr(args, key) is not None and getattr(args, key) != plan[key]
                for key in ('vit_epochs', 'fusion_epochs', 'profiles'))):
            raise ValueError('Resume retains recorded full/smoke protocol, profiles and horizons')
        suite = Path(args.resume)
    else:
        suite = create_suite(args)
    run_suite(suite)


if __name__ == '__main__':
    main()
