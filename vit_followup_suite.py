"""Run the requested independent-ViT and CNN feature-fusion experiments.

The finite queue is ViT100 -> joint late fusion30, shallow64 attention30,
and two cross-attention/MLP blocks30. This module never sends email or stops
a VM; run_vit_followup_overnight.py adds those delivery operations.
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
from generalization_suite import ActiveBudget, atomic_json
from literature_experiment import ROOT, new_name, recipe_hash, save_json
from mechanism_diagnostics import load_spatial_parent
from spatial_experiment import execution_environment, source_hashes as spatial_sources


PROTOCOL = 'tiny-vit-followup-suite-v1'
DEFAULT_PARENT = ROOT/'runs/spatial_tiny/suite_20261103T175622_a19c86/cnn8_seed42'
SOURCES = ('vit_followup_suite.py', 'vit_followup_experiment.py',
           'vit_followup_diagnostics.py', 'models/vit_followup.py',
           'run_vit_followup_overnight.py', 'generalization_suite.py',
           'generalization_experiment.py', 'fusion30_experiment.py',
           'models/attention_fusion.py', 'models/vit.py',
           'attention_fusion_experiment.py', 'attention_fusion_diagnostics.py',
           'run_notified.py', 'overnight_twostage.py')


def source_hashes():
    return {**spatial_sources(), **{name: file_hash(ROOT/name) for name in SOURCES}}


def _outside_parent(output, parent):
    if output == parent or parent in output.parents or output in parent.parents:
        raise ValueError('Output must be a separate directory outside the original CNN run')


def make_plan(args):
    vit_epochs = 1 if args.smoke else (100 if args.vit_epochs is None else args.vit_epochs)
    fusion_epochs = 1 if args.smoke else (30 if args.fusion_epochs is None else args.fusion_epochs)
    if not 1 <= vit_epochs <= 100 or not 1 <= fusion_epochs <= 30:
        raise ValueError('Need ViT1..100 and follow-up1..30 epochs')
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
    _outside_parent(output, parent)
    train, selection, _, split, manifest = clean_datasets(config, summary['seed'], summary['split_hash'], 0)
    if (len(train) != 80000 or len(selection) != 10000
            or set(manifest['train_clean']) & set(manifest['selection'])):
        raise ValueError('Expected the original disjoint80K/10K development partitions')
    return dict(protocol=PROTOCOL, parent_run=str(parent), parent_checkpoint_sha256=digest,
        parent_best_epoch=summary['best_epoch'], seed=summary['seed'], split_hash=split,
        config=config, config_recipe_hash=recipe_hash(config), source_hashes=source_hashes(),
        execution_environment=execution_environment(config, summary['seed']),
        vit_epochs=vit_epochs, fusion_epochs=fusion_epochs,
        complete_all=True, max_hours=None, smoke=bool(args.smoke), examples=args.examples,
        stages=['training', 'feature_analysis'],
        jobs=['vit_pretrain', 'dual_fusion', 'shallow64_attention', 'cross16_mlp'],
        total_model_epochs=vit_epochs+3*fusion_epochs,
        vit=dict(image_size=64, patch_size=4, width=224, depth=5, heads=4,
                 mlp_ratio=4., dropout=.1, initialization='scratch'),
        dual_fusion=dict(cnn='original CNN8 best', vit='vit_pretrain selected best',
                         features='CNN pooled2048 plus ViT CLS224',
                         parameter_sharing=False, cnn_trainable=True, vit_trainable=True,
                         operation='feature concatenation then one classifier'),
        shallow64=dict(feature_source='same CNN stem kernel applied at stride1',
                       tokens=4096, qkv_source='same native64 shallow feature map',
                       width=128, heads=4, query_chunk_size=64,
                       attention='exact global attention with query chunks; checkpointed in training',
                       original_cnn_path_preserved=True),
        cross16_mlp=dict(query='deep8x8', key_value='middle16x16',
                         width=128, heads=4, depth=2, mlp_ratio=4.,
                         normalization='pre-LayerNorm with attention and MLP residuals'),
        comparison_note='New architecture screens. Dual training includes ViT100 plus joint30 '
                        'and the existing CNN pretraining; this is not an equal-cost fresh100 comparison. '
                        'Cross16-MLP changes depth and latent processing as well as adding an MLP.',
        audit_partition_evaluated=False, official_validation_evaluated=False)


def create_suite(args, plan=None):
    plan = make_plan(args) if plan is None else plan
    root, parent = Path(args.output).resolve(), Path(plan['parent_run']).resolve()
    _outside_parent(root, parent)
    suite = root/new_name('smoke' if plan['smoke'] else 'suite')
    suite.mkdir(parents=True, exist_ok=False)
    save_json(suite/'suite.json', plan)
    print(f'ViT follow-up suite: {suite}', flush=True)
    return suite


def validate_resume(path):
    suite = Path(path).resolve()
    plan = json.loads((suite/'suite.json').read_text(encoding='utf-8'))
    if plan.get('protocol') != PROTOCOL or plan.get('source_hashes') != source_hashes():
        raise ValueError('Resume requires the recorded ViT follow-up protocol and source files')
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
    from vit_followup_experiment import run_followup_study
    from vit_followup_diagnostics import run_followup_diagnostics
    suite = Path(path).resolve()
    plan = validate_resume(suite)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; ViT follow-up suite not started on CPU')
    outcomes = {}
    with ActiveBudget(suite, None) as budget:
        stages = (
            ('training', lambda: run_followup_study(plan['parent_run'], suite/'training',
                config=copy.deepcopy(plan['config']), vit_epochs=plan['vit_epochs'],
                fusion_epochs=plan['fusion_epochs'], examples=plan['examples'],
                smoke=plan['smoke'], resume=True)),
            ('feature_analysis', lambda: run_followup_diagnostics(plan['parent_run'], suite/'training',
                suite/'feature_analysis', config=copy.deepcopy(plan['config']), examples=plan['examples'])),
        )
        for name, action in stages:
            print(f'\nStarting ViT follow-up stage: {name}', flush=True)
            atomic_json(suite/'status.json', dict(protocol=PROTOCOL, active_stage=name, outcomes=outcomes))
            try:
                result = action()
                outcomes[name] = dict(status=result.get('status', 'complete'))
            except Exception as exc:
                outcomes[name] = dict(status='failed', error_type=type(exc).__name__, traceback=traceback.format_exc())
                directory = suite/name
                directory.mkdir(exist_ok=True)
                save_json(directory/'stage_failure.json', outcomes[name])
                print(f'Follow-up stage {name} failed ({type(exc).__name__}); partial outputs retained.', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        unchanged = file_hash(Path(plan['parent_run'])/'best.pt') == plan['parent_checkpoint_sha256']
        status = 'complete' if unchanged and all(item['status'] == 'complete' for item in outcomes.values()) else 'failed'
        result = dict(protocol=PROTOCOL, status=status, outcomes=outcomes,
            complete_all=True, elapsed_active_hours=budget.elapsed()/3600,
            parent_checkpoint_unchanged=unchanged, vit_epochs=plan['vit_epochs'],
            fusion_epochs=plan['fusion_epochs'], smoke=plan['smoke'],
            audit_partition_evaluated=False, official_validation_evaluated=False)
        atomic_json(suite/'completion.json', result)
        atomic_json(suite/'status.json', result)
        (suite/'report.md').write_text('# Independent ViT and CNN feature-fusion follow-up\n\n'
            f'Status: **{status}**. The finite queue is ViT100, joint fusion30, '
            'native64 self-attention30 and cross-attention/MLP30 (formal defaults).\n\n'
            '- [Training comparisons](training/report.md)\n'
            '- [Feature/branch interventions](feature_analysis/report.md)\n\n'
            'Pure ViT and CNN have independent parameters in the late-fusion model. '
            'Both receive the same augmented image and are updated by the fused classification loss. '
            'The fusion schedule includes earlier pretraining costs. Audit and official validation remain unused.\n',
            encoding='utf-8')
        if status != 'complete':
            raise RuntimeError('ViT follow-up suite contains failures; partial reports have been retained')
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', default=str(DEFAULT_PARENT))
    parser.add_argument('--data-dir')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--vit-epochs', type=int, help='Independent ViT schedule, default100, maximum100')
    parser.add_argument('--fusion-epochs', type=int, help='Each of the three follow-up schedules, default30, maximum30')
    parser.add_argument('--complete-all', action='store_true', help='Finish the four planned jobs without a time cutoff (default)')
    parser.add_argument('--examples', type=int, default=8)
    parser.add_argument('--smoke', action='store_true', help='Separate one-epoch check; not a formal result')
    parser.add_argument('--output', '--output-dir', default=str(ROOT/'runs/vit_followup_tiny'))
    parser.add_argument('--resume')
    return parser


def main(argv=None):
    args = arguments().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no follow-up suite created')
    if args.resume:
        plan = validate_resume(args.resume)
        if args.smoke or any(getattr(args, key) is not None and getattr(args, key) != plan[key]
                             for key in ('vit_epochs', 'fusion_epochs')):
            raise ValueError('Resume retains its recorded full/smoke protocol and epoch schedules')
        suite = Path(args.resume)
    else:
        suite = create_suite(args)
    run_suite(suite)


if __name__ == '__main__':
    main()
