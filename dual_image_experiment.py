"""Four independent-image fusion jobs. Never shuts down or sends email."""
import argparse
import copy
import gc
import json
from pathlib import Path

import pandas as pd
import torch

from correction_experiment import find_parent, file_hash, base_evaluation
from literature_experiment import ROOT, new_name, save_json, source_hashes, train_run
from models.literature_resnet import build_model
from models.dual_image_fusion import DualImageFusion, AACNNTiny

JOBS = {'prediction100': 100, 'feature100': 100, 'prediction25': 25, 'aacnn100': 100}


def factory(plan):
    def build(config, variant, seed, fingerprint):
        if fingerprint != plan['split_hash']:
            raise ValueError('Split differs from parent CNN')
        aacnn = variant == 'aacnn100'
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            if aacnn:
                model = AACNNTiny()
            else:
                if file_hash(plan['base_checkpoint']) != plan['base_sha256']:
                    raise ValueError('Parent checkpoint changed')
                state = torch.load(plan['base_checkpoint'], map_location='cpu', weights_only=False)
                base = build_model('resnet50_preact', seed)
                base.load_state_dict(state['model'], strict=True)
                model = DualImageFusion(base, 'feature' if variant == 'feature100' else 'prediction')
        return model, dict(
            architecture=dict(name=variant, independent_raw_image_branches=True,
                              image_size=64, vit_patch=8, vit_tokens=65, vit_width=192,
                              vit_depth=4, vit_heads=4, vit_mlp_ratio=2,
                              cnn='3 conv blocks 32/64/128' if aacnn else 'frozen resnet50_preact',
                              fusion='256D two-way softmax feature gate, 64D hidden' if aacnn
                              else model.mode,
                              auxiliary_vit_ce_weight=0 if aacnn else .5,
                              external_pretraining=False,
                              method_note='AACNN fusion-structure adaptation only; both branches scratch, native64, common CE recipe'
                              if aacnn else 'Experimental dynamic fusion; novelty unverified'),
            experiment_protocol=plan['protocol'], parent_training_epochs=0 if aacnn else 100,
            parent_train_seconds=0 if aacnn else plan['base_train_seconds'],
            parent_checkpoint_sha256=None if aacnn else plan['base_sha256'])
    return build


def create_suite(args):
    parent, summary, checkpoint = find_parent(args.base_run)
    config = copy.deepcopy(checkpoint['config'])
    config['training']['early_stopping_patience'] = getattr(args, 'patience', 10)
    if config['training']['early_stopping_patience'] < 0:
        raise ValueError('Patience must be nonnegative')
    data = Path(config['dataset']['data_dir'])
    config['dataset']['data_dir'] = str(data if data.is_absolute() else ROOT/data)
    plan = dict(protocol='tiny-dual-image-v1', config=config, seed=42,
                base_run=str(parent), base_checkpoint=str(parent/'best.pt'),
                base_sha256=file_hash(parent/'best.pt'), split_hash=summary['split_hash'],
                base_train_seconds=summary['total_train_seconds'], source_hashes=source_hashes(),
                smoke=args.smoke,
                jobs=[dict(variant=v, epochs=1 if args.smoke else JOBS[v],
                           intended_full_epochs=JOBS[v]) for v in args.variants])
    suite = ROOT/'runs/dual_image'/new_name('smoke' if args.smoke else 'suite')
    suite.mkdir(parents=True)
    save_json(suite/'suite.json', plan)
    save_json(suite/'parent_summary.json', summary)
    return suite


def comparison(suite, plan):
    rows = []
    base = json.loads((suite/'frozen_base.json').read_text(encoding='utf-8'))
    for job in plan['jobs']:
        path = suite/job['variant']/'summary.json'
        if not path.exists():
            continue
        s = json.loads(path.read_text(encoding='utf-8'))
        rows.append(dict(variant=job['variant'], epochs=s['completed_epochs'],
                         planned_epochs=s.get('planned_epochs', job.get('epochs', job['intended_full_epochs'])),
                         stopped_early=s.get('stopped_early', False),
                         best_epoch=s['best_epoch'],
                         top1_pct=100*s['heldout']['accuracy'],
                         delta_vs_frozen_cnn_pp=100*(s['heldout']['accuracy']-base['heldout']['accuracy']),
                         top5_pct=100*s['heldout']['top5_accuracy'],
                         params=s['params'], trainable_params=s['trainable_params'],
                         gflops=s['forward_flops_estimate']/1e9,
                         flops_change_pct=100*(s['forward_flops_estimate']/base['forward_flops_estimate']-1),
                         train_minutes=s['total_train_seconds']/60,
                         parent_train_minutes=s['parent_train_seconds']/60,
                         total_train_minutes=(s['total_train_seconds']+s['parent_train_seconds'])/60,
                         experiment_minutes=s['total_experiment_seconds']/60,
                         latency_batch1_ms=s['inference']['1']['latency_per_batch_ms'],
                         throughput=s['inference'][str(s['inference_batch_size'])]['throughput_images_per_second'],
                         rough_full_train_minutes=s['total_train_seconds']/60/s['completed_epochs']*job['intended_full_epochs']))
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(suite/'comparison.csv', index=False)
        print(frame.to_string(index=False), flush=True)


def run_suite(suite):
    plan = json.loads((suite/'suite.json').read_text(encoding='utf-8'))
    if plan['source_hashes'] != source_hashes():
        raise ValueError('Source changed; do not silently resume')
    if file_hash(plan['base_checkpoint']) != plan['base_sha256']:
        raise ValueError('Parent checkpoint changed')
    if not (suite/'frozen_base.json').exists():
        base_evaluation(plan, suite)
    for job in plan['jobs']:
        run = suite/job['variant']
        if (run/'summary.json').exists():
            continue
        config = copy.deepcopy(plan['config'])
        config['training'].update(epochs=job['epochs'], mixed_precision=False,
                                  learning_rate=.0003)
        config['training']['scheduler']['warmup_epochs'] = min(2, job['epochs']-1)
        train_run(config, job['variant'], 42, run, resume=(run/'last.pt').exists(),
                  model_factory=factory(plan))
        comparison(suite, plan)
        gc.collect()
        torch.cuda.empty_cache()
    print(f'Completed: {suite}\nNo shutdown requested. Instance remains running.', flush=True)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run', help='Completed 100-epoch resnet50_preact seed42 directory')
    p.add_argument('--patience', type=int, default=10, help='Selection Top-1 patience; 0 disables early stopping')
    p.add_argument('--smoke', action='store_true', help='Separate one-epoch screening for every job')
    p.add_argument('--variants', nargs='+', choices=JOBS, default=list(JOBS))
    p.add_argument('--resume', help='Resume a suite without changing its saved schedule')
    return p


def main():
    args = arguments().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing CPU training')
    if args.resume and args.smoke:
        raise ValueError('Resume uses saved schedule; omit --smoke')
    suite = Path(args.resume).resolve() if args.resume else create_suite(args)
    print(f'Suite: {suite}', flush=True)
    run_suite(suite)


if __name__ == '__main__':
    main()
