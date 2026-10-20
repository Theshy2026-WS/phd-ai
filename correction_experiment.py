"""Frozen 100-epoch CNN correction + compact pure ViT on identical Tiny splits.

No email, downloads, shutdown or cloud mutations in this training entry point.
Use overnight_correction.py for automatic bundling and managed VM shutdown.
Only load trusted local checkpoints (torch pickle).
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import torch

from literature_experiment import (ROOT, PROTOCOL, new_name, save_json, train_run,
                                  evaluate, source_hashes)
from models.literature_resnet import build_model
from models.attention_correction import FrozenAttentionCorrection, TinyViTFP32
from datasets.resnet_tiny import prepare_datasets
from datasets.utility_tiny import loader
from utility_experiment import runtime
from utils.literature_profiling import estimate_flops
from utils.profiling import benchmark_inference

VARIANTS = ('attention_correction', 'pooled_correction', 'vit_tiny64')


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def find_parent(path=None):
    if path:
        candidates = [Path(path).resolve()]
    else:
        candidates = []
        for summary_path in (ROOT/'runs').rglob('summary.json'):
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            if (summary.get('variant') == 'resnet50_preact' and
                    summary.get('completed_epochs') == 100 and
                    summary.get('seed') == 42 and (summary_path.parent/'best.pt').exists()):
                candidates.append(summary_path.parent.resolve())
    if len(candidates) != 1:
        raise ValueError('Specify --base-run with ONE completed 100-epoch resnet50_preact directory. '
                         f'Candidates: {[str(p) for p in candidates]}')
    parent = candidates[0]
    summary = json.loads((parent/'summary.json').read_text(encoding='utf-8'))
    if (summary.get('variant') != 'resnet50_preact' or
            summary.get('completed_epochs') != 100 or summary.get('seed') != 42 or
            summary.get('protocol') != PROTOCOL):
        raise ValueError('Need seed42 resnet50_preact, completed 100 epochs, literature split')
    checkpoint = torch.load(parent/'best.pt', map_location='cpu', weights_only=False)
    for key in ('variant', 'seed', 'split_hash', 'source_hashes', 'recipe_hash'):
        if checkpoint['metadata'][key] != summary[key]:
            raise ValueError(f'Parent summary/checkpoint mismatch: {key}')
    if checkpoint['epoch'] != summary['best_epoch']:
        raise ValueError('Parent best epoch mismatch')
    # Native model definition must still match the one used by the trained CNN.
    for name in ('models/literature_twostage.py', 'models/literature_alternet.py'):
        if summary['source_hashes'].get(name) != file_hash(ROOT/name):
            raise ValueError(f'Parent CNN implementation differs: {name}')
    model = build_model('resnet50_preact', 42)
    model.load_state_dict(checkpoint['model'], strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point()):
        raise ValueError('Non-finite parent checkpoint')
    return parent, summary, checkpoint


def make_plan(args):
    parent, summary, checkpoint = find_parent(args.base_run)
    config = copy.deepcopy(checkpoint['config'])
    # Resolve data relative to repository, not to the historical run directory.
    data = Path(config['dataset']['data_dir'])
    config['dataset']['data_dir'] = str(data if data.is_absolute() else (ROOT/data).resolve())
    return dict(protocol='tiny-correction-v1', base_run=str(parent),
                base_checkpoint=str(parent/'best.pt'),
                base_sha256=file_hash(parent/'best.pt'), base_best_epoch=checkpoint['epoch'],
                base_training_epochs=100, base_train_seconds=summary['total_train_seconds'],
                split_hash=summary['split_hash'], seed=42, config=config,
                correction_epochs=args.correction_epochs, vit_epochs=args.vit_epochs,
                variants=args.variants, source_hashes=source_hashes())


def factory(plan):
    def build(config, variant, seed, fingerprint):
        if fingerprint != plan['split_hash']:
            raise ValueError('Dataset split does not match trained CNN')
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            if variant == 'vit_tiny64':
                model = TinyViTFP32()
                description = dict(name=variant, pure_vit=True, input_size=64,
                                   patch_size=4, tokens=257, embed_dim=224, depth=5,
                                   heads=4, pretrained=False, precision='float32',
                                   source='existing models/vit.py compact baseline')
            else:
                if file_hash(plan['base_checkpoint']) != plan['base_sha256']:
                    raise ValueError('Base checkpoint changed')
                state = torch.load(plan['base_checkpoint'], map_location='cpu', weights_only=False)
                base = build_model('resnet50_preact', seed)
                base.load_state_dict(state['model'], strict=True)
                model = FrozenAttentionCorrection(base, mode='pooled' if variant == 'pooled_correction' else 'attention')
                description = dict(name=variant, backbone='resnet50_preact', frozen=True,
                                   spatial_features=[1024, 8, 8], width=192, heads=4,
                                   fusion='original FP32 logits + zero-initialized correction',
                                   readout='prediction-conditioned cross-attention' if variant == 'attention_correction' else 'mean pooling + prediction query',
                                   precision='float32', novelty='unverified candidate; side-tuning precedent')
        return model, dict(architecture=description, experiment_protocol=plan['protocol'],
                           parent_checkpoint_sha256=None if variant == 'vit_tiny64' else plan['base_sha256'],
                           parent_training_epochs=0 if variant == 'vit_tiny64' else 100,
                           parent_train_seconds=0 if variant == 'vit_tiny64' else plan['base_train_seconds'],
                           parent_best_epoch=None if variant == 'vit_tiny64' else plan['base_best_epoch'])
    return build


def base_evaluation(plan, suite):
    config = plan['config']
    device, amp, dtype, _ = runtime(config, 42)
    datasets, _, fingerprint = prepare_datasets(config, 42)
    if fingerprint != plan['split_hash']:
        raise ValueError('Parent split mismatch')
    state = torch.load(plan['base_checkpoint'], map_location='cpu', weights_only=False)
    model = build_model('resnet50_preact', 42).to(device)
    model.load_state_dict(state['model'], strict=True)
    score = evaluate(model, loader(datasets['audit'], config['dataset'], 42), device, False, dtype)
    profile = {str(b): benchmark_inference(model, (3, 64, 64), device, b,
               amp_enabled=False, **config['profiling']) for b in (1, config['dataset']['batch_size'])}
    save_json(suite/'frozen_base.json', dict(variant='frozen_base', heldout=score,
              params=sum(p.numel() for p in model.parameters()), trainable_params=0,
              forward_flops_estimate=estimate_flops(model, device),
              total_train_seconds=0, parent_train_seconds=plan['base_train_seconds'],
              inference=profile, inference_batch_size=config['dataset']['batch_size'],
              parent_checkpoint_sha256=plan['base_sha256'], split_hash=fingerprint))
    print(f'Frozen base held-out Top-1: {score["accuracy"]:.4f}', flush=True)
    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


def write_comparison(suite):
    paths = [suite/'frozen_base.json', *[suite/v/'summary.json' for v in VARIANTS]]
    rows = []
    for path in paths:
        if not path.exists():
            continue
        s = json.loads(path.read_text(encoding='utf-8'))
        rows.append(dict(variant=s['variant'], top1_pct=100*s['heldout']['accuracy'],
                         top5_pct=100*s['heldout']['top5_accuracy'], params=s['params'],
                         trainable_params=s['trainable_params'],
                         gflops=s['forward_flops_estimate']/1e9,
                         additional_train_minutes=s['total_train_seconds']/60,
                         total_train_minutes=(s['total_train_seconds']+s['parent_train_seconds'])/60,
                         latency_batch1_ms=s['inference']['1']['latency_per_batch_ms'],
                         images_per_second=s['inference'][str(s['inference_batch_size'])]['throughput_images_per_second'],
                         peak_training_memory_mb=s.get('peak_training_memory_bytes', 0)/1024**2,
                         best_epoch=s.get('best_epoch'),
                         epochs=s.get('completed_epochs', 0)))
    if rows:
        base = next(row['top1_pct'] for row in rows if row['variant'] == 'frozen_base')
        for row in rows:
            row['delta_vs_base_pp'] = row['top1_pct']-base
        pd.DataFrame(rows).to_csv(suite/'comparison.csv', index=False)


def run_suite(suite):
    suite = Path(suite)
    plan = json.loads((suite/'suite.json').read_text(encoding='utf-8'))
    if source_hashes() != plan['source_hashes']:
        raise ValueError('Source changed since suite creation; cannot resume silently')
    if file_hash(plan['base_checkpoint']) != plan['base_sha256']:
        raise ValueError('Parent checkpoint changed')
    if not (suite/'frozen_base.json').exists():
        base_evaluation(plan, suite)
    for variant in plan['variants']:
        run = suite/variant
        if (run/'summary.json').exists():
            continue
        config = copy.deepcopy(plan['config'])
        t = config['training']
        t['epochs'] = plan['vit_epochs'] if variant == 'vit_tiny64' else plan['correction_epochs']
        t['mixed_precision'] = False  # Record actual precision; avoid fp16 instability.
        t['scheduler']['warmup_epochs'] = min(2, t['epochs']-1)
        if variant != 'vit_tiny64':
            t['learning_rate'] = .0003
        train_run(config, variant, 42, run, resume=(run/'last.pt').exists(),
                  model_factory=factory(plan))
        write_comparison(suite)
        gc.collect()
        torch.cuda.empty_cache()


def create_suite(args):
    plan = make_plan(args)
    suite = ROOT/'runs/attention_correction'/new_name('suite')
    suite.mkdir(parents=True)
    save_json(suite/'suite.json', plan)
    parent_copy = suite/'parent_provenance'
    parent_copy.mkdir()
    for name in ('summary.json', 'metadata.json', 'config.yaml', 'epochs.csv'):
        path = Path(plan['base_run'])/name
        if path.exists():
            shutil.copy2(path, parent_copy/name)
    return suite


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', help='100-epoch resnet50_preact run; auto-detect only if unique')
    parser.add_argument('--correction-epochs', type=int, default=25)
    parser.add_argument('--vit-epochs', type=int, default=100)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS,
                        default=['attention_correction', 'vit_tiny64'])
    parser.add_argument('--resume', help='Correction suite directory')
    return parser


def main():
    args = arguments().parse_args()
    if min(args.correction_epochs, args.vit_epochs) < 1:
        raise ValueError('Epochs must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing CPU training')
    suite = Path(args.resume).resolve() if args.resume else create_suite(args)
    print(f'Suite: {suite}', flush=True)
    run_suite(suite)


if __name__ == '__main__':
    main()
