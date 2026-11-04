"""Evaluate actual last checkpoints from the previous three-arm fusion screen.

These reports deliberately distinguish last.pt from the previously selected
best.pt (epoch zero). No optimization, dataset changes or final-test access.
"""
import copy
import csv
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from baseline_diagnostics import clean_datasets, class_labels, class_comparison
from correction_experiment import file_hash
from datasets.utility_tiny import loader
from fusion30_experiment import _checkpoint_matches, _validate_history, paired_predictions
from literature_experiment import ROOT, new_name, save_json
from mechanism_diagnostics import load_spatial_parent
from models.attention_correction import state_digest
from models.spatial_fusion import FUSION_VARIANTS, build_fusion_model
from models.spatial_preact import build_spatial_model
from spatial_experiment import execution_environment
from utils.baseline_diagnostics import evaluate_clean


PROTOCOL = 'tiny-previous-last-diagnosis-v1'


def _check_time(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Previous-checkpoint diagnosis reached its stage deadline')


@torch.inference_mode()
def branch_contribution(model, dataset, config, output, device, deadline=None):
    """Ablate only the learned correction, retaining its co-adapted encoder.

    Removing a branch at inference is an intervention, not another trained
    baseline. It indicates whether small net gains hide many offsetting flips.
    """
    correction = getattr(model, 'correction', None)
    if correction is None:
        return dict(status='not_applicable', reason='CNN control has no added correction branch')
    output = Path(output)
    captured, original_logits = [], []
    hook = correction.register_forward_hook(lambda module, inputs, result: captured.append(result))
    encoder = getattr(model, 'encoder', None)
    if encoder is None or not hasattr(encoder, 'head'):
        hook.remove()
        raise ValueError('Expected an explicit original encoder classifier for branch ablation')
    head_hook = encoder.head.register_forward_hook(lambda module, inputs, result: original_logits.append(result))
    counts = dict(n=0, full_correct=0, without_branch_correct=0, corrected=0, harmed=0,
                  prediction_changes=0, full_nll=0., without_branch_nll=0.)
    square_base, square_delta, elements = 0., 0., 0
    model.eval()
    try:
        with (output/'branch_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_id', 'target', 'prediction_without_branch', 'prediction_full',
                             'corrected_by_branch', 'harmed_by_branch', 'correction_rms'])
            for images, targets, identifiers in loader(dataset, config['dataset'], 42):
                _check_time(deadline)
                captured.clear()
                original_logits.clear()
                images, targets = images.to(device).float(), targets.to(device)
                with torch.autocast(device, enabled=False):
                    full = model(images)
                if (len(captured) != 1 or len(original_logits) != 1
                        or captured[0].shape != full.shape or original_logits[0].shape != full.shape):
                    raise RuntimeError('Expected one class-logit correction per forward')
                delta = captured[0]
                base = original_logits[0]
                if not torch.equal(full, base+delta):
                    raise RuntimeError('Model output is not the declared sum of original and correction logits')
                if not torch.isfinite(full).all() or not torch.isfinite(delta).all():
                    raise ValueError('Non-finite branch contribution')
                p, q = base.argmax(1), full.argmax(1)
                a, b = p.eq(targets), q.eq(targets)
                counts['n'] += len(targets)
                counts['full_correct'] += int(b.sum())
                counts['without_branch_correct'] += int(a.sum())
                counts['corrected'] += int((~a & b).sum())
                counts['harmed'] += int((a & ~b).sum())
                counts['prediction_changes'] += int(p.ne(q).sum())
                counts['full_nll'] += float(F.cross_entropy(full, targets, reduction='sum'))
                counts['without_branch_nll'] += float(F.cross_entropy(base, targets, reduction='sum'))
                square_base += float(base.double().square().sum())
                square_delta += float(delta.double().square().sum())
                elements += delta.numel()
                rms = delta.square().mean(1).sqrt().cpu().tolist()
                writer.writerows(zip(identifiers, targets.cpu().tolist(), p.cpu().tolist(), q.cpu().tolist(),
                    (~a & b).cpu().tolist(), (a & ~b).cpu().tolist(), rms))
    finally:
        hook.remove()
        head_hook.remove()
    n = counts['n']
    if n != len(dataset) or not n:
        raise RuntimeError('Branch evaluation count differs from its clean dataset')
    counts.update(status='complete', full_top1_pct=100*counts['full_correct']/n,
        without_branch_top1_pct=100*counts['without_branch_correct']/n,
        delta_top1_pp=100*(counts['corrected']-counts['harmed'])/n,
        correction_logit_rms=(square_delta/elements)**.5,
        base_logit_rms=(square_base/elements)**.5,
        rms_ratio=(square_delta/max(square_base, 1e-30))**.5,
        interpretation='Inference branch ablation on the same co-adapted encoder; not a separately trained baseline')
    counts['full_nll'] /= n
    counts['without_branch_nll'] /= n
    save_json(output/'branch_contribution.json', counts)
    return counts


def run_previous_last(parent_run, previous_fusion, output_dir, config=None, examples=8, deadline=None):
    parent, parent_summary, parent_state, parent_hash = load_spatial_parent(parent_run)
    previous = Path(previous_fusion).expanduser().resolve()
    output = Path(output_dir).resolve()
    if output == parent or parent in output.parents or output == previous or previous in output.parents:
        raise ValueError('New last-checkpoint output must be outside the original runs')
    if not (previous/'suite.json').is_file():
        output.mkdir(parents=True, exist_ok=True)
        result = dict(status='unavailable', reason='Previous fusion suite not found', previous_fusion=str(previous))
        save_json(output/'summary.json', result)
        (output/'report.md').write_text('# Previous last-checkpoint analysis\n\nThe previous suite was not found. '
            'New controlled experiments can still run.\n', encoding='utf-8')
        return result
    previous_plan = json.loads((previous/'suite.json').read_text(encoding='utf-8'))
    if (previous_plan.get('protocol') != 'tiny-cnn8-fusion30-v1' or previous_plan.get('smoke')
            or previous_plan['parent_checkpoint_sha256'] != parent_hash
            or previous_plan['split_hash'] != parent_summary['split_hash']):
        raise ValueError('Previous fusion suite is not the matching formal parent/split study')
    interpretation_sources = {name: file_hash(ROOT/name)
        for name in ('fusion30_experiment.py', 'models/spatial_fusion.py')}
    for name, digest in interpretation_sources.items():
        if previous_plan.get('source_hashes', {}).get(name) != digest:
            raise ValueError(f'Previous fusion model/loader implementation changed: {name}')
    requested = copy.deepcopy(parent_state['config'] if config is None else config)
    train, selection, _, fingerprint, manifest = clean_datasets(
        requested, parent_summary['seed'], parent_summary['split_hash'], 0)
    names = class_labels(train)
    identity = dict(protocol=PROTOCOL, parent_checkpoint_sha256=parent_hash,
        previous_fusion=str(previous), config=requested, examples=examples,
        split_hash=fingerprint, source_sha256=file_hash(Path(__file__)),
        interpretation_sources=interpretation_sources,
        environment=execution_environment(requested, parent_summary['seed']),
        checkpoints={variant: file_hash(previous/variant/'last.pt') for variant in FUSION_VARIANTS})
    if (output/'summary.json').exists():
        saved = json.loads((output/'summary.json').read_text(encoding='utf-8'))
        if saved.get('status') == 'complete':
            if saved.get('identity') != identity:
                raise ValueError('Completed previous-checkpoint report has a different identity')
            return saved
    output.mkdir(parents=True, exist_ok=True)
    attempt = output/new_name('attempt')
    attempt.mkdir()
    save_json(attempt/'evaluation_manifest.json', manifest)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reference = build_spatial_model('cnn8', parent_summary['seed']).to(device).float().eval()
    reference.load_state_dict(parent_state['model'], strict=True)
    reference_dir = attempt/'parent_selection'
    _check_time(deadline)
    reference_score = evaluate_clean(reference, selection, device=device,
        batch_size=requested['dataset']['batch_size'], num_workers=requested['dataset']['num_workers'],
        output_dir=reference_dir, class_names=names, example_count=0)
    del reference
    rows, scores, pairs, contributions = [], {}, {}, {}
    for variant in FUSION_VARIANTS:
        _check_time(deadline)
        print(f'Previous actual LAST checkpoint: {variant}', flush=True)
        state = torch.load(previous/variant/'last.pt', map_location='cpu', weights_only=False)
        _checkpoint_matches(state, previous_plan, variant)
        _validate_history(state)
        if state['epoch'] != 30:
            raise ValueError('Expected the actual completed epoch30 last checkpoint')
        model = build_fusion_model(variant, parent_state['model'], parent_summary['seed']).to(device).float().eval()
        model.load_state_dict(state['model'], strict=True)
        state_epoch, recorded = state['epoch'], state['history'][-1]
        del state
        model_digest = state_digest(model)
        directory = attempt/variant
        directory.mkdir()
        scores[variant] = {}
        for split, dataset in (('train_clean', train), ('selection', selection)):
            _check_time(deadline)
            scores[variant][split] = evaluate_clean(model, dataset, device=device,
                batch_size=requested['dataset']['batch_size'], num_workers=requested['dataset']['num_workers'],
                output_dir=directory/split, class_names=names, example_count=examples)
        class_comparison(directory)
        pairs[variant] = paired_predictions(reference_dir/'predictions.npz',
            directory/'selection/predictions.npz', directory/'paired_last_vs_parent.csv')
        contributions[variant] = branch_contribution(model, selection, requested, directory, device, deadline)
        clean, selected = scores[variant]['train_clean'], scores[variant]['selection']
        rows.append(dict(variant=variant, checkpoint='last.pt', added_epoch=state_epoch,
            train_clean_top1_pct=100*clean['accuracy'], selection_top1_pct=100*selected['accuracy'],
            gap_pp=100*(clean['accuracy']-selected['accuracy']), selection_nll=selected['nll'],
            recorded_selection_top1_pct=100*recorded['selection_accuracy'],
            **pairs[variant]))
        if state_digest(model) != model_digest:
            raise RuntimeError('Evaluation changed a last-checkpoint model')
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    with (attempt/'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    save_json(attempt/'paired_summary.json', pairs)
    save_json(attempt/'branch_summary.json', contributions)
    lines = ['# Actual epoch30 checkpoints from the previous fusion study', '',
        'These are last.pt models, not the selected epoch-zero best.pt models. No new training was performed.', '',
        '| Model | Clean train Top1 | Selection Top1 | Gap pp | Selection NLL | Corrected / harmed vs parent |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["variant"]} | {row["train_clean_top1_pct"]:.3f}% | '
            f'{row["selection_top1_pct"]:.3f}% | {row["gap_pp"]:.3f} | {row["selection_nll"]:.6f} | '
            f'{row["corrected"]} / {row["harmed"]} |')
    lines += ['', 'Branch summaries remove the learned correction at inference while keeping its co-adapted encoder. '
        'This measures its immediate contribution and magnitude, not the outcome of training another model.', '',
        'Per-class confusion and both corrected/harmed examples accompany each actual last checkpoint. '
        'Neither audit nor official validation was evaluated.', '']
    (attempt/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'report.md').write_text(f'# Previous actual last-checkpoint report\n\n'
        f'[Authoritative report]({attempt.name}/report.md)\n', encoding='utf-8')
    result = dict(status='complete', identity=identity, analysis_directory=attempt.name,
        parent_selection=reference_score, scores=scores, paired=pairs, branch_contribution=contributions,
        audit_partition_evaluated=False, official_validation_evaluated=False)
    for variant, digest in identity['checkpoints'].items():
        if file_hash(previous/variant/'last.pt') != digest:
            raise RuntimeError('Original last checkpoint changed during evaluation')
    if file_hash(parent/'best.pt') != parent_hash:
        raise RuntimeError('Parent checkpoint changed during evaluation')
    save_json(output/'summary.json', result)
    return result
