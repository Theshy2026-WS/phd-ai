"""Read-only evaluation of independent AugReg ViTs and fixed-CNN fusion.

Selection remains development data. Counterfactual branch inputs reuse the
same encoders and classifier; they are not separately trained model scores.
"""
import copy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from attention_fusion_diagnostics import _write_csv, _galleries
from baseline_diagnostics import clean_datasets, class_labels
from correction_experiment import file_hash
from datasets.utility_tiny import loader
from fusion_continuation_diagnostics import (
    _baselines, _prediction_source, _validate_arrays, _write_result, fixed_mixtures, paired_result,
    probability_result, threeway_counts, mean_vit_feature as old_mean_vit_feature,
    evaluate_modes as old_evaluate_modes)
from literature_experiment import new_name, save_json
from models.attention_correction import state_digest
from models.vit_followup import build_followup_model


PROTOCOL = 'tiny-vit-augreg-interventions-v1'
MISMATCH_POLICY = ('Fixed evaluation-loader order, ViT CLS and patch features rolled by one '
    'within each batch; a final singleton uses the preceding batch last image. '
    'Every replacement must have a different image ID. Labels are not used.')


@torch.inference_mode()
def mean_vit_features(model, dataset, config, seed, device):
    """Fit CLS and position-wise patch means on clean training images only."""
    if getattr(dataset, 'split', None) not in ('train_clean', 'train_probe'):
        raise ValueError('Mean-feature fitting requires a clean training probe')
    model.eval()
    before, cls_sum, patch_sum, ids = state_digest(model), None, None, []
    for images, _, identifiers in loader(dataset, config['dataset'], seed):
        with torch.autocast(device, enabled=False):
            tokens = model.vit.forward_tokens(images.to(device).float()).float()
        if tokens.ndim != 3 or tokens.shape[1] < 2 or not torch.isfinite(tokens).all():
            raise ValueError('Invalid ViT tokens for training-only means')
        cls, patch = tokens[:, 0].double().sum(0), tokens[:, 1:].double().sum(0)
        cls_sum = cls if cls_sum is None else cls_sum + cls
        patch_sum = patch if patch_sum is None else patch_sum + patch
        ids.extend(identifiers)
    if len(ids) != len(dataset) or not ids or len(ids) != len(set(ids)):
        raise ValueError('Training mean probe IDs/count differ')
    if state_digest(model) != before:
        raise RuntimeError('Training mean calculation changed model state')
    return (cls_sum/len(ids)).float(), (patch_sum/len(ids)).float(), dict(
        n=len(ids), split='train_clean_probe', uses_labels=False, uses_selection_images=False,
        image_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())


def _mismatch(cls, patches, ids, previous):
    if len(ids) > 1:
        source_ids = list(ids[-1:]) + list(ids[:-1])
        wrong_cls, wrong_patches = cls.roll(1, 0), patches.roll(1, 0)
    elif previous is not None:
        source_ids = [previous[2]]
        wrong_cls, wrong_patches = previous[0], previous[1]
    else:
        raise ValueError('Image mismatch needs at least two distinct images')
    if any(source == target for source, target in zip(source_ids, ids)):
        raise ValueError('Image mismatch accidentally reused the matching image')
    return wrong_cls, wrong_patches, source_ids


@torch.inference_mode()
def evaluate_modes(model, dataset, config, seed, device, names, *, mean_feature=None, mean_patches=None):
    """Reuse one encoder pass for every same-checkpoint intervention."""
    model.eval()
    before = state_digest(model)
    dual = hasattr(model, 'extract_features') and hasattr(model, 'forward_from_features')
    modes = tuple(model.analysis_modes) if dual else ('learned',)
    if dual and (mean_feature is None or mean_patches is None):
        raise ValueError('Fusion diagnostics require training-only CLS and patch means')
    probabilities, losses = {m: [] for m in modes}, {m: [] for m in modes}
    contributions = {k: [] for k in ('base_logit_rms', 'correction_logit_rms',
        'base_true_margin', 'correction_true_margin', 'full_true_margin')}
    correction_scale = getattr(model, 'correction_scale', None) if dual else None
    scale = float(torch.as_tensor(correction_scale).detach().cpu()) if correction_scale is not None else None
    if scale is not None and (not np.isfinite(scale) or scale < 0):
        raise ValueError('Invalid bounded-correction scale')
    bounded = (dict(learned_correction_scale=scale,
        residual_bound=float(getattr(model, 'residual_bound', scale)),
        correction_max_abs_logit=0., safe_anchor_zone_count=0,
        safe_anchor_zone_prediction_changes=0,
        safe_anchor_zone_rule='anchor Top1 minus runner-up logit > 2 * learned_correction_scale',
        interpretation='Algebraic consequence of a bounded logit correction, not a new-method claim')
        if scale is not None else None)
    identifiers, targets, mismatch_ids, previous = [], [], [], None
    for images, labels, ids in loader(dataset, config['dataset'], seed):
        images, truth = images.to(device).float(), labels.to(device)
        with torch.autocast(device, enabled=False):
            if dual:
                cnn, cls, patches = model.extract_features(images)
                logits = {}
                for mode in modes:
                    if mode == 'vit_mismatch':
                        other_cls, other_patches, other_ids = _mismatch(cls, patches, list(ids), previous)
                        logits[mode] = model.forward_from_features(cnn, other_cls, other_patches, mode='learned')
                        mismatch_ids.extend(other_ids)
                    else:
                        logits[mode] = model.forward_from_features(cnn, cls, patches, mode=mode,
                            mean_feature=mean_feature, mean_patches=mean_patches)
                previous = (cls[-1:].detach().clone(), patches[-1:].detach().clone(), ids[-1])
                full, base = logits['learned'].float(), logits['cnn_only'].float()
                correction = full-base
                if bounded is not None:
                    maximum = float(correction.abs().max())
                    tolerance = 8*torch.finfo(full.dtype).eps*max(
                        1., float(full.abs().max()), float(base.abs().max()))
                    if maximum > scale+tolerance:
                        raise ValueError('Actual logit correction exceeds its declared bound')
                    anchor_top = base.topk(2, dim=1).values
                    safe = anchor_top[:, 0]-anchor_top[:, 1] > 2*scale
                    changed = full.argmax(1) != base.argmax(1)
                    violations = int((safe & changed).sum())
                    if violations:
                        raise RuntimeError('A prediction changed inside the bounded-correction safe anchor zone')
                    bounded['correction_max_abs_logit'] = max(bounded['correction_max_abs_logit'], maximum)
                    bounded['safe_anchor_zone_count'] += int(safe.sum())
                    bounded['safe_anchor_zone_prediction_changes'] += violations
                rivals = full.clone()
                rivals.scatter_(1, truth[:, None], -torch.inf)
                competitor = rivals.argmax(1)
                index = torch.arange(len(truth), device=truth.device)
                for key, value in (
                    ('base_logit_rms', base.square().mean(1).sqrt()),
                    ('correction_logit_rms', correction.square().mean(1).sqrt()),
                    ('base_true_margin', base[index, truth]-base[index, competitor]),
                    ('correction_true_margin', correction[index, truth]-correction[index, competitor]),
                    ('full_true_margin', full[index, truth]-full[index, competitor])):
                    contributions[key].append(value.cpu().numpy())
            else:
                logits = {'learned': model(images)}
            for mode, value in logits.items():
                value = value.float()
                if value.shape != (len(labels), len(names)) or not torch.isfinite(value).all():
                    raise ValueError('Invalid AugReg diagnostic logits')
                probabilities[mode].append(value.softmax(1).cpu().numpy())
                losses[mode].append((-value.log_softmax(1).gather(1, truth[:, None]).squeeze(1)).double().cpu().numpy())
        identifiers.extend(ids)
        targets.extend(labels.tolist())
    if state_digest(model) != before:
        raise RuntimeError('AugReg intervention evaluation changed model state')
    if len(identifiers) != len(dataset):
        raise ValueError('AugReg diagnostic evaluation skipped images')
    values = {mode: probability_result(np.concatenate(probabilities[mode]), targets, identifiers,
        names, nll=np.concatenate(losses[mode])) for mode in modes}
    contribution = {key: np.concatenate(value) for key, value in contributions.items()} if dual else {}
    if bounded is not None:
        bounded['safe_anchor_zone_fraction'] = bounded['safe_anchor_zone_count']/len(identifiers)
        bounded['scale_fraction_of_maximum'] = scale/max(bounded['residual_bound'], np.finfo(float).tiny)
        bounded['bound_verified'] = True
    return values, contribution, dict(mismatch_policy=MISMATCH_POLICY, bounded_correction=bounded,
        mismatched_images=len(mismatch_ids), all_mismatches_have_different_ids=all(
            a != b for a, b in zip(mismatch_ids, identifiers)) if mismatch_ids else None)


def contribution_groups(contribution, learned, original):
    if not contribution:
        return []
    a = learned['predictions'] == learned['targets']
    b = original['predictions'] == original['targets']
    groups = [('all', np.ones(len(a), dtype=bool)), ('corrected_vs_original_cnn', a & ~b),
        ('harmed_vs_original_cnn', ~a & b), ('both_correct', a & b), ('both_wrong', ~a & ~b)]
    return [dict(group=name, n=int(mask.sum()), **{key: float(value[mask].mean()) if mask.any() else None
        for key, value in contribution.items()}) for name, mask in groups]


def _selected_record(training):
    path = training/'selected_vit.json'
    if not path.is_file() or path.is_symlink():
        return None
    selected = json.loads(path.read_text(encoding='utf-8'))
    if not selected.get('complete'):
        raise ValueError('Selected ViT search has not completed')
    checkpoint = Path(selected['checkpoint']).resolve()
    if (not checkpoint.is_relative_to(training) or checkpoint.is_symlink()
            or not checkpoint.is_file() or file_hash(checkpoint) != selected['checkpoint_sha256']):
        raise ValueError('Selected ViT checkpoint identity/location differs')
    return selected


def run_augreg_diagnostics(plan, training_dir, output_dir, examples=8):
    from fusion_continuation_sources import load_sources
    from vit_augreg_experiment import load_analysis_model
    sources = load_sources(plan['source_suite'], config=plan['config'])
    training, output = Path(training_dir).resolve(), Path(output_dir).resolve()
    protected = [training, Path(sources['source_suite']).resolve(), Path(sources['parent_run']).resolve()]
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('AugReg diagnostics must use a separate output directory')
    recorded_path = training/'suite.json'
    if not recorded_path.is_file():
        raise ValueError('No recorded AugReg training plan exists')
    recorded = json.loads(recorded_path.read_text(encoding='utf-8'))
    for key in set(recorded) | set(plan):
        if recorded.get(key) != plan.get(key):
            raise ValueError(f'AugReg training/diagnostic plan mismatch: {key}')
    selected = _selected_record(training)
    output.mkdir(parents=True, exist_ok=True)
    if selected is None:
        result = dict(protocol=PROTOCOL, status='unavailable', reason='No selected ViT checkpoint; inspect training failures')
        save_json(output/'summary.json', result)
        (output/'report.md').write_text('# AugReg analysis unavailable\n\nNo selected ViT checkpoint is available.\n', encoding='utf-8')
        return result
    config, seed = copy.deepcopy(plan['config']), plan['seed']
    probe, selection, _, split_hash, manifest = clean_datasets(config, seed, plan['split_hash'], 20)
    names, device = class_labels(probe), 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoints, summaries = {}, {}
    for run in sorted(training.iterdir()):
        if not run.is_dir() or run.name.startswith('pretrain_') or not (run/'summary.json').is_file():
            continue
        summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
        if summary.get('status') != 'complete':
            continue
        filenames = ['best.pt', 'last.pt']
        if summary.get('best_epoch') == 0 and summary.get('best_trained_checkpoint'):
            filenames.append(summary['best_trained_checkpoint'])
        checkpoints[run.name] = {name: file_hash(run/name) for name in filenames}
        summaries[run.name] = summary
    cache_identity = {}
    for kind in ('cnn', 'vit'):
        path = _prediction_source(sources, kind)
        cache_identity[kind] = file_hash(path) if path is not None and path.is_file() else None
    identity = dict(protocol=PROTOCOL, source_provenance=plan['source_provenance'],
        source_hashes=plan['source_hashes'], config=config, seed=seed, split_hash=split_hash,
        examples=examples, selected_vit=selected, checkpoints=checkpoints,
        cached_reference_probabilities=cache_identity,
        mean_policy='fixed clean training probe, label-blind CLS and per-position patch means',
        mismatch_policy=MISMATCH_POLICY)
    if (output/'summary.json').is_file():
        prior = json.loads((output/'summary.json').read_text(encoding='utf-8'))
        if prior.get('identity') == identity and prior.get('status') == 'complete':
            return prior
        old = prior.get('identity', {})
        if old and any(old.get(key) != value for key, value in identity.items() if key not in ('checkpoints', 'selected_vit')):
            raise ValueError('Saved AugReg diagnostics belong to a different experiment')
    attempt = output/new_name('attempt')
    attempt.mkdir()
    save_json(attempt/'evaluation_manifest.json', manifest)
    references, cache = _baselines(sources, probe, selection, manifest, config, seed, device, names, attempt)
    rows, paired, trios, contributions, mixture_rows = [], {}, {}, [], []
    failures = {job: dict(error_type='IncompleteTraining', message='No complete downstream checkpoint is available')
        for job in plan.get('jobs', list(summaries)) if not job.startswith('pretrain_') and job not in summaries}
    independent, old_dual = {}, {}
    pretrain_job = selected.get('job', 'pretrain_'+selected['profile'])
    model, state = load_analysis_model(pretrain_job, training/pretrain_job, plan, sources)
    model = model.to(device).float().eval()
    for split, dataset in (('train_probe', probe), ('selection', selection)):
        results, _, _ = evaluate_modes(model, dataset, config, seed, device, names)
        independent[split] = results['learned']
        _validate_arrays(independent[split], names, references[split]['cnn']['image_ids'], references[split]['cnn']['targets'])
        _write_result(attempt/'independent_references'/'new_vit'/split, independent[split])
        rows.append(dict(job=pretrain_job, checkpoint='selected_best', checkpoint_epoch=state['epoch'],
            split=split, mode='learned', **independent[split]['metrics']))
        paired[f'new_vit/{split}/vs_old_vit'] = paired_result(independent[split], references[split]['vit'])
        for name, vit in [('old_vit', references[split]['vit']), ('new_vit', independent[split])]:
            current_rows, mixes = fixed_mixtures(references[split]['cnn'], vit)
            mixture_rows.extend(dict(reference=name, split=split, **row) for row in current_rows)
            for alpha, value in mixes.items():
                _write_result(attempt/'probability_fusion'/name/split/f'vit_weight_{alpha:g}', value)
    del model, state
    model = build_followup_model('dual_fusion', parent_state=sources['parent_model'],
        vit_state=sources['vit_model'], seed=seed).to(device).float().eval()
    model.load_state_dict(sources['dual_best']['model'], strict=True)
    old_mean, _ = old_mean_vit_feature(model, probe, config, seed, device)
    for split, dataset in (('train_probe', probe), ('selection', selection)):
        values, _ = old_evaluate_modes(model, dataset, config, seed, device, names, mean_feature=old_mean)
        old_dual[split] = values['learned']
        _write_result(attempt/'source_dual30'/split, old_dual[split])
        rows.append(dict(job='source_dual30', checkpoint='source_best', checkpoint_epoch=sources['dual_best']['epoch'],
            split=split, mode='learned', **old_dual[split]['metrics']))
    del model
    for job, summary in summaries.items():
        specs = [('best.pt', 'best'), ('last.pt', 'last_raw')]
        if summary.get('best_epoch') == 0 and summary.get('best_trained_checkpoint'):
            specs.append((summary['best_trained_checkpoint'], 'best_trained'))
        for filename, label in specs:
            try:
                model, state = load_analysis_model(job, training/job, plan, sources, checkpoint=filename)
                model = model.to(device).float().eval()
                mean, patches, mean_info = (mean_vit_features(model, probe, config, seed, device)
                    if hasattr(model, 'extract_features') else (None, None, None))
                if mean_info is not None:
                    (attempt/job/label).mkdir(parents=True, exist_ok=True)
                    save_json(attempt/job/label/'training_mean_provenance.json', dict(mean_info,
                        cls_shape=list(mean.shape), patch_shape=list(patches.shape)))
                for split, dataset in (('train_probe', probe), ('selection', selection)):
                    print(f'AugReg evidence: {job}/{label}/{split}', flush=True)
                    values, contribution, mismatch = evaluate_modes(model, dataset, config, seed, device, names,
                        mean_feature=mean, mean_patches=patches)
                    for mode, result in values.items():
                        _validate_arrays(result, names, references[split]['cnn']['image_ids'], references[split]['cnn']['targets'])
                        _write_result(attempt/job/label/split/mode, result)
                        rows.append(dict(job=job, checkpoint=label, checkpoint_epoch=state['epoch'],
                            split=split, mode=mode, **result['metrics']))
                    learned = values['learned']
                    save_json(attempt/job/label/split/'intervention_policy.json', mismatch)
                    for reference_name, reference in [('original_cnn', references[split]['cnn']),
                            ('source_dual30', old_dual[split])]:
                        paired[f'{job}/{label}/{split}/vs_{reference_name}'] = paired_result(learned, reference)
                    for mode, value in values.items():
                        if mode != 'learned':
                            paired[f'{job}/{label}/{split}/learned_vs_{mode}'] = paired_result(learned, value)
                    if split == 'selection':
                        for refname, vit in [('old_vit', references[split]['vit']), ('new_vit', independent[split])]:
                            trios[f'{job}/{label}/{refname}'] = threeway_counts(references[split]['cnn'], vit, learned)
                        if examples > 0 and hasattr(model, 'analysis'):
                            _galleries(model, dataset, learned, references[split]['cnn'],
                                attempt/job/label/'galleries_vs_original_cnn', names, device, examples)
                    contributions.extend(dict(job=job, checkpoint=label, split=split, **entry)
                        for entry in contribution_groups(contribution, learned, references[split]['cnn']))
                del model, state
                if device == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as exc:
                failures[f'{job}/{label}'] = dict(error_type=type(exc).__name__, message=str(exc))
                print(f'AugReg evidence failed: {job}/{label} ({type(exc).__name__})', flush=True)
            finally:
                model, state = None, None
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
    _write_csv(attempt/'comparison.csv', rows)
    _write_csv(attempt/'probability_fusion.csv', mixture_rows)
    _write_csv(attempt/'contribution_summary.csv', contributions)
    _write_csv(attempt/'threeway_counts.csv', [dict(job_checkpoint_reference=key, **row)
        for key, value in trios.items() for row in value['rows']])
    save_json(attempt/'paired_summary.json', paired)
    save_json(attempt/'threeway_summary.json', trios)
    save_json(attempt/'reference_prediction_sources.json', cache)
    save_json(attempt/'failures.json', failures)
    lines = ['# Independent ViT AugReg and fixed-CNN fusion evidence', '',
        'All scores use development selection data, not final test data. The same original CNN is used '
        'throughout. Old and new ViT probability mixtures are independent-model references.', '',
        '| Job | Checkpoint/epoch | Mode | Selection Top1 | NLL |',
        '|---|---|---|---:|---:|']
    for row in rows:
        if row['split'] == 'selection':
            lines.append(f'|{row["job"]}|{row["checkpoint"]}@{row["checkpoint_epoch"]}|{row["mode"]}|'
                f'{100*row["accuracy"]:.3f}%|{row["nll"]:.6f}|')
    lines.extend(['', '## Interpretation', '',
        '- This is a Tiny ImageNet adaptation of AugReg principles, not a reproduction of the paper\'s ImageNet results.',
        '- Concat/normalized cnn_only masks the ViT source inside the trained head; it is not a standalone CNN. '
        'Anchored residual/retrieval cnn_only returns the fixed original CNN logits.',
        '- vit_train_mean replaces CLS and each patch position with a mean estimated only on the clean training probe. '
        'vit_mismatch pairs CNN features with another image\'s ViT features without using labels. '+MISMATCH_POLICY,
        '- A within-image permutation of encoded ViT tokens would be a vacuous intervention for order-invariant '
        'cross-attention pooling; it is not used as evidence for spatial reasoning.',
        '- Base/correction margins use the full model\'s strongest non-target competitor; their sum is the full margin. '
        'Logit magnitudes are not causal contribution percentages. Groups selected by outcome are descriptive.',
        '- For anchored bounded-logit variants, each split records the learned correction scale, maximum actual '
        'logit correction and the fraction of images whose anchor Top1 margin exceeds twice that scale. '
        'The evaluator verifies the logit bound and that no Top1 prediction changes inside this safe zone. '
        'This is an algebraic property of the design, not proof that the corrections improve accuracy.',
        '- Static mixture weights, model profiles and checkpoints are selected or compared using development data. '
        'No switching oracle is deployable or a bound on arbitrary feature fusion.',
        '- Best and actual last are distinct. Audit and official validation are not evaluated. '
        'Probability NPZs and checkpoints remain on the instance; aggregate evidence is included in the report ZIP.', ''])
    (attempt/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'report.md').write_text(f'# AugReg feature evidence\n\n[Authoritative report]({attempt.name}/report.md)\n', encoding='utf-8')
    if _selected_record(training) != selected:
        raise RuntimeError('Selected ViT record changed during analysis')
    for job, files in checkpoints.items():
        if any(file_hash(training/job/name) != digest for name, digest in files.items()):
            raise RuntimeError('Trained checkpoint changed during AugReg analysis')
    if any(file_hash(path) != digest for path, digest in sources['source_files'].items()):
        raise RuntimeError('Original source files changed during AugReg analysis')
    for kind, digest in cache_identity.items():
        path = _prediction_source(sources, kind)
        if digest is not None and (not path.is_file() or file_hash(path) != digest):
            raise RuntimeError('Cached original prediction probabilities changed during analysis')
    result = dict(protocol=PROTOCOL, status='failed' if failures else 'complete', identity=identity,
        analysis_directory=attempt.name, checkpoints_analyzed=sum(len(v) for v in checkpoints.values()),
        failures=failures, paired=paired, threeway=trios, audit_partition_evaluated=False,
        official_validation_evaluated=False)
    save_json(output/'summary.json', result)
    if failures:
        raise RuntimeError('Some AugReg interventions failed; partial reports retained')
    return result
