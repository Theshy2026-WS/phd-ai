"""Read-only evidence for the strongest independent CNN/ViT joint fusion.

Global mismatch is a deterministic random derangement of all image IDs in a
split. Labels are consulted only afterwards to report the donor-class rate.
This avoids the class-sorted, within-batch mismatch confound in older reports.
"""
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
    _validate_arrays, _write_result, fixed_mixtures, paired_result,
    probability_result, threeway_counts)
from literature_experiment import new_name, save_json
from models.attention_correction import state_digest
from models.spatial_preact import build_spatial_model
from models.vit_augreg import build_vit


PROTOCOL = 'tiny-best-pair-joint-interventions-v1'
MODES = ('learned', 'cnn_only', 'vit_only', 'vit_train_mean', 'global_mismatch')
MISMATCH_POLICY = ('Sort all unique image IDs, randomly permute them with a fixed '
    'seed, and rotate that random cycle by one. Map the resulting donor IDs '
    'back to evaluation order. No labels or batch boundaries choose donors.')


def global_derangement(image_ids, seed):
    """Return donor row indexes, invariant to input order and never self-paired."""
    ids = list(image_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('Global mismatch needs at least two unique nonempty image IDs')
    canonical = sorted(ids)
    permutation = np.random.default_rng(int(seed)).permutation(len(ids))
    donors = {canonical[int(target)]: canonical[int(donor)]
        for target, donor in zip(permutation, np.roll(permutation, 1))}
    positions = {name: i for i, name in enumerate(ids)}
    result = np.asarray([positions[donors[name]] for name in ids], dtype=np.int64)
    if np.any(result == np.arange(len(ids))) or len(set(result.tolist())) != len(ids):
        raise RuntimeError('Global mismatch is not a one-to-one derangement')
    return result


def mismatch_policy(image_ids, targets, seed):
    """Pairing is label blind; label agreement is a descriptive audit only."""
    indexes = global_derangement(image_ids, seed)
    truth = np.asarray(targets, dtype=np.int64)
    if truth.shape != (len(indexes),):
        raise ValueError('Mismatch audit label shape differs')
    _, counts = np.unique(truth, return_counts=True)
    n = len(truth)
    expected = float(np.sum(counts*(counts-1))/(n*(n-1)))
    donor_ids = [image_ids[i] for i in indexes]
    return indexes, dict(policy=MISMATCH_POLICY, seed=int(seed), n=n,
        pairing_uses_labels=False, pairing_uses_batch_boundaries=False,
        same_image_pairs=int((indexes == np.arange(n)).sum()),
        same_class_pairs=int((truth == truth[indexes]).sum()),
        same_class_fraction=float((truth == truth[indexes]).mean()),
        expected_same_class_fraction_under_random_distinct_donor=expected,
        target_ids_sha256=hashlib.sha256('\n'.join(image_ids).encode()).hexdigest(),
        donor_ids_sha256=hashlib.sha256('\n'.join(donor_ids).encode()).hexdigest(),
        note='A two-image intervention, never a deployable single-image model score.')


@torch.inference_mode()
def collect_concat_features(model, dataset, config, seed, device):
    """One encoder pass; cache pooled vectors on CPU, not all patch tokens."""
    if getattr(model, 'variant', None) != 'concat':
        raise ValueError('Pooled feature caching supports the concat model only')
    model.eval()
    before = state_digest(model)
    cnn, vit, identifiers, targets = [], [], [], []
    for images, labels, ids in loader(dataset, config['dataset'], seed):
        with torch.autocast(device, enabled=False):
            c, v, _ = model.extract_features(images.to(device).float())
        if (c.ndim != 2 or v.ndim != 2 or len(c) != len(labels) or len(v) != len(labels)
                or not torch.isfinite(c).all() or not torch.isfinite(v).all()):
            raise ValueError('Invalid pooled CNN/ViT feature layout')
        cnn.append(c.detach().cpu().float())
        vit.append(v.detach().cpu().float())
        identifiers.extend(ids)
        targets.extend(labels.tolist())
    if len(identifiers) != len(dataset) or not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError('Feature cache image IDs/count differ')
    if state_digest(model) != before:
        raise RuntimeError('Feature extraction changed model state')
    return dict(cnn=torch.cat(cnn), vit=torch.cat(vit), image_ids=list(identifiers),
        targets=np.asarray(targets, dtype=np.int64), split=getattr(dataset, 'split', None))


def training_mean(cache):
    if cache.get('split') not in ('train_clean', 'train_probe'):
        raise ValueError('Mean feature requires a clean training probe')
    value = cache['vit'].double().mean(0).float()
    return value, dict(n=len(cache['image_ids']), split='clean_training_probe',
        uses_labels=False, uses_selection_images=False,
        image_ids_sha256=hashlib.sha256('\n'.join(cache['image_ids']).encode()).hexdigest())


@torch.inference_mode()
def evaluate_cached_modes(model, cache, config, seed, device, names, *, mean_feature):
    """Exact concat-head interventions with a global, label-blind donor map."""
    if getattr(model, 'variant', None) != 'concat':
        raise ValueError('Only concat features can be evaluated by this cache')
    if mean_feature is None or mean_feature.shape != cache['vit'].shape[1:]:
        raise ValueError('A training-only ViT mean with matching width is required')
    model.eval()
    before = state_digest(model)
    pairing_seed = int(seed)+730003
    donors, policy = mismatch_policy(cache['image_ids'], cache['targets'], pairing_seed)
    probabilities, losses = {m: [] for m in MODES}, {m: [] for m in MODES}
    margins = {k: [] for k in ('cnn_head_logit_rms', 'vit_logit_rms',
        'cnn_head_true_margin', 'vit_true_margin', 'full_true_margin')}
    step = int(config['dataset']['batch_size'])
    if step < 1:
        raise ValueError('Batch size must be positive')
    for start in range(0, len(donors), step):
        stop = min(start+step, len(donors))
        c = cache['cnn'][start:stop].to(device)
        v = cache['vit'][start:stop].to(device)
        y = torch.as_tensor(cache['targets'][start:stop], device=device)
        # Concat uses CLS only. The one-token placeholder is never read by its
        # head, avoiding a cache of64/256 unused patch vectors per image.
        placeholder = v[:, None, :]
        with torch.autocast(device, enabled=False):
            logits = {mode: model.forward_from_features(c, v, placeholder,
                mode=mode, mean_feature=mean_feature) for mode in MODES if mode != 'global_mismatch'}
            other = cache['vit'][donors[start:stop]].to(device)
            logits['global_mismatch'] = model.forward_from_features(c, other,
                other[:, None, :], mode='learned')
            full, base = logits['learned'].float(), logits['cnn_only'].float()
            correction = full-base
            rivals = full.clone(); rivals.scatter_(1, y[:, None], -torch.inf)
            rival = rivals.argmax(1); index = torch.arange(len(y), device=device)
            for name, value in (
                ('cnn_head_logit_rms', base.square().mean(1).sqrt()),
                ('vit_logit_rms', correction.square().mean(1).sqrt()),
                ('cnn_head_true_margin', base[index, y]-base[index, rival]),
                ('vit_true_margin', correction[index, y]-correction[index, rival]),
                ('full_true_margin', full[index, y]-full[index, rival])):
                margins[name].append(value.cpu().numpy())
            for mode, value in logits.items():
                value = value.float()
                if value.shape != (len(y), len(names)) or not torch.isfinite(value).all():
                    raise ValueError('Invalid cached concat-head logits')
                probabilities[mode].append(value.softmax(1).cpu().numpy())
                losses[mode].append(-value.log_softmax(1).gather(1, y[:, None]).squeeze(1).double().cpu().numpy())
    if state_digest(model) != before:
        raise RuntimeError('Cached interventions changed model state')
    values = {mode: probability_result(np.concatenate(probabilities[mode]), cache['targets'],
        cache['image_ids'], names, nll=np.concatenate(losses[mode])) for mode in MODES}
    policy['donor_image_ids'] = [cache['image_ids'][i] for i in donors]
    return values, {k: np.concatenate(v) for k, v in margins.items()}, policy


@torch.inference_mode()
def evaluate_plain(model, dataset, config, seed, device, names):
    model.eval(); before = state_digest(model)
    probabilities, nll, targets, ids = [], [], [], []
    for images, labels, identifiers in loader(dataset, config['dataset'], seed):
        with torch.autocast(device, enabled=False):
            logits = model(images.to(device).float()).float()
            y = labels.to(device)
            if logits.shape != (len(labels), len(names)) or not torch.isfinite(logits).all():
                raise ValueError('Invalid independent reference logits')
            probabilities.append(logits.softmax(1).cpu().numpy())
            nll.append(-logits.log_softmax(1).gather(1, y[:, None]).squeeze(1).double().cpu().numpy())
        targets.extend(labels.tolist()); ids.extend(identifiers)
    if state_digest(model) != before or len(ids) != len(dataset):
        raise RuntimeError('Independent evaluation changed state or skipped images')
    return probability_result(np.concatenate(probabilities), targets, ids, names, nll=np.concatenate(nll))


def contribution_groups(margins, result, reference):
    _validate_arrays(result, reference['class_names'], reference['image_ids'], reference['targets'])
    a, b = result['predictions'] == result['targets'], reference['predictions'] == reference['targets']
    groups = [('all', np.ones(len(a), dtype=bool)), ('corrected_vs_strong_cnn', a & ~b),
        ('harmed_vs_strong_cnn', ~a & b), ('both_correct', a & b), ('both_wrong', ~a & ~b)]
    return [dict(group=name, n=int(mask.sum()), **{k: float(v[mask].mean()) if mask.any() else None
        for k, v in margins.items()}) for name, mask in groups]


def build_previous_joint_model(sources, seed):
    """Restore the complete earlier joint model, including its original CNN."""
    from vit_augreg_experiment import build_job_model
    record = sources.get('previous_joint')
    if record is None:
        return None, None
    state = record['state']
    model = build_job_model('new_joint_concat', sources, sources['selected'], seed)
    # Loading the whole previous state replaces both backbones and the head.
    # Merely loading the old head onto today's stronger CNN would not reproduce
    # the65.25% reference and is deliberately not done.
    model.load_state_dict(state['model'], strict=True)
    return model, state


def _checkpoint_files(training):
    run = training/'new_joint_concat'
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'complete':
        raise ValueError('Joint30 training has not completed')
    filenames = ['best.pt', 'last.pt']
    if summary.get('best_epoch') == 0 and summary.get('best_trained_checkpoint'):
        filenames.append(summary['best_trained_checkpoint'])
    if any(name not in ('best.pt', 'last.pt', 'best_trained.pt') for name in filenames):
        raise ValueError('Unexpected trained checkpoint path')
    files = {}
    for name in filenames:
        path = run/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Need regular trained checkpoints')
        files[name] = file_hash(path)
    return summary, files


def run_best_pair_diagnostics(plan, training_dir, output_dir, examples=8):
    from best_pair_sources import load_best_sources
    from best_pair_joint import load_best_pair_model
    sources = load_best_sources(plan['cnn_suite'], plan['vit_suite'], config=plan['config'])
    if sources['provenance'] != plan['source_provenance'] or sources['source_files'] != plan['source_files']:
        raise ValueError('Best-pair source identity changed')
    training, output = Path(training_dir).resolve(), Path(output_dir).resolve()
    for protected in (training, Path(plan['cnn_suite']).resolve(), Path(plan['vit_suite']).resolve()):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Diagnostics output must be separate from sources/training')
    recorded = json.loads((training/'suite.json').read_text(encoding='utf-8'))
    if recorded != plan:
        raise ValueError('Training and diagnostics plans differ')
    summary, files = _checkpoint_files(training)
    config, seed = plan['config'], plan['seed']
    identity = dict(protocol=PROTOCOL, source_provenance=plan['source_provenance'],
        source_hashes=plan['source_hashes'], config=config, seed=seed,
        split_hash=plan['split_hash'], checkpoints=files, examples=examples,
        mismatch_policy=MISMATCH_POLICY)
    output.mkdir(parents=True, exist_ok=True)
    marker = output/'summary.json'
    if marker.exists():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved.get('status') == 'complete' and saved.get('identity') == identity:
            return saved
        old = saved.get('identity', {})
        if old and any(old.get(k) != v for k, v in identity.items() if k != 'checkpoints'):
            raise ValueError('Saved diagnostics belong to another best-pair experiment')
    probe, selection, _, split, manifest = clean_datasets(config, seed, plan['split_hash'], 20)
    if split != plan['split_hash']:
        raise ValueError('Diagnostic data partition changed')
    names, device = class_labels(probe), 'cuda' if torch.cuda.is_available() else 'cpu'
    attempt = output/new_name('attempt'); attempt.mkdir()
    save_json(attempt/'evaluation_manifest.json', manifest)
    references = {k: {} for k in ('train_probe', 'selection')}
    rows, mixture_rows, paired, trios, contributions = [], [], {}, {}, []
    selected = sources['selected']
    model = build_spatial_model('cnn8', seed)
    model.load_state_dict(sources['parent_model'], strict=True)
    for kind in ('strong_cnn', 'independent_vit'):
        if kind == 'independent_vit':
            model = build_vit(selected['profile'], seed)
            model.load_state_dict(selected['state']['model'], strict=True)
        model = model.to(device).float().eval()
        for label, data in (('train_probe', probe), ('selection', selection)):
            result = evaluate_plain(model, data, config, seed, device, names)
            references[label][kind] = result
            _write_result(attempt/'independent_references'/kind/label, result)
            source_epoch = sources['provenance'].get('source_cnn_best_epoch' if kind == 'strong_cnn' else 'source_vit_best_epoch')
            rows.append(dict(job=kind, checkpoint='source_best', checkpoint_epoch=source_epoch,
                split=label, mode='learned', **result['metrics']))
        del model; gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()
    model, previous_state = build_previous_joint_model(sources, seed)
    if model is not None:
        model = model.to(device).float().eval()
        for label, data in (('train_probe', probe), ('selection', selection)):
            result = evaluate_plain(model, data, config, seed, device, names)
            reference = references[label]['strong_cnn']
            _validate_arrays(result, names, reference['image_ids'], reference['targets'])
            references[label]['previous_joint'] = result
            _write_result(attempt/'independent_references'/'previous_joint'/label, result)
            rows.append(dict(job='previous_joint', checkpoint='source_best',
                checkpoint_epoch=previous_state['epoch'], split=label, mode='learned', **result['metrics']))
        del model, previous_state; gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()
    for label in references:
        current, mixes = fixed_mixtures(references[label]['strong_cnn'], references[label]['independent_vit'])
        mixture_rows.extend(dict(split=label, **row) for row in current)
        for alpha, result in mixes.items():
            _write_result(attempt/'probability_fusion'/label/f'vit_weight_{alpha:g}', result)
    failures = {}
    for filename in files:
        checkpoint = {'best.pt':'best', 'last.pt':'last_raw', 'best_trained.pt':'best_trained'}[filename]
        try:
            model, state = load_best_pair_model(plan, training, checkpoint=filename, sources=sources)
            model = model.to(device).float().eval()
            for label, data in (('train_probe', probe), ('selection', selection)):
                cache = collect_concat_features(model, data, config, seed, device)
                if label == 'train_probe':
                    mean, mean_info = training_mean(cache)
                    (attempt/checkpoint).mkdir(parents=True, exist_ok=True)
                    save_json(attempt/checkpoint/'training_mean_provenance.json', mean_info)
                values, margins, policy = evaluate_cached_modes(model, cache, config, seed, device,
                    names, mean_feature=mean)
                donors = policy.pop('donor_image_ids')
                (attempt/checkpoint/label).mkdir(parents=True, exist_ok=True)
                save_json(attempt/checkpoint/label/'intervention_policy.json', policy)
                _write_csv(attempt/checkpoint/label/'global_mismatch_pairs.csv',
                    [dict(image_id=target, donor_image_id=donor, target=int(y), donor_target=int(dy))
                     for target, donor, y, dy in zip(cache['image_ids'], donors, cache['targets'],
                        cache['targets'][global_derangement(cache['image_ids'], seed+730003)])])
                for mode, result in values.items():
                    reference = references[label]['strong_cnn']
                    _validate_arrays(result, names, reference['image_ids'], reference['targets'])
                    _write_result(attempt/checkpoint/label/mode, result)
                    rows.append(dict(job='strong_pair_joint', checkpoint=checkpoint,
                        checkpoint_epoch=state['epoch'], split=label, mode=mode, **result['metrics']))
                    if mode != 'learned':
                        paired[f'{checkpoint}/{label}/learned_vs_{mode}'] = paired_result(values['learned'], result)
                learned = values['learned']
                paired[f'{checkpoint}/{label}/vs_strong_cnn'] = paired_result(learned, references[label]['strong_cnn'])
                if 'previous_joint' in references[label]:
                    paired[f'{checkpoint}/{label}/vs_previous_joint'] = paired_result(learned, references[label]['previous_joint'])
                if label == 'selection':
                    trios[checkpoint] = threeway_counts(references[label]['strong_cnn'], references[label]['independent_vit'], learned)
                    if examples > 0 and hasattr(model, 'analysis'):
                        _galleries(model, data, learned, references[label]['strong_cnn'],
                            attempt/checkpoint/'galleries_vs_strong_cnn', names, device, examples)
                contributions.extend(dict(checkpoint=checkpoint, split=label, **entry)
                    for entry in contribution_groups(margins, learned, references[label]['strong_cnn']))
                del cache, values
            del model, state; gc.collect()
            if device == 'cuda': torch.cuda.empty_cache()
        except Exception as exc:
            failures[checkpoint] = dict(error_type=type(exc).__name__, message=str(exc))
            print(f'Best-pair analysis failed: {checkpoint} ({type(exc).__name__})', flush=True)
    _write_csv(attempt/'comparison.csv', rows)
    _write_csv(attempt/'probability_fusion.csv', mixture_rows)
    _write_csv(attempt/'contribution_summary.csv', contributions)
    _write_csv(attempt/'threeway_counts.csv', [dict(checkpoint=key, **row) for key, value in trios.items() for row in value['rows']])
    save_json(attempt/'paired_summary.json', paired); save_json(attempt/'threeway_summary.json', trios)
    save_json(attempt/'failures.json', failures)
    lines = ['# Strongest independent CNN + ViT: joint30 evidence', '',
        'All scores are development-selection evidence. The source CNN and independent ViT are '
        'reevaluated in the same environment. Joint30 updates both independent branches.', '',
        '|Job|Checkpoint/epoch|Mode|Selection Top1|NLL|', '|---|---|---|---:|---:|']
    for row in rows:
        if row['split'] == 'selection':
            lines.append(f'|{row["job"]}|{row["checkpoint"]}@{row["checkpoint_epoch"]}|{row["mode"]}|'
                f'{100*row["accuracy"]:.3f}%|{row["nll"]:.6f}|')
    lines += ['', '## Interpretation', '',
        '- Same-image learned scores are the model results. Global mismatch uses a second image and is a diagnostic intervention only.',
        '- Global donor mapping is label blind and independent of batch boundaries. Actual donor-class agreement and the random-donor expectation are recorded; the old within-batch rotation is not used.',
        '- ViT mean features are fitted on the clean training probe only. Masked branches are interventions on a co-adapted classifier, not independent model scores.',
        '- The CNN-head margin includes the common classifier bias. Its margin plus the ViT margin equals the full margin against the same full-model strongest competing class.',
        '- Raw/EMA best and actual last remain distinct. A peak selected on this development set is not independent confirmation or an architectural novelty claim.',
        '- When available, previous_joint restores the complete earlier65.25% checkpoint, including its earlier CNN, and reevaluates it on the same images. New fusion is paired against that result as well as the stronger independent CNN.',
        '- All probability arrays and checkpoints remain on the VM; aggregate reports can be emailed. No audit or official validation is evaluated.', '']
    (attempt/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'report.md').write_text(f'# Best-pair analysis\n\n[Authoritative report]({attempt.name}/report.md)\n', encoding='utf-8')
    if any(file_hash(training/'new_joint_concat'/name) != digest for name, digest in files.items()):
        raise RuntimeError('Trained checkpoints changed during analysis')
    if any(file_hash(path) != digest for path, digest in sources['source_files'].items()):
        raise RuntimeError('Source files changed during analysis')
    result = dict(protocol=PROTOCOL, status='failed' if failures else 'complete', identity=identity,
        analysis_directory=attempt.name, checkpoints_analyzed=len(files), failures=failures,
        previous_joint_reference_available='previous_joint' in references['selection'],
        paired=paired, threeway=trios, audit_partition_evaluated=False, official_validation_evaluated=False)
    save_json(marker, result)
    if failures: raise RuntimeError('Some best-pair analyses failed; partial reports retained')
    return result
