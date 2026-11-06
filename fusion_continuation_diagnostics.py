"""Read-only evidence for staged CNN/ViT continuation and matched controls.

All interventions keep co-adapted weights fixed. Independent probability
mixtures and three-way error counts are computed before email compaction, so
the aggregate report does not depend on emailing large probability arrays.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from attention_fusion_diagnostics import _write_csv, compare_decisions, _galleries
from baseline_diagnostics import clean_datasets, class_labels
from correction_experiment import file_hash
from datasets.utility_tiny import loader
from literature_experiment import new_name, recipe_hash, save_json
from models.attention_correction import state_digest
from models.spatial_preact import build_spatial_model
from models.vit_followup import build_followup_model


PROTOCOL = 'tiny-fusion-continuation-interventions-v1'
MIXTURE_WEIGHTS = (0., .05, .1, .2, .25, .5, .75, 1.)
DUAL_MODES = ('learned', 'cnn_only', 'vit_only', 'vit_train_mean')


def _inside(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Prediction path must be relative to its source directory')
    path = root/relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Prediction path escapes its source directory')
    return path


def _validate_arrays(result, class_names=None, expected_ids=None, expected_targets=None):
    ids = list(result['image_ids'])
    truth = np.asarray(result['targets'])
    probabilities = np.asarray(result['probabilities'])
    names = list(result['class_names'])
    if (not ids or len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids)
            or truth.shape != (len(ids),) or probabilities.shape != (len(ids), len(names))):
        raise ValueError('Invalid prediction IDs, labels or probability layout')
    if not np.issubdtype(truth.dtype, np.integer) or (truth < 0).any() or (truth >= len(names)).any():
        raise ValueError('Prediction labels do not match the class mapping')
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(1), 1., atol=2e-6, rtol=0)):
        raise ValueError('Invalid probability values or row normalization')
    if class_names is not None and names != list(class_names):
        raise ValueError('Prediction class mapping differs')
    if expected_ids is not None and ids != list(expected_ids):
        raise ValueError('Prediction image order differs')
    if expected_targets is not None and not np.array_equal(truth, expected_targets):
        raise ValueError('Prediction targets differ')
    prediction = probabilities.argmax(1)
    if 'predictions' in result and not np.array_equal(prediction, result['predictions']):
        raise ValueError('Stored predictions differ from probability argmax')
    return result


def probability_result(probabilities, targets, image_ids, class_names, *, nll=None):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    result = dict(image_ids=list(image_ids), targets=targets, probabilities=probabilities,
                  predictions=probabilities.argmax(1), class_names=list(class_names))
    _validate_arrays(result)
    order = np.argsort(-probabilities, axis=1, kind='stable')
    ranks = (order == targets[:, None]).argmax(1)+1
    losses = (-np.log(np.maximum(probabilities[np.arange(len(targets)), targets].astype(np.float64),
                                np.finfo(np.float64).tiny)) if nll is None else np.asarray(nll, dtype=np.float64))
    if losses.shape != targets.shape or not np.isfinite(losses).all():
        raise ValueError('Invalid per-image NLL')
    result.update(nll=losses, ranks=ranks, metrics=dict(n=len(targets),
        accuracy=float((result['predictions'] == targets).mean()),
        top5_accuracy=float((ranks <= min(5, probabilities.shape[1])).mean()), nll=float(losses.mean())))
    return result


def load_probabilities(path, names=None, expected_ids=None, expected_targets=None):
    with np.load(path, allow_pickle=False) as data:
        result = probability_result(data['probabilities'], data['targets'], data['image_ids'].tolist(),
            data['class_names'].tolist(), nll=data['nll'] if 'nll' in data.files else None)
        if 'predicted_labels' in data.files:
            result['predictions'] = data['predicted_labels']
        elif 'predictions' in data.files:
            result['predictions'] = data['predictions']
        _validate_arrays(result, names, expected_ids, expected_targets)
        return result


def paired_result(candidate, reference):
    _validate_arrays(candidate, reference['class_names'], reference['image_ids'], reference['targets'])
    a, b = candidate['predictions'] == candidate['targets'], reference['predictions'] == reference['targets']
    return dict(n=len(a), corrected=int((a & ~b).sum()), harmed=int((~a & b).sum()),
        both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum()),
        delta_accuracy_pp=100*float((a.astype(float)-b.astype(float)).mean()),
        prediction_changes=int((candidate['predictions'] != reference['predictions']).sum()),
        mean_nll_improvement=float((reference['nll']-candidate['nll']).mean()))


def threeway_counts(cnn, vit, candidate):
    _validate_arrays(vit, cnn['class_names'], cnn['image_ids'], cnn['targets'])
    _validate_arrays(candidate, cnn['class_names'], cnn['image_ids'], cnn['targets'])
    c = cnn['predictions'] == cnn['targets']
    v = vit['predictions'] == vit['targets']
    d = candidate['predictions'] == candidate['targets']
    rows = [dict(cnn_correct=bool(a), vit_correct=bool(b), candidate_correct=bool(e),
        n=int(((c == a) & (v == b) & (d == e)).sum()))
        for a in (False, True) for b in (False, True) for e in (False, True)]
    return dict(n=len(c), rows=rows,
        vit_only_available=int((~c & v).sum()),
        vit_only_retained=int((~c & v & d).sum()),
        newly_correct_when_both_wrong=int((~c & ~v & d).sum()),
        harmed_when_both_correct=int((c & v & ~d).sum()),
        switching_oracle_accuracy=float((c | v).mean()),
        oracle_note='Uses true labels to choose an existing Top1; not deployable or a bound on arbitrary feature fusion.')


def fixed_mixtures(cnn, vit):
    _validate_arrays(vit, cnn['class_names'], cnn['image_ids'], cnn['targets'])
    results, rows = {}, []
    for alpha in MIXTURE_WEIGHTS:
        result = probability_result((1-alpha)*cnn['probabilities']+alpha*vit['probabilities'],
            cnn['targets'], cnn['image_ids'], cnn['class_names'])
        results[alpha] = result
        paired = paired_result(result, cnn)
        rows.append(dict(vit_probability_weight=alpha, **result['metrics'],
            corrected=paired['corrected'], harmed=paired['harmed'],
            delta_vs_original_cnn_pp=paired['delta_accuracy_pp']))
    return rows, results


def _write_result(output, result):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'metrics.json', result['metrics'])
    names, truth, prediction = result['class_names'], result['targets'], result['predictions']
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    _write_csv(output/'per_class.csv', [dict(label=i, class_name=name, n=int((truth == i).sum()),
        correct=int(confusion[i, i]), recall=float(confusion[i, i]/max(1, (truth == i).sum())))
        for i, name in enumerate(names)])
    _write_csv(output/'confusion_matrix.csv', [dict(true_class=name,
        **{str(j): int(value) for j, value in enumerate(confusion[i])}) for i, name in enumerate(names)])
    _write_csv(output/'predictions.csv', [dict(image_id=image_id, target=int(y), prediction=int(p),
        correct=bool(y == p), nll=float(loss), true_rank=int(rank))
        for image_id, y, p, loss, rank in zip(result['image_ids'], truth, prediction, result['nll'], result['ranks'])])
    np.savez_compressed(output/'predictions.npz', image_ids=np.asarray(result['image_ids']),
        targets=truth, probabilities=result['probabilities'], predicted_labels=prediction,
        true_ranks=result['ranks'], nll=result['nll'], class_names=np.asarray(names))


@torch.inference_mode()
def mean_vit_feature(model, dataset, config, seed, device):
    """Mean is fit exclusively on the supplied clean training probe, without labels."""
    if getattr(dataset, 'split', None) not in ('train_clean', 'train_probe'):
        raise ValueError('Mean-feature fitting requires a clean training probe')
    before = state_digest(model)
    model.eval()
    total, count, identifiers = None, 0, []
    for images, _, ids in loader(dataset, config['dataset'], seed):
        with torch.autocast(device, enabled=False):
            features = model.vit.forward_features(images.to(device).float())
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise ValueError('Invalid ViT features for train-probe mean')
        current = features.double().sum(0)
        total = current if total is None else total+current
        count += len(images)
        identifiers.extend(ids)
    if count != len(dataset) or count == 0 or len(set(identifiers)) != count:
        raise ValueError('Mean-feature probe IDs/count differ')
    if state_digest(model) != before:
        raise RuntimeError('Mean-feature fitting changed model parameters or buffers')
    return (total/count).float(), dict(n=count, split='train_clean_probe',
        image_ids_sha256=hashlib.sha256('\n'.join(identifiers).encode()).hexdigest(),
        uses_selection_images=False, uses_labels=False)


@torch.inference_mode()
def evaluate_modes(model, dataset, config, seed, device, names, *, mean_feature=None):
    """One encoder pass per batch; branch interventions reuse identical features."""
    model.eval()
    before = state_digest(model)
    dual = hasattr(model, 'vit') and hasattr(model, 'fusion_classifier')
    modes = DUAL_MODES if dual else ('learned',)
    if dual and mean_feature is None:
        raise ValueError('Dual interventions require a training-probe mean feature')
    probabilities = {mode: [] for mode in modes}
    losses = {mode: [] for mode in modes}
    ids, truth = [], []
    contribution = {key: [] for key in ('cnn_logit_rms', 'vit_logit_rms', 'cnn_true_margin',
        'vit_true_margin', 'bias_true_margin', 'true_margin')}
    for images, labels, identifiers in loader(dataset, config['dataset'], seed):
        images, y = images.to(device).float(), labels.to(device)
        with torch.autocast(device, enabled=False):
            if dual:
                cnn = model.encoder(images).float()
                vit = model.vit.forward_features(images).float()
                if cnn.ndim != 2 or vit.ndim != 2 or len(cnn) != len(vit):
                    raise ValueError('Expected two pooled feature matrices')
                head = model.fusion_classifier
                if head.in_features != cnn.shape[1]+vit.shape[1] or mean_feature.shape != vit.shape[1:]:
                    raise ValueError('Feature concat/classifier/mean layout differs')
                features = dict(learned=torch.cat((cnn, vit), 1),
                    cnn_only=torch.cat((cnn, torch.zeros_like(vit)), 1),
                    vit_only=torch.cat((torch.zeros_like(cnn), vit), 1),
                    vit_train_mean=torch.cat((cnn, mean_feature.to(device).expand_as(vit)), 1))
                logits = {mode: head(value) for mode, value in features.items()}
                wc, wv = head.weight[:, :cnn.shape[1]], head.weight[:, cnn.shape[1]:]
                zc, zv = F.linear(cnn, wc), F.linear(vit, wv)
                learned = logits['learned']
                competitors = learned.clone()
                competitors.scatter_(1, y[:, None], -torch.inf)
                competitor = competitors.argmax(1)
                index = torch.arange(len(y), device=device)
                contribution['cnn_logit_rms'].append(zc.square().mean(1).sqrt().cpu().numpy())
                contribution['vit_logit_rms'].append(zv.square().mean(1).sqrt().cpu().numpy())
                contribution['cnn_true_margin'].append((zc[index, y]-zc[index, competitor]).cpu().numpy())
                contribution['vit_true_margin'].append((zv[index, y]-zv[index, competitor]).cpu().numpy())
                bias = head.bias if head.bias is not None else learned.new_zeros(len(names))
                contribution['bias_true_margin'].append((bias[y]-bias[competitor]).cpu().numpy())
                contribution['true_margin'].append((learned[index, y]-learned[index, competitor]).cpu().numpy())
            else:
                logits = {'learned': model(images)}
            for mode, value in logits.items():
                if value.shape != (len(y), len(names)) or not torch.isfinite(value).all():
                    raise ValueError('Invalid diagnostic logits')
                logp = value.float().log_softmax(1)
                probabilities[mode].append(logp.exp().cpu().numpy())
                losses[mode].append((-logp.gather(1, y[:, None]).squeeze(1)).cpu().double().numpy())
        ids.extend(identifiers)
        truth.extend(labels.tolist())
    if len(ids) != len(dataset) or len(set(ids)) != len(ids) or not ids:
        raise ValueError('Diagnostic dataset IDs/count differ')
    if state_digest(model) != before:
        raise RuntimeError('Read-only feature interventions changed parameters or buffers')
    results = {mode: probability_result(np.concatenate(probabilities[mode]), truth, ids, names,
        nll=np.concatenate(losses[mode])) for mode in modes}
    contributions = {key: np.concatenate(values) for key, values in contribution.items()} if dual else {}
    return results, contributions


def contribution_groups(contribution, candidate, reference):
    if not contribution:
        return []
    _validate_arrays(candidate, reference['class_names'], reference['image_ids'], reference['targets'])
    a = candidate['predictions'] == candidate['targets']
    b = reference['predictions'] == reference['targets']
    groups = {'all': np.ones(len(a), dtype=bool), 'corrected_vs_original_cnn': a & ~b,
        'harmed_vs_original_cnn': ~a & b, 'both_correct': a & b, 'both_wrong': ~a & ~b}
    return [dict(group=name, n=int(mask.sum()), **{
        key: float(values[mask].mean()) if mask.any() else None for key, values in contribution.items()})
        for name, mask in groups.items()]


def _prediction_source(sources, kind):
    training = Path(sources.get('source_training_dir', Path(sources['source_suite'])/'training'))
    if kind == 'cnn':
        marker = training/'parent_reference.json'
        if not marker.is_file():
            return None
        record = json.loads(marker.read_text(encoding='utf-8'))
        return _inside(training, Path(record['directory'])/'predictions.npz')
    summary = training/'vit_pretrain/summary.json'
    if not summary.is_file():
        return None
    record = json.loads(summary.read_text(encoding='utf-8'))
    return _inside(training/'vit_pretrain', Path(record['diagnostic_directory'])/'selection/predictions.npz')


def _known_targets(dataset):
    folder, indices = getattr(dataset, 'folder', None), getattr(dataset, 'indices', None)
    if folder is not None and indices is not None and hasattr(folder, 'samples'):
        return np.asarray([folder.samples[i][1] for i in indices], dtype=np.int64)
    return None


def _baselines(sources, probe, selection, manifest, config, seed, device, names, attempt):
    results = {'train_probe': {}, 'selection': {}}
    cache = {}
    for kind in ('cnn', 'vit'):
        path = _prediction_source(sources, kind)
        selected = None
        if path is not None and path.is_file():
            selected = load_probabilities(path, names, manifest['selection'], _known_targets(selection))
            cache[kind] = dict(path=str(path), sha256=file_hash(path), reused=True)
        model = (build_spatial_model('cnn8', seed) if kind == 'cnn' else
                 build_followup_model('vit', vit_state=sources['vit_model'], seed=seed))
        if kind == 'cnn':
            model.load_state_dict(sources['parent_model'], strict=True)
        model = model.to(device).float().eval()
        for split, dataset in (('train_probe', probe), ('selection', selection)):
            if split == 'selection' and selected is not None:
                result = selected
            else:
                evaluated, _ = evaluate_modes(model, dataset, config, seed, device, names)
                result = evaluated['learned']
            expected = manifest['train_clean' if split == 'train_probe' else 'selection']
            _validate_arrays(result, names, expected)
            results[split][kind] = result
            _write_result(attempt/'independent_references'/kind/split, result)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    return results, cache


def run_continuation_diagnostics(source_suite, training_dir, output_dir, config=None, examples=8):
    from fusion_continuation_sources import load_sources
    from fusion_continuation_experiment import build_job_model, source_hashes
    sources = load_sources(source_suite, config=config)
    training, output = Path(training_dir).resolve(), Path(output_dir).resolve()
    protected = [Path(sources['source_suite']).resolve(), Path(sources['parent_run']).resolve(), training]
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('Continuation analysis must use a separate output directory')
    plan_path = training/'suite.json'
    if not plan_path.is_file():
        output.mkdir(parents=True, exist_ok=True)
        result = dict(status='unavailable', protocol=PROTOCOL, reason='No continuation training plan is available')
        save_json(output/'summary.json', result)
        (output/'report.md').write_text('# Continuation analysis unavailable\n\nNo training plan was created.\n', encoding='utf-8')
        return result
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('source_hashes') != source_hashes() or plan.get('source_provenance') != sources['provenance']:
        raise ValueError('Continuation diagnostic source or original checkpoint provenance differs')
    if plan.get('seed') != sources['seed'] or plan.get('split_hash') != sources['split_hash']:
        raise ValueError('Continuation training seed or split differs')
    config = copy.deepcopy(sources['config'])
    probe, selection, _, split_hash, manifest = clean_datasets(config, sources['seed'], sources['split_hash'], 20)
    names, device = class_labels(probe), 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoints = {}
    summaries = {}
    for job in plan['jobs']:
        path = training/job/'summary.json'
        if path.is_file():
            summary = json.loads(path.read_text(encoding='utf-8'))
            if summary.get('status') != 'complete':
                continue
            files = ['best.pt', 'last.pt']
            if summary['best_epoch'] == 0 and summary.get('best_trained_checkpoint'):
                files.append(summary['best_trained_checkpoint'])
            checkpoints[job] = {name: file_hash(training/job/name) for name in files}
            summaries[job] = summary
    cache_identity = {}
    for kind in ('cnn', 'vit'):
        path = _prediction_source(sources, kind)
        cache_identity[kind] = file_hash(path) if path is not None and path.is_file() else None
    identity = dict(protocol=PROTOCOL, source_provenance=sources['provenance'],
        split_hash=split_hash, config=config, examples=examples, checkpoints=checkpoints,
        source_hashes=source_hashes(), mixture_weights=list(MIXTURE_WEIGHTS),
        mean_policy='global ViT mean from fixed clean training probe only', cached_predictions=cache_identity)
    if (output/'summary.json').is_file():
        saved = json.loads((output/'summary.json').read_text(encoding='utf-8'))
        if saved.get('identity') == identity and saved.get('status') == 'complete':
            return saved
        prior = saved.get('identity', {})
        if any(prior.get(key) != value for key, value in identity.items() if key != 'checkpoints'):
            raise ValueError('Saved analysis differs from this continuation request')
    output.mkdir(parents=True, exist_ok=True)
    attempt = output/new_name('attempt')
    attempt.mkdir()
    save_json(attempt/'evaluation_manifest.json', manifest)
    references, cache = _baselines(sources, probe, selection, manifest, config, sources['seed'], device, names, attempt)
    mixture_results = {}
    mixture_rows = []
    for split in ('train_probe', 'selection'):
        rows, mixes = fixed_mixtures(references[split]['cnn'], references[split]['vit'])
        mixture_results[split] = mixes[.2]
        mixture_rows += [dict(split=split, **row) for row in rows]
        for alpha, result in mixes.items():
            _write_result(attempt/'static_probability_fusion'/split/f'vit_weight_{alpha:g}', result)
    _write_csv(attempt/'static_probability_fusion.csv', mixture_rows)
    rows, paired, trios, contribution_rows, failures = [], {}, {}, [], {}
    old_results = {}
    specs = [('source_dual30', 'source_best', sources['dual_best'], sources['dual_best']['epoch'], None)]
    for job, summary in summaries.items():
        for filename, label, epoch in [('best.pt', 'best', summary['best_epoch']),
                                      ('last.pt', 'last_raw', summary['completed_epochs'])]:
            specs.append((job, label, training/job/filename, epoch, summary))
        if summary['best_epoch'] == 0 and summary.get('best_trained_checkpoint'):
            specs.append((job, 'best_trained', training/job/summary['best_trained_checkpoint'],
                          summary['best_trained_selection']['epoch'], summary))
    for job, label, checkpoint, expected_epoch, summary in specs:
        try:
            state = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location='cpu', weights_only=False)
            if state['epoch'] != expected_epoch:
                raise ValueError('Continuation checkpoint chronology differs')
            if summary is not None:
                metadata = state['metadata']
                if metadata.get('source_hashes') != plan['source_hashes']:
                    raise ValueError('Continuation checkpoint source differs')
                if metadata.get('job') != job or any(metadata.get(key) != plan.get(key)
                        for key in ('seed', 'split_hash', 'source_provenance')):
                    raise ValueError('Continuation checkpoint differs from the requested job/source/split')
                for key in ('job', 'variant', 'seed', 'split_hash', 'source_provenance', 'recipe_hash'):
                    if metadata.get(key) != summary.get(key):
                        raise ValueError(f'Continuation checkpoint/summary mismatch: {key}')
                if metadata['recipe_hash'] != recipe_hash(state['config']):
                    raise ValueError('Continuation checkpoint recipe differs')
            model = build_job_model('dual_continue' if job == 'source_dual30' else job, sources, seed=sources['seed'])
            model.load_state_dict(state['model'], strict=True)
            model = model.to(device).float().eval()
            inherited = 0 if job == 'source_dual30' else int(summary.get('inherited_joint_epochs', 0))
            total_epoch = inherited+expected_epoch
            mean, mean_info = (mean_vit_feature(model, probe, config, sources['seed'], device)
                if hasattr(model, 'vit') else (None, None))
            if mean is not None:
                (attempt/job/label).mkdir(parents=True, exist_ok=True)
                save_json(attempt/job/label/'train_probe_mean.json', dict(mean_info, feature_mean=mean.cpu().tolist()))
            for split, dataset in (('train_probe', probe), ('selection', selection)):
                print(f'Continuation feature analysis: {job}/{label}/{split}', flush=True)
                evaluated, contribution = evaluate_modes(model, dataset, config, sources['seed'], device, names,
                    mean_feature=mean)
                original = references[split]['cnn']
                for mode, result in evaluated.items():
                    directory = attempt/job/label/split/mode
                    _write_result(directory, result)
                    _validate_arrays(result, names, original['image_ids'], original['targets'])
                    rows.append(dict(job=job, checkpoint=label, checkpoint_epoch=expected_epoch,
                        total_adaptation_epochs=total_epoch, split=split, mode=mode, **result['metrics']))
                learned = evaluated['learned']
                if job == 'source_dual30':
                    old_results[split] = learned
                for reference_name, reference in (('original_cnn', original),
                        ('source_dual30', old_results.get(split)), ('static_0.8cnn_0.2vit', mixture_results[split])):
                    if reference is not None:
                        paired[f'{job}/{label}/{split}/vs_{reference_name}'] = paired_result(learned, reference)
                for mode, result in evaluated.items():
                    if mode != 'learned':
                        paired[f'{job}/{label}/{split}/learned_vs_{mode}'] = compare_decisions(learned, result)
                        if split == 'selection':
                            _galleries(model, dataset, learned, result, attempt/job/label/f'galleries_vs_{mode}',
                                names, device, examples)
                if split == 'selection':
                    trios[f'{job}/{label}'] = threeway_counts(original, references[split]['vit'], learned)
                contribution_rows += [dict(job=job, checkpoint=label, split=split, **item)
                    for item in contribution_groups(contribution, learned, original)]
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
        except Exception as exc:
            failures[f'{job}/{label}'] = dict(error_type=type(exc).__name__, message=str(exc))
            print(f'Continuation feature analysis failed for {job}/{label}: {type(exc).__name__}', flush=True)
    _write_csv(attempt/'comparison.csv', rows)
    _write_csv(attempt/'contribution_summary.csv', contribution_rows)
    trio_rows = [dict(job_checkpoint=key, **row) for key, value in trios.items() for row in value['rows']]
    _write_csv(attempt/'threeway_counts.csv', trio_rows)
    save_json(attempt/'threeway_summary.json', trios)
    save_json(attempt/'paired_summary.json', paired)
    save_json(attempt/'failures.json', failures)
    save_json(attempt/'reference_prediction_sources.json', cache)
    lines = ['# CNN/ViT continuation feature evidence', '',
        'Static mixtures use independent original CNN and ViT probabilities; all branch interventions '
        'use the same co-adapted trained model. The mean replacement uses only a fixed clean training probe.', '',
        '| Job | State/local epoch | Total adaptation epochs | Mode | Selection Top1 | NLL |',
        '|---|---|---:|---|---:|---:|']
    for row in rows:
        if row['split'] == 'selection':
            lines.append(f'|{row["job"]}|{row["checkpoint"]}@{row["checkpoint_epoch"]}|{row["total_adaptation_epochs"]}|'
                f'{row["mode"]}|{100*row["accuracy"]:.3f}%|{row["nll"]:.6f}|')
    lines += ['', '## Interpretation', '',
        '- vit_train_mean replaces each image-specific ViT feature by the mean fit on clean training '
        'probe images; it keeps the actual classifier and CNN. Compare it with learned and cnn_only '
        'to distinguish image-specific contribution from a constant shift.',
        '- Logit RMS values are magnitudes, not percentages of causal contribution. The reported '
        'CNN/ViT margins compare the true class with the full learned model\'s strongest competing class.',
        '- Static probability weights are a fixed exploratory grid on the development selection set; '
        'the best grid entry is not a final-test performance claim.',
        '- Three-way counts use independent original CNN/ViT correctness and each new model. A true-label '
        'Top1 switching oracle is diagnostic only, not a deployable score or arbitrary feature-fusion bound.',
        '- Epoch-zero candidates, actual last, best, and best actually trained states are separate. '
        'Continuation starts after the original30 joint epochs and uses a separately declared LR stage.',
        '- Audit and official validation are not evaluated. Scripts, checkpoints and probability arrays '
        'remain on the VM; aggregate mixtures, paired results and three-way counts fit in the email report.', '']
    (attempt/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'report.md').write_text(f'# Continuation feature analysis\n\n[Authoritative report]({attempt.name}/report.md)\n', encoding='utf-8')
    for job, hashes in checkpoints.items():
        if any(file_hash(training/job/name) != digest for name, digest in hashes.items()):
            raise RuntimeError('Trained checkpoint changed during continuation analysis')
    for path, digest in sources['source_files'].items():
        if file_hash(Path(path)) != digest:
            raise RuntimeError('Original source checkpoint/report changed during analysis')
    for kind, digest in cache_identity.items():
        path = _prediction_source(sources, kind)
        if digest is not None and (not path.is_file() or file_hash(path) != digest):
            raise RuntimeError('Cached original probabilities changed during analysis')
    result = dict(protocol=PROTOCOL, status='failed' if failures else 'complete', identity=identity,
        analysis_directory=attempt.name, checkpoints_analyzed=len(specs), failures=failures,
        paired=paired, threeway=trios, audit_partition_evaluated=False, official_validation_evaluated=False)
    save_json(output/'summary.json', result)
    if failures:
        raise RuntimeError('Some continuation feature analyses failed; partial evidence retained')
    return result
