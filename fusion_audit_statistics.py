"""Predeclared paired statistics for fixed CNN/ViT confirmation models.

This module never selects a checkpoint, a fusion weight or a new winning model.
It needs only NumPy and the existing prediction-record validator; SciPy is not
required. Confidence intervals describe image sampling for these fixed models,
not uncertainty over training seeds or future datasets.
"""
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from fusion_continuation_diagnostics import _validate_arrays, probability_result


PROTOCOL = 'tiny-fixed-fusion-confirmation-statistics-v1'
MODELS = ('source_cnn', 'source_vit', 'existing_joint', 'probability_90_10',
    'random_vit_cls_linear', 'reset_cnn_head_cls_linear', 'patch_mean_linear')
PRIMARY = ('joint_vs_cnn', 'existing_joint', 'source_cnn')
SECONDARY = (
    ('joint_vs_probability_90_10', 'existing_joint', 'probability_90_10'),
    ('joint_vs_random_vit', 'existing_joint', 'random_vit_cls_linear'),
    ('joint_vs_reset_cnn_head', 'existing_joint', 'reset_cnn_head_cls_linear'),
    ('joint_vs_patch_mean', 'existing_joint', 'patch_mean_linear'))
_PLAN = dict(protocol=PROTOCOL, alpha=.05,
    primary=dict(id=PRIMARY[0], candidate=PRIMARY[1], reference=PRIMARY[2],
        endpoint='Top-1 accuracy difference in percentage points',
        test='exact two-sided McNemar: discordant outcomes ~ Binomial(n, 0.5)',
        multiplicity='one predeclared primary hypothesis'),
    secondary=[dict(id=key, candidate=candidate, reference=reference)
        for key, candidate, reference in SECONDARY],
    secondary_multiplicity='Holm adjustment across all four predeclared secondary tests',
    secondary_confirmatory_gate='Primary positive effect must pass exact p<.05 and its 95% interval must exclude zero',
    confidence_interval=dict(method='class-stratified paired-image percentile bootstrap through within-class multinomial +1/-1/0 counts',
        confidence=.95, replicates=10000, seed=20261109,
        comparison_seed_policy='base seed plus fixed comparison index, primary first',
        sampling_unit='evaluation image within its true class; both model decisions resampled together and each class count retained',
        coverage='pointwise 95%; secondary intervals are not simultaneous intervals',
        interpretation='Conditional on fixed trained models and sampled classes; excludes training-run variability'),
    probability_mixture=dict(cnn=.9, vit=.1, fit_on_confirmation_images=False),
    ancillary_metrics='Top-5 and mean NLL plus paired NLL improvement are descriptive; no calibration claim',
    model_order=list(MODELS), select_model_using_confirmation_scores=False,
    references=dict(exact_binomial='https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html',
        holm='https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html'))
_PLAN_JSON = json.dumps(_PLAN, sort_keys=True, separators=(',', ':'))
STATISTICAL_PLAN = copy.deepcopy(_PLAN)


def statistical_plan():
    """Return the complete fixed rule before any confirmation inference."""
    return json.loads(_PLAN_JSON)


def _validated_plan(plan):
    plan = statistical_plan() if plan is None else plan
    if json.dumps(plan, sort_keys=True, separators=(',', ':')) != _PLAN_JSON:
        raise ValueError('Statistical plan changed after registration')
    return copy.deepcopy(plan)


def exact_mcnemar(corrected, harmed):
    """Symmetric exact two-sided binomial test, including double underflow."""
    for value in (corrected, harmed):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError('McNemar counts must be nonnegative integers')
    corrected, harmed = int(corrected), int(harmed)
    n, lower = corrected+harmed, min(corrected, harmed)
    if n == 0 or abs(corrected-harmed) <= 1:
        log_p, value = 0., 1.
    else:
        log_mass = (math.lgamma(n+1)-math.lgamma(lower+1)
                    -math.lgamma(n-lower+1)-n*math.log(2.))
        weight, terms = 1., [1.]
        for k in range(lower, 0, -1):
            weight *= k/(n-k+1)
            terms.append(weight)
            if weight == 0.:
                break
        log_p = min(0., math.log(2.)+log_mass+math.log(math.fsum(terms)))
        value = min(1., math.exp(log_p))
    return dict(discordant=corrected+harmed, pvalue=value, log_pvalue=log_p,
        numeric_underflow=bool(value == 0. and log_p < 0.), alternative='two-sided')


def holm_adjust(pvalues):
    """Holm adjusted p-values in original fixed-comparison order."""
    items = list(pvalues.items())
    if not items or any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for _, p in items):
        raise ValueError('Holm adjustment needs finite probabilities')
    ordered = sorted(enumerate(items), key=lambda item: (float(item[1][1]), item[0]))
    adjusted, previous = {}, 0.
    for rank, (_, (name, value)) in enumerate(ordered):
        previous = min(1., max(previous, (len(items)-rank)*float(value)))
        adjusted[name] = previous
    return {name: adjusted[name] for name, _ in items}


def stratified_accuracy_interval(candidate_correct, reference_correct, targets, *,
                                 seed, replicates=10000, confidence=.95):
    """Resample paired outcomes within classes with bounded B x 3 memory."""
    a, b, truth = np.asarray(candidate_correct), np.asarray(reference_correct), np.asarray(targets)
    if (truth.ndim != 1 or len(truth) < 1 or a.shape != truth.shape or b.shape != truth.shape
            or a.dtype != np.bool_ or b.dtype != np.bool_
            or not np.issubdtype(truth.dtype, np.integer)
            or not isinstance(replicates, (int, np.integer)) or replicates < 1
            or not 0 < confidence < 1):
        raise ValueError('Invalid stratified paired bootstrap outcomes or settings')
    rng = np.random.default_rng(int(seed))
    totals = np.zeros(int(replicates), dtype=np.int64)
    strata = []
    for label in np.unique(truth):
        mask = truth == label
        n_class = int(mask.sum())
        corrected, harmed = int((a[mask] & ~b[mask]).sum()), int((~a[mask] & b[mask]).sum())
        frequencies = np.asarray([corrected, harmed, n_class-corrected-harmed], dtype=np.float64)/n_class
        sampled = rng.multinomial(n_class, frequencies, size=int(replicates))
        totals += sampled[:, 0]-sampled[:, 1]
        strata.append(dict(label=int(label), n=n_class, corrected=corrected, harmed=harmed))
    differences = 100*totals/len(truth)
    tail = (1-confidence)/2
    low, high = np.quantile(differences, [tail, 1-tail], method='linear')
    return dict(low_pp=float(low), high_pp=float(high), confidence=float(confidence),
        replicates=int(replicates), seed=int(seed), pointwise=True,
        training_seed_resampling=False, class_counts_preserved=True, strata=strata)


def _normalize_results(results):
    if not isinstance(results, dict) or set(results) != set(MODELS):
        raise ValueError('Need exactly the seven registered model results')
    baseline = results['source_cnn']
    _validate_arrays(baseline)
    names = list(baseline['class_names'])
    if not names or len(names) != len(set(names)) or any(not isinstance(v, str) or not v for v in names):
        raise ValueError('Class names must be unique nonempty strings')
    normalized = {}
    for key in MODELS:
        result = results[key]
        _validate_arrays(result, names, baseline['image_ids'], baseline['targets'])
        losses = np.asarray(result.get('nll'), dtype=np.float64)
        if losses.shape != np.asarray(result['targets']).shape or not np.isfinite(losses).all() or (losses < 0).any():
            raise ValueError('Finite nonnegative per-image NLL is required')
        # Recompute all aggregates from arrays instead of trusting cached metrics.
        normalized[key] = probability_result(result['probabilities'], result['targets'],
            result['image_ids'], names, nll=losses)
    expected = (.9*normalized['source_cnn']['probabilities']+
                .1*normalized['source_vit']['probabilities'])
    if not np.allclose(normalized['probability_90_10']['probabilities'], expected, rtol=0, atol=2e-7):
        raise ValueError('Registered 90/10 probability mixture changed')
    return normalized


def _comparison(key, candidate_key, reference_key, results, plan, index):
    candidate, reference = results[candidate_key], results[reference_key]
    truth = candidate['targets']
    a, b = candidate['predictions'] == truth, reference['predictions'] == truth
    corrected, harmed = int((a & ~b).sum()), int((~a & b).sum())
    settings = plan['confidence_interval']
    interval = stratified_accuracy_interval(a, b, truth,
        seed=settings['seed']+index, replicates=settings['replicates'], confidence=settings['confidence'])
    exact = exact_mcnemar(corrected, harmed)
    return dict(id=key, candidate=candidate_key, reference=reference_key, n=len(truth),
        corrected=corrected, harmed=harmed, both_correct=int((a & b).sum()),
        both_wrong=int((~a & ~b).sum()), prediction_changes=int((candidate['predictions'] != reference['predictions']).sum()),
        candidate_top1_pct=100*candidate['metrics']['accuracy'],
        reference_top1_pct=100*reference['metrics']['accuracy'],
        delta_accuracy_pp=100*(corrected-harmed)/len(truth),
        mean_nll_improvement=float((reference['nll']-candidate['nll']).mean()),
        nll_is_descriptive=True, confidence_interval=interval, exact_mcnemar=exact,
        multiplicity_adjusted_pvalue=None, confirmatory_gate_open=False,
        positive_improvement_established=False)


def evaluate_audit_results(results, *, plan=None):
    """Apply all predeclared rules to aligned, fixed confirmation predictions."""
    plan = _validated_plan(plan)
    results = _normalize_results(results)
    primary = _comparison(*PRIMARY, results, plan, 0)
    primary['family'] = 'primary'
    primary['confirmatory_gate_open'] = True
    primary['multiplicity_adjusted_pvalue'] = primary['exact_mcnemar']['pvalue']
    primary_positive = (primary['delta_accuracy_pp'] > 0
        and primary['exact_mcnemar']['pvalue'] < plan['alpha']
        and primary['confidence_interval']['low_pp'] > 0)
    primary['positive_improvement_established'] = bool(primary_positive)
    secondary = [_comparison(*item, results, plan, index+1) for index, item in enumerate(SECONDARY)]
    adjusted = holm_adjust({row['id']: row['exact_mcnemar']['pvalue'] for row in secondary})
    for row in secondary:
        row['family'] = 'secondary'
        row['multiplicity_adjusted_pvalue'] = adjusted[row['id']]
        row['confirmatory_gate_open'] = bool(primary_positive)
        row['positive_improvement_established'] = bool(primary_positive
            and row['delta_accuracy_pp'] > 0 and adjusted[row['id']] < plan['alpha']
            and row['confidence_interval']['low_pp'] > 0)
    metrics = [dict(model=key, **results[key]['metrics']) for key in MODELS]
    baseline = results['source_cnn']
    fingerprint = hashlib.sha256(json.dumps(dict(image_ids=baseline['image_ids'],
        targets=baseline['targets'].tolist(), class_names=baseline['class_names']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(protocol=PROTOCOL, status='complete', statistical_plan=plan,
        n=len(baseline['targets']), aligned_prediction_identity_sha256=fingerprint,
        metrics=metrics, primary=primary, secondary=secondary,
        conclusions=dict(primary_positive_improvement_established=bool(primary_positive),
            joint_vs_probability_90_10_positive_improvement_established=secondary[0]['positive_improvement_established'],
            selected_best_model=None, fixed_candidate='existing_joint',
            checkpoint_selection_on_confirmation_images=False,
            probability_weight_selection_on_confirmation_images=False,
            training_seed_variability_estimated=False),
        interpretation=('This is one fixed-model confirmation comparison. Paired image uncertainty '
            'does not estimate variability over training seeds or future datasets. A positive effect '
            'on one dataset does not establish a universal CNN ceiling, a new architecture or method novelty. '
            'No model or mixture weight is selected using these scores.'),
        evaluation_provenance_note=('The evaluation core must record the actual partition and prior-exposure '
            'evidence. Statistical calculations alone do not establish that a partition was previously unused.'))


def _atomic_text(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Report destination cannot be a symlink')
    temporary = path.with_name(path.name+'.tmp')
    if temporary.is_symlink():
        raise ValueError('Report temporary destination cannot be a symlink')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def _csv_text(rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return buffer.getvalue()


def _p_text(value):
    return '<1e-300' if value == 0 else f'{value:.6g}'


def write_report(output_dir, summary):
    """Write fixed-order aggregates, never rank models by confirmation score."""
    if summary.get('protocol') != PROTOCOL or summary.get('status') != 'complete':
        raise ValueError('Need a complete fixed-protocol statistics result')
    _validated_plan(summary.get('statistical_plan'))
    output = Path(output_dir).absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError('Report directory cannot contain symlinks')
    output.mkdir(parents=True, exist_ok=True)
    rows = [summary['primary'], *summary['secondary']]
    comparison_rows = [dict(id=row['id'], family=row['family'], candidate=row['candidate'],
        reference=row['reference'], n=row['n'], candidate_top1_pct=row['candidate_top1_pct'],
        reference_top1_pct=row['reference_top1_pct'], delta_accuracy_pp=row['delta_accuracy_pp'],
        corrected=row['corrected'], harmed=row['harmed'], both_correct=row['both_correct'],
        both_wrong=row['both_wrong'], prediction_changes=row['prediction_changes'],
        ci95_low_pp=row['confidence_interval']['low_pp'], ci95_high_pp=row['confidence_interval']['high_pp'],
        exact_mcnemar_p=row['exact_mcnemar']['pvalue'], adjusted_p=row['multiplicity_adjusted_pvalue'],
        mean_nll_improvement=row['mean_nll_improvement'], confirmatory_gate_open=row['confirmatory_gate_open'],
        positive_improvement_established=row['positive_improvement_established']) for row in rows]
    lines = ['# Fixed CNN/ViT confirmation', '',
        'Checkpoints, the 90/10 mixture and all statistical comparisons were fixed before confirmation inference. '
        'The existing joint model remains the candidate; this report does not choose a new winner.', '',
        '|Model, fixed order|Top-1|Top-5|NLL|', '|---|---:|---:|---:|']
    for row in summary['metrics']:
        lines.append(f'|{row["model"]}|{100*row["accuracy"]:.2f}%|'
            f'{100*row["top5_accuracy"]:.2f}%|{row["nll"]:.6f}|')
    lines.extend(['', '|Comparison|Delta pp|95% image CI|Corrected / harmed|Exact p|Adjusted p|Positive effect established|',
        '|---|---:|---|---:|---:|---:|---|'])
    for row in rows:
        interval = row['confidence_interval']
        lines.append(f'|{row["id"]}|{row["delta_accuracy_pp"]:+.3f}|'
            f'[{interval["low_pp"]:+.3f}, {interval["high_pp"]:+.3f}]|'
            f'{row["corrected"]} / {row["harmed"]}|{_p_text(row["exact_mcnemar"]["pvalue"])}|'
            f'{_p_text(row["multiplicity_adjusted_pvalue"])}|{row["positive_improvement_established"]}|')
    lines.extend(['', 'The primary hypothesis is joint versus CNN at alpha 0.05. Four secondary tests use '
        'Holm adjustment and are confirmatory only when the positive primary result passes. The 95% '
        'bootstrap intervals are pointwise, not multiplicity-adjusted simultaneous intervals. A positive '
        'effect is marked established only when its permitted p-value is below alpha, the effect is '
        'positive and its interval excludes zero. Otherwise it remains unestablished, which is not proof '
        'of equivalence.', '',
        'Bootstrap resampling uses 10,000 class-stratified draws of paired image outcomes and a fixed '
        'statistical RNG seed; each original class count stays fixed. '
        'It does not retrain models or change the training seed. Top-5, NLL and paired NLL differences are '
        'descriptive; lower NLL alone is not a demonstrated calibration improvement.', '',
        summary['interpretation'], '', summary['evaluation_provenance_note']])
    if 'evaluation_provenance' in summary:
        lines.extend(['', '## Evaluation provenance', '',
            '```json', json.dumps(summary['evaluation_provenance'], ensure_ascii=False, indent=2), '```'])
    _atomic_text(output/'metrics.csv', _csv_text(summary['metrics']))
    _atomic_text(output/'comparisons.csv', _csv_text(comparison_rows))
    _atomic_text(output/'statistics.json', json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    _atomic_text(output/'report.md', '\n'.join(lines)+'\n')
    return summary
