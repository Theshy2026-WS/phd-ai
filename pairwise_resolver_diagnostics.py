"""Read-only, cached-feature evidence for the frozen-anchor pairwise study.

Selection remains a development split. This module never opens the reserved
audit partition or official validation set. Global patch mismatch uses all
image IDs and never class-sorted batch positions. It is an intervention, not a
single-image model score.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from attention_fusion_diagnostics import _write_csv
from best_pair_diagnostics import mismatch_policy
from correction_experiment import file_hash
from fusion_continuation_diagnostics import probability_result, paired_result, _write_result
from literature_experiment import new_name


PROTOCOL = 'tiny-frozen-pairwise-resolver-diagnostics-v1'
JOBS = ('pairwise', 'static_query', 'mlp', 'mean_tokens')


class AnalysisDeadline(RuntimeError):
    pass


def save_json(path, value):
    """Retain existing reports if a Windows sync scanner briefly holds a file."""
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError('Report JSON must not be a symlink')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05*(attempt+1))


def _check_deadline(deadline_unix):
    if deadline_unix is not None and time.time() >= float(deadline_unix):
        raise AnalysisDeadline('Analysis deadline reached; incomplete evaluations are not final scores')


def _cache_layout(cache):
    required = ('features', 'patch_tokens', 'anchor_logits', 'labels', 'image_ids', 'class_names')
    if any(key not in cache for key in required):
        raise ValueError('Resolver feature cache is missing required fields')
    n = len(cache['image_ids'])
    if (not n or len(set(cache['image_ids'])) != n
            or cache['features'].ndim != 2 or len(cache['features']) != n
            or cache['patch_tokens'].ndim != 3 or len(cache['patch_tokens']) != n
            or cache['anchor_logits'].shape != (n, len(cache['class_names']))
            or cache['labels'].shape != (n,)):
        raise ValueError('Resolver feature-cache layout differs')
    for key in ('features', 'patch_tokens', 'anchor_logits'):
        if not torch.isfinite(cache[key]).all():
            raise ValueError('Feature cache contains nonfinite values')
    return n


def _as_result(logits, cache):
    value = torch.as_tensor(logits, dtype=torch.float32).cpu()
    truth = torch.as_tensor(cache['labels'], dtype=torch.long).cpu()
    if value.shape != cache['anchor_logits'].shape or not torch.isfinite(value).all():
        raise ValueError('Resolver evaluation logits are invalid')
    losses = -value.log_softmax(1).gather(1, truth[:, None]).squeeze(1).double().numpy()
    return probability_result(value.softmax(1).numpy(), truth.numpy(),
        cache['image_ids'], cache['class_names'], nll=losses)


def anchor_result(cache):
    _cache_layout(cache)
    return _as_result(cache['anchor_logits'], cache)


@torch.inference_mode()
def evaluate_cached_resolver(model, cache, batch_size, device, *, deadline_unix=None,
        patch_mode='learned', train_mean=None, mismatch_seed=730045):
    """Evaluate a complete cached split; no partial count is returned as final."""
    n = _cache_layout(cache)
    if int(batch_size) < 1 or patch_mode not in ('learned', 'train_mean', 'global_mismatch'):
        raise ValueError('Invalid cached resolver evaluation policy')
    model.eval()
    pairing = None
    if patch_mode == 'global_mismatch':
        donors, pairing = mismatch_policy(cache['image_ids'],
            torch.as_tensor(cache['labels']).cpu().numpy(), mismatch_seed)
    elif patch_mode == 'train_mean':
        if train_mean is None or tuple(train_mean.shape) != tuple(cache['patch_tokens'].shape[1:]):
            raise ValueError('Train-only mean token layout differs')
    logits, deltas, candidates, etas = [], [], [], []
    route_statistics = {}
    anchor = cache['anchor_logits'].float().cpu()
    for start in range(0, n, int(batch_size)):
        _check_deadline(deadline_unix)
        stop = min(n, start+int(batch_size))
        features = cache['features'][start:stop].to(device).float()
        patches = cache['patch_tokens'][start:stop]
        if patch_mode == 'global_mismatch':
            patches = cache['patch_tokens'][torch.as_tensor(donors[start:stop])]
        elif patch_mode == 'train_mean':
            patches = train_mean.unsqueeze(0).expand(stop-start, -1, -1)
        base = anchor[start:stop].to(device)
        with torch.autocast(device, enabled=False):
            detail = model.forward_cached(features, patches.to(device).float(),
                anchor_logits=base, collect=True)
        for key in ('logits', 'delta', 'candidates', 'eta'):
            if key not in detail:
                raise ValueError(f'Resolver did not expose {key}')
        out, delta = detail['logits'].float(), detail['delta'].float()
        eta = float(torch.as_tensor(detail['eta']).detach().cpu())
        selected = detail['candidates'].long()
        if (out.shape != base.shape or delta.shape != base.shape
                or selected.ndim != 2 or len(selected) != len(base)
                or not np.isfinite(eta) or eta <= 0
                or not torch.isfinite(out).all() or not torch.isfinite(delta).all()):
            raise ValueError('Resolver produced invalid details')
        if (not torch.allclose(out, base+delta, atol=2e-5, rtol=1e-6)
                or float(delta.abs().max()) > eta+2e-5
                or float(delta.sum(1).abs().max()) > 2e-5):
            raise RuntimeError('Resolver violated its stated bounded zero-sum update')
        if 'anchor_logits' in detail and not torch.equal(detail['anchor_logits'].float(), base):
            raise RuntimeError('Cached inference altered the frozen anchor logits')
        logits.append(out.cpu()); deltas.append(delta.cpu())
        candidates.append(selected.cpu()); etas.extend([eta]*len(base))
        for key in ('attention_entropy', 'attention_peak', 'attention_logit_std',
                'signed_weight_l1', 'query_rms', 'key_rms', 'value_rms',
                'correction_rms', 'pair_score_rms'):
            if key in detail:
                value = torch.as_tensor(detail[key]).detach().float().cpu().reshape(-1)
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite routing statistics')
                route_statistics.setdefault(key, []).append(value)
    _check_deadline(deadline_unix)
    logits, delta, selected = torch.cat(logits), torch.cat(deltas), torch.cat(candidates)
    eta = torch.tensor(etas, dtype=torch.float64)
    # Stable Top-k is also the model candidate policy and resolves boundary ties.
    expected = torch.argsort(anchor, dim=1, descending=True, stable=True)[:, :selected.shape[1]]
    if not torch.equal(selected, expected):
        raise RuntimeError('Reported candidates differ from the actual anchor Top-k')
    result, reference = _as_result(logits, cache), anchor_result(cache)
    top = torch.argsort(anchor, dim=1, descending=True, stable=True)
    margin = anchor.gather(1, top[:, :1]).squeeze(1)-anchor.gather(1, top[:, 1:2]).squeeze(1)
    protected = margin.double() > 2*eta+2e-5
    predictions = torch.as_tensor(result['predictions'])
    outside = ~(selected == predictions[:, None]).any(1)
    targets = torch.as_tensor(cache['labels']).cpu()
    coverage = (selected == targets[:, None]).any(1)
    changed = predictions != torch.as_tensor(reference['predictions'])
    if bool((protected & changed).any()) or bool(outside.any()):
        raise RuntimeError('Resolver violated its margin protection or Top-k candidate constraint')
    stats = dict(n=n, patch_mode=patch_mode, eta=float(eta[0]),
        trainable_parameters=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        total_parameters=int(sum(p.numel() for p in model.parameters())),
        candidate_k=int(selected.shape[1]), candidate_target_coverage=float(coverage.double().mean()),
        candidate_coverage_note='An architecture-specific upper bound when updates are zero-sum within the true anchor Top-k.',
        predicted_outside_candidates=int(outside.sum()),
        protected_by_margin=int(protected.sum()), protected_prediction_changes=int((protected & changed).sum()),
        protected_anchor_correct=int((protected & (targets == torch.as_tensor(reference['predictions']))).sum()),
        max_abs_delta=float(delta.abs().max()), max_abs_delta_sum=float(delta.sum(1).abs().max()),
        mean_abs_delta=float(delta.abs().mean()), paired_vs_anchor=paired_result(result, reference))
    if pairing is not None:
        stats['mismatch_policy'] = pairing
    stats['routing_statistics'] = {key: dict(mean=float(torch.cat(values).mean()),
        std=float(torch.cat(values).std(unbiased=False)),
        p10=float(torch.quantile(torch.cat(values), .1)),
        p50=float(torch.quantile(torch.cat(values), .5)),
        p90=float(torch.quantile(torch.cat(values), .9))) for key, values in route_statistics.items()}
    return result, stats


def control_comparisons(results, anchor=None, *, all_training_complete=False, full_best_is_trained=False):
    full = results.get('pairwise')
    if full is None:
        return dict(status='incomplete', reason='Input-conditioned pairwise result is missing', comparisons={})
    comparisons = {job: paired_result(full, results[job]) for job in JOBS[1:] if job in results}
    complete = len(comparisons) == len(JOBS)-1
    beats_controls = complete and all(item['delta_accuracy_pp'] > 0 for item in comparisons.values())
    anchor_comparison = paired_result(full, anchor) if anchor is not None else None
    beats_anchor = anchor_comparison is not None and anchor_comparison['delta_accuracy_pp'] > 0
    advance = beats_controls and beats_anchor and all_training_complete and full_best_is_trained
    return dict(status='complete' if complete else 'incomplete', comparisons=comparisons,
        full_vs_anchor=anchor_comparison,
        full_beats_all_three_controls_on_selection=bool(beats_controls),
        full_beats_anchor_on_selection=bool(beats_anchor),
        all_four_training_horizons_complete=bool(all_training_complete),
        full_best_checkpoint_has_training_updates=bool(full_best_is_trained),
        eligible_for_held_out_confirmation=bool(advance),
        recommendation=('Candidate for one predeclared held-out confirmation; selection gains remain exploratory.' if advance else
            'Do not extend epochs or repeat seeds to rescue this hypothesis; inspect the specified controls.'),
        novelty_note='Control wins test this mechanism in this setting; they do not establish research novelty or publication sufficiency.')


def _safe_summary(path):
    path = Path(path)
    if path.is_file() and not path.is_symlink():
        return json.loads(path.read_text(encoding='utf-8'))
    return {}


def run_pairwise_diagnostics(plan, training_dir, output_dir, examples=8, deadline_unix=None):
    from pairwise_resolver_experiment import load_feature_cache, load_resolver_model
    started = time.time()
    training, output = Path(training_dir).resolve(), Path(output_dir).resolve()
    if output == training or output.is_relative_to(training) or training.is_relative_to(output):
        raise ValueError('Diagnostics must use a separate directory from training')
    output.mkdir(parents=True, exist_ok=True)
    _check_deadline(deadline_unix)
    cache = load_feature_cache(plan, training/'selection_cache.pt')
    _cache_layout(cache)
    if cache.get('identity', {}).get('split') != 'selection':
        raise ValueError('Only development selection is permitted in this first-round analysis')
    run = output/new_name('analysis')
    run.mkdir()
    reference = anchor_result(cache)
    _write_result(run/'anchor', reference)
    save_json(run/'identity.json', dict(protocol=PROTOCOL, cache_identity=cache.get('identity'),
        image_ids_sha256=hashlib.sha256('\n'.join(cache['image_ids']).encode()).hexdigest(),
        selection_cache_sha256=file_hash(training/'selection_cache.pt'),
        audit_partition_evaluated=False, official_validation_evaluated=False))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = int(plan['config']['dataset']['batch_size'])
    results, rows, records, failures = {}, [], {}, {}
    for job in JOBS:
        path = training/job/'best.pt'
        summary = _safe_summary(training/job/'summary.json')
        if not path.is_file() or path.is_symlink():
            failures[job] = dict(status='missing', reason='No regular best checkpoint is available')
            continue
        try:
            _check_deadline(deadline_unix)
            model, state = load_resolver_model(plan, training, job, checkpoint='best.pt')
            model.to(device).eval()
            result, stats = evaluate_cached_resolver(model, cache, batch, device, deadline_unix=deadline_unix)
            _write_result(run/job, result)
            results[job] = result
            records[job] = dict(metrics=result['metrics'], properties=stats,
                checkpoint_sha256=file_hash(path), epoch=state.get('epoch'),
                source=state.get('source'), training_summary=summary,
                training_complete=(summary.get('status') == 'complete'
                    and summary.get('completed_epochs', -1) == plan['epochs']))
            trained_path = training/job/'best_trained.pt'
            if int(state.get('epoch', 0)) == 0 and trained_path.is_file() and not trained_path.is_symlink():
                trained_model, trained_state = load_resolver_model(plan, training, job, checkpoint='best_trained.pt')
                trained_model.to(device).eval()
                trained_result, trained_stats = evaluate_cached_resolver(trained_model, cache, batch, device,
                    deadline_unix=deadline_unix)
                _write_result(run/(job+'_best_trained'), trained_result)
                records[job]['best_trained_checkpoint'] = dict(metrics=trained_result['metrics'],
                    properties=trained_stats, epoch=trained_state.get('epoch'), source=trained_state.get('source'),
                    checkpoint_sha256=file_hash(trained_path))
                del trained_model
            else:
                records[job]['best_trained_checkpoint'] = dict(metrics=summary.get('best_trained_selection'),
                    epoch=summary.get('best_trained_epoch'), source=summary.get('best_trained_source'),
                    note='Training summary; main best-checkpoint evaluation is listed separately.')
            row = dict(job=job, epoch=state.get('epoch'), source=state.get('source'),
                training_status=summary.get('status', 'unverified'), **result['metrics'],
                corrected_vs_anchor=stats['paired_vs_anchor']['corrected'],
                harmed_vs_anchor=stats['paired_vs_anchor']['harmed'],
                delta_vs_anchor_pp=stats['paired_vs_anchor']['delta_accuracy_pp'],
                eta=stats['eta'], trainable_parameters=stats['trainable_parameters'],
                candidate_coverage=stats['candidate_target_coverage'])
            rows.append(row)
            if job == 'pairwise':
                # These quick interventions reuse the same cached frozen anchor.
                # They are never selected as competing trained model scores.
                interventions = {}
                for mode in ('global_mismatch', 'train_mean'):
                    try:
                        _check_deadline(deadline_unix)
                        mean = None
                        if mode == 'train_mean':
                            probe = load_feature_cache(plan, training/'train_probe_cache.pt')
                            identity = probe.get('identity', {})
                            if identity.get('split') not in ('train_probe', 'train_clean', 'clean_training_probe'):
                                raise ValueError('Mean tokens require a verified clean training probe')
                            if not bool(model.resolver.train_patch_mean_ready):
                                raise ValueError('Checkpoint has no fitted training-only mean')
                            mean = model.train_patch_mean.detach().cpu().float()
                            if not torch.equal(mean, probe['patch_tokens'].mean(0)):
                                raise ValueError('Checkpoint mean differs from the fixed training probe')
                        intervention, detail = evaluate_cached_resolver(model, cache, batch, device,
                            deadline_unix=deadline_unix, patch_mode=mode, train_mean=mean,
                            mismatch_seed=int(plan.get('seed', 42))+730003)
                        interventions[mode] = dict(metrics=intervention['metrics'], properties=detail,
                            note='Fixed-weight intervention; not a trained or deployable model score.')
                        save_json(run/job/f'{mode}_intervention.json', interventions[mode])
                    except AnalysisDeadline:
                        interventions[mode] = dict(status='not_performed', reason='Time budget')
                        break
                records[job]['interventions'] = interventions
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
        except AnalysisDeadline:
            failures[job] = dict(status='incomplete', reason='Analysis time budget')
            break
        except Exception as exc:
            failures[job] = dict(status='failed', error_type=type(exc).__name__, reason=str(exc))
    complete = not failures and len(results) == len(JOBS) and all(r['training_complete'] for r in records.values())
    comparisons = control_comparisons(results, reference, all_training_complete=complete,
        full_best_is_trained=int(records.get('pairwise', {}).get('epoch') or 0) > 0)
    result = dict(protocol=PROTOCOL, status='complete' if complete else 'partial',
        directory=str(run.relative_to(output)), anchor_metrics=reference['metrics'], jobs=records,
        failures=failures, controls=comparisons, elapsed_seconds=time.time()-started,
        audit_partition_evaluated=False, official_validation_evaluated=False,
        galleries=dict(status='not_performed', reason='Cached vectors/tokens suffice for this predeclared mechanism test; raw-image visualization is deferred.'),
        interpretation='All model selection and paired comparisons here are exploratory development-set evidence.')
    _write_csv(run/'comparison.csv', rows)
    save_json(run/'completion.json', result)
    save_json(output/'completion.json', result)
    text = ['# Frozen-anchor pairwise resolver evidence', '',
        f'Status: **{result["status"]}**. Development selection only; audit and official validation were not evaluated.', '',
        f'Frozen anchor: Top1 **{100*reference["metrics"]["accuracy"]:.2f}%**, NLL{reference["metrics"]["nll"]:.4f}.', '',
        '|Arm|Top1|NLL|Corrected|Harmed|Delta pp|Epoch|', '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f'|{row["job"]}|{100*row["accuracy"]:.2f}%|{row["nll"]:.4f}|{row["corrected_vs_anchor"]}|{row["harmed_vs_anchor"]}|{row["delta_vs_anchor_pp"]:+.2f}|{row["epoch"]}|')
    text.extend(['', 'Best checkpoint after at least one completed training epoch (kept separately when initialization wins):', ''])
    for job, record in records.items():
        trained = record.get('best_trained_checkpoint', {})
        metric = trained.get('metrics')
        if metric:
            text.append(f'- {job}: {100*metric["accuracy"]:.2f}%, epoch{trained.get("epoch")}, source{trained.get("source")}.')
    text.extend(['', comparisons['recommendation'] if 'recommendation' in comparisons else comparisons['reason'], '',
        'Candidate Top-k coverage bounds this architecture, not general CNN/ViT fusion. '
        'The margin bound protects predictions, including some confident errors. '
        'Attention maps alone do not prove a causal explanation. The global mismatch intervention is label blind and spans the full split.', '',
        f'Detailed evidence: [{run.name}/comparison.csv]({run.name}/comparison.csv). '
        'Prediction arrays and checkpoints remain on the VM; the email contains ordinary report formats.'])
    (output/'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return result
