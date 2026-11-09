"""Read-only evidence for the Tiny ImageNet HCSI mechanism adaptation.

The study uses the development selection split, not the reserved audit or
official validation images. Fixed-weight no-interaction and uniform-routing
interventions describe a co-adapted model; they are not trained controls.
"""
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from attention_fusion_diagnostics import _write_csv
from correction_experiment import file_hash
from datasets.resnet_tiny import prepare_datasets
from datasets.utility_tiny import loader
from fusion_continuation_diagnostics import probability_result, paired_result, _write_result
from literature_experiment import new_name
from models.attention_correction import state_digest
from models.spatial_preact import build_spatial_model
from models.vit_augreg import build_vit


PROTOCOL = 'tiny-hcsi-adaptation-diagnostics-v1'
JOBS = ('cnn_continue', 'late_mlp', 'hcsi', 'cnn_mlp', 'late_mlp_random_vit')


class AnalysisDeadline(RuntimeError):
    pass


def check_deadline(deadline_unix):
    if deadline_unix is not None and time.time() >= float(deadline_unix):
        raise AnalysisDeadline('Analysis time limit; incomplete evaluations are not final scores')


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError('Analysis JSON cannot be a symlink')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05*(attempt+1))


@torch.inference_mode()
def evaluate_predictions(model, dataset, config, seed, device, *, deadline_unix=None,
                         collect_statistics=False):
    """Collect a complete clean split and reject any inference-state mutation."""
    if getattr(dataset, 'split', None) not in ('selection', 'train_probe'):
        raise ValueError('Only selection and the clean training probe are allowed')
    model.eval()
    before = state_digest(model)
    rows, losses, targets, image_ids = [], [], [], []
    statistics = {}
    names = list(dataset.folder.classes)
    for images, labels, ids in loader(dataset, config['dataset'], seed):
        check_deadline(deadline_unix)
        with torch.autocast(device, enabled=False):
            if collect_statistics and hasattr(model, 'forward_decomposed'):
                detail = model.forward_decomposed(images.to(device).float())
                logits = detail['logits'].float()
            else:
                detail = {}
                logits = model(images.to(device).float()).float()
        if logits.shape != (len(labels), len(names)) or not torch.isfinite(logits).all():
            raise ValueError('Invalid HCSI evaluation logits')
        truth = labels.to(device)
        losses.append(-logits.log_softmax(1).gather(1, truth[:, None]).squeeze(1).double().cpu().numpy())
        rows.append(logits.softmax(1).cpu().numpy())
        targets.extend(labels.tolist())
        image_ids.extend(ids)
        for key, value in detail.items():
            if (key.startswith('stage') and isinstance(value, torch.Tensor)
                    and value.shape == (len(labels),)):
                value = value.detach().float().cpu()
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite HCSI routing or gate statistic')
                statistics.setdefault(key, []).append(value)
    check_deadline(deadline_unix)
    if len(image_ids) != len(dataset) or not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError('Clean evaluation IDs or full split denominator differ')
    if state_digest(model) != before:
        raise RuntimeError('Read-only evaluation changed model parameters or buffers')
    result = probability_result(np.concatenate(rows), targets, image_ids, names,
                                nll=np.concatenate(losses))
    stats = {}
    for key, values in statistics.items():
        values = torch.cat(values)
        stats[key] = dict(mean=float(values.mean()), std=float(values.std(unbiased=False)),
            minimum=float(values.min()), maximum=float(values.max()),
            p10=float(torch.quantile(values, .1)), median=float(torch.quantile(values, .5)),
            p90=float(torch.quantile(values, .9)))
    return result, stats


def _summary(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Training summary cannot be a symlink')
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def interpret_comparisons(best_results, planned_epochs, summaries):
    """Separate the user hypothesis, head capacity and HCSI-specific evidence."""
    comparisons = {}
    pairs = (
        ('continued_cnn_vs_source_cnn', 'cnn_continue', 'source_cnn'),
        ('cnn_mlp_vs_continued_cnn', 'cnn_mlp', 'cnn_continue'),
        ('late_mlp_vs_cnn_mlp', 'late_mlp', 'cnn_mlp'),
        ('trained_vit_vs_random_vit', 'late_mlp', 'late_mlp_random_vit'),
        ('random_vit_vs_cnn_mlp', 'late_mlp_random_vit', 'cnn_mlp'),
        ('hcsi_vs_late_mlp', 'hcsi', 'late_mlp'),
        ('hcsi_vs_continued_cnn', 'hcsi', 'cnn_continue'),
        ('existing_joint_vs_continued_cnn', 'existing_joint', 'cnn_continue'),
        ('hcsi_vs_existing_joint', 'hcsi', 'existing_joint'),
    )
    for label, method, reference in pairs:
        if method in best_results and reference in best_results:
            comparisons[label] = paired_result(best_results[method], best_results[reference])
    complete = all(job in best_results and summaries.get(job, {}).get('status') == 'complete'
        and summaries[job].get('completed_epochs') == planned_epochs for job in JOBS)
    hcsi_trained = summaries.get('hcsi', {}).get('best_epoch', 0) > 0
    gate_gain = comparisons.get('hcsi_vs_late_mlp', {}).get('delta_accuracy_pp')
    baseline_gain = comparisons.get('hcsi_vs_continued_cnn', {}).get('delta_accuracy_pp')
    supported = complete and hcsi_trained and gate_gain is not None and gate_gain > 0 \
        and baseline_gain is not None and baseline_gain > 0
    trained_vit_gain = comparisons.get('trained_vit_vs_random_vit', {}).get('delta_accuracy_pp')
    mlp_gain = comparisons.get('late_mlp_vs_cnn_mlp', {}).get('delta_accuracy_pp')
    pretraining_supported = complete and trained_vit_gain is not None and trained_vit_gain > 0 \
        and mlp_gain is not None and mlp_gain > 0
    return dict(comparisons=comparisons, all_training_horizons_complete=complete,
        hcsi_best_has_training_updates=hcsi_trained,
        hcsi_mechanism_supported_on_selection=bool(supported),
        independent_vit_pretraining_supported_on_selection=bool(pretraining_supported),
        interpretation=('HCSI exceeds both its matched late-MLP control and continued CNN on development selection; '
            'one independently held-out confirmation is still required.' if supported else
            'The specified HCSI advantage is not established. Do not extend training or repeat seeds merely to rescue it.'),
        hypothesis_note='A gain over source CNN alone does not exclude ordinary continuation. The existing joint model '
            'versus matched continued CNN addresses the late-fusion hypothesis; HCSI versus late_mlp addresses '
            'intermediate gating with matched pooling and head. late_mlp versus the same architecture with '
            'a randomly initialized ViT tests the value of independent ViT training, rather than extra capacity alone.',
        selection_is_exploratory=True)


def _build_source_models(plan, sources):
    cnn = build_spatial_model('cnn8', seed=plan['seed'])
    cnn.load_state_dict(sources['parent_model'], strict=True)
    vit = build_vit(sources['selected']['profile'], seed=plan['seed'])
    vit.load_state_dict(sources['selected']['state']['model'], strict=True)
    return {'source_cnn': (cnn, dict(epoch=sources['provenance'].get('source_cnn_best_epoch'), source='source_best')),
            'source_vit': (vit, dict(epoch=sources['selected'].get('best_epoch'), source='source_best'))}


def run_hcsi_diagnostics(plan, training_dir, output_dir, examples=8, deadline_unix=None):
    from hcsi_tiny_experiment import load_sources, load_reference_model, load_model
    started = time.time()
    training, output = Path(training_dir).resolve(), Path(output_dir).resolve()
    if output == training or output.is_relative_to(training) or training.is_relative_to(output):
        raise ValueError('Diagnostics output must be separate from training')
    if output.is_symlink():
        raise ValueError('Analysis directory cannot be a symlink')
    output.mkdir(parents=True, exist_ok=True)
    run = output/new_name('analysis')
    run.mkdir()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sources = load_sources(plan)
    datasets, manifest, fingerprint = prepare_datasets(plan['config'], plan['seed'])
    if (fingerprint != plan['split_hash'] or len(datasets['selection']) != 10000
            or len(datasets['train_probe']) != 4000
            or set(manifest['train']) & set(manifest['selection'])):
        raise ValueError('Expected original disjoint80K/10K selection and4K training probe')
    rows, records, best_results, failures = [], {}, {}, {}
    summaries = {job: _summary(training/job/'summary.json') for job in JOBS}
    try:
        check_deadline(deadline_unix)
        source_models = _build_source_models(plan, sources)
        reference = load_reference_model(plan, sources=sources)
        if reference is not None:
            source_models['existing_joint'] = reference
        work = [(name, 'source_best', model, state) for name, (model, state) in source_models.items()]
        # Load each large trained checkpoint only when evaluated.
        work.extend((job, checkpoint, None, None) for job in JOBS
            for checkpoint in ('best.pt', 'last.pt', 'last_ema'))
        for job, checkpoint, model, state in work:
            key = job+'/'+checkpoint
            try:
                check_deadline(deadline_unix)
                if model is None:
                    if not (training/job/'last.pt').is_file():
                        failures[key] = dict(status='missing', reason='No committed training checkpoint')
                        continue
                    model, state = load_model(plan, training, job, checkpoint=checkpoint)
                model.to(device).eval()
                path = run/job/checkpoint.replace('.pt', '')
                result, statistics = evaluate_predictions(model, datasets['selection'], plan['config'],
                    plan['seed'], device, deadline_unix=deadline_unix, collect_statistics=job == 'hcsi')
                _write_result(path/'selection', result)
                probe, _ = evaluate_predictions(model, datasets['train_probe'], plan['config'],
                    plan['seed'], device, deadline_unix=deadline_unix)
                _write_result(path/'train_probe', probe)
                record = dict(metrics=result['metrics'], train_probe_metrics=probe['metrics'],
                    train_probe_gap_pp=100*(probe['metrics']['accuracy']-result['metrics']['accuracy']),
                    epoch=state.get('epoch'), source=state.get('source'), statistics=statistics,
                    params=sum(p.numel() for p in model.parameters()),
                    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
                if job in JOBS:
                    record['training_summary'] = summaries[job]
                    filename = 'last.pt' if checkpoint == 'last_ema' else checkpoint
                    record['checkpoint_sha256'] = file_hash(training/job/filename)
                records[key] = record
                rows.append(dict(job=job, checkpoint=checkpoint, epoch=state.get('epoch'), source=state.get('source'),
                    **result['metrics'], train_probe_accuracy=probe['metrics']['accuracy'],
                    train_probe_gap_pp=record['train_probe_gap_pp'], params=record['params']))
                if checkpoint in ('best.pt', 'source_best'):
                    best_results[job] = result
                if job == 'hcsi' and checkpoint == 'best.pt':
                    interventions = {}
                    for mode in ('no_interaction', 'uniform'):
                        try:
                            check_deadline(deadline_unix)
                            with model.analysis(mode):
                                changed, detail = evaluate_predictions(model, datasets['selection'], plan['config'],
                                    plan['seed'], device, deadline_unix=deadline_unix, collect_statistics=True)
                            _write_result(path/mode, changed)
                            interventions[mode] = dict(metrics=changed['metrics'], statistics=detail,
                                paired_vs_learned=paired_result(changed, result),
                                note='Fixed-weight intervention on co-adapted encoders/head, not a trained baseline.')
                        except AnalysisDeadline:
                            interventions[mode] = dict(status='not_performed', reason='Time budget')
                            break
                    record['interventions'] = interventions
                del model
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
            except AnalysisDeadline:
                failures[key] = dict(status='partial', reason='Analysis time budget')
                break
            except Exception as exc:
                failures[key] = dict(status='failed', error_type=type(exc).__name__, reason=str(exc))
    except AnalysisDeadline:
        failures['analysis'] = dict(status='partial', reason='Analysis time budget')
    interpretation = interpret_comparisons(best_results, plan['epochs'], summaries)
    complete = not failures and all(job in best_results for job in JOBS) \
        and interpretation['all_training_horizons_complete']
    result = dict(protocol=PROTOCOL, status='complete' if complete else 'partial',
        directory=run.name, jobs=records, failures=failures, comparisons=interpretation,
        elapsed_seconds=time.time()-started, audit_partition_evaluated=False,
        official_validation_evaluated=False,
        method_scope='Tiny ImageNet mechanism adaptation of HCSI with four aligned interaction stages; '
            'not author-benchmark SBI/deepfake reproduction.',
        galleries=dict(status='not_performed', reason='Aggregate paired decisions and routing/gate statistics prioritized within the10h budget.'),
        interpretation='All checkpoint selection and comparisons use development data; no independent confirmation is claimed.')
    _write_csv(run/'comparison.csv', rows)
    save_json(run/'summary.json', result)
    save_json(output/'completion.json', result)
    text = ['# HCSI adaptation and matched controls', '', f'Status: **{result["status"]}**. Development selection only.', '',
        '|Model|Checkpoint|Top1|NLL|Probe gap pp|Epoch|', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        text.append(f'|{row["job"]}|{row["checkpoint"]}|{100*row["accuracy"]:.2f}%|'
            f'{row["nll"]:.4f}|{row["train_probe_gap_pp"]:.2f}|{row["epoch"]}|')
    text.extend(['', interpretation['interpretation'], '',
        'HCSI versus late_mlp compares the interaction mechanism using the same patch-mean pooling and MLP head. '
        'The previous65.45% joint model instead uses CLS pooling and a linear head; its comparison cannot isolate gating. '
        'CNN continuation tests additional training; cnn_mlp tests the deeper head without ViT. '
        'late_mlp_random_vit holds the fusion architecture and parameter count fixed while removing independent ViT pretraining. '
        'No-interaction and uniform interventions use already co-adapted weights and are not trained controls.', '',
        'The original paper uses different backbones, SBI supervision and cross-dataset deepfake AUC. '
        'These Tiny ImageNet results are not numerical reproductions of its benchmark. '
        'Raw image galleries are deferred; full arrays and checkpoints remain on the VM.'])
    (output/'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return result
