"""Read-only, paired evidence for the winning-family fusion controls.

The held-out audit and official validation partitions are never evaluated.
Checkpoint selection, probability-mixture grids and mechanism comparisons here
are development diagnostics, not independent confirmation of a contribution.
"""
import gc
import json
from pathlib import Path
import time

import torch

from attention_fusion_diagnostics import _write_csv
from baseline_diagnostics import clean_datasets, class_labels
from best_pair_diagnostics import (
    collect_concat_features, contribution_groups, evaluate_cached_modes,
    evaluate_plain, training_mean, MISMATCH_POLICY)
from correction_experiment import file_hash
from fusion_continuation_diagnostics import (
    _validate_arrays, _write_result, fixed_mixtures, load_probabilities,
    paired_result, threeway_counts)
from hcsi_tiny_diagnostics import check_deadline, save_json
from literature_experiment import new_name
from models.spatial_preact import build_spatial_model
from models.vit_augreg import build_vit


PROTOCOL = 'tiny-winning-fusion-mechanism-diagnostics-v1'
JOBS = ('random_vit_cls_linear', 'reset_cnn_head_cls_linear', 'patch_mean_linear')
SPLITS = ('train_probe', 'selection')
INTERVENTION_NOTE = ('Masked, mean-replaced and mismatched branches use jointly adapted '
    'weights and the same classifier. They are interventions, not independently trained '
    'CNN/ViT baselines. The CNN term contains the shared bias.')


def _json(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Diagnostic metadata cannot be a symlink')
    return json.loads(path.read_text(encoding='utf-8'))


def _allowed_data(probe, selection, split_hash, expected_hash, manifest):
    if split_hash != expected_hash:
        raise ValueError('Diagnostic partition changed')
    if getattr(probe, 'split', None) not in ('train_clean', 'train_probe') \
            or getattr(selection, 'split', None) != 'selection':
        raise ValueError('Only clean training probe and selection are allowed')
    if set(manifest['train_clean']) & set(manifest['selection']):
        raise ValueError('Training probe and development selection overlap')
    if len(probe) != len(manifest['train_clean']) or len(selection) != len(manifest['selection']):
        raise ValueError('Diagnostic image counts differ from manifest')


def _protect_output(output, training, plan):
    protected = [training, plan['cnn_suite'], plan['vit_suite'], plan['reference']['suite']]
    for value in protected:
        path = Path(value).resolve()
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('Diagnostics output must be separate from immutable sources and training')
    if output.is_symlink():
        raise ValueError('Diagnostics output cannot be a symlink')


def _checkpoint_manifest(training):
    files, summaries, work = {}, {}, []
    for job in JOBS:
        run = training/job
        summary_path = run/'summary.json'
        summaries[job] = _json(summary_path) if summary_path.is_file() else {}
        for name in ('metadata.json', 'summary.json', 'epochs.csv', 'history.csv'):
            path = run/name
            if path.is_symlink():
                raise ValueError('Training artifact cannot be a symlink')
            if path.is_file():
                files[str(Path(job)/name)] = file_hash(path)
        for name in ('best.pt', 'best_trained.pt', 'last.pt'):
            path = run/name
            if path.is_symlink():
                raise ValueError('Training checkpoint cannot be a symlink')
            if path.is_file():
                files[str(Path(job)/name)] = file_hash(path)
                work.append((job, name))
        if (run/'last.pt').is_file():
            work.append((job, 'last_ema'))
    return files, summaries, work


def interpret_comparisons(best_results, summaries, epochs):
    """State precisely which fixed-budget hypotheses the results support."""
    pairs = (
        ('winner_vs_source_cnn', 'existing_joint', 'source_cnn'),
        ('winner_vs_random_vit', 'existing_joint', 'random_vit_cls_linear'),
        ('winner_vs_reset_cnn_head', 'existing_joint', 'reset_cnn_head_cls_linear'),
        ('patch_mean_vs_cls', 'patch_mean_linear', 'existing_joint'),
        ('random_vit_vs_source_cnn', 'random_vit_cls_linear', 'source_cnn'),
        ('reset_head_vs_source_cnn', 'reset_cnn_head_cls_linear', 'source_cnn'),
        ('patch_mean_vs_source_cnn', 'patch_mean_linear', 'source_cnn'))
    comparisons = {label: paired_result(best_results[method], best_results[reference])
        for label, method, reference in pairs if method in best_results and reference in best_results}
    complete = all(summaries.get(job, {}).get('status') == 'complete'
        and summaries[job].get('completed_epochs') == epochs and job in best_results for job in JOBS)
    def greater(name):
        value = comparisons.get(name, {}).get('delta_accuracy_pp')
        return None if not complete or value is None else bool(value > 0)
    return dict(comparisons=comparisons, all_training_horizons_complete=complete,
        independently_trained_vit_advantage_on_selection=greater('winner_vs_random_vit'),
        inherited_cnn_head_advantage_on_selection=greater('winner_vs_reset_cnn_head'),
        patch_mean_advantage_on_selection=greater('patch_mean_vs_cls'),
        scope=f'Only this matched {epochs}-epoch joint fine-tuning recipe; no claim about arbitrary training budgets.',
        selection_is_exploratory=True, independent_confirmation=False,
        interpretation=('Use the same CLS+Linear family to test ViT initialization and inherited CNN head. '
            'Patch-mean+Linear changes only the readout. A positive development difference is a mechanism '
            'lead, not a significance, universal necessity, novelty or audit-set claim. '
            'A reset/random arm peaking late may be optimization-limited within this fixed budget.'))


def _cache_files(path):
    result = {}
    for item in path.rglob('*'):
        if item.is_symlink():
            raise ValueError('Cached analysis cannot contain symlinks')
        if item.is_file() and item.name != 'record.json':
            result[item.relative_to(path).as_posix()] = file_hash(item)
    return result


def _load_cache(path, identity):
    marker = path/'record.json'
    if not marker.is_file():
        return None
    record = _json(marker)
    if record.get('identity') != identity:
        raise ValueError('Cached checkpoint analysis belongs to another model')
    for relative, digest in record.get('files', {}).items():
        part = Path(relative)
        item = path/part
        if part.is_absolute() or '..' in part.parts or item.is_symlink() or not item.is_file() \
                or not item.resolve().is_relative_to(path.resolve()) or file_hash(item) != digest:
            raise ValueError('Cached checkpoint analysis changed or escaped its directory')
    if not record.get('files'):
        raise ValueError('Cached checkpoint analysis has no validated outputs')
    results = {split: load_probabilities(path/split/'learned'/'predictions.npz') for split in SPLITS}
    return record, results


def _prepare_cache(path):
    """Keep an interrupted write as evidence and start a clean owned directory."""
    if path.is_symlink():
        raise ValueError('Analysis cache directory cannot be a symlink')
    if path.exists():
        path.rename(path.with_name(path.name+'__incomplete_'+new_name('attempt')))


def _source_models(plan, sources):
    cnn = build_spatial_model('cnn8', seed=plan['seed'])
    cnn.load_state_dict(sources['parent_model'], strict=True)
    yield 'source_cnn', cnn, dict(epoch=sources['provenance']['source_cnn_best_epoch'], source='source_best')
    vit = build_vit(sources['selected']['profile'], seed=plan['seed'])
    vit.load_state_dict(sources['selected']['state']['model'], strict=True)
    yield 'source_vit', vit, dict(epoch=sources['provenance']['source_vit_best_epoch'], source='source_best')
    from fusion_mechanism_experiment import load_reference_model
    # This helper loads the historical winner with its canonical independent
    # source scope, never passing an augmented source_files map to the old guard.
    model, state = load_reference_model(plan, sources=sources)
    yield 'existing_joint', model, state


def _independent_record(model, datasets, plan, path, names, device, identity, state, deadline):
    results, rows = {}, []
    for split in SPLITS:
        check_deadline(deadline)
        result = evaluate_plain(model, datasets[split], plan['config'], plan['seed'], device, names)
        results[split] = result
        _write_result(path/split/'learned', result)
        rows.append(dict(split=split, mode='learned', **result['metrics']))
    record = dict(identity=identity, epoch=state.get('epoch'), source=state.get('source'),
        rows=rows, metrics=results['selection']['metrics'], train_probe_metrics=results['train_probe']['metrics'],
        train_probe_gap_pp=100*(results['train_probe']['metrics']['accuracy']-results['selection']['metrics']['accuracy']),
        params=sum(p.numel() for p in model.parameters()), interventions={}, contributions=[])
    return record, results


def _concat_record(model, datasets, plan, path, names, device, identity, state, references, deadline):
    results, rows, interventions, contributions, trio = {}, [], {}, [], None
    for split in SPLITS:
        check_deadline(deadline)
        cache = collect_concat_features(model, datasets[split], plan['config'], plan['seed'], device)
        if split == 'train_probe':
            mean, mean_info = training_mean(cache)
            path.mkdir(parents=True, exist_ok=True)
            save_json(path/'training_mean_provenance.json', mean_info)
        values, margins, policy = evaluate_cached_modes(model, cache, plan['config'], plan['seed'],
            device, names, mean_feature=mean)
        policy.pop('donor_image_ids', None)
        save_json(path/f'{split}_intervention_policy.json', policy)
        learned = values['learned']
        for mode, result in values.items():
            _validate_arrays(result, names, references[split]['source_cnn']['image_ids'],
                references[split]['source_cnn']['targets'])
            _write_result(path/split/mode, result)
            rows.append(dict(split=split, mode=mode, **result['metrics']))
        results[split] = learned
        interventions[split] = {mode: dict(metrics=result['metrics'],
            learned_vs_intervention=paired_result(learned, result), note=INTERVENTION_NOTE)
            for mode, result in values.items() if mode != 'learned'}
        contributions.extend(dict(split=split, **entry) for entry in
            contribution_groups(margins, learned, references[split]['source_cnn']))
        if split == 'selection':
            trio = threeway_counts(references[split]['source_cnn'], references[split]['source_vit'], learned)
        del cache, values
    record = dict(identity=identity, epoch=state.get('epoch'), source=state.get('source'),
        readout_policy=getattr(model, 'readout_policy', 'CLS'), rows=rows,
        metrics=results['selection']['metrics'], train_probe_metrics=results['train_probe']['metrics'],
        train_probe_gap_pp=100*(results['train_probe']['metrics']['accuracy']-results['selection']['metrics']['accuracy']),
        params=sum(p.numel() for p in model.parameters()), interventions=interventions,
        contributions=contributions, threeway=trio)
    return record, results


def run_fusion_mechanism_diagnostics(plan, training_dir, output_dir, examples=0, deadline_unix=None):
    """Resume only exact, checksum-verified per-checkpoint diagnostic outputs."""
    from fusion_mechanism_experiment import load_sources, load_model
    started = time.time()
    training, output = Path(training_dir).resolve(), Path(output_dir).resolve()
    _protect_output(output, training, plan)
    if _json(training/'suite.json') != plan:
        raise ValueError('Training and diagnostic plans differ')
    sources = load_sources(plan)
    checkpoints, summaries, work = _checkpoint_manifest(training)
    identity = dict(protocol=PROTOCOL, source_provenance=plan['source_provenance'],
        source_files=plan['source_files'], source_hashes=plan['source_hashes'],
        config=plan['config'], seed=plan['seed'], split_hash=plan['split_hash'],
        checkpoints=checkpoints, reference=plan['reference'], epochs=plan['epochs'],
        mismatch_policy=MISMATCH_POLICY)
    output.mkdir(parents=True, exist_ok=True)
    progress = output/'progress.json'
    if progress.is_file():
        previous = _json(progress)
        if previous.get('identity') != identity:
            old = dict(previous.get('identity', {})); old.pop('checkpoints', None)
            current = dict(identity); current.pop('checkpoints')
            if old != current:
                raise ValueError('Saved diagnostic progress belongs to another experiment')
            # A failed training arm may have completed after a legitimate resume.
            # Preserve the old analysis and evaluate a fresh consistent snapshot.
            run = output/new_name('analysis')
            run.mkdir()
            save_json(progress, dict(identity=identity, analysis_directory=run.name))
        else:
            run = output/previous['analysis_directory']
            if run.is_symlink() or not run.resolve().is_relative_to(output) or not run.is_dir():
                raise ValueError('Saved analysis directory is invalid')
    else:
        run = output/new_name('analysis')
        run.mkdir()
        save_json(progress, dict(identity=identity, analysis_directory=run.name))
    probe, selection, _, split, manifest = clean_datasets(plan['config'], plan['seed'], plan['split_hash'], 20)
    _allowed_data(probe, selection, split, plan['split_hash'], manifest)
    datasets, names = dict(train_probe=probe, selection=selection), class_labels(probe)
    save_json(run/'evaluation_manifest.json', manifest)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    references, records, best_results, failures, rows = {split: {} for split in SPLITS}, {}, {}, {}, []
    for job in JOBS:
        if summaries[job].get('status') == 'complete':
            for checkpoint in ('best.pt', 'best_trained.pt', 'last.pt'):
                if str(Path(job)/checkpoint) not in checkpoints:
                    failures[job+'/'+checkpoint] = dict(error_type='MissingCheckpoint',
                        message='A completed arm must retain best, best-trained and actual last checkpoints')
    reference_hashes = dict(source_cnn=plan['source_provenance']['parent_checkpoint_sha256'],
        source_vit=sources['selected']['checkpoint_sha256'],
        existing_joint=plan['reference']['checkpoint_sha256'])
    for job, model, state in _source_models(plan, sources):
        try:
            check_deadline(deadline_unix)
            path = run/job/'source_best'
            item_identity = dict(identity=identity, job=job, checkpoint='source_best',
                checkpoint_sha256=reference_hashes[job])
            cached = _load_cache(path, item_identity)
            if cached is None:
                _prepare_cache(path)
                model = model.to(device).float().eval()
                if job == 'existing_joint':
                    record, values = _concat_record(model, datasets, plan, path, names, device,
                        item_identity, state, references, deadline_unix)
                else:
                    record, values = _independent_record(model, datasets, plan, path, names, device,
                        item_identity, state, deadline_unix)
                record['files'] = _cache_files(path)
                save_json(path/'record.json', record)
            else:
                record, values = cached
            records[job+'/source_best'] = record
            for label, result in values.items():
                references[label][job] = result
            best_results[job] = values['selection']
        except Exception as exc:
            failures[job+'/source_best'] = dict(error_type=type(exc).__name__, message=str(exc))
        finally:
            del model
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    # Derived probability grids require no extra model training or encoder pass.
    mixtures = []
    if all({'source_cnn', 'source_vit'} <= set(references[s]) for s in SPLITS):
        for split in SPLITS:
            values, probabilities = fixed_mixtures(references[split]['source_cnn'], references[split]['source_vit'])
            for row in values:
                paired = (paired_result(probabilities[row['vit_probability_weight']], references[split]['existing_joint'])
                    if 'existing_joint' in references[split] else None)
                mixtures.append(dict(split=split, selection_grid_is_exploratory=True, **row,
                    corrected_vs_existing_joint=None if paired is None else paired['corrected'],
                    harmed_vs_existing_joint=None if paired is None else paired['harmed'],
                    delta_vs_existing_joint_pp=None if paired is None else paired['delta_accuracy_pp'],
                    mean_nll_improvement_vs_existing_joint=None if paired is None else paired['mean_nll_improvement']))
    _write_csv(run/'probability_fusion.csv', mixtures)
    for job, checkpoint in work:
        key = job+'/'+checkpoint
        model = None
        try:
            check_deadline(deadline_unix)
            if any(len(references[split]) != 3 for split in SPLITS):
                raise ValueError('All three aligned source references are required before control analysis')
            physical = 'last.pt' if checkpoint == 'last_ema' else checkpoint
            item_identity = dict(identity=identity, job=job, checkpoint=checkpoint,
                checkpoint_sha256=checkpoints[str(Path(job)/physical)])
            path = run/job/checkpoint.replace('.pt', '')
            cached = _load_cache(path, item_identity)
            if cached is None:
                _prepare_cache(path)
                model, state = load_model(plan, training, job, checkpoint=checkpoint, sources=sources)
                model = model.to(device).float().eval()
                record, values = _concat_record(model, datasets, plan, path, names, device,
                    item_identity, state, references, deadline_unix)
                record['files'] = _cache_files(path)
                save_json(path/'record.json', record)
            else:
                record, values = cached
            record['training_summary'] = summaries[job]
            record['paired_reference'] = {split: {reference: paired_result(values[split], result)
                for reference, result in references[split].items()} for split in SPLITS}
            records[key] = record
            if checkpoint == 'best.pt':
                best_results[job] = values['selection']
        except Exception as exc:
            failures[key] = dict(error_type=type(exc).__name__, message=str(exc))
        finally:
            if model is not None:
                del model
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    for key, record in records.items():
        job, checkpoint = key.split('/')
        rows.extend(dict(job=job, checkpoint=checkpoint, epoch=record['epoch'], source=record['source'],
            params=record['params'], **row) for row in record['rows'])
    interpretation = interpret_comparisons(best_results, summaries, plan['epochs'])
    expected = len(work)+3
    complete = not failures and len(records) == expected and interpretation['all_training_horizons_complete']
    for relative, digest in checkpoints.items():
        if file_hash(training/relative) != digest:
            raise RuntimeError('Training artifacts changed during read-only analysis')
    for path, digest in sources['source_files'].items():
        if file_hash(path) != digest:
            raise RuntimeError('Immutable source artifacts changed during read-only analysis')
    result = dict(protocol=PROTOCOL, status='complete' if complete else 'partial', identity=identity,
        analysis_directory=run.name, jobs=records, failures=failures, comparisons=interpretation,
        probability_mixtures=mixtures, elapsed_seconds=time.time()-started,
        audit_partition_evaluated=False, official_validation_evaluated=False,
        galleries=dict(status='not_performed', reason='Paired decisions and branch contribution statistics prioritized.'),
        interpretation='All peaks and mixture-grid comparisons are exploratory development results; '
            'no new architecture or universal CNN performance ceiling is established.')
    _write_csv(run/'comparison.csv', rows)
    contribution_rows = [dict(job=key.split('/')[0], checkpoint=key.split('/')[1], **row)
        for key, record in records.items() for row in record.get('contributions', [])]
    _write_csv(run/'contribution_summary.csv', contribution_rows)
    save_json(run/'summary.json', result)
    save_json(output/'completion.json', result)
    lines = ['# Winning-family fusion mechanism controls', '',
        f'Status: **{result["status"]}**. Development selection only; no audit/official validation.', '',
        '|Model|Checkpoint|Top1|NLL|Probe gap pp|Epoch|', '|---|---|---:|---:|---:|---:|']
    for key, record in records.items():
        lines.append(f'|{key.split("/")[0]}|{key.split("/")[1]}|'
            f'{100*record["metrics"]["accuracy"]:.2f}%|{record["metrics"]["nll"]:.5f}|'
            f'{record["train_probe_gap_pp"]:.2f}|{record["epoch"]}|')
    lines.extend(['', interpretation['interpretation'], '', interpretation['scope'], '',
        'Existing_joint is the complete historical winning checkpoint, not a retrained reference. '
        'Random ViT changes independent ViT initialization only; resetting the CNN head tests the inherited '
        'decision boundary under this horizon; patch mean changes the readout while preserving the linear head. '
        'Best may be epoch0; best_trained and last raw/EMA are reported separately.', '',
        INTERVENTION_NOTE, '',
        'Branch margin contributions are measured against the same competing class chosen by the full model. '
        'CNN+bias and ViT margins sum to the full margin. The training-mean replacement uses only the clean '
        'training probe. Global mismatch uses a label-blind derangement, never batch rotation.', '',
        'The probability-mixture table is a fixed, predefined grid computed from the two independent models. '
        'Selecting its best development weight is exploratory and does not constitute an independently '
        'validated ensemble or an arbitrary-fusion upper bound.', '', '|Arm|Pure training minutes|Completed epochs|',
        '|---|---:|---:|'])
    for job, summary in summaries.items():
        seconds = summary.get('total_train_seconds')
        lines.append(f'|{job}|{seconds/60:.2f}|' + str(summary.get('completed_epochs')) + '|'
            if isinstance(seconds, (int, float)) else f'|{job}|unavailable|{summary.get("completed_epochs")}|')
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
