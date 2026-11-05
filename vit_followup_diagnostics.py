"""Read-only branch and Transformer interventions for the ViT follow-up study.

Independent-image fusion is evaluated with either feature source masked.
Attention variants additionally receive uniform-routing and branch-removal
interventions; full Transformer fusion also receives a no-MLP intervention.
These are the SAME co-adapted weights, not separately trained model scores.
"""
import copy
import json
from pathlib import Path

import torch

from attention_fusion_diagnostics import (_write_csv, evaluate_intervention,
    compare_decisions, _galleries)
from baseline_diagnostics import clean_datasets, class_labels
from correction_experiment import file_hash
from literature_experiment import ROOT, new_name, recipe_hash, save_json
from mechanism_diagnostics import load_spatial_parent, _relocation_only
from models.vit_followup import build_followup_model


PROTOCOL = 'tiny-vit-followup-interventions-v1'
MODES = dict(vit=('learned',), dual_fusion=('learned', 'cnn_only', 'vit_only'),
    shallow64=('learned', 'uniform', 'removed'),
    cross16_mlp=('learned', 'uniform', 'removed', 'no_mlp'))


def run_followup_diagnostics(parent_run, training_dir, output_dir, config=None, examples=8):
    from vit_followup_experiment import source_hashes, load_vit_dependency
    parent, parent_summary, parent_state, parent_hash = load_spatial_parent(parent_run)
    training, output = Path(training_dir).resolve(), Path(output_dir).resolve()
    if output == parent or parent in output.parents or output == training or training in output.parents:
        raise ValueError('Follow-up analysis must be outside original and training directories')
    plan = json.loads((training/'suite.json').read_text(encoding='utf-8'))
    if plan['parent_checkpoint_sha256'] != parent_hash or plan['split_hash'] != parent_summary['split_hash']:
        raise ValueError('Follow-up training has a different parent or split')
    if plan['source_hashes'] != source_hashes():
        raise ValueError('Follow-up diagnostic source differs from the trained source')
    requested = copy.deepcopy(parent_state['config'] if config is None else config)
    _relocation_only(parent_state['config'], requested)
    probe, selection, _, fingerprint, manifest = clean_datasets(
        requested, parent_summary['seed'], parent_summary['split_hash'], 20)
    names, device = class_labels(probe), 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoints = {}
    for job in plan['jobs']:
        if (training/job/'summary.json').is_file():
            item = json.loads((training/job/'summary.json').read_text(encoding='utf-8'))
            files = ['best.pt', 'last.pt']
            if item['best_epoch'] == 0 and item.get('best_trained_checkpoint'):
                files.append(item['best_trained_checkpoint'])
            checkpoints[job] = {name: file_hash(training/job/name) for name in files}
    identity = dict(protocol=PROTOCOL, parent_checkpoint_sha256=parent_hash, split_hash=fingerprint,
        config=requested, examples=examples, checkpoints=checkpoints,
        source_hashes=source_hashes(), mode_policy={key: list(value) for key, value in MODES.items()})
    if (output/'summary.json').is_file():
        saved = json.loads((output/'summary.json').read_text(encoding='utf-8'))
        if saved.get('identity') == identity and saved.get('status') == 'complete':
            return saved
        old = saved.get('identity', {})
        if any(old.get(key) != value for key, value in identity.items() if key != 'checkpoints'):
            raise ValueError('Saved follow-up analysis differs from this source/config/parent request')
    output.mkdir(parents=True, exist_ok=True)
    if not checkpoints:
        result = dict(status='unavailable', identity=identity, checkpoints_analyzed=0)
        save_json(output/'summary.json', result)
        (output/'report.md').write_text('# Follow-up analysis unavailable\n\n'
            'No successful checkpointed training runs are available. See the training failures.\n', encoding='utf-8')
        return result
    attempt = output/new_name('attempt')
    attempt.mkdir()
    save_json(attempt/'evaluation_manifest.json', manifest)
    rows, comparisons, failures = [], {}, {}
    vit_state, vit_hash = None, None
    for job in checkpoints:
        run = training/job
        summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
        variant, scope = summary['variant'], summary['scope']
        specs = [('best.pt', 'best', summary['best_epoch']),
                 ('last.pt', 'last_raw', summary['completed_epochs'])]
        if summary['best_epoch'] == 0 and summary.get('best_trained_checkpoint'):
            specs.append((summary['best_trained_checkpoint'], 'best_trained',
                          summary['best_trained_selection']['epoch']))
        for checkpoint, label, expected_epoch in specs:
            try:
                state = torch.load(run/checkpoint, map_location='cpu', weights_only=False)
                for key in ('job', 'variant', 'seed', 'parent_checkpoint_sha256', 'source_hashes',
                            'recipe_hash', 'vit_checkpoint_sha256'):
                    if state['metadata'].get(key) != summary.get(key):
                        raise ValueError(f'Follow-up checkpoint/summary mismatch: {key}')
                if state['metadata']['source_hashes'] != plan['source_hashes']:
                    raise ValueError('Follow-up checkpoint source differs from its suite')
                if state['metadata']['recipe_hash'] != recipe_hash(state['config']):
                    raise ValueError('Follow-up checkpoint has an inconsistent recipe')
                if state['epoch'] != expected_epoch:
                    raise ValueError('Follow-up checkpoint chronology differs')
                if variant == 'dual_fusion':
                    if vit_state is None:
                        vit_state, vit_hash = load_vit_dependency(plan, training)
                    if state['metadata']['vit_checkpoint_sha256'] != vit_hash:
                        raise ValueError('Dual fusion was initialized from another ViT')
                model = build_followup_model(variant,
                    parent_state=parent_state['model'] if scope == 'joint' else None,
                    vit_state=vit_state if variant == 'dual_fusion' else None,
                    seed=parent_summary['seed']).to(device).float().eval()
                model.load_state_dict(state['model'], strict=True)
                del state
                for split, dataset in (('train_probe', probe), ('selection', selection)):
                    results = {}
                    for mode in MODES[variant]:
                        print(f'Follow-up intervention: {job}/{label}/{split}/{mode}', flush=True)
                        directory = attempt/job/label/split/mode
                        directory.parent.mkdir(parents=True, exist_ok=True)
                        result = evaluate_intervention(model, dataset, requested, mode=mode,
                            device=device, output=directory, class_names=names)
                        results[mode] = result
                        rows.append(dict(job=job, scope=scope, variant=variant, checkpoint=label,
                            checkpoint_epoch=expected_epoch, split=split, **result['metrics']))
                    for mode in MODES[variant][1:]:
                        comparisons[f'{job}/{label}/{split}/learned_vs_{mode}'] = compare_decisions(
                            results['learned'], results[mode])
                        if split == 'selection':
                            _galleries(model, dataset, results['learned'], results[mode],
                                attempt/job/label/f'galleries_vs_{mode}', names, device, examples)
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as exc:
                failures[f'{job}/{label}'] = dict(error_type=type(exc).__name__, message=str(exc))
                print(f'Follow-up diagnostic failed for {job}/{label} ({type(exc).__name__}).', flush=True)
    _write_csv(attempt/'comparison.csv', rows)
    save_json(attempt/'paired_interventions.json', comparisons)
    save_json(attempt/'failures.json', failures)
    lines = ['# ViT/CNN feature-source and Transformer interventions', '',
        'All interventions keep the SAME co-adapted trained weights. Masking one source is not '
        'an independently trained CNN or ViT. Uniform routing and removal are also inference interventions.', '',
        '| Job | State/epoch | Mode | Selection Top1 | NLL | Relative feature change |',
        '|---|---|---|---:|---:|---:|']
    for row in rows:
        if row['split'] == 'selection':
            lines.append(f'|{row["job"]}|{row["checkpoint"]}@{row["checkpoint_epoch"]}|{row["mode"]}|'
                f'{100*row["accuracy"]:.3f}%|{row["nll"]:.6f}|{row["relative_feature_change"]:.5f}|')
    lines += ['', '## Interpretation', '',
        '- Dual fusion consumes independent original-image CNN and full-ViT features. cnn_only masks '
        'the ViT feature source; vit_only masks the CNN feature source at the joint classifier.',
        '- Shallow64 uses4096 true64x64 shallow tokens. Query chunks change memory use, not the global '
        'key set or attention formula. Uniform routing can use the exact mean-V shortcut.',
        '- No-MLP keeps the attention path and removes only Transformer feed-forward processing; '
        'removed bypasses the whole inserted path. Interpret both on their co-adapted encoder.',
        '- Feature statistics use different feature layouts across models. Compare within a model, '
        'not their absolute RMS across unrelated representations.',
        '- Retrieval maps describe attention weights; they do not establish class-specific causal explanations.',
        '- Best, last and best-actually-trained checkpoints are distinguished. Audit and official validation remain unused.', '']
    (attempt/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'report.md').write_text(f'# Follow-up analysis\n\n[Authoritative report]({attempt.name}/report.md)\n',
                                  encoding='utf-8')
    for job, hashes in checkpoints.items():
        if any(file_hash(training/job/name) != digest for name, digest in hashes.items()):
            raise RuntimeError('A follow-up trained checkpoint changed during analysis')
    if file_hash(parent/'best.pt') != parent_hash:
        raise RuntimeError('Original CNN parent checkpoint changed')
    result = dict(status='failed' if failures else 'complete', identity=identity,
        analysis_directory=attempt.name, checkpoints_analyzed=len(checkpoints), failures=failures,
        comparisons=comparisons, audit_partition_evaluated=False, official_validation_evaluated=False)
    save_json(output/'summary.json', result)
    if failures:
        raise RuntimeError('Some follow-up interventions failed; partial analysis retained')
    return result
