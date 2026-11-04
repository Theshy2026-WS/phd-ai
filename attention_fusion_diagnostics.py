"""Inference interventions for learned CNN/attention feature fusion.

Uniform/removal interventions use the SAME co-adapted trained encoder. They
complement separately trained convolution controls; attention pictures alone
are not causal evidence. No optimization or final-test evaluation occurs.
"""
import copy
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from baseline_diagnostics import clean_datasets, class_labels
from correction_experiment import file_hash
from datasets.utility_tiny import loader
from literature_experiment import ROOT, new_name, recipe_hash, save_json
from mechanism_diagnostics import load_spatial_parent
from models.attention_correction import state_digest
from utils.baseline_diagnostics import _font, _source_image, _spread


PROTOCOL = 'tiny-attention-feature-interventions-v1'
ATTENTION_VARIANTS = {'self8', 'self16', 'cross16', 'cross32', 'bidir16', 'gated_cross16'}


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Attention analysis reached its explicit time budget')


def _write_csv(path, rows, fields=None):
    fields = list(rows[0]) if rows and fields is None else (fields or [])
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _number(value):
    if value is None:
        return None
    if isinstance(value, dict):
        values = [_number(x) for x in value.values()]
        values = [x for x in values if x is not None]
        return sum(values)/len(values) if values else None
    if isinstance(value, torch.Tensor):
        return float(value.detach().double().mean())
    if isinstance(value, (list, tuple)):
        return _number(dict(enumerate(value)))
    return float(value)


@torch.inference_mode()
def evaluate_intervention(model, dataset, config, *, mode, device, output, class_names, deadline=None):
    """Record scalar feature-use statistics and compact per-image decisions."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = state_digest(model)
    model.eval()
    predictions, targets, ids, losses = [], [], [], []
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    top5 = correct = count = 0
    square_before = square_change = 0.
    elements = 0
    scalars = {key: [0., 0] for key in ('gamma', 'gate', 'attention_entropy', 'attention_peak')}
    with model.analysis(mode):
        with (output/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_id', 'target', 'prediction', 'correct', 'true_rank', 'nll'])
            for images, labels, identifiers in loader(dataset, config['dataset'], config['training'].get('seed', 42)):
                _deadline(deadline)
                images, truth = images.to(device).float(), labels.to(device)
                with torch.autocast(device, enabled=False):
                    values = model.forward_decomposed(images)
                logits = values['logits'].float()
                if logits.shape != (len(labels), len(class_names)) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid attention diagnostic logits')
                order = logits.argsort(dim=1, descending=True, stable=True)
                ranks = (order == truth[:, None]).long().argmax(1)+1
                loss = -logits.log_softmax(1).gather(1, truth[:, None]).squeeze(1)
                prediction = order[:, 0]
                pred, y = prediction.cpu().numpy(), labels.numpy()
                correct += int(prediction.eq(truth).sum())
                top5 += int((ranks <= min(5, len(class_names))).sum())
                count += len(labels)
                np.add.at(confusion, (y, pred), 1)
                predictions.extend(pred.tolist()); targets.extend(y.tolist()); ids.extend(identifiers)
                losses.extend(loss.cpu().double().tolist())
                writer.writerows(zip(identifiers, y.tolist(), pred.tolist(), prediction.eq(truth).cpu().tolist(),
                                      ranks.cpu().tolist(), loss.cpu().tolist()))
                original, fused = values['features_before'], values['features_after']
                if original.shape != fused.shape or not torch.isfinite(fused).all():
                    raise ValueError('Feature intervention changed layout or produced non-finite features')
                square_before += float(original.detach().float().square().sum())
                square_change += float((fused.detach().float()-original.detach().float()).square().sum())
                elements += original.numel()
                for key, accumulator in scalars.items():
                    value = _number(values.get(key))
                    if value is not None:
                        if not np.isfinite(value):
                            raise ValueError(f'Non-finite diagnostic statistic: {key}')
                        accumulator[0] += value*len(labels); accumulator[1] += len(labels)
    if count != len(dataset) or len(set(ids)) != count or not count:
        raise ValueError('Diagnostic sample count/ID mismatch')
    if state_digest(model) != before:
        raise RuntimeError('Inference intervention changed model parameters or buffers')
    metrics = dict(n=count, mode=mode, accuracy=correct/count, top5_accuracy=top5/count,
        nll=float(np.mean(losses)), feature_change_rms=(square_change/elements)**.5,
        feature_before_rms=(square_before/elements)**.5,
        relative_feature_change=(square_change/max(square_before, 1e-30))**.5,
        **{key: value[0]/value[1] if value[1] else None for key, value in scalars.items()})
    save_json(output/'metrics.json', metrics)
    counts = confusion.sum(1)
    _write_csv(output/'per_class.csv', [dict(label=i, class_name=name, n=int(counts[i]),
        correct=int(confusion[i, i]), recall=float(confusion[i, i]/counts[i]) if counts[i] else None)
        for i, name in enumerate(class_names)])
    with (output/'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['true_class', *class_names])
        writer.writerows([[name, *row.tolist()] for name, row in zip(class_names, confusion)])
    np.savez_compressed(output/'decisions.npz', image_ids=np.asarray(ids), targets=np.asarray(targets),
        predictions=np.asarray(predictions), nll=np.asarray(losses))
    return dict(metrics=metrics, image_ids=ids, targets=np.asarray(targets),
                predictions=np.asarray(predictions), nll=np.asarray(losses))


def compare_decisions(learned, ablated):
    if learned['image_ids'] != ablated['image_ids'] or not np.array_equal(learned['targets'], ablated['targets']):
        raise ValueError('Interventions must evaluate identical ordered IDs and targets')
    a = learned['predictions'] == learned['targets']
    b = ablated['predictions'] == ablated['targets']
    return dict(n=len(a), corrected_by_learned=int((a & ~b).sum()), harmed_by_learned=int((~a & b).sum()),
        delta_learned_top1_pp=100*float((a.astype(float)-b.astype(float)).mean()),
        learned_nll_improvement=float((ablated['nll']-learned['nll']).mean()),
        prediction_changes=int((learned['predictions'] != ablated['predictions']).sum()))


@torch.inference_mode()
def _galleries(model, dataset, learned, ablated, output, names, device, examples):
    if examples <= 0:
        return
    output.mkdir(parents=True, exist_ok=True)
    truth, identifiers = learned['targets'], learned['image_ids']
    a, b = learned['predictions'] == truth, ablated['predictions'] == truth
    font = _font(13)
    old_capture = getattr(model, 'capture_attention', False)
    try:
        model.capture_attention = True
        for label, indices in (('corrected', np.flatnonzero(a & ~b)), ('harmed', np.flatnonzero(~a & b))):
            chosen = _spread(indices.tolist(), examples, identifiers)
            canvas = Image.new('RGB', (1200, 60+max(1, (len(chosen)+3)//4)*245), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 10), f'{label}: learned versus ablation; retrieval weights are descriptive, not causal explanations',
                      fill='black', font=font)
            for number, index in enumerate(chosen):
                source = _source_image(dataset, index, identifiers[index]).resize((112, 112), Image.Resampling.NEAREST)
                x, y = number % 4*300, 48+number//4*245
                canvas.paste(source, (x+8, y))
                tensor, _, replay_id = dataset[index]
                if replay_id != identifiers[index]:
                    raise ValueError('Gallery replay changed an image ID')
                with model.analysis('learned'), torch.autocast(device, enabled=False):
                    detail = model.forward_decomposed(tensor[None].to(device).float())
                maps = detail.get('retrieval_maps') or {}
                if maps:
                    values = next(iter(maps.values()))[0].detach().float().cpu().numpy()
                    values = np.asarray(values).squeeze()
                    if values.ndim == 2:
                        values = (values-values.min())/max(float(np.ptp(values)), 1e-12)
                        heat = np.stack([255*values, 70*(1-values), 255*(1-values)], axis=-1).astype(np.uint8)
                        canvas.paste(Image.fromarray(heat).resize((112, 112), Image.Resampling.NEAREST), (x+158, y))
                lines = [identifiers[index], f'True: {names[int(truth[index])]}',
                         f'Learned: {names[int(learned["predictions"][index])]}',
                         f'Ablated: {names[int(ablated["predictions"][index])]}']
                for offset, line in enumerate(lines):
                    while draw.textbbox((0, 0), line, font=font)[2] > 286:
                        line = line[:-4]+'...'
                    draw.text((x+7, y+120+offset*22), line, fill='black', font=font)
            canvas.save(output/f'{label}.png')
    finally:
        model.capture_attention = old_capture


def run_attention_diagnostics(parent_run, training_dir, output_dir, config=None, examples=8, deadline=None):
    from models.attention_fusion import build_attention_model
    parent, parent_summary, parent_state, parent_hash = load_spatial_parent(parent_run)
    training, output = Path(training_dir).resolve(), Path(output_dir).resolve()
    if output == parent or parent in output.parents or output == training or training in output.parents:
        raise ValueError('Attention analysis must be outside original checkpoint/training directories')
    plan = json.loads((training/'suite.json').read_text(encoding='utf-8'))
    if plan['parent_checkpoint_sha256'] != parent_hash or plan['split_hash'] != parent_summary['split_hash']:
        raise ValueError('Training suite has a different parent or split')
    requested = copy.deepcopy(parent_state['config'] if config is None else config)
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
        source_hashes={name: file_hash(ROOT/name) for name in ('attention_fusion_diagnostics.py',
            'attention_fusion_experiment.py', 'models/attention_fusion.py')})
    if (output/'summary.json').is_file():
        saved = json.loads((output/'summary.json').read_text(encoding='utf-8'))
        if saved.get('identity') == identity and saved.get('status') == 'complete':
            return saved
        old_identity = saved.get('identity', {})
        if any(old_identity.get(key) != value for key, value in identity.items() if key != 'checkpoints'):
            raise ValueError('Saved attention analysis differs from this source/config/parent request')
        # A resumed training study may add checkpoints or advance a previously
        # partial model. Preserve the old attempt and analyze the new hashes.
    output.mkdir(parents=True, exist_ok=True)
    if not checkpoints:
        (output/'report.md').write_text('# Attention analysis unavailable\n\n'
            'No successful checkpointed training runs are available. See training status and failures.\n', encoding='utf-8')
        result = dict(status='unavailable', identity=identity, checkpoints_analyzed=0)
        save_json(output/'summary.json', result)
        return result
    attempt = output/new_name('attempt'); attempt.mkdir()
    save_json(attempt/'evaluation_manifest.json', manifest)
    rows, comparisons, failures = [], {}, {}
    for job in checkpoints:
        _deadline(deadline)
        run = training/job
        summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
        variant = summary.get('model_variant', summary['variant'])
        scope = summary.get('scope', job.split('_', 1)[0])
        checkpoint_specs = [('best.pt', 'best', summary['best_epoch']),
                            ('last.pt', 'last_raw', summary['completed_epochs'])]
        if summary['best_epoch'] == 0 and summary.get('best_trained_checkpoint'):
            checkpoint_specs.append((summary['best_trained_checkpoint'], 'best_trained',
                                     summary['best_trained_selection']['epoch']))
        for checkpoint, label, expected_epoch in checkpoint_specs:
            try:
                _deadline(deadline)
                state = torch.load(run/checkpoint, map_location='cpu', weights_only=False)
                for key in ('job', 'variant', 'seed', 'parent_checkpoint_sha256', 'source_hashes', 'recipe_hash'):
                    if state['metadata'].get(key) != summary.get(key):
                        raise ValueError(f'Checkpoint/summary mismatch: {key}')
                if state['metadata'].get('recipe_hash') != recipe_hash(state['config']):
                    raise ValueError('Diagnostic checkpoint has an inconsistent training recipe')
                if state['epoch'] != expected_epoch:
                    raise ValueError('Checkpoint chronology differs from its report')
                model = build_attention_model(variant,
                    parent_state['model'] if scope == 'warm' else None, parent_summary['seed'],
                    width=plan.get('width', 128), heads=plan.get('heads', 4)).to(device).float().eval()
                model.load_state_dict(state['model'], strict=True)
                del state
                modes = (('learned', 'uniform', 'removed') if variant in ATTENTION_VARIANTS else
                         ('learned',) if variant == 'control' else ('learned', 'removed'))
                for split, dataset in (('train_probe', probe), ('selection', selection)):
                    results = {}
                    for mode in modes:
                        print(f'Attention intervention: {job}/{label}/{split}/{mode}', flush=True)
                        directory = attempt/job/label/split/mode
                        directory.parent.mkdir(parents=True, exist_ok=True)
                        result = evaluate_intervention(model, dataset, requested, mode=mode,
                            device=device, output=directory, class_names=names, deadline=deadline)
                        results[mode] = result
                        rows.append(dict(job=job, scope=scope, variant=variant, checkpoint=label,
                            checkpoint_epoch=expected_epoch, split=split, **result['metrics']))
                    for mode in modes[1:]:
                        key = f'{job}/{label}/{split}/learned_vs_{mode}'
                        comparisons[key] = compare_decisions(results['learned'], results[mode])
                        if split == 'selection':
                            _galleries(model, dataset, results['learned'], results[mode],
                                attempt/job/label/f'galleries_vs_{mode}', names, device, examples)
                del model
            except TimeoutError:
                raise
            except Exception as exc:
                failures[f'{job}/{label}'] = dict(error_type=type(exc).__name__, message=str(exc))
                print(f'Attention diagnostic failed for {job}/{label} ({type(exc).__name__}).', flush=True)
    _write_csv(attempt/'comparison.csv', rows)
    save_json(attempt/'paired_interventions.json', comparisons)
    save_json(attempt/'failures.json', failures)
    lines = ['# Attention feature-fusion interventions', '',
        'Best and actual last checkpoints are evaluated separately. Uniform attention and branch removal '
        'use the same co-adapted encoder; compare separately trained convolution controls for primary method claims.', '',
        '| Job | State/epoch | Readout mode | Selection Top1 | NLL | Relative feature change |',
        '|---|---|---|---:|---:|---:|']
    for row in rows:
        if row['split'] == 'selection':
            lines.append(f'| {row["job"]} | {row["checkpoint"]}@{row["checkpoint_epoch"]} | {row["mode"]} | '
                f'{100*row["accuracy"]:.3f}% | {row["nll"]:.6f} | {row["relative_feature_change"]:.5f} |')
    lines += ['', '## Interpretation', '',
        '- Learned versus uniform tests content-dependent routing given the same learned projections and encoder. '
        'It is not the outcome of training a separate uniform model.',
        '- Learned versus removed measures the immediate contribution of the feature branch, including co-adaptation.',
        '- Gamma, feature-change magnitude, gate values and entropy help detect collapsed or weak branches. '
        'Nonzero parameters alone do not prove useful input-dependent behavior.',
        '- Retrieval maps are head/query-averaged descriptive weights; they do not independently establish explanations.',
        '- Late-stage maps average all64 queries although phase00 directly reads only the even/even grid. '
        'They are not class-specific explanations of just the directly read queries.',
        '- Feature RMS at stage2 and stage4 describes different representations. Compare interventions within '
        'the same model rather than ranking methods by absolute RMS across insertion layers.',
        '- A best epoch of zero has not learned during this run; random insertion effects must not be called training gains.',
        '- Corrected and harmed images are both reported. Audit and official validation remain unused.', '']
    (attempt/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output/'report.md').write_text(f'# Attention analysis\n\n[Authoritative report]({attempt.name}/report.md)\n', encoding='utf-8')
    for job, hashes in checkpoints.items():
        if any(file_hash(training/job/name) != digest for name, digest in hashes.items()):
            raise RuntimeError('A trained checkpoint changed during its analysis')
    if file_hash(parent/'best.pt') != parent_hash:
        raise RuntimeError('Original parent checkpoint changed')
    complete = dict(status='failed' if failures else 'complete', identity=identity,
        analysis_directory=attempt.name, checkpoints_analyzed=len(checkpoints), comparisons=comparisons,
        failures=failures, audit_partition_evaluated=False, official_validation_evaluated=False)
    save_json(output/'summary.json', complete)
    if failures:
        raise RuntimeError('Some attention interventions failed; partial analysis retained')
    return complete
