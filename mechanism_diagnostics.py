"""Fixed-checkpoint spatial readout and train-only BatchNorm diagnostics.

No optimizer, label fitting, audit evaluation, email or instance control occurs
here. The original checkpoint remains immutable. All five readouts share each
encoder forward; the recalibrated copy uses clean training images exclusively.
"""
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_diagnostics import clean_datasets, class_labels
from correction_experiment import file_hash
from literature_experiment import ROOT, new_name, recipe_hash, save_json
from models.attention_correction import state_digest
from models.spatial_preact import build_spatial_model
from utils.baseline_diagnostics import _image_id, _font, _source_image, _spread


PROTOCOL = 'tiny-fixed-checkpoint-mechanisms-v1'
SPATIAL_PROTOCOL = 'tiny-spatial-cnn-v1-80-10-10'
READOUTS = ('phase00', 'phase01', 'phase10', 'phase11', 'full8')
TRAIN_IMAGES, SELECTION_IMAGES = 80000, 10000
MODEL_SOURCES = ('models/spatial_preact.py', 'models/literature_twostage.py',
                 'models/literature_alternet.py')
DIAGNOSTIC_SOURCES = ('mechanism_diagnostics.py', *MODEL_SOURCES,
                      'baseline_diagnostics.py', 'utils/baseline_diagnostics.py',
                      'datasets/utility_tiny.py', 'datasets/tiny_imagenet.py')


def diagnostic_sources():
    return {name: file_hash(ROOT/name) for name in DIAGNOSTIC_SOURCES}


def load_spatial_parent(path):
    """Load only a trusted completed 100-epoch cnn8 development checkpoint."""
    run = Path(path).expanduser().resolve()
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if (summary.get('protocol') != SPATIAL_PROTOCOL or summary.get('variant') != 'cnn8'
            or summary.get('completed_epochs') != 100):
        raise ValueError('Need the completed 100-epoch spatial cnn8 run, not cnn4 or a smoke run')
    checkpoint = run/'best.pt'
    checkpoint_hash = file_hash(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    metadata = state['metadata']
    for key in ('protocol', 'variant', 'seed', 'split_hash', 'source_hashes',
                'recipe_hash', 'initial_state_sha256', 'environment', 'params'):
        if metadata[key] != summary[key]:
            raise ValueError(f'Spatial summary/checkpoint mismatch: {key}')
    if (state['epoch'] != summary['best_epoch'] or
            not 1 <= state['epoch'] <= summary['completed_epochs'] == state['config']['training']['epochs']
            or state['selection'] != summary['best_selection']):
        raise ValueError('Spatial checkpoint chronology/selection does not match the completed run')
    if recipe_hash(state['config']) != summary['recipe_hash']:
        raise ValueError('Spatial parent recipe hash mismatch')
    # New diagnostic/training files need not match an older run. Only the code
    # defining its actual architecture must still match the recorded hashes.
    for name in MODEL_SOURCES:
        if summary['source_hashes'].get(name) != file_hash(ROOT/name):
            raise ValueError(f'Parent architecture implementation changed: {name}')
    if any(t.is_floating_point() and not torch.isfinite(t).all()
           for t in state['model'].values()):
        raise ValueError('Non-finite spatial parent state')
    return run, summary, state, checkpoint_hash


def _relocation_only(original, requested):
    left, right = copy.deepcopy(original), copy.deepcopy(requested)
    for config in (left, right):
        for key in ('data_dir', 'num_workers', 'batch_size', 'download'):
            config['dataset'].pop(key, None)
    if left != right:
        raise ValueError('Fixed-checkpoint diagnosis permits only dataset path/loader relocation')


def readout_logits(model, features):
    """Use the unchanged head at four phases and at all spatial locations."""
    if features.ndim != 4 or features.shape[-2:] != (8, 8):
        raise ValueError('Mechanism diagnosis requires the cnn8 final 8x8 feature map')
    if model.training or any(module.training for module in model.head.modules()):
        raise ValueError('Readout identity requires an evaluation-mode classifier')
    # TF32 rounds each GAP vector before the linear layer. Averaging and that
    # input rounding need not commute, so use strict FP32 for the tiny head
    # calls only. Encoder convolution/matmul retains the inherited policy.
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('highest')
        outputs = {f'phase{row}{column}': model.head(features[..., row::2, column::2])
                   for row in range(2) for column in range(2)}
        outputs['full8'] = model.head(features)
        mean_phase = torch.stack([outputs[name] for name in READOUTS[:4]]).mean(0)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    if not torch.allclose(mean_phase, outputs['full8'], atol=1e-4, rtol=1e-4):
        raise RuntimeError('Full8 head no longer equals mean phase logits in eval mode')
    return outputs, float((mean_phase-outputs['full8']).detach().abs().max())


def _non_bn_digest(model):
    allowed = {f'{name}.{suffix}' if name else suffix
               for name, module in model.named_modules()
               if isinstance(module, nn.modules.batchnorm._BatchNorm)
               for suffix in ('running_mean', 'running_var', 'num_batches_tracked')}
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name not in allowed:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest(), allowed


@torch.no_grad()
def recalibrate_bn(model, dataset, *, allowed_ids, batch_size, num_workers, device, seed=42):
    """Return a separate model with clean-train BN statistics; never fit weights.

    Deterministic shuffled, equal-size batches avoid class-sorted calibration
    and unequal final-batch weighting. Every BN uses cumulative batch moments,
    as in standard BN-statistic recalibration, rather than a claim of exact
    full-dataset activation moments. Deeper BNs see batch-normalized upstream
    activations. This is a controlled inference-state intervention.
    """
    if getattr(dataset, 'split', None) != 'train_clean':
        raise ValueError('BN calibration requires a clean training view, without augmentation')
    allowed_ids = list(allowed_ids)
    if len(dataset) == 0 or batch_size < 2 or len(dataset) % batch_size:
        raise ValueError('BN calibration requires nonempty clean data and equal batches of size >=2')
    if len(allowed_ids) != len(dataset) or len(set(allowed_ids)) != len(allowed_ids):
        raise ValueError('Calibration authorization must name every clean training ID exactly once')
    before = state_digest(model)
    calibrated = copy.deepcopy(model).to(device).float().eval()
    immutable_digest, allowed_buffers = _non_bn_digest(calibrated)
    original_buffers = {name: value.detach().cpu().clone()
                        for name, value in calibrated.state_dict().items() if name in allowed_buffers}
    modules = [module for module in calibrated.modules()
               if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not modules or any(not module.track_running_stats for module in modules):
        raise ValueError('Expected BatchNorm modules with running statistics')
    momenta = [module.momentum for module in modules]
    for module in modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    generator = torch.Generator().manual_seed(seed+450001)
    batches = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         generator=generator, num_workers=num_workers,
                         pin_memory=str(device).startswith('cuda'), persistent_workers=False)
    visited, permitted = [], set(allowed_ids)
    try:
        for images, _, identifiers in tqdm(batches, desc='Clean-train BN recalibration', leave=False):
            identifiers = [_image_id(value) for value in identifiers]
            if len(set(identifiers)) != len(identifiers) or not set(identifiers) <= permitted:
                raise ValueError('BN calibration encountered unauthorized or duplicated image IDs')
            visited.extend(identifiers)
            images = images.to(device, dtype=torch.float32)
            if not torch.isfinite(images).all():
                raise ValueError('Non-finite calibration input')
            with torch.autocast(torch.device(device).type, enabled=False):
                logits = calibrated(images)  # Original phase00 path, never select a phase using labels.
            if not torch.isfinite(logits).all():
                raise ValueError('Non-finite logits during BN recalibration')
    finally:
        for module, momentum in zip(modules, momenta):
            module.momentum = momentum
        calibrated.eval()
    if len(visited) != len(allowed_ids) or set(visited) != permitted or len(set(visited)) != len(visited):
        raise ValueError('BN calibration did not visit exactly the authorized training partition')
    if _non_bn_digest(calibrated)[0] != immutable_digest or state_digest(model) != before:
        raise RuntimeError('Calibration changed weights/non-BN buffers or the original model')
    changed = [name for name, old in original_buffers.items()
               if not torch.equal(old, calibrated.state_dict()[name].detach().cpu())]
    if any(value.is_floating_point() and not torch.isfinite(value).all()
           for value in calibrated.state_dict().values()):
        raise ValueError('BN recalibration produced non-finite state')
    return calibrated, dict(training_images=len(visited), batch_size=batch_size,
        batches=len(visited)//batch_size, deterministic_shuffle_seed=seed+450001,
        calibration_ids=visited, labels_used=False, optimizer_used=False,
        only_bn_buffers_changed=True, changed_buffers=changed,
        original_state_sha256=before, recalibrated_state_sha256=state_digest(calibrated),
        method='Reset all tracked BN buffers; cumulative equal-batch moments on clean train only; '
               'non-BN modules in eval; original phase00 forward; FP32; one pass',
        caveat='Batch-averaged moments, not exact population moments; deeper BNs see batch-normalized upstream activations')


def _write_csv(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


@torch.inference_mode()
def evaluate_readouts(model, dataset, *, output_dir, class_names, batch_size,
                      num_workers, device):
    """One encoder pass/batch; export five auditable result sets without huge arrays."""
    if getattr(dataset, 'split', None) == 'train':
        raise ValueError('Readout evaluation requires deterministic clean inputs')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite readout results')
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    n, classes = len(dataset), len(class_names)
    if n < 1 or classes < 2 or batch_size < 1 or num_workers < 0:
        raise ValueError('Invalid evaluation dimensions/loader settings')
    for value in model.state_dict().values():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError('Mechanism evaluation requires FP32 model state')
    records = {name: dict(predictions=np.empty(n, dtype=np.int64),
                         ranks=np.empty(n, dtype=np.int32), losses=np.empty(n, dtype=np.float64),
                         confidence=np.empty(n, dtype=np.float32),
                         confusion=np.zeros((classes, classes), dtype=np.int64),
                         class_nll=np.zeros(classes, dtype=np.float64),
                         class_top5=np.zeros(classes, dtype=np.int64)) for name in READOUTS}
    targets, identifiers, seen = np.empty(n, dtype=np.int64), [], set()
    streams, writers = {}, {}
    for name in READOUTS:
        folder = output/name
        folder.mkdir()
        streams[name] = (folder/'predictions.csv').open('w', newline='', encoding='utf-8')
        writers[name] = csv.writer(streams[name])
        writers[name].writerow(['image_id', 'true_label', 'predicted_label', 'correct',
            'true_rank', 'nll', 'p_true', 'confidence', 'top5_labels'])
    max_identity_error, offset = 0., 0
    try:
        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.device(device).type == 'cuda', persistent_workers=False)
        for images, labels, ids in tqdm(batches, desc='Five fixed readouts', leave=False):
            ids = [_image_id(value) for value in ids]
            if len(ids) != len(set(ids)) or seen.intersection(ids):
                raise ValueError('Duplicate evaluation image IDs')
            seen.update(ids)
            if labels.dtype != torch.int64 or labels.ndim != 1 or (labels < 0).any() or (labels >= classes).any():
                raise ValueError('Invalid evaluation class labels')
            if images.ndim != 4 or not torch.isfinite(images).all() or offset+len(labels) > n:
                raise ValueError('Invalid clean evaluation batch')
            images = images.to(device, dtype=torch.float32)
            with torch.autocast(torch.device(device).type, enabled=False):
                features = model.stages(model.stem(images))
                logits_by_readout, error = readout_logits(model, features)
            max_identity_error = max(max_identity_error, error)
            y = labels.to(device)
            truth = labels.numpy()
            targets[offset:offset+len(truth)] = truth
            identifiers.extend(ids)
            for name, logits in logits_by_readout.items():
                if logits.dtype != torch.float32 or logits.shape != (len(labels), classes) or not torch.isfinite(logits).all():
                    raise ValueError('Readout produced invalid/non-FP32 logits')
                log_probs = logits.log_softmax(-1)
                losses = -log_probs.gather(1, y[:, None]).squeeze(1)
                if not torch.isfinite(log_probs).all() or not torch.isfinite(losses).all():
                    raise ValueError('Readout produced non-finite log probabilities/NLL')
                order = logits.argsort(dim=1, descending=True, stable=True)
                rank = (order == y[:, None]).long().argmax(1)+1
                predictions, ranks = order[:, 0].cpu().numpy(), rank.cpu().numpy()
                loss_values = losses.cpu().double().numpy()
                confidence = log_probs.gather(1, order[:, :1]).exp().squeeze(1).cpu().numpy()
                top = order[:, :min(5, classes)].cpu().numpy()
                item = records[name]
                item['predictions'][offset:offset+len(truth)] = predictions
                item['ranks'][offset:offset+len(truth)] = ranks
                item['losses'][offset:offset+len(truth)] = loss_values
                item['confidence'][offset:offset+len(truth)] = confidence
                np.add.at(item['confusion'], (truth, predictions), 1)
                np.add.at(item['class_nll'], truth, loss_values)
                np.add.at(item['class_top5'], truth, ranks <= min(5, classes))
                writers[name].writerows([identifier, int(target), int(pred), int(target == pred),
                    int(r), f'{loss:.9g}', f'{np.exp(-loss):.9g}', f'{conf:.9g}', json.dumps(top[j].tolist())]
                    for j, (identifier, target, pred, r, loss, conf) in enumerate(
                        zip(ids, truth, predictions, ranks, loss_values, confidence)))
            offset += len(truth)
    finally:
        for stream in streams.values():
            stream.close()
    if offset != n:
        raise ValueError('Evaluation count does not match the dataset')
    for name, item in records.items():
        folder = output/name
        confusion = item['confusion']
        counts, correct = confusion.sum(1), confusion.diagonal()
        present = counts > 0
        item['metrics'] = dict(n=n, accuracy=float(correct.sum()/n),
            top5_accuracy=float(item['class_top5'].sum()/n), nll=float(item['losses'].mean()),
            mean_confidence=float(item['confidence'].mean()),
            macro_accuracy=float((correct[present]/counts[present]).mean()),
            precision='float32; head matmul highest (TF32 disabled); encoder retains inherited policy',
            ranking_tie_rule='lower class index first')
        save_json(folder/'metrics.json', item['metrics'])
        _write_csv(folder/'per_class.csv', ['label', 'class_name', 'n', 'correct', 'recall', 'nll'],
            [[label, label_name, int(counts[label]), int(correct[label]),
              float(correct[label]/counts[label]) if counts[label] else '',
              float(item['class_nll'][label]/counts[label]) if counts[label] else '']
             for label, label_name in enumerate(class_names)])
        _write_csv(folder/'confusion_matrix.csv', ['true_class\\predicted_class', *class_names],
                   [[label_name, *row.tolist()] for label_name, row in zip(class_names, confusion)])
        np.savez_compressed(folder/'predictions.npz', image_ids=np.asarray(identifiers), targets=targets,
            predicted_labels=item['predictions'], true_ranks=item['ranks'], nll=item['losses'],
            confidence=item['confidence'], class_names=np.asarray(class_names))
    phase_predictions = np.stack([records[name]['predictions'] for name in READOUTS[:4]], axis=1)
    phase_correct = phase_predictions == targets[:, None]
    sensitivity = dict(n=n, any_phase_disagreement=int((phase_predictions != phase_predictions[:, :1]).any(1).sum()),
        all_phases_correct=int(phase_correct.all(1).sum()), any_phase_correct=int(phase_correct.any(1).sum()),
        mean_phase_full8_max_logit_error=max_identity_error,
        note='Any-phase-correct is an oracle description, never a deployable accuracy')
    save_json(output/'phase_sensitivity.json', sensitivity)
    _write_csv(output/'phase_predictions.csv', ['image_id', 'target', *READOUTS[:4], 'full8', 'phase_disagreement'],
        [[identifier, int(target), *row.tolist(), int(records['full8']['predictions'][j]),
          int(not (row == row[0]).all())]
         for j, (identifier, target, row) in enumerate(zip(identifiers, targets, phase_predictions))])
    return dict(records=records, targets=targets, image_ids=identifiers, sensitivity=sensitivity)


def _pair(reference, alternative, targets):
    left, right = reference['predictions'] == targets, alternative['predictions'] == targets
    return dict(n=len(targets), corrected=int((~left & right).sum()), harmed=int((left & ~right).sum()),
        prediction_changes=int((reference['predictions'] != alternative['predictions']).sum()),
        delta_accuracy_pp=100*float((right.astype(float)-left.astype(float)).mean()),
        nll_improvement=float((reference['losses']-alternative['losses']).mean()))


def _changed_gallery(dataset, output, original, alternative, identifiers, targets, names, count):
    if not count:
        return
    left, right = original['predictions'] == targets, alternative['predictions'] == targets
    output.mkdir(parents=True, exist_ok=True)
    font = _font(13)
    for name, candidates in (('corrected', np.flatnonzero(~left & right)),
                             ('harmed', np.flatnonzero(left & ~right))):
        chosen = _spread(candidates.tolist(), count, identifiers)
        canvas = Image.new('RGB', (1200, 60+max(1, (len(chosen)+3)//4)*230), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f'{name}: original phase00 vs this full8 readout; sorted-ID systematic sample', fill='black', font=font)
        rows = []
        for position, j in enumerate(chosen):
            image = _source_image(dataset, j, identifiers[j]).resize((128, 128), Image.Resampling.NEAREST)
            x, y = position % 4*300, 50+position//4*230
            canvas.paste(image, (x+80, y))
            old, new, true = int(original['predictions'][j]), int(alternative['predictions'][j]), int(targets[j])
            lines = [identifiers[j], f'True: {names[true]}', f'Before: {names[old]}', f'After: {names[new]}']
            for line_number, line in enumerate(lines):
                while draw.textbbox((0, 0), line, font=font)[2] > 285:
                    line = line[:-4]+'...'
                draw.text((x+6, y+130+line_number*21), line, fill='black', font=font)
            rows.append([identifiers[j], true, old, new])
        canvas.save(output/f'{name}.png')
        _write_csv(output/f'{name}.csv', ['image_id', 'target', 'original_phase00', 'new_full8'], rows)


def _plot_comparison(rows, path):
    canvas = Image.new('RGB', (1100, 660), 'white')
    draw, font = ImageDraw.Draw(canvas), _font(15)
    draw.text((20, 12), 'Fixed checkpoint: clean train (blue) / selection (orange) Top-1', fill='black', font=font)
    for tick in (0, 20, 40, 60, 80, 100):
        x = 300+tick*7
        draw.line((x, 45, x, 620), fill='#dddddd')
        draw.text((x-10, 625), str(tick), fill='black', font=font)
    for j, row in enumerate(rows):
        y = 55+j*55
        draw.text((12, y+10), f'{row["bn_state"]}/{row["readout"]}', fill='black', font=font)
        for offset, key, color in ((0, 'train_clean_top1_pct', '#377eb8'), (20, 'selection_top1_pct', '#ff7f00')):
            value = row[key]
            draw.rectangle((300, y+offset, 300+value*7, y+offset+14), fill=color)
            draw.text((min(1010, 306+value*7), y+offset-2), f'{value:.2f}', fill='black', font=font)
    canvas.save(path)


def run_diagnostics(parent_run, output_dir, config=None, examples=12):
    """Run/resume the complete fixed-checkpoint analysis; return its summary."""
    parent, summary, state, checkpoint_hash = load_spatial_parent(parent_run)
    requested = copy.deepcopy(state['config'] if config is None else config)
    _relocation_only(state['config'], requested)
    if examples < 0 or requested['dataset']['num_workers'] < 0:
        raise ValueError('Invalid gallery/worker settings')
    batch = requested['dataset']['batch_size']
    if batch < 2 or TRAIN_IMAGES % batch:
        raise ValueError('Use a batch size >=2 dividing 80000 for equal-weight BN calibration')
    output = Path(output_dir).resolve()
    if output == parent or parent in output.parents:
        raise ValueError('Diagnostics must not write into the original training run')
    identity = dict(protocol=PROTOCOL, parent_run=str(parent), parent_checkpoint_sha256=checkpoint_hash,
        split_hash=summary['split_hash'], source_hashes=diagnostic_sources(),
        config=requested, examples=examples)
    if (output/'summary.json').exists():
        complete = json.loads((output/'summary.json').read_text(encoding='utf-8'))
        if complete.get('identity') != identity or complete.get('status') != 'complete':
            raise ValueError('Completed mechanism output does not match this request')
        return complete
    output.mkdir(parents=True, exist_ok=True)
    if (output/'request.json').exists():
        if json.loads((output/'request.json').read_text(encoding='utf-8')) != identity:
            raise ValueError('Interrupted mechanism request has different source/config/parent identity')
    else:
        if any(output.iterdir()):
            raise ValueError('Refusing a nonempty, unrecognized mechanism output directory')
        save_json(output/'request.json', identity)
    attempt = output/new_name('attempt')
    attempt.mkdir()
    train, selection, _, split_hash, manifest = clean_datasets(requested, summary['seed'], summary['split_hash'], 0)
    train_ids, selection_ids = manifest['train_clean'], manifest['selection']
    if set(train_ids) & set(selection_ids) or len(train_ids) != TRAIN_IMAGES or len(selection_ids) != SELECTION_IMAGES:
        raise ValueError('Calibration/evaluation partitions are not the required disjoint 80K/10K split')
    save_json(attempt/'evaluation_manifest.json', manifest)
    names = class_labels(train)
    save_json(attempt/'class_mapping.json', dict(enumerate(names)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_spatial_model('cnn8', summary['seed'])
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.to(device).float().eval()
    original_digest = state_digest(model)
    scores, evaluated, sensitivity_rows = {}, {}, []
    for label, current in (('original', model),):
        evaluated[label] = {}
        for split, dataset in (('train_clean', train), ('selection', selection)):
            print(f'Mechanism diagnosis: {label}/{split}; five shared-encoder readouts', flush=True)
            result = evaluate_readouts(current, dataset, output_dir=attempt/label/split,
                class_names=names, batch_size=batch, num_workers=requested['dataset']['num_workers'], device=device)
            if result['image_ids'] != manifest[split]:
                raise RuntimeError('Evaluation image order differs from the verified split manifest')
            evaluated[label][split] = result
            sensitivity_rows.append(dict(bn_state=label, split=split, **result['sensitivity']))
    calibrated, calibration = recalibrate_bn(model, train, allowed_ids=train_ids,
        batch_size=batch, num_workers=requested['dataset']['num_workers'], device=device, seed=summary['seed'])
    save_json(attempt/'bn_calibration_ids.json', calibration.pop('calibration_ids'))
    save_json(attempt/'bn_calibration.json', calibration)
    evaluated['recalibrated'] = {}
    for split, dataset in (('train_clean', train), ('selection', selection)):
        print(f'Mechanism diagnosis: recalibrated/{split}; five shared-encoder readouts', flush=True)
        result = evaluate_readouts(calibrated, dataset, output_dir=attempt/'recalibrated'/split,
            class_names=names, batch_size=batch, num_workers=requested['dataset']['num_workers'], device=device)
        if result['image_ids'] != manifest[split]:
            raise RuntimeError('Recalibrated evaluation image order differs from the split manifest')
        evaluated['recalibrated'][split] = result
        sensitivity_rows.append(dict(bn_state='recalibrated', split=split, **result['sensitivity']))
    rows, paired = [], {}
    for label in ('original', 'recalibrated'):
        scores[label] = {}
        for readout in READOUTS:
            clean = evaluated[label]['train_clean']['records'][readout]
            selected = evaluated[label]['selection']['records'][readout]
            scores[label][readout] = dict(train_clean=clean['metrics'], selection=selected['metrics'])
            baseline = evaluated['original']['selection']['records']['phase00']
            relative = _pair(baseline, selected, evaluated[label]['selection']['targets'])
            paired[f'{label}_{readout}_vs_original_phase00'] = relative
            if label == 'recalibrated':
                paired[f'recalibrated_vs_original_{readout}'] = _pair(
                    evaluated['original']['selection']['records'][readout], selected,
                    evaluated[label]['selection']['targets'])
            row = dict(bn_state=label, readout=readout,
                train_clean_top1_pct=100*clean['metrics']['accuracy'],
                selection_top1_pct=100*selected['metrics']['accuracy'],
                gap_pp=100*(clean['metrics']['accuracy']-selected['metrics']['accuracy']),
                selection_top5_pct=100*selected['metrics']['top5_accuracy'],
                selection_nll=selected['metrics']['nll'], **relative)
            rows.append(row)
        selected = evaluated[label]['selection']
        _changed_gallery(selection, attempt/label/'changed_galleries',
            evaluated['original']['selection']['records']['phase00'], selected['records']['full8'],
            selected['image_ids'], selected['targets'], names, examples)
        paired[f'{label}_full8_vs_same_state_phase00'] = _pair(
            selected['records']['phase00'], selected['records']['full8'], selected['targets'])
    if state_digest(model) != original_digest or file_hash(parent/'best.pt') != checkpoint_hash:
        raise RuntimeError('The source model/checkpoint changed during mechanism analysis')
    _write_csv(attempt/'comparison.csv', list(rows[0]), [list(row.values()) for row in rows])
    _write_csv(attempt/'phase_sensitivity.csv', list(sensitivity_rows[0]), [list(row.values()) for row in sensitivity_rows])
    save_json(attempt/'paired_summary.json', paired)
    _plot_comparison(rows, attempt/'comparison.png')
    observed = scores['original']['phase00']['selection']
    replay = dict(recorded=summary['best_selection'], observed=observed,
                  accuracy_delta_pp=100*(observed['accuracy']-summary['best_selection']['accuracy']))
    lines = ['# Fixed-checkpoint spatial and BN mechanism analysis', '',
        'No optimizer or parameter fitting was used. Every condition uses the same original cnn8 checkpoint; '
        'a separate copy receives one pass of clean training images to recalibrate BN buffers. '
        'Neither audit nor official validation is evaluated.', '',
        '| BN state | Readout | Clean train Top-1 | Selection Top-1 | Gap pp | Selection NLL | Corrected / harmed vs original00 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["bn_state"]} | {row["readout"]} | {row["train_clean_top1_pct"]:.3f}% | '
            f'{row["selection_top1_pct"]:.3f}% | {row["gap_pp"]:.3f} | {row["selection_nll"]:.6f} | '
            f'{row["corrected"]} / {row["harmed"]} |')
    lines.extend(['', '![Fixed-checkpoint comparison](comparison.png)', '',
        '## Interpretation and limits', '',
        '- The four fixed phases were declared before evaluation. Do not select the best phase after looking at selection labels.',
        '- Full8 uses all 64 positions. Its logits equal the mean of the four phase logits in eval mode; '
        'this numerical identity is checked batch by batch. Averaging probabilities is a different operation.',
        '- The five small classifier-head calls temporarily use highest FP32 matmul precision (TF32 disabled), '
        'then restore the inherited policy. Encoder precision is unchanged. This avoids TF32 input rounding '
        'breaking the phase-logit identity; the original00 replay honestly records any difference from archived evaluation.',
        '- The original head and its BN statistics were trained with phase00. These are inference interventions, '
        'not results from separately retraining five architectures.',
        '- BN recalibration uses every clean training image once in deterministically shuffled equal-size batches. '
        'Labels are ignored. All affine/convolution weights remain unchanged. Only tracked BN buffers may change.',
        '- Calibration resets statistics and averages batch moments. This does not claim exact population moments '
        'for a network whose upstream normalization changes, or prove that augmented-data BN is the sole source of the gap.',
        '- Report corrected and harmed images together. A smaller train-selection gap caused only by worse training accuracy '
        'is not evidence of improved generalization.',
        '- Any-phase-correct is an oracle diagnostic and cannot be deployed without labels.',
        '- The unchanged original phase00 replay is included in summary.json; small numerical differences from the archived '
        'training evaluation can occur across software/hardware.', '',
        '## Files', '',
        '`comparison.csv`, `phase_sensitivity.csv`, and `paired_summary.json` summarize all declared conditions. '
        'Each state/split/readout folder contains metrics, per-class recall/NLL, confusion, predictions CSV and a compact NPZ. '
        'NPZ files remain on the instance. Changed-image galleries show both corrections and harms. '
        'The calibration ID list records the exact clean training images used.', ''])
    (attempt/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    complete = dict(status='complete', identity=identity, analysis_directory=attempt.relative_to(output).as_posix(),
        scores=scores, calibration=calibration, paired=paired, selection_replay=replay,
        source_checkpoint_unchanged=True, original_model_unchanged=True,
        original_state_sha256=original_digest, split_hash=split_hash,
        audit_partition_evaluated=False, official_validation_evaluated=False)
    index = ('# Fixed-checkpoint mechanism report\n\n'
             'All five spatial readouts were evaluated before and after clean-training-only BN recalibration. '
             'No optimizer or weight fitting was used.\n\n'
             f'- [Authoritative analysis and interpretation]({attempt.name}/report.md)\n'
             f'- [Condition comparison]({attempt.name}/comparison.csv)\n'
             f'- [Phase sensitivity]({attempt.name}/phase_sensitivity.csv)\n'
             f'- [Paired corrections and harms]({attempt.name}/paired_summary.json)\n\n'
             'summary.json identifies the completed attempt. Older incomplete attempts, if any, are retained separately.\n')
    (output/'report.md').write_text(index, encoding='utf-8')
    save_json(output/'summary.json', complete)
    return complete
