"""Clean, label-preserving classification diagnostics; never fits a model.

The caller supplies a deterministic clean dataset and an FP32 model. Probabilities
are retained in one float32 array; per-image CSV records are streamed by batch.
Ranking ties are resolved by the lower class index for reproducibility.
"""
import csv
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding='utf-8')


def _base_dataset(dataset):
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    return dataset


def _image_id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Every image must have a nonempty string identifier')
    path = PurePosixPath(value.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or Path(value).is_absolute():
        raise ValueError('Image identifiers must be dataset-relative paths')
    return value


def _font(size):
    for candidate in ('C:/Windows/Fonts/arial.ttf',
                      '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _source_image(dataset, index, image_id):
    base = _base_dataset(dataset)
    folder = getattr(base, 'folder', None)
    if folder is not None and hasattr(folder, 'root'):
        root = Path(folder.root).resolve()
        source = (root / image_id).resolve()
        if not source.is_relative_to(root):
            raise ValueError('Gallery source escapes the dataset root')
        if source.is_file():
            with Image.open(source) as image:
                return image.convert('RGB')
    tensor, _, replay_id = dataset[index]
    if replay_id != image_id:
        raise ValueError('Dataset identifiers changed during gallery replay')
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError('Gallery fallback expects ImageNet-normalized RGB tensors')
    pixels = tensor.detach().float().cpu()
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    pixels = ((pixels * std + mean).clamp(0, 1) * 255).round().byte()
    return Image.fromarray(pixels.permute(1, 2, 0).numpy(), mode='RGB')


def _spread(indices, count, image_ids):
    ordered = sorted(indices, key=lambda j: image_ids[j])
    if len(ordered) <= count:
        return ordered
    return [ordered[j] for j in np.linspace(0, len(ordered)-1, count, dtype=int)]


def _galleries(dataset, output, image_ids, targets, probabilities, predictions, ranks,
               class_names, count):
    output.mkdir()
    confidence = probabilities[np.arange(len(targets)), predictions]
    wrong = np.flatnonzero(predictions != targets).tolist()
    groups = {
        'true_top5_wrong': (_spread([j for j in wrong if ranks[j] <= min(5, len(class_names))],
                                    count, image_ids), 'Systematic sample over sorted image identifiers'),
        'true_outside_top5': (_spread([j for j in wrong if ranks[j] > min(5, len(class_names))],
                                      count, image_ids), 'Systematic sample over sorted image identifiers'),
        'confident_wrong': (sorted(wrong, key=lambda j: (-float(confidence[j]), image_ids[j]))[:count],
                            'Highest predicted-class confidence; identifier breaks ties'),
        'correct_controls': (_spread(np.flatnonzero(predictions == targets).tolist(), count, image_ids),
                              'Systematic sample over sorted image identifiers'),
    }
    header_font, font = _font(18), _font(14)
    descriptions = {}
    for name, (indices, rule) in groups.items():
        descriptions[name] = dict(n=len(indices), selection_rule=rule)
        rows = max(1, math.ceil(len(indices)/4))
        canvas = Image.new('RGB', (1360, 70 + rows*270 if indices else 130), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), name, fill='black', font=header_font)
        draw.text((12, 35), rule, fill='black', font=font)
        records = []
        for position, j in enumerate(indices):
            image = _source_image(dataset, j, image_ids[j])
            image.thumbnail((160, 160), Image.Resampling.NEAREST)
            # Enlarge the original native image without inventing fine detail.
            scale = min(160/image.width, 160/image.height)
            image = image.resize((max(1, round(image.width*scale)),
                                  max(1, round(image.height*scale))), Image.Resampling.NEAREST)
            x, y = position % 4 * 340, 70 + position // 4 * 270
            canvas.paste(image, (x+(340-image.width)//2, y))
            pred, true = int(predictions[j]), int(targets[j])
            record = dict(image_id=image_ids[j], true_label=true, true_class=class_names[true],
                          predicted_label=pred, predicted_class=class_names[pred],
                          confidence=float(confidence[j]), true_rank=int(ranks[j]))
            records.append(record)
            lines = [image_ids[j], f'True: {class_names[true]}', f'Pred: {class_names[pred]}',
                     f'Confidence: {confidence[j]:.3f}   True rank: {ranks[j]}']
            for line_number, line in enumerate(lines):
                # Keep long custom identifiers inside a card; full text is in CSV.
                while draw.textbbox((0, 0), line, font=font)[2] > 326:
                    line = line[:-4] + '...'
                draw.text((x+7, y+165+line_number*21), line, fill='black', font=font)
        if not indices:
            draw.text((12, 82), 'No examples in this category.', fill='black', font=font)
        canvas.save(output / f'{name}.png')
        with (output / f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image_id', 'true_label', 'true_class',
                                    'predicted_label', 'predicted_class', 'confidence', 'true_rank'])
            writer.writeheader()
            writer.writerows(records)
    _write_json(output/'selection_rules.json', descriptions)
    return descriptions


@torch.inference_mode()
def evaluate_clean(model, dataset, *, device, batch_size, num_workers, output_dir,
                   class_names, example_count=24):
    """Evaluate clean samples and write an auditable, newly-created result bundle.

    ``class_names`` is the class-index mapping (typically ImageFolder.classes).
    ``example_count=0`` omits gallery generation. A nonempty output directory is
    rejected. The caller must make sure dataset transforms are deterministic;
    obvious training-mode datasets are rejected as an additional guard.
    """
    class_names = list(class_names)
    classes, total = len(class_names), len(dataset)
    if classes < 2 or any(not isinstance(x, str) or not x for x in class_names):
        raise ValueError('Need at least two nonempty class names')
    if len(set(class_names)) != classes:
        raise ValueError('Class names must identify unique class indices')
    if total == 0:
        raise ValueError('Empty evaluation dataset')
    if batch_size < 1 or num_workers < 0 or example_count < 0:
        raise ValueError('Invalid batch size, worker count or example count')
    if getattr(_base_dataset(dataset), 'split', None) == 'train':
        raise ValueError('Clean evaluation cannot use the augmented training dataset')
    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in model.state_dict().values()):
        raise ValueError('Clean evaluation requires FP32 model parameters and buffers')
    device = torch.device(device)
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Refusing to overwrite a nonempty diagnostic output')
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output/'classes.json', {str(i): name for i, name in enumerate(class_names)})
    probabilities = np.empty((total, classes), dtype=np.float32)
    targets = np.empty(total, dtype=np.int64)
    predictions = np.empty(total, dtype=np.int64)
    ranks = np.empty(total, dtype=np.int32)
    image_ids, seen = [], set()
    confusion = np.zeros((classes, classes), dtype=np.int64)
    class_nll = np.zeros(classes, dtype=np.float64)
    class_top5 = np.zeros(classes, dtype=np.int64)
    k, offset, nll_total = min(5, classes), 0, 0.
    model.eval()
    batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         pin_memory=device.type == 'cuda', persistent_workers=False)
    fields = ['image_id', 'true_label', 'true_class', 'predicted_label', 'predicted_class',
              'correct', 'true_rank', 'p_true', 'confidence', 'margin', 'entropy',
              'top5_labels', 'top5_classes']
    with (output/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for x, y, identifiers in tqdm(batches, desc='Clean classification', leave=False):
            if x.ndim != 4 or not torch.is_floating_point(x) or not torch.isfinite(x).all():
                raise ValueError('Expected finite floating-point image batches')
            if y.ndim != 1 or y.dtype != torch.int64 or len(y) != len(x):
                raise ValueError('Expected one int64 class index per image')
            if (y < 0).any() or (y >= classes).any():
                raise ValueError('Class index outside the supplied class mapping')
            ids = [_image_id(value) for value in identifiers]
            if len(ids) != len(x) or len(ids) != len(set(ids)) or seen.intersection(ids):
                raise ValueError('Duplicate or missing evaluation image identifiers')
            seen.update(ids)
            if offset + len(x) > total:
                raise ValueError('Evaluation yielded more examples than its declared length')
            x = x.to(device, dtype=torch.float32)
            with torch.autocast(device.type, enabled=False):
                logits = model(x)
            if not isinstance(logits, torch.Tensor) or logits.shape != (len(y), classes):
                raise ValueError('Model must produce one logit vector matching the class mapping')
            if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                raise ValueError('Model produced non-finite or non-FP32 logits')
            labels = y.to(device)
            log_probs = logits.log_softmax(-1)
            losses = -log_probs.gather(1, labels[:, None]).squeeze(1)
            if not torch.isfinite(log_probs).all() or not torch.isfinite(losses).all():
                raise ValueError('Non-finite clean evaluation log probabilities or NLL')
            probs = log_probs.exp()
            order = logits.argsort(dim=1, descending=True, stable=True)
            true_rank = (order == labels[:, None]).to(torch.int64).argmax(1)+1
            values = probs.gather(1, order[:, :k])
            pred = order[:, 0]
            entropy = -(probs * log_probs).sum(1)
            probs, top_labels = probs.cpu().numpy(), order[:, :k].cpu().numpy()
            loss_values = losses.cpu().numpy().astype(np.float64)
            rank_values, predicted = true_rank.cpu().numpy(), pred.cpu().numpy()
            confidences = values[:, 0].cpu().numpy()
            margins = (values[:, 0]-values[:, 1]).cpu().numpy()
            entropies, true_labels = entropy.cpu().numpy(), y.numpy()
            probabilities[offset:offset+len(y)] = probs
            targets[offset:offset+len(y)] = true_labels
            predictions[offset:offset+len(y)] = predicted
            ranks[offset:offset+len(y)] = rank_values
            image_ids.extend(ids)
            np.add.at(confusion, (true_labels, predicted), 1)
            np.add.at(class_nll, true_labels, loss_values)
            np.add.at(class_top5, true_labels, rank_values <= k)
            nll_total += loss_values.sum()
            for j, identifier in enumerate(ids):
                true, prediction = int(true_labels[j]), int(predicted[j])
                writer.writerow([identifier, true, class_names[true], prediction, class_names[prediction],
                                 int(true == prediction), int(rank_values[j]), f'{probs[j, true]:.9g}',
                                 f'{confidences[j]:.9g}', f'{margins[j]:.9g}', f'{entropies[j]:.9g}',
                                 json.dumps(top_labels[j].tolist()),
                                 json.dumps([class_names[i] for i in top_labels[j]], ensure_ascii=False)])
            offset += len(y)
    if offset != total:
        raise ValueError('Evaluation yielded fewer examples than its declared length')
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1., atol=1e-5):
        raise ValueError('Invalid probability normalization')
    counts, correct = confusion.sum(1), confusion.diagonal()
    present = counts > 0
    with (output/'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['true_class\\predicted_class', *class_names])
        writer.writerows([[name, *row.tolist()] for name, row in zip(class_names, confusion)])
    with (output/'per_class.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['label', 'class_name', 'n', 'correct', 'recall', 'top5_correct',
                         'top5_accuracy', 'nll'])
        for i, name in enumerate(class_names):
            n = int(counts[i])
            writer.writerow([i, name, n, int(correct[i]), float(correct[i]/n) if n else '',
                             int(class_top5[i]), float(class_top5[i]/n) if n else '',
                             float(class_nll[i]/n) if n else ''])
    with (output/'confusion_pairs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['true_label', 'true_class', 'predicted_label', 'predicted_class',
                         'count', 'fraction_of_true_class', 'reverse_count'])
        pairs = [(i, j) for i, j in zip(*np.nonzero(confusion)) if i != j]
        for i, j in sorted(pairs, key=lambda pair: (-int(confusion[pair]), *pair)):
            writer.writerow([i, class_names[i], j, class_names[j], int(confusion[i, j]),
                             float(confusion[i, j]/counts[i]), int(confusion[j, i])])
    np.savez_compressed(output/'predictions.npz', image_ids=np.asarray(image_ids), targets=targets,
                        probabilities=probabilities, predicted_labels=predictions,
                        true_ranks=ranks, class_names=np.asarray(class_names))
    result = dict(n=total, classes=classes, classes_present=int(present.sum()),
                  accuracy=float(correct.sum()/total), top5_accuracy=float(class_top5.sum()/total),
                  nll=float(nll_total/total), macro_accuracy=float((correct[present]/counts[present]).mean()),
                  top5_wrong_count=int(class_top5.sum()-correct.sum()),
                  outside_top5_count=int(total-class_top5.sum()),
                  class_counts={name: int(n) for name, n in zip(class_names, counts)},
                  precision='float32', loss='unsmoothed cross entropy; natural logarithm',
                  ranking_tie_rule='lower class index first',
                  gallery_fallback='ImageNet-denormalized tensor when source image is unavailable')
    if example_count:
        result['galleries'] = _galleries(dataset, output/'galleries', image_ids, targets,
                                         probabilities, predictions, ranks, class_names, example_count)
    _write_json(output/'metrics.json', result)
    return result
