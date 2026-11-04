"""Data provenance, image content and fixed-model generalization diagnostics.

No dataset, labels or checkpoint are changed. Only the existing development
train/selection partitions are inspected. These descriptive tests can expose
pipeline errors or sensitivities; they cannot establish semantic distribution
equivalence from low-dimensional image statistics.
"""
import copy
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from tqdm import tqdm

from baseline_diagnostics import clean_datasets, class_labels
from correction_experiment import file_hash
from datasets.utility_tiny import UtilityDataset
from literature_experiment import ROOT, new_name, save_json, environment
from main import configure_runtime
from mechanism_diagnostics import load_spatial_parent, _relocation_only
from models.attention_correction import state_digest
from models.spatial_preact import build_spatial_model
from utils.baseline_diagnostics import _font


PROTOCOL = 'tiny-generalization-data-audit-v1'
TRAIN_IMAGES, SELECTION_IMAGES, PROBE_PER_CLASS = 80000, 10000, 20
VIEWS = ('clean', 'flip', 'shift_x_minus2', 'shift_x_plus2',
         'shift_y_minus2', 'shift_y_plus2', 'crop_flip',
         'randaugment_flip', 'inherited_aug')
SOURCES = ('generalization_audit.py', 'mechanism_diagnostics.py',
           'baseline_diagnostics.py', 'datasets/utility_tiny.py',
           'datasets/resnet_tiny.py', 'models/spatial_preact.py',
           'models/literature_twostage.py', 'models/literature_alternet.py')
MEAN, STD = (.485, .456, .406), (.229, .224, .225)


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Data audit reached its monotonic deadline; partial attempt retained')


def _csv(path, rows, fields=None):
    path = Path(path)
    fields = list(rows[0]) if fields is None and rows else (fields or [])
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def average_hash(rgb):
    """64-bit grayscale average hash: bucket candidates, not duplicate proof."""
    values = np.asarray(rgb.convert('L').resize((8, 8), Image.Resampling.LANCZOS))
    bits = values >= values.mean()
    return f'{int.from_bytes(np.packbits(bits).tobytes(), "big"):016x}'


def image_record(path, identifier, target, split):
    contents = Path(path).read_bytes()
    with Image.open(io.BytesIO(contents)) as source:
        mode, image_format, size = source.mode, source.format, source.size
        rgb = source.convert('RGB')
        pixels = np.asarray(rgb, dtype=np.uint8)
        perceptual = average_hash(rgb)
    normalized = pixels.astype(np.float64)/255.
    luminance = normalized @ np.array([.2126, .7152, .0722])
    pixel_digest = hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
    return dict(image_id=identifier, label=int(target), split=split,
        width=size[0], height=size[1], original_mode=mode, original_format=image_format,
        file_bytes=len(contents), file_sha256=hashlib.sha256(contents).hexdigest(),
        rgb_pixel_sha256=pixel_digest, average_hash64=perceptual,
        red_mean=float(normalized[..., 0].mean()), green_mean=float(normalized[..., 1].mean()),
        blue_mean=float(normalized[..., 2].mean()), luminance_mean=float(luminance.mean()),
        luminance_std=float(luminance.std()))


def clean_pipeline_replay(train, count=32):
    """Feed identical source IDs through clean-train and selection code paths."""
    if not len(train) or count < 1 or getattr(train, 'split', None) != 'train_clean':
        raise ValueError('Same-image replay requires nonempty unaugmented clean training data')
    selection_view = UtilityDataset(train.folder, train.indices, 'selection', train.seed)
    positions = np.linspace(0, len(train)-1, min(count, len(train)), dtype=int)
    matches = dict(tensors=True, targets=True, image_ids=True)
    failures, max_error = [], 0.
    for position in positions:
        left, target_a, id_a = train[int(position)]
        right, target_b, id_b = selection_view[int(position)]
        tensor_match = left.dtype == right.dtype and torch.equal(left, right)
        target_match, id_match = target_a == target_b, id_a == id_b
        matches['tensors'] &= bool(tensor_match)
        matches['targets'] &= bool(target_match)
        matches['image_ids'] &= bool(id_match)
        if left.shape == right.shape:
            max_error = max(max_error, float((left.float()-right.float()).abs().max()))
        if not (tensor_match and target_match and id_match) and len(failures) < 8:
            failures.append(dict(position=int(position), clean_train_id=id_a,
                selection_replay_id=id_b, tensor_equal=bool(tensor_match),
                target_equal=bool(target_match), image_id_equal=bool(id_match)))
    return dict(images_checked=len(positions), all_equal=all(matches.values()),
        same_image_tensor_equal=matches['tensors'], same_target_equal=matches['targets'],
        same_image_id_equal=matches['image_ids'], max_absolute_tensor_difference=max_error,
        examples_of_mismatch=failures,
        scope='Systematic same-source replay through clean training and selection UtilityDataset views; no augmentation')


def scan_sources(train, selection, output, deadline=None):
    """Decode each source once; hash exactly the images used by the two loaders."""
    output = Path(output)
    rows, failures, class_counts = [], [], {}
    folders = [Path(dataset.folder.root).resolve() for dataset in (train, selection)]
    if folders[0] != folders[1] or train.folder.class_to_idx != selection.folder.class_to_idx:
        raise ValueError('Train/selection do not share the same source root and class mapping')
    root = folders[0]
    for split, dataset in (('train_clean', train), ('selection', selection)):
        counts = Counter()
        for position in tqdm(range(len(dataset)), desc=f'Source decode/hash {split}', leave=False):
            _check_deadline(deadline)
            index = dataset.indices[position]
            path, target = dataset.folder.samples[index]
            path = Path(path)
            try:
                identifier = path.relative_to(dataset.folder.root).as_posix()
                if not path.resolve().is_relative_to(root):
                    raise ValueError('Source resolves outside the shared dataset root')
                synset = Path(identifier).parts[0]
                if (dataset.folder.class_to_idx.get(synset) != target or
                        dataset.folder.targets[index] != target):
                    raise ValueError('Path, loader class mapping and target disagree')
                record = image_record(path, identifier, target, split)
                if record['width'] != 64 or record['height'] != 64:
                    raise ValueError('Expected native 64x64 source image')
                rows.append(record)
                counts[int(target)] += 1
            except Exception as exc:
                failures.append(dict(split=split, index=index, path=str(path),
                                     error_type=type(exc).__name__, error=str(exc)))
        class_counts[split] = dict(counts)
    _csv(output/'image_inventory.csv', rows)
    _csv(output/'source_failures.csv', failures,
         ['split', 'index', 'path', 'error_type', 'error'])
    if failures:
        save_json(output/'source_scan.json', dict(status='issues_found', failures=len(failures),
            successfully_decoded=len(rows), shared_root=str(root), class_counts=class_counts))
        raise ValueError('Source decoding/provenance errors found; inspect source_failures.csv')
    by_split = {split: [row for row in rows if row['split'] == split]
                for split in ('train_clean', 'selection')}
    id_sets = {split: {row['image_id'] for row in values} for split, values in by_split.items()}
    if (id_sets['train_clean'] & id_sets['selection'] or
            any(len(id_sets[split]) != len(values) for split, values in by_split.items())):
        raise ValueError('Image IDs overlap or repeat across the source partitions')
    duplicates, duplicate_counts = [], {}
    for column in ('file_sha256', 'rgb_pixel_sha256', 'average_hash64'):
        groups = defaultdict(lambda: defaultdict(list))
        for row in rows:
            groups[row[column]][row['split']].append(row)
        cross_groups = [value for value in groups.values() if 'train_clean' in value and 'selection' in value]
        duplicate_counts[column] = dict(cross_split_groups=len(cross_groups),
            train_images=sum(len(value['train_clean']) for value in cross_groups),
            selection_images=sum(len(value['selection']) for value in cross_groups),
            pair_count=sum(len(value['train_clean'])*len(value['selection']) for value in cross_groups),
            exact_duplicate_proof=column != 'average_hash64')
        # Bounded examples prevent huge all-black/low-information hash buckets
        # from creating a quadratic output. Aggregate pair_count remains exact.
        remaining = 1000
        for value in cross_groups:
            for left in value['train_clean'][:20]:
                for right in value['selection'][:20]:
                    if remaining <= 0:
                        break
                    duplicates.append(dict(kind=column, train_id=left['image_id'],
                        selection_id=right['image_id'], train_label=left['label'],
                        selection_label=right['label'], label_conflict=left['label'] != right['label'],
                        exact_duplicate_proof=column != 'average_hash64'))
                    remaining -= 1
    _csv(output/'hash_duplicates.csv', duplicates,
         ['kind', 'train_id', 'selection_id', 'train_label', 'selection_label',
          'label_conflict', 'exact_duplicate_proof'])
    statistics = []
    metrics = ('red_mean', 'green_mean', 'blue_mean', 'luminance_mean', 'luminance_std', 'file_bytes')
    for split, values in by_split.items():
        for label in sorted(train.folder.class_to_idx.values()):
            subset = [row for row in values if row['label'] == label]
            statistics.append(dict(split=split, label=label, n=len(subset),
                **{metric: float(np.mean([row[metric] for row in subset])) for metric in metrics}))
    _csv(output/'distribution_by_class.csv', statistics)
    differences = []
    for label in [-1, *sorted(train.folder.class_to_idx.values())]:
        left = [row for row in by_split['train_clean'] if label == -1 or row['label'] == label]
        right = [row for row in by_split['selection'] if label == -1 or row['label'] == label]
        for metric in metrics:
            a, b = np.array([row[metric] for row in left]), np.array([row[metric] for row in right])
            pooled = np.sqrt((a.var()+b.var())/2)
            differences.append(dict(label=label, metric=metric, train_n=len(a), selection_n=len(b),
                train_mean=float(a.mean()), selection_mean=float(b.mean()),
                standardized_mean_difference=float((b.mean()-a.mean())/pooled) if pooled else 0.))
    _csv(output/'distribution_effect_sizes.csv', differences)
    replay = clean_pipeline_replay(train)
    report = dict(status='complete' if replay['all_equal'] else 'issues_found',
        shared_root=str(root), shared_class_mapping=True,
        class_mapping=train.folder.class_to_idx, class_counts=class_counts,
        counts={key: len(value) for key, value in by_split.items()},
        ids_disjoint=True, source_failures=0, all_decoded_dimensions_64=True,
        original_modes={key: dict(Counter(row['original_mode'] for row in values)) for key, values in by_split.items()},
        formats={key: dict(Counter(row['original_format'] for row in values)) for key, values in by_split.items()},
        duplicate_counts=duplicate_counts, same_image_pipeline_replay=replay,
        caveat='Average-hash equality only produces candidates; no exhaustive perceptual nearest-neighbor search. '
               'Image moments and class balance cannot prove semantic distribution equivalence or correct labels.')
    save_json(output/'source_scan.json', report)
    if not replay['all_equal']:
        raise ValueError('Same-image clean-training/selection tensor, target or ID replay mismatch')
    return report


class AuditView(Dataset):
    """Counterfactual views sharing the inherited crop/flip/RandAugment choices."""
    def __init__(self, base, view, augmentation, epoch=101):
        if view not in VIEWS:
            raise ValueError('Unknown audit view')
        if base.split == 'train':
            raise ValueError('Audit views require the unaugmented dataset')
        self.base, self.view, self.epoch, self.split = base, view, epoch, 'diagnostic'
        self.folder, self.indices, self.seed = base.folder, base.indices, base.seed
        self.crop = T.RandomResizedCrop(64, scale=tuple(augmentation['crop_scale']),
                                      interpolation=T.InterpolationMode.BICUBIC)
        self.flip = T.RandomHorizontalFlip()
        self.randaugment = T.RandAugment(num_ops=augmentation['randaugment_ops'],
            magnitude=augmentation['randaugment_magnitude'],
            interpolation=T.InterpolationMode.BILINEAR, fill=[124, 116, 104])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, position):
        if self.view in ('crop_flip', 'randaugment_flip', 'inherited_aug'):
            index = self.indices[position]
            path, label = self.folder.samples[index]
            image = self.folder.loader(path)
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(self.seed+self.epoch*1000003+index)
                cropped = self.crop(image)
                before_flip = torch.get_rng_state()
                cropped = self.flip(cropped)
                after_flip = torch.get_rng_state()
                if self.view == 'randaugment_flip':
                    torch.set_rng_state(before_flip)
                    selected = self.flip(image)
                    torch.set_rng_state(after_flip)
                else:
                    selected = cropped
                if self.view != 'crop_flip':
                    selected = self.randaugment(selected)
                x = TF.normalize(TF.to_tensor(selected), MEAN, STD)
            identifier = Path(path).relative_to(self.folder.root).as_posix()
            return x, label, identifier
        x, label, identifier = self.base[position]
        if self.view == 'flip':
            x = x.flip(-1)
        elif self.view.startswith('shift_'):
            dx = (-2 if 'minus2' in self.view else 2) if '_x_' in self.view else 0
            dy = (-2 if 'minus2' in self.view else 2) if '_y_' in self.view else 0
            padded = TF.pad(x, [2, 2, 2, 2], padding_mode='reflect')
            x = padded[:, 2-dy:66-dy, 2-dx:66-dx]
        return x, label, identifier


@torch.inference_mode()
def evaluate_view(model, dataset, output, *, batch_size, num_workers, device, classes, deadline=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    labels, predictions, losses, identifiers = [], [], [], []
    top5_count, confusion = 0, np.zeros((classes, classes), dtype=np.int64)
    rows = []
    for images, targets, ids in DataLoader(dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=str(device).startswith('cuda'), persistent_workers=False):
        _check_deadline(deadline)
        if targets.dtype != torch.long or (targets < 0).any() or (targets >= classes).any():
            raise ValueError('Invalid audit target mapping')
        with torch.autocast(torch.device(device).type, enabled=False):
            logits = model(images.to(device, dtype=torch.float32))
        if logits.shape != (len(targets), classes) or not torch.isfinite(logits).all():
            raise ValueError('Invalid/non-finite audit logits')
        log_probs = logits.log_softmax(-1)
        y = targets.to(device)
        nll = -log_probs.gather(1, y[:, None]).squeeze(1)
        ordering = logits.argsort(dim=1, descending=True, stable=True)
        p = ordering[:, 0].cpu().numpy()
        truth = targets.numpy()
        top5_count += int((ordering[:, :min(5, classes)] == y[:, None]).any(1).sum())
        np.add.at(confusion, (truth, p), 1)
        labels.extend(truth.tolist()); predictions.extend(p.tolist())
        losses.extend(nll.cpu().double().tolist()); identifiers.extend(ids)
        rows.extend(dict(image_id=identifier, target=int(target), predicted=int(prediction),
            correct=int(target == prediction), nll=float(loss))
            for identifier, target, prediction, loss in zip(ids, truth, p, nll.cpu().tolist()))
    if not labels or len(labels) != len(dataset) or len(set(identifiers)) != len(identifiers):
        raise ValueError('Audit view has missing or duplicate image IDs')
    labels, predictions, losses = np.array(labels), np.array(predictions), np.array(losses)
    metrics = dict(n=len(labels), accuracy=float((labels == predictions).mean()),
        top5_accuracy=top5_count/len(labels), nll=float(losses.mean()))
    save_json(output/'metrics.json', metrics)
    _csv(output/'predictions.csv', rows)
    per_class = [dict(label=label, n=int(confusion[label].sum()), correct=int(confusion[label, label]),
        recall=float(confusion[label, label]/confusion[label].sum()) if confusion[label].sum() else '')
        for label in range(classes)]
    _csv(output/'per_class.csv', per_class)
    with (output/'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['true_class', *range(classes)])
        writer.writerows([[j, *row.tolist()] for j, row in enumerate(confusion)])
    return dict(metrics=metrics, image_ids=identifiers, targets=labels,
                predictions=predictions, losses=losses)


def _gallery(base, augmentation, destination, examples):
    if not examples:
        return
    selected = np.linspace(0, len(base)-1, min(examples, len(base)), dtype=int)
    views = ('clean', 'flip', 'crop_flip', 'randaugment_flip', 'inherited_aug')
    canvas = Image.new('RGB', (len(views)*220, 55+len(selected)*180), 'white')
    draw, font = ImageDraw.Draw(canvas), _font(12)
    for j, view in enumerate(views):
        draw.text((j*220+8, 8), view, fill='black', font=font)
        dataset = AuditView(base, view, augmentation)
        for i, position in enumerate(selected):
            x, label, identifier = dataset[int(position)]
            rgb = ((x*torch.tensor(STD)[:, None, None]+torch.tensor(MEAN)[:, None, None]).clamp(0, 1)*255).round().byte()
            image = Image.fromarray(rgb.permute(1, 2, 0).numpy()).resize((128, 128), Image.Resampling.NEAREST)
            canvas.paste(image, (j*220+45, 40+i*180))
            draw.text((j*220+8, 170+i*180), f'label={label} {Path(identifier).name[:19]}', fill='black', font=font)
    canvas.save(destination)


def run_data_audit(parent_run, output_dir, config=None, examples=12, previous_fusion=None, deadline=None):
    """Audit source data and fixed-model sensitivities; never optimize a model.

    ``deadline`` is an optional absolute ``time.monotonic()`` timestamp. A
    timeout preserves a partial attempt for diagnosis; a later call starts a
    separate attempt. ``previous_fusion`` is recorded for provenance only.
    """
    began = time.monotonic()
    _check_deadline(deadline)
    parent, summary, state, checkpoint_hash = load_spatial_parent(parent_run)
    requested = copy.deepcopy(state['config'] if config is None else config)
    _relocation_only(state['config'], requested)
    if examples < 0 or requested['dataset']['num_workers'] < 0:
        raise ValueError('Invalid gallery/loader settings')
    output = Path(output_dir).resolve()
    if output == parent or parent in output.parents:
        raise ValueError('Audit outputs must be outside the original model run')
    configure_runtime(requested['training'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    identity = dict(protocol=PROTOCOL, parent_run=str(parent), checkpoint_sha256=checkpoint_hash,
        source_hashes={name: file_hash(ROOT/name) for name in SOURCES}, config=requested,
        examples=examples, previous_fusion=str(previous_fusion) if previous_fusion else None,
        split_hash=summary['split_hash'], seed=summary['seed'],
        environment=environment(device, False, torch.float32))
    if (output/'summary.json').exists():
        recorded = json.loads((output/'summary.json').read_text(encoding='utf-8'))
        if recorded.get('identity') != identity or recorded.get('status') != 'complete':
            raise ValueError('Completed data audit does not match this source/config/checkpoint')
        return recorded
    output.mkdir(parents=True, exist_ok=True)
    if (output/'request.json').exists():
        if json.loads((output/'request.json').read_text(encoding='utf-8')) != identity:
            raise ValueError('Interrupted data audit identity differs from this request')
    elif any(output.iterdir()):
        raise ValueError('Refusing an unrecognized nonempty audit output')
    else:
        save_json(output/'request.json', identity)
    attempt = output/new_name('attempt')
    attempt.mkdir()

    def fatal_result(message):
        if file_hash(parent/'best.pt') != checkpoint_hash:
            raise RuntimeError('Source checkpoint changed during failed data audit')
        text = ('# Data audit found a concrete input problem\n\n'
            f'{message}\n\nTraining is not authorized by this audit result. '
            'No files or labels were modified. Inspect source_failures.csv and source_scan.json '
            'when present; fix the source/protocol before using this suite for training.\n')
        (attempt/'report.md').write_text(text, encoding='utf-8')
        (output/'report.md').write_text(f'# Data audit\n\n[Input problem]({attempt.name}/report.md)\n', encoding='utf-8')
        result = dict(status='complete', identity=identity, analysis_directory=attempt.name,
            training_allowed=False, fatal_issues=[message], view_scores={},
            elapsed_seconds=time.monotonic()-began, checkpoint_unchanged=True,
            labels_modified=False, audit_partition_evaluated=False,
            official_validation_evaluated=False)
        save_json(output/'summary.json', result)
        return result

    train, selection, names, fingerprint, manifest = clean_datasets(
        requested, summary['seed'], summary['split_hash'], 0)
    if len(train) != TRAIN_IMAGES or len(selection) != SELECTION_IMAGES:
        return fatal_result('Expected complete fixed development partitions')
    save_json(attempt/'evaluation_manifest.json', manifest)
    try:
        scan = scan_sources(train, selection, attempt, deadline=deadline)
    except ValueError as exc:
        return fatal_result(str(exc))
    expected_train, expected_selection = TRAIN_IMAGES//len(names), SELECTION_IMAGES//len(names)
    if (set(scan['class_counts']['train_clean']) != set(range(len(names))) or
            set(scan['class_counts']['selection']) != set(range(len(names))) or
            any(value != expected_train for value in scan['class_counts']['train_clean'].values()) or
            any(value != expected_selection for value in scan['class_counts']['selection'].values())):
        return fatal_result('Source class balance does not match the fixed stratified split')
    probe, replay_selection, _, replay_hash, probe_manifest = clean_datasets(
        requested, summary['seed'], summary['split_hash'], PROBE_PER_CLASS)
    if replay_hash != fingerprint or probe_manifest['selection'] != manifest['selection']:
        raise ValueError('Probe construction changed the evaluation partition')
    model = build_spatial_model('cnn8', summary['seed'])
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.to(device).float().eval()
    digest = state_digest(model)
    conditions, rows = {}, []
    for split, base in (('train_probe', probe), ('selection', replay_selection)):
        results = {}
        for view in VIEWS:
            _check_deadline(deadline)
            print(f'Data/generalization audit: {split}/{view}', flush=True)
            result = evaluate_view(model, AuditView(base, view, requested['augmentation']),
                attempt/split/view, batch_size=requested['dataset']['batch_size'],
                num_workers=requested['dataset']['num_workers'], device=device, classes=len(names), deadline=deadline)
            results[view] = result
            clean = results['clean']
            if result['image_ids'] != clean['image_ids'] or not np.array_equal(result['targets'], clean['targets']):
                raise ValueError('Audit conditions changed ordered image IDs or target mapping')
            correct_clean = clean['predictions'] == clean['targets']
            correct_view = result['predictions'] == result['targets']
            rows.append(dict(split=split, view=view, **result['metrics'],
                prediction_agreement=float((clean['predictions'] == result['predictions']).mean()),
                corrected=int((~correct_clean & correct_view).sum()),
                harmed=int((correct_clean & ~correct_view).sum()),
                delta_accuracy_pp=100*float((correct_view.astype(float)-correct_clean.astype(float)).mean())))
        conditions[split] = {view: result['metrics'] for view, result in results.items()}
        _gallery(base, requested['augmentation'], attempt/f'{split}_augmentation_examples.png', examples)
    _csv(attempt/'view_comparison.csv', rows)
    if state_digest(model) != digest or file_hash(parent/'best.pt') != checkpoint_hash:
        raise RuntimeError('Audit mutated the original model or source checkpoint')
    text = ['# Data provenance and fixed-model generalization audit', '',
        'The same Tiny ImageNet train folder supplies both fixed development partitions. '
        'No labels, files, optimizer or model weights were changed. Audit and official validation were not read.', '',
        '## Source checks', '',
        f'- Decoded {TRAIN_IMAGES} training and {SELECTION_IMAGES} selection images; shared mapping and native64 dimensions verified.',
        '- See source_scan.json for class counts, formats, original modes and cross-split hash groups.',
        '- A systematic32-image same-source replay checks that clean-train and selection views produce '
        'exactly equal tensors, targets and IDs. This checks preprocessing paths without requiring two different '
        'images to have the same prediction.',
        '- File/pixel SHA256 equality is exact content evidence. Average-hash equality is only a candidate match; '
        'low-information images can collide. No dataset cleaning is performed automatically.',
        '- Image brightness/color/contrast statistics are descriptive. Similarity cannot prove semantic equivalence, '
        'and differences alone cannot explain a generalization gap.', '',
        '## Fixed-model view sensitivity', '',
        '| Split | View | Top1 | Top5 | NLL | Prediction agreement | Corrected / harmed |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f'| {row["split"]} | {row["view"]} | {100*row["accuracy"]:.3f}% | '
            f'{100*row["top5_accuracy"]:.3f}% | {row["nll"]:.6f} | '
            f'{100*row["prediction_agreement"]:.3f}% | {row["corrected"]} / {row["harmed"]} |')
    text += ['', '## Interpretation limits', '',
        '- Clean training uses a fixed stratified probe, not an augmented training-loop metric. Selection uses all10K.',
        '- Reflection-padded translations move the input by2 pixels; they are different from changing internal readout phases. '
        'Cropped borders and rare orientation-sensitive images can change semantic evidence.',
        '- Crop/flip and RandAugment/flip views share random choices with the inherited full augmentation. '
        'The full view matches the existing per-image augmentation at data epoch101. No Mixup/CutMix is applied here, '
        'because a mixed image has soft labels and is not an ordinary hard-label accuracy comparison.',
        '- Strong augmentation can legitimately lower accuracy. A drop can expose sensitivity or removed evidence; '
        'it does not automatically mean the augmentation is harmful during training.',
        '- Validation transformations are diagnostic only; no fitting, augmentation selection or label correction uses validation images.',
        '- Source hash overlap could inflate held-out performance; it does not by itself explain low held-out accuracy.', '']
    (attempt/'report.md').write_text('\n'.join(text), encoding='utf-8')
    (output/'report.md').write_text('# Data/generalization audit\n\n'
        f'[Completed analysis]({attempt.name}/report.md)\n\n'
        f'[Source scan]({attempt.name}/source_scan.json)\n\n'
        f'[View comparison]({attempt.name}/view_comparison.csv)\n', encoding='utf-8')
    completed = dict(status='complete', identity=identity, analysis_directory=attempt.name,
        source_scan=scan, view_scores=conditions, elapsed_seconds=time.monotonic()-began,
        training_allowed=True, fatal_issues=[], model_unchanged=True,
        checkpoint_unchanged=True, labels_modified=False,
        audit_partition_evaluated=False, official_validation_evaluated=False)
    save_json(output/'summary.json', completed)
    # JSON object keys are strings on disk. Return the persisted representation
    # so initial and reused completion results have identical public schemas.
    return json.loads((output/'summary.json').read_text(encoding='utf-8'))
